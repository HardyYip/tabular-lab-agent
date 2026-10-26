"""V2 Docker boundary: no key, no network, no validation labels in code containers."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .models import score_predictions
from .process import run_process


def _docker_candidates() -> list[Path]:
    if os.name != "nt":
        return []
    candidates = []
    if local := os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(local) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe")
    if program_files := os.environ.get("ProgramFiles"):
        candidates.append(Path(program_files) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe")
    return candidates


def docker_executable() -> str:
    """Find a fresh Desktop install even when this parent process has a stale PATH."""
    if executable := shutil.which("docker"):
        return executable
    for candidate in _docker_candidates():
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("V2 requires Docker Desktop with a Linux engine. Install/start it; V1 needs no Docker.")


def preflight(image: str) -> dict:
    executable = docker_executable()
    try:
        info = subprocess.run([executable, "info", "--format", "{{.OSType}}"],
                              capture_output=True, text=True, timeout=15, check=True)
        if info.stdout.strip() != "linux":
            raise RuntimeError("V2 requires Linux containers, not Windows containers")
        inspect = subprocess.run([executable, "image", "inspect", image, "--format", "{{.Id}}"],
                                 capture_output=True, text=True, timeout=15, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError("Docker engine/image unavailable. Run: docker build -t " + image +
                           " -f docker/Dockerfile .") from exc
    return {"engine": "linux", "image": image, "image_id": inspect.stdout.strip()}


def validate_code(code: str):
    # Syntax/interface check only; Docker provides process/filesystem/resource restrictions.
    tree = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "build_pipeline" for node in tree.body):
        raise ValueError("source must define build_pipeline(seed, feature_schema)")


def docker_command(task, image_id: str, input_dir: Path, output_dir: Path, name: str,
                   executable: str = "docker"):
    return [executable, "run", "--rm", "--name", name, "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128",
            "--memory=" + task.docker.memory, "--memory-swap=" + task.docker.memory,
            "--cpus=" + str(task.docker.cpus), "--user=65534:65534",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=256m", "--ulimit=nofile=256:256",
            "--env=OMP_NUM_THREADS=" + str(task.docker.cpus), "--env=OPENBLAS_NUM_THREADS=1",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONHASHSEED=" + str(task.seed),
            "--mount", f"type=bind,source={input_dir.resolve()},target=/input,readonly",
            "--mount", f"type=bind,source={output_dir.resolve()},target=/output",
            image_id]


def _execute(task, folder, timeout, image_id):
    name = "tabular-lab-" + uuid.uuid4().hex[:16]
    input_dir, output_dir = folder / "container-input", folder / "container-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o777)  # dedicated bind mount writable by the nonroot container UID on Linux
    executable = docker_executable()
    command = docker_command(task, image_id, input_dir, output_dir, name, executable)
    started = time.monotonic()
    try:
        result = run_process(command, folder / "docker.log", timeout, task.docker.cpus)
    finally:
        # Killing the docker CLI alone does not necessarily stop its container.
        try:
            subprocess.run([executable, "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15, check=False)
        except (OSError, subprocess.SubprocessError):
            # Preserve the original failure and make cleanup problems observable.
            write_json(folder / "cleanup-warning.json", {"container": name, "cleanup_failed": True})
    result["seconds"] = time.monotonic() - started
    if result["status"] == "ok" and (output_dir / "error.json").exists():
        result = {"status": "failed", "seconds": result["seconds"],
                  "safe_error": read_json(output_dir / "error.json")}
    return result


def _inputs(folder, X_train, y_train, X_eval, code, seed, class_count, action="train"):
    input_dir = folder / "container-input"
    input_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(input_dir / "train.csv", index=False)
    np.save(input_dir / "train_labels.npy", y_train, allow_pickle=False)
    X_eval.to_csv(input_dir / "eval.csv", index=False)
    (input_dir / "candidate.py").write_text(code, encoding="utf-8")
    write_json(input_dir / "request.json", {"action": action, "seed": seed, "class_count": class_count,
               "feature_schema": {c: str(X_train[c].dtype) for c in X_train.columns}})


def _values(folder, class_count, y=None):
    # Host reads bounded numeric arrays, never pickle generated by a candidate.
    path = folder / "container-output" / "predictions.npz"
    if path.is_symlink():
        raise ValueError("candidate artifacts must not be symbolic links")
    if path.stat().st_size > 100_000_000:
        raise ValueError("candidate prediction artifact is too large")
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 100_000_000:
            raise ValueError("uncompressed prediction artifact is too large")
    with np.load(path, allow_pickle=False) as values:
        predictions = values["predictions"]
        probabilities = values["probabilities"]
        if (predictions.ndim != 1 or probabilities.shape != (len(predictions), class_count) or
                not np.isin(predictions, np.arange(class_count)).all() or
                not np.isfinite(probabilities).all() or (probabilities < 0).any() or
                (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("invalid candidate prediction arrays")
        if y is not None:
            metrics = score_predictions(y, predictions, probabilities, class_count)
        else:
            metrics = None
    return predictions, probabilities, metrics


def evaluate_code(task, dataset, code, folder, timeout):
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        validate_code(code)
        image = preflight(task.docker.image)
        write_json(folder / "image.json", image)
        _inputs(folder, dataset.train.X, dataset.train.y, dataset.validation.X, code,
                task.seed, len(dataset.classes))
        result = _execute(task, folder, timeout, image["image_id"])
        if result["status"] == "ok":
            artifact = folder / "container-output" / "pipeline.joblib"
            if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size > 200_000_000:
                raise ValueError("missing or invalid model artifact")
            _, _, metrics = _values(folder, len(dataset.classes), dataset.validation.y)
            result["metrics"] = metrics
        return result
    except (ValueError, SyntaxError, OSError, RuntimeError) as exc:
        return {"status": "failed", "safe_error": {"error_type": type(exc).__name__},
                "seconds": time.monotonic() - started}


def finalize_code(task, dataset, candidate, folder, model_folder, timeout):
    code = (candidate / "container-input" / "candidate.py").read_text(encoding="utf-8")
    image_id = read_json(candidate / "image.json")["image_id"]
    _inputs(folder, pd.concat([dataset.train.X, dataset.validation.X], ignore_index=True),
            np.concatenate([dataset.train.y, dataset.validation.y]), dataset.test.X, code,
            task.seed, len(dataset.classes))
    result = _execute(task, folder, timeout, image_id)
    if result["status"] == "ok":
        try:
            artifact = folder / "container-output" / "pipeline.joblib"
            if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size > 200_000_000:
                raise ValueError("invalid final model artifact")
            predictions, probabilities, metrics = _values(folder, len(dataset.classes), dataset.test.y)
        except (ValueError, OSError) as exc:
            result = {"status": "failed", "seconds": result["seconds"],
                      "safe_error": {"error_type": type(exc).__name__}}
    source = folder if result["status"] == "ok" else candidate
    # Preserve the already-fitted candidate if refitting fails; never load it on the host.
    shutil.copy2(source / "container-output" / "pipeline.joblib", model_folder / "pipeline.joblib")
    (model_folder / "candidate.py").write_text(code, encoding="utf-8")
    write_json(model_folder / "image.json", {"image_id": image_id, "image": task.docker.image})
    if result["status"] == "ok":
        result["metrics"] = metrics
        np.savez(folder / "predictions.npz", predictions=predictions, probabilities=probabilities)
    return result


def predict_code(model, frame, metadata):
    from .config import TaskConfig
    preflight(metadata["docker_image"])
    image_id = read_json(model / "image.json")["image_id"]
    # Use a local temporary workspace; no mounts of user data directories or API credentials.
    with tempfile.TemporaryDirectory(prefix="tabular-lab-predict-") as temporary:
        folder = Path(temporary)
        inputs = folder / "container-input"
        inputs.mkdir()
        frame.to_csv(inputs / "eval.csv", index=False)
        shutil.copy2(model / "candidate.py", inputs / "candidate.py")
        shutil.copy2(model / "pipeline.joblib", inputs / "pipeline.joblib")
        write_json(inputs / "request.json", {"action": "predict", "seed": metadata["seed"],
                   "class_count": len(metadata["classes"]),
                   "feature_schema": metadata.get("feature_schema", {})})
        task = TaskConfig(data_path=Path("unused"), target=metadata["target"], seed=metadata["seed"])
        result = _execute(task, folder, 120, image_id)
        if result["status"] != "ok":
            raise RuntimeError("container prediction failed: " + str(result.get("safe_error", result["status"])))
        predictions, probabilities, _ = _values(folder, len(metadata["classes"]))
        if len(predictions) != len(frame):
            raise ValueError("candidate prediction row count mismatch")
        return predictions, probabilities
