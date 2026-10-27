import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabular_lab.data import load_dataset
from tabular_lab.sandbox import _inputs, docker_command, preflight, validate_code


def require_docker(image):
    try:
        preflight(image)
    except RuntimeError as exc:
        if os.environ.get("CI"):
            pytest.fail(f"Docker CI prerequisite failed: {exc}")
        pytest.skip(str(exc))


def test_codegen_syntax_and_interface():
    validate_code("def build_pipeline(seed, feature_schema):\n    return None\n")
    with pytest.raises(ValueError):
        validate_code("def wrong(): pass")
    with pytest.raises(SyntaxError):
        validate_code("def build_pipeline(:")


def test_syntax_error_preserves_code_and_reports_location_without_raw_message(task, tmp_path):
    from tabular_lab.sandbox import evaluate_code
    code = "def build_pipeline(:\n    pass\n"
    result = evaluate_code(task, load_dataset(task), code, tmp_path / "invalid", 10)
    assert result["status"] == "failed"
    assert result["safe_error"]["error_type"] == "SyntaxError"
    assert result["safe_error"]["line"] == 1 and result["safe_error"]["offset"] > 0
    assert set(result["safe_error"]) == {"error_type", "line", "offset"}
    assert (tmp_path / "invalid" / "candidate.py").read_text() == code


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_docker_boundary_has_no_api_secret_or_eval_labels(task, tmp_path, monkeypatch, device):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-in-container")
    task.device = device
    data = load_dataset(task)
    _inputs(tmp_path, data.train.X, data.train.y, data.validation.X,
            "def build_pipeline(seed, feature_schema): pass", 42, 3)
    command = docker_command(task, "sha256:fixed-image", tmp_path / "container-input",
                             tmp_path / "container-output", "test-container")
    text = " ".join(command)
    assert "--network=none" in command and "--read-only" in command
    assert "--cap-drop=ALL" in command and "--user=65534:65534" in command
    assert ("--gpus=all" in command) == (device == "gpu")
    assert "never-in-container" not in text and "DEEPSEEK" not in text
    names = {p.name for p in (tmp_path / "container-input").iterdir()}
    assert names == {"train.csv", "train_labels.npy", "eval.csv", "candidate.py", "request.json"}
    assert not any("test" in name or "eval_label" in name for name in names)


def test_missing_docker_error_is_actionable(monkeypatch):
    import tabular_lab.sandbox as sandbox
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sandbox, "_docker_candidates", lambda: [])
    with pytest.raises(RuntimeError, match="V1 needs no Docker"):
        preflight("tabular-lab-worker:0.1.0")


def test_new_docker_install_is_found_without_refreshing_parent_path(monkeypatch, tmp_path):
    import tabular_lab.sandbox as sandbox
    executable = tmp_path / "docker.exe"
    executable.write_text("discovery test fixture", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sandbox, "_docker_candidates", lambda: [tmp_path / "missing.exe", executable])
    assert sandbox.docker_executable() == str(executable)


@pytest.mark.parametrize("device,probe_result", [
    ("cpu", "unused"), ("auto", "available"), ("auto", "failed"),
    ("auto", "timeout"), ("gpu", "available"), ("gpu", "failed"), ("gpu", "timeout")])
def test_gpu_preflight_resolves_device_and_cleans_up(monkeypatch, device, probe_result):
    import tabular_lab.sandbox as sandbox
    calls = []
    monkeypatch.setattr(sandbox, "docker_executable", lambda: "docker")

    def invoke(command, **kwargs):
        calls.append(command)
        if command[1] == "info":
            return SimpleNamespace(stdout="linux\n")
        if command[1] == "image":
            return SimpleNamespace(stdout="sha256:fixed-image\n")
        if command[1] == "run":
            assert "--gpus=all" in command and "--network=none" in command
            assert "sha256:fixed-image" in command and "--user=65534:65534" in command
            if probe_result == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if probe_result == "failed":
                raise subprocess.CalledProcessError(1, command)
            return SimpleNamespace(stdout="NVIDIA test GPU\n")
        assert command[1:3] == ["rm", "-f"]
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", invoke)
    if device == "gpu" and probe_result != "available":
        with pytest.raises(RuntimeError, match="GPU requested"):
            preflight("worker:test", device)
    else:
        result = preflight("worker:test", device)
        enabled = probe_result == "available"
        assert result["device"] == ("gpu" if enabled else "cpu")
        assert result["gpu_enabled"] is enabled
        assert result["requested_device"] == device
        assert result["gpu_probe"] == ("not_requested" if device == "cpu" else
                                       "available" if enabled else "unavailable")
    if device == "cpu":
        assert len(calls) == 2
    else:
        assert calls[-1][1:3] == ["rm", "-f"]
        assert calls[-1][-1] == calls[-2][calls[-2].index("--name") + 1]


@pytest.mark.parametrize("saved_device", [None, "gpu"])
def test_prediction_restores_recorded_device_and_immutable_image(task, tmp_path, monkeypatch, saved_device):
    import numpy as np

    import tabular_lab.sandbox as sandbox
    from tabular_lab.io import write_json
    data = load_dataset(task)
    model = tmp_path / "model"
    model.mkdir()
    image = {"image_id": "sha256:saved", "image": "old-tag:removed"}
    if saved_device:
        image["device"] = saved_device
    write_json(model / "image.json", image)
    (model / "candidate.py").write_text("", encoding="utf-8")
    (model / "pipeline.joblib").write_bytes(b"not loaded on host")
    checks, executions = [], []
    monkeypatch.setattr(sandbox, "preflight", lambda image, device: checks.append((image, device)))

    def execute(config, folder, timeout, image_id):
        executions.append((config.device, image_id))
        return {"status": "ok"}

    monkeypatch.setattr(sandbox, "_execute", execute)
    monkeypatch.setattr(sandbox, "_values", lambda *args: (
        np.zeros(len(data.test.y)), np.full((len(data.test.y), 3), 1 / 3), None))
    sandbox.predict_code(model, data.test.X, {"seed": 42, "target": task.target, "classes": data.classes})
    expected = saved_device or "cpu"
    assert checks == [("sha256:saved", expected)]
    assert executions == [(expected, "sha256:saved")]


@pytest.mark.docker
def test_real_container_train_score_and_export(task, tmp_path):
    from tabular_lab.sandbox import evaluate_code, finalize_code, predict_code
    require_docker(task.docker.image)
    code = '''from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
def build_pipeline(seed, feature_schema):
    numeric = make_pipeline(SimpleImputer(), StandardScaler())
    category = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    preprocess = make_column_transformer((numeric, make_column_selector(dtype_include="number")),
                                        (category, make_column_selector(dtype_exclude="number")))
    return make_pipeline(preprocess, LogisticRegression(max_iter=1000, random_state=seed))
'''
    data = load_dataset(task)
    trial = tmp_path / "trial"
    result = evaluate_code(task, data, code, trial, 60)
    assert result["status"] == "ok", result
    model = tmp_path / "model"
    model.mkdir()
    final = finalize_code(task, data, trial, tmp_path / "final", model, 60)
    assert final["status"] == "ok", final
    predictions, probabilities = predict_code(model, data.test.X,
        {"docker_image": task.docker.image, "seed": 42, "target": task.target, "classes": data.classes,
         "feature_schema": {c: str(data.train.X[c].dtype) for c in data.columns}})
    assert len(predictions) == len(data.test.y) and probabilities.shape == (len(data.test.y), 3)


@pytest.mark.docker
def test_catboost_default_file_output_uses_scratch_directory(task, tmp_path):
    from tabular_lab.sandbox import evaluate_code
    require_docker(task.docker.image)
    code = '''from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
def build_pipeline(seed, feature_schema):
    numeric = SimpleImputer()
    category = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    preprocess = make_column_transformer((numeric, make_column_selector(dtype_include="number")),
                                        (category, make_column_selector(dtype_exclude="number")))
    return make_pipeline(preprocess, CatBoostClassifier(iterations=20, depth=3, verbose=False,
                                                       random_seed=seed, thread_count=2))
'''
    data = load_dataset(task)
    result = evaluate_code(task, data, code, tmp_path / "catboost", 60)
    assert result["status"] == "ok", result
    assert 0 <= result["metrics"]["macro_f1"] <= 1


@pytest.mark.docker
def test_timed_out_generated_code_leaves_no_container(task, tmp_path, monkeypatch):
    import tabular_lab.sandbox as sandbox
    require_docker(task.docker.image)
    suffix = uuid.uuid4().hex
    monkeypatch.setattr(sandbox.uuid, "uuid4", lambda: SimpleNamespace(hex=suffix))
    code = "def build_pipeline(seed, feature_schema):\n    while True:\n        pass\n"
    result = sandbox.evaluate_code(task, load_dataset(task), code, tmp_path / "timeout", 3)
    assert result["status"] == "timeout", result
    name = "tabular-lab-" + suffix[:16]
    containers = subprocess.run([sandbox.docker_executable(), "ps", "-a", "--filter", f"name=^/{name}$",
                                 "--format", "{{.Names}}"], capture_output=True, text=True, check=True, timeout=15)
    assert not containers.stdout.strip()


@pytest.mark.docker
@pytest.mark.parametrize("gpu_access", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_codegen_controller_repairs_failed_candidate_with_replay(task, tmp_path, monkeypatch, gpu_access):
    import tabular_lab.runner as runner
    from tabular_lab.config import LLMConfig
    from tabular_lab.io import append_event, read_json
    from tabular_lab.llm import JSONClient
    from tabular_lab.runner import run_task
    if gpu_access and os.environ.get("TABULAR_LAB_GPU_TESTS") != "1":
        pytest.skip("Set TABULAR_LAB_GPU_TESTS=1 to opt in to local GPU integration")
    require_docker(task.docker.image)
    if gpu_access:
        preflight(task.docker.image, "gpu")
        task.device = "auto"
    responses = tmp_path / "responses.jsonl"
    broken = "def build_pipeline(seed, feature_schema):\n    raise ValueError('intentional test failure')\n"
    repaired = '''from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
def build_pipeline(seed, feature_schema):
    numeric = SimpleImputer()
    category = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    preprocess = make_column_transformer((numeric, make_column_selector(dtype_include="number")),
                                        (category, make_column_selector(dtype_exclude="number")))
    return make_pipeline(preprocess, ExtraTreesClassifier(n_estimators=40, random_state=seed, n_jobs=2))
'''
    for code in (broken, repaired):
        append_event(responses, {"event": "llm_response", "content": json.dumps(
            {"action": "code", "hypothesis": "Exercise controlled repair", "code": code})})
    append_event(responses, {"event": "llm_response", "content": json.dumps(
        {"action": "stop", "reason": "Successful candidate is available"})})
    contexts = []

    class CaptureReplay(JSONClient):
        def ask(self, system, context, timeout=None, **kwargs):
            contexts.append(context)
            return super().ask(system, context, timeout, **kwargs)

    monkeypatch.setattr(runner, "JSONClient", CaptureReplay)
    task.llm = LLMConfig(provider="replay", replay_path=responses)
    task.strategy = "codegen"
    task.limits.max_trials = 5
    result = run_task(task)
    assert result["status"] == "completed", result
    assert result["trials"] == 4 and result["failed_trials"] == 1
    assert result["llm_calls"] == 3 and result["cost_cny_estimate"] == 0
    history = read_json(Path(result["path"]) / "experiments.json")
    assert history[2]["status"] == "failed" and history[3]["status"] == "ok"
    assert history[2]["spec"]["model"] == history[3]["spec"]["model"] == "generated"
    assert contexts[1]["repair"]["code"] == broken
    assert contexts[2]["best_generated"]["code"] == repaired
    assert contexts[2]["best_generated"]["metrics"] == history[3]["metrics"]
    assert contexts[2]["history"][3]["runtime"]["device_used"] == "cpu"
    assert contexts[0]["execution_limits"]["device"] == ("gpu" if gpu_access else "cpu")
    assert contexts[0]["execution_limits"]["gpu_enabled"] is gpu_access
    assert result["execution_environment"]["gpu_enabled"] is gpu_access
    saved = read_json(Path(result["path"]) / "config.json")
    assert saved["device"] == ("auto" if gpu_access else "cpu")
    assert "test_metrics" not in json.dumps(contexts)


@pytest.mark.docker
@pytest.mark.gpu
def test_real_gpu_train_export_and_fresh_process_prediction(task, tmp_path):
    import sys

    import numpy as np
    import pandas as pd

    from tabular_lab.io import read_json, write_json
    from tabular_lab.sandbox import evaluate_code, finalize_code
    if os.environ.get("TABULAR_LAB_GPU_TESTS") != "1":
        pytest.skip("Set TABULAR_LAB_GPU_TESTS=1 to opt in to local GPU integration")
    preflight(task.docker.image, "gpu")
    task.device = "auto"
    code = '''from tabular_lab.early_stopping import EarlyStoppingCatBoost
def build_pipeline(seed, feature_schema):
    return EarlyStoppingCatBoost(params={"iterations": 120, "depth": 4, "thread_count": 4,
                       "task_type": "GPU", "devices": "0", "gpu_ram_part": 0.25},
                       patience=8, random_state=seed)
'''
    data = load_dataset(task)
    trial = tmp_path / "gpu-trial"
    result = evaluate_code(task, data, code, trial, 90)
    assert result["status"] == "ok", result
    assert result["device_used"] == "gpu"
    early = result["runtime"]["early_stopping"][0]
    assert early["fit_rows"] + early["monitor_rows"] == len(data.train.y)
    assert early["refit_rows"] == len(data.train.y) and early["selection_device"] == "gpu"
    assert early["selected_iterations"] == result["runtime"]["estimators"][0]["fitted_iterations"]
    assert read_json(trial / "image.json")["device"] == "gpu"
    model = tmp_path / "gpu-model"
    model.mkdir()
    task.device = "cpu"
    final = finalize_code(task, data, trial, tmp_path / "gpu-final", model, 90)
    assert final["status"] == "ok", final
    assert final["runtime"]["device_used"] == "gpu"
    assert final["runtime"]["early_stopping"][0]["refit_rows"] == len(data.train.y) + len(data.validation.y)
    assert read_json(model / "runtime.json") == final["runtime"]
    assert read_json(model / "image.json")["device"] == "gpu"
    write_json(model / "metadata.json", {"backend": "docker", "seed": 42, "target": task.target,
               "classes": data.classes, "columns": data.columns, "id_column": None,
               "feature_schema": {c: str(data.train.X[c].dtype) for c in data.columns},
               "docker_image": "unused-tag:missing"})
    source, output = tmp_path / "gpu-input.csv", tmp_path / "gpu-predictions.csv"
    data.test.X.to_csv(source, index=False)
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    subprocess.run([sys.executable, "-m", "tabular_lab", "predict", "--model", str(model),
                    "--input", str(source), "--output", str(output)], env=env, check=True, timeout=90)
    predicted = pd.read_csv(output)
    saved = np.load(tmp_path / "gpu-final" / "predictions.npz", allow_pickle=False)
    assert predicted.prediction.tolist() == [data.classes[int(v)] for v in saved["predictions"]]
    np.testing.assert_allclose(predicted.filter(like="probability_").to_numpy(), saved["probabilities"])


@pytest.mark.docker
@pytest.mark.parametrize("verified", [False, True])
def test_container_requires_verified_catboost_early_stopping(task, tmp_path, verified):
    from tabular_lab.sandbox import evaluate_code
    require_docker(task.docker.image)
    if verified:
        code = '''from tabular_lab.early_stopping import EarlyStoppingCatBoost
def build_pipeline(seed, feature_schema):
    return EarlyStoppingCatBoost(params={"iterations": 20, "depth": 3, "thread_count": 2},
                                 patience=3, random_state=seed)
'''
    else:
        code = '''from tabular_lab.early_stopping import EarlyStoppingCatBoost
from catboost import CatBoostClassifier
from tabular_lab.models import CatFrame
class SelfMonitored(EarlyStoppingCatBoost):
    def fit(self, X, y):
        self.preprocessor_ = CatFrame().fit(X)
        features = self.preprocessor_.transform(X)
        self.model_ = CatBoostClassifier(iterations=5, depth=2, thread_count=2, verbose=False,
                         allow_writing_files=False, cat_features=self.preprocessor_.categorical_)
        self.model_.fit(features, y, eval_set=(features, y), early_stopping_rounds=2)
        self.classes_ = self.model_.classes_
        return self
def build_pipeline(seed, feature_schema):
    return SelfMonitored()
'''
    result = evaluate_code(task, load_dataset(task), code, tmp_path / "early-stopping", 60)
    if verified:
        assert result["status"] == "ok", result
        assert result["runtime"]["device_used"] == "cpu"
        assert result["runtime"]["early_stopping"][0]["refit_rows"] == 108
    else:
        assert result["status"] == "failed", result
        assert result["safe_error"]["error_type"] == "UnverifiedEarlyStoppingError"


@pytest.mark.docker
@pytest.mark.parametrize("verified", [False, True])
def test_container_audits_custom_members_and_nested_stopping(task, tmp_path, verified):
    import numpy as np

    from tabular_lab.io import read_json
    from tabular_lab.sandbox import evaluate_code, finalize_code, predict_code

    require_docker(task.docker.image)
    code = '''import numpy as np
from catboost import CatBoostClassifier
from tabular_lab.early_stopping import EarlyStoppingCatBoost
from tabular_lab.models import CatFrame
class Ensemble:
    def fit(self, X, y):
        self.members_ = [EarlyStoppingCatBoost(params={"iterations": 12, "depth": 2, "thread_count": 2},
                            patience=2, random_state=seed).fit(X, y) for seed in (1, 2)]
        self.classes_ = self.members_[0].classes_
        if not VERIFIED:
            frame = CatFrame().fit(X)
            features = frame.transform(X)
            direct = CatBoostClassifier(iterations=4, depth=2, thread_count=2, verbose=False,
                                       allow_writing_files=False, cat_features=frame.categorical_)
            direct.fit(features, y, eval_set=(features, y), early_stopping_rounds=2)
            self.members_.append(direct)
        return self
    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_[:2]], axis=0)
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
def build_pipeline(seed, feature_schema):
    return Ensemble()
'''.replace("VERIFIED", repr(verified))
    data = load_dataset(task)
    trial = tmp_path / "members"
    result = evaluate_code(task, data, code, trial, 60)
    if not verified:
        assert result["status"] == "failed", result
        assert result["safe_error"]["error_type"] == "UnverifiedEarlyStoppingError"
        return
    assert result["status"] == "ok", result
    assert result["runtime"]["complete"] and result["device_used"] == "cpu"
    assert len(result["runtime"]["estimators"]) == len(result["runtime"]["early_stopping"]) == 2
    model = tmp_path / "model"
    model.mkdir()
    final_folder = tmp_path / "final-members"
    final = finalize_code(task, data, trial, final_folder, model, 60)
    assert final["status"] == "ok", final
    runtime = read_json(model / "runtime.json")
    assert runtime == final["runtime"] and runtime["complete"]
    assert all(entry["refit_rows"] == len(data.train.y) + len(data.validation.y)
               for entry in runtime["early_stopping"])
    metadata = {"seed": task.seed, "target": task.target, "classes": data.classes,
                "feature_schema": {c: str(data.train.X[c].dtype) for c in data.columns},
                "docker_image": task.docker.image}
    predictions, probabilities = predict_code(model, data.test.X, metadata)
    with np.load(final_folder / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(predictions, saved["predictions"])
        np.testing.assert_allclose(probabilities, saved["probabilities"])
