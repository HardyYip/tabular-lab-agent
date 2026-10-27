from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from .config import TaskConfig, load_task
from .data import load_dataset
from .io import append_event, code_version, read_json, write_json
from .runner import run_task


def _plans(suite_path):
    suite_path = suite_path.resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    allowed = {"tasks", "seeds", "strategies", "alternate_strategy_order", "provider", "output_dir"}
    if not isinstance(suite, dict) or set(suite) - allowed:
        raise ValueError("invalid benchmark suite fields")
    for key, default in (("tasks", []), ("seeds", [17, 42, 73]), ("strategies", ["baseline", "random", "agent"])):
        values = suite.get(key, default)
        if not isinstance(values, list) or not values or len(set(map(str, values))) != len(values):
            raise ValueError(f"suite {key} must be a nonempty list without duplicates")
    if any(not isinstance(name, str) or not name.strip() for name in suite["tasks"]):
        raise ValueError("suite tasks must be file paths")
    if any(type(seed) is not int or seed < 0 for seed in suite.get("seeds", [17, 42, 73])):
        raise ValueError("suite seeds must be nonnegative integers")
    if any(not isinstance(strategy, str) or strategy not in {"baseline", "random", "agent", "codegen"}
           for strategy in suite.get("strategies", ["baseline", "random", "agent"])):
        raise ValueError("unknown benchmark strategy")
    if type(suite.get("alternate_strategy_order", False)) is not bool:
        raise ValueError("alternate_strategy_order must be a boolean")
    if not isinstance(suite.get("output_dir", "../runs/benchmark"), str):
        raise ValueError("benchmark output_dir must be a path string")
    if len(suite["tasks"]) * len(suite.get("seeds", [17, 42, 73])) * len(
            suite.get("strategies", ["baseline", "random", "agent"])) > 1000:
        raise ValueError("benchmark suite exceeds the 1000-run safety limit")
    output = (suite_path.parent / suite.get("output_dir", "../runs/benchmark")).resolve()
    plans = []
    for task_file in suite["tasks"]:
        for seed_index, seed in enumerate(suite.get("seeds", [17, 42, 73])):
            strategies = list(suite.get("strategies", ["baseline", "random", "agent"]))
            if suite.get("alternate_strategy_order") and seed_index % 2:
                strategies.reverse()
            for strategy in strategies:
                values = load_task(suite_path.parent / task_file).model_dump()
                values.update(seed=seed, strategy=strategy, run_dir=output)
                if suite.get("provider"):
                    values["llm"]["provider"] = suite["provider"]
                plans.append((task_file, TaskConfig.model_validate(values)))
    if len({Path(name).stem for name in suite["tasks"]}) != len(suite["tasks"]):
        raise ValueError("benchmark task filenames must have distinct stems")
    return output, plans


def plan_benchmark(suite_path):
    output, plans = _plans(suite_path)
    return {"output_dir": str(output), "runs": len(plans), "new_api_requests": 0,
            "configured_budget_cny": sum(task.limits.budget_cny for _, task in plans
                if task.strategy in {"agent", "codegen"} and task.llm.provider == "deepseek"),
            "plan": [{"task": name, "seed": task.seed, "strategy": task.strategy,
                      "provider": task.llm.provider} for name, task in plans]}


@contextmanager
def _lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".benchmark.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another benchmark is using this output directory") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _images(plans):
    from .sandbox import preflight
    images = {task.docker.image for _, task in plans if task.strategy == "codegen"}
    return {image: preflight(image)["image_id"] for image in sorted(images)}


def _protocol(plans):
    inputs = {}
    for name, task in plans:
        if name not in inputs:
            inputs[name] = load_dataset(task).hashes
    return {"schema_version": 1, "package_source_sha256": code_version()["package_source_sha256"],
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "machine": platform.machine(), "versions": {name: importlib.metadata.version(name)
                                for name in ("numpy", "pandas", "scikit-learn", "catboost", "pydantic",
                                             "httpx", "pillow")}},
            "inputs": inputs, "images": _images(plans),
            "plans": [{"task_file": name, "config": task.model_dump(mode="json")} for name, task in plans]}


def _row(index, name, task, summary, wall_seconds):
    label = "offline_mock_agent" if task.strategy == "agent" and task.llm.provider == "mock" else task.strategy
    return {"run_index": index, "task": Path(name).stem, "seed": task.seed, "strategy": label,
            "status": summary["status"], "run_path": summary["path"], "seconds": summary["seconds"],
            "trials": summary["trials"], "wall_seconds": wall_seconds,
            "wall_seconds_measured": wall_seconds is not None,
            "failed_trials": summary["failed_trials"], "cost_cny_estimate": summary["cost_cny_estimate"],
            "usage_unknown": summary.get("usage_unknown", False), "llm_calls": summary["llm_calls"],
            "stop_reason": summary["stop_reason"], "test_macro_f1": summary.get("test_metrics", {}).get("macro_f1"),
            "test_accuracy": summary.get("test_metrics", {}).get("accuracy"),
            "test_roc_auc": summary.get("test_metrics", {}).get("roc_auc")}


def _saved_summary(folder, name, task, protocol):
    if not (folder / "summary.json").is_file():
        raise RuntimeError("interrupted run has no final summary; inspect usage before starting a new protocol")
    manifest = read_json(folder / "manifest.json")
    if (read_json(folder / "config.json") != task.model_dump(mode="json") or
            manifest["package_source_sha256"] != protocol["package_source_sha256"] or
            manifest.get("data_hashes") != protocol["inputs"][name] or
            (task.strategy == "codegen" and manifest.get("execution_environment", {}).get("image_id") !=
             protocol["images"][task.docker.image])):
        raise ValueError("saved run does not match the benchmark protocol")
    summary = read_json(folder / "summary.json")
    if Path(summary["path"]).resolve() != folder.resolve():
        raise ValueError("saved run path differs from the checkpoint")
    return summary


def run_benchmark(suite_path: Path, resume=False):
    output, plans = _plans(suite_path)
    with _lock(output):
        occupied = any(path.name != ".benchmark.lock" for path in output.iterdir())
        if occupied and not resume:
            raise FileExistsError("benchmark output is not empty; use --resume or a new output directory")
        protocol = _protocol(plans)
        protocol_file = output / "protocol.json"
        if resume:
            if not protocol_file.is_file() or read_json(protocol_file) != protocol:
                raise ValueError("resume requires matching source, data, images, configuration and run order")
        else:
            write_json(protocol_file, protocol)
        rows = read_json(output / "results.json") if resume and (output / "results.json").exists() else []
        if not isinstance(rows, list) or len(rows) > len(plans):
            raise ValueError("invalid benchmark checkpoint")
        for index, row in enumerate(rows):
            name, task = plans[index]
            folder = output / f"run-{index:04d}"
            summary = _saved_summary(folder, name, task, protocol)
            expected = _row(index, name, task, summary, row.get("wall_seconds"))
            if row != expected:
                raise ValueError("checkpoint row differs from saved run")
            if row["usage_unknown"]:
                raise RuntimeError("Benchmark stopped because API usage is unknown; inspect saved results.")
        for index in range(len(rows), len(plans)):
            name, task = plans[index]
            folder = output / f"run-{index:04d}"
            if folder.exists():
                summary = _saved_summary(folder, name, task, protocol)
                elapsed = None
            else:
                started = time.monotonic()
                summary = run_task(task, output=folder, expected_protocol={
                    "data_hashes": protocol["inputs"][name],
                    "package_source_sha256": protocol["package_source_sha256"],
                    "image_id": protocol["images"].get(task.docker.image)})
                elapsed = time.monotonic() - started
            rows.append(_row(index, name, task, summary, elapsed))
            _save(output, rows)
            if summary.get("usage_unknown"):
                raise RuntimeError("Benchmark stopped because API usage is unknown; inspect saved results.")
        _save(output, rows)
    return output / "results.csv"


def _save(output, rows):
    write_json(output / "results.json", rows)
    csv_temp = output / "results.csv.tmp"
    with csv_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temp, output / "results.csv")
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame["test_macro_f1"] = frame["test_macro_f1"].where(frame["status"] == "completed")
    aggregate = frame.groupby(["task", "strategy"]).agg(
        seeds=("seed", "count"), macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"), seconds_mean=("seconds", "mean"),
        cost_mean=("cost_cny_estimate", "mean"), failed_trials=("failed_trials", "sum"),
        successful_runs=("status", lambda values: int((values == "completed").sum())))
    aggregate.to_csv(output / "aggregate.csv.tmp")
    os.replace(output / "aggregate.csv.tmp", output / "aggregate.csv")


def curate_run(source: Path, destination: Path):
    """Publish only explicitly selected summaries, never arbitrary logs or absolute local paths."""
    summary = read_json(source / "summary.json")
    for key in ("path", "finalization"):
        summary.pop(key, None)
    write_json(destination / "summary.json", summary)
    experiments = read_json(source / "experiments.json")
    clean = [{k: row[k] for k in ("trial", "spec", "fingerprint", "status", "metrics", "seconds") if k in row}
             for row in experiments]
    write_json(destination / "experiments.json", clean)
    manifest = read_json(source / "manifest.json")
    write_json(destination / "manifest.json", manifest)
    # These built-in example reports contain no raw training rows or filenames.
    (destination / "report.md").write_text((source / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
    # Small public audit trace; prompts, raw content, error strings and machine paths stay private.
    import json
    public_log = destination / "events.sample.jsonl"
    public_log.write_text("", encoding="utf-8")
    allowed = {"time", "event", "trial", "spec", "fingerprint", "status", "metrics", "seconds",
               "device_used", "metric", "value", "call", "provider", "model", "usage", "cost_cny",
               "spent_cny", "budget_cny", "reserved_cny", "reservation_cny", "error_type"}
    for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        append_event(public_log, {key: value for key, value in event.items() if key in allowed})
