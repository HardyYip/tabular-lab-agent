from __future__ import annotations

import csv
from pathlib import Path

import yaml

from .config import load_task
from .io import append_event, read_json, write_json
from .runner import run_task


def run_benchmark(suite_path: Path):
    suite_path = suite_path.resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    output = (suite_path.parent / suite.get("output_dir", "../runs/benchmark")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Never compare test scores while proposing experiments; each run sees only its validation split.
    for task_file in suite["tasks"]:
        for seed in suite.get("seeds", [17, 42, 73]):
            for strategy in suite.get("strategies", ["baseline", "random", "agent"]):
                task = load_task(suite_path.parent / task_file)
                task.seed, task.strategy = seed, strategy
                task.run_dir = output
                if suite.get("provider"):
                    task.llm.provider = suite["provider"]
                if strategy == "agent" and task.llm.provider == "mock":
                    label = "offline_mock_agent"
                else:
                    label = strategy
                summary = run_task(task)
                row = {"task": Path(task_file).stem, "seed": seed, "strategy": label,
                       "status": summary["status"], "run_path": summary["path"],
                       "seconds": summary["seconds"], "trials": summary["trials"],
                       "failed_trials": summary["failed_trials"], "cost_cny_estimate": summary["cost_cny_estimate"],
                       "llm_calls": summary["llm_calls"], "stop_reason": summary["stop_reason"],
                       "test_macro_f1": summary.get("test_metrics", {}).get("macro_f1"),
                       "test_accuracy": summary.get("test_metrics", {}).get("accuracy")}
                rows.append(row)
                _save(output, rows)
    return output / "results.csv"


def _save(output, rows):
    write_json(output / "results.json", rows)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame["test_macro_f1"] = frame["test_macro_f1"].where(frame["status"] == "completed")
    aggregate = frame.groupby(["task", "strategy"]).agg(
        seeds=("seed", "count"), macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"), seconds_mean=("seconds", "mean"),
        cost_mean=("cost_cny_estimate", "mean"), failed_trials=("failed_trials", "sum"),
        successful_runs=("status", lambda values: int((values == "completed").sum())))
    aggregate.to_csv(output / "aggregate.csv")


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
