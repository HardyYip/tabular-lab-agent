from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from tabular_lab.config import ExperimentSpec, TaskConfig
from tabular_lab.data import load_dataset
from tabular_lab.io import read_json, write_json
from tabular_lab.search import allowed_fingerprints


def main():
    parser = argparse.ArgumentParser(description="Audit recorded V1 results without loading models or calling APIs.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rows = read_json(args.input / "results.json")
    assert rows, "No recorded runs"
    keys = {(r["task"], r["seed"], r["strategy"]) for r in rows}
    assert len(keys) == len(rows), "Duplicate task/seed/strategy records"
    references = {}
    calls, cost, failures = 0, 0.0, 0
    for row in rows:
        folder = Path(row["run_path"])
        summary = read_json(folder / "summary.json")
        task = TaskConfig.model_validate(read_json(folder / "config.json"))
        manifest = read_json(folder / "manifest.json")
        history = read_json(folder / "experiments.json")
        splits = read_json(folder / "splits.json")
        data = load_dataset(task)
        assert summary["status"] == "completed", f"Incomplete run: {row['task']}/{row['seed']}/{row['strategy']}"
        assert manifest["data_hashes"] == data.hashes, "Dataset changed after experiment"
        assert all(splits[name] == getattr(data, name).indices for name in ("train", "validation", "test"))
        identity = (data.hashes, splits)
        key = (row["task"], row["seed"])
        assert references.setdefault(key, identity) == identity, "Strategies used different data or splits"
        assert len(history) <= task.limits.max_trials
        best = max((h for h in history if h["status"] == "ok"), key=lambda h: h["metrics"][task.metric])
        assert best["trial"] == summary["best_trial"], "Winner does not match validation selection"
        for candidate in history[2:]:
            assert ExperimentSpec.model_validate(candidate["spec"]).fingerprint() in allowed_fingerprints()
        with np.load(folder / "final" / "predictions.npz", allow_pickle=False) as predictions:
            labels, probabilities = predictions["predictions"], predictions["probabilities"]
            assert len(labels) == len(data.test.y)
            expected = {"macro_f1": f1_score(data.test.y, labels, average="macro", zero_division=0),
                        "accuracy": accuracy_score(data.test.y, labels)}
            if len(data.classes) == 2:
                expected["roc_auc"] = roc_auc_score(data.test.y, probabilities[:, 1])
        for name, value in expected.items():
            assert abs(value - summary["test_metrics"][name]) < 1e-12, f"Incorrect saved {name}"
        assert abs(row["test_macro_f1"] - expected["macro_f1"]) < 1e-12
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        responses = [e for e in events if e.get("event") == "llm_response"]
        assert len(responses) == summary["llm_calls"] and not summary["usage_unknown"]
        assert abs(sum(e.get("cost_cny", 0) for e in responses) - summary["cost_cny_estimate"]) < 1e-12
        for event in events:
            if event.get("event") == "llm_request":
                context = json.loads(event["messages"][1]["content"])
                assert not {"test", "test_metrics", "test_rows"}.intersection(context)
        if task.strategy in {"baseline", "random"}:
            assert summary["llm_calls"] == 0 and summary["cost_cny_estimate"] == 0
        calls += summary["llm_calls"]
        cost += summary["cost_cny_estimate"]
        failures += summary["failed_trials"]
    report = {"audited_runs": len(rows), "task_seed_pairs": len(references), "llm_calls": calls,
              "estimated_cost_cny": cost, "failed_trials": failures,
              "data_hashes_and_splits_match": True, "matched_splits_between_strategies": True,
              "validation_based_winner_verified": True, "test_metrics_independently_recomputed": True,
              "candidate_registry_and_usage_ledger_verified": True, "model_objects_loaded_on_host": False}
    write_json(args.input / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
