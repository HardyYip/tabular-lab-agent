import pytest
import yaml

from tabular_lab import benchmark
from tabular_lab.io import read_json


def test_unknown_api_usage_stops_entire_suite_after_saving_partial_result(task, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "load_task", lambda path: task.model_copy(deep=True))

    def run(config):
        calls.append(config.seed)
        return {"status": "partial", "path": "private-run", "seconds": 1, "trials": 2,
                "failed_trials": 0, "cost_cny_estimate": 0, "llm_calls": 1,
                "stop_reason": "unknown usage", "usage_unknown": True}

    monkeypatch.setattr(benchmark, "run_task", run)
    output = tmp_path / "output"
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"tasks": ["task.yaml"], "seeds": [17, 42],
                                    "strategies": ["agent"], "output_dir": str(output)}))
    with pytest.raises(RuntimeError, match="usage is unknown"):
        benchmark.run_benchmark(suite)
    assert calls == [17]
    rows = read_json(output / "results.json")
    assert len(rows) == 1 and rows[0]["usage_unknown"] is True
    assert rows[0]["wall_seconds"] >= 0


def test_paired_comparison_alternates_order_and_preserves_limits(task, tmp_path, monkeypatch):
    task.llm.provider = "deepseek"
    calls = []
    monkeypatch.setattr(benchmark, "load_task", lambda path: task.model_copy(deep=True))

    def run(config):
        calls.append((config.seed, config.strategy, config.limits.model_dump()))
        return {"status": "completed", "path": "fixture-run", "seconds": 1, "trials": 2,
                "failed_trials": 0, "cost_cny_estimate": 0, "llm_calls": 0,
                "stop_reason": "fixture", "test_metrics": {"macro_f1": 0.5, "accuracy": 0.5}}

    monkeypatch.setattr(benchmark, "run_task", run)
    suite = tmp_path / "paired.yaml"
    suite.write_text(yaml.safe_dump({"tasks": ["task.yaml"], "seeds": [17, 42, 73],
                     "strategies": ["agent", "codegen"], "alternate_strategy_order": True,
                     "output_dir": str(tmp_path / "results")}))
    benchmark.run_benchmark(suite)
    assert [(seed, strategy) for seed, strategy, _ in calls] == [
        (17, "agent"), (17, "codegen"), (42, "codegen"), (42, "agent"),
        (73, "agent"), (73, "codegen")]
    assert all(limits == task.limits.model_dump() for _, _, limits in calls)
