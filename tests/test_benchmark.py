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
