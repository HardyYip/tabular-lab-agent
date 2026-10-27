import json

import pandas as pd
import pytest
import yaml

from tabular_lab import benchmark
from tabular_lab.io import code_version, read_json, write_json


@pytest.fixture
def suite(task, tmp_path, monkeypatch):
    task.strategy = "baseline"
    monkeypatch.setattr(benchmark, "load_task", lambda path: task.model_copy(deep=True))
    path = tmp_path / "suite.yaml"
    output = tmp_path / "results"
    path.write_text(yaml.safe_dump({"tasks": ["task.yaml"], "seeds": [17, 42],
                                    "strategies": ["baseline"], "output_dir": str(output)}))
    calls = []

    def run(config, output, **kwargs):
        calls.append(config.seed)
        output.mkdir()
        summary = {"status": "completed", "path": str(output), "seconds": 1, "trials": 2,
                   "failed_trials": 0, "cost_cny_estimate": 0, "llm_calls": 0,
                   "stop_reason": "fixture", "usage_unknown": False, "test_metrics": {"macro_f1": 0.5}}
        write_json(output / "summary.json", summary)
        write_json(output / "config.json", config.model_dump(mode="json"))
        write_json(output / "manifest.json", {**code_version(), "data_hashes": benchmark.load_dataset(config).hashes})
        return summary

    monkeypatch.setattr(benchmark, "run_task", run)
    return path, output, calls, run, task


def test_resume_skips_saved_runs_and_never_overwrites_fresh_output(suite, monkeypatch):
    path, output, calls, run, _ = suite

    def interrupt(config, output, **kwargs):
        if config.seed == 42:
            raise KeyboardInterrupt()
        return run(config, output)

    monkeypatch.setattr(benchmark, "run_task", interrupt)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(path)
    saved = (output / "results.json").read_bytes()
    with pytest.raises(FileExistsError, match="not empty"):
        benchmark.run_benchmark(path)
    assert (output / "results.json").read_bytes() == saved
    monkeypatch.setattr(benchmark, "run_task", run)
    benchmark.run_benchmark(path, resume=True)
    assert calls == [17, 42]
    benchmark.run_benchmark(path, resume=True)
    assert calls == [17, 42]
    assert len(read_json(output / "results.json")) == 2


@pytest.mark.parametrize("changed", ["source", "data", "config"])
def test_resume_rejects_protocol_changes_without_new_execution(suite, monkeypatch, changed):
    path, _, calls, _, task = suite
    benchmark.run_benchmark(path)
    if changed == "source":
        monkeypatch.setattr(benchmark, "code_version", lambda: {"package_source_sha256": "changed"})
    elif changed == "data":
        frame = pd.read_csv(task.data_path)
        frame.loc[0, "x0"] += 0.01
        frame.to_csv(task.data_path, index=False)
    else:
        task.limits.max_trials += 1
    with pytest.raises(ValueError, match="matching source"):
        benchmark.run_benchmark(path, resume=True)
    assert calls == [17, 42]


def test_resume_preserves_unknown_usage_stop(suite, monkeypatch):
    path, output, calls, run, _ = suite

    def unknown(config, output, **kwargs):
        summary = run(config, output)
        summary.update(status="partial", usage_unknown=True)
        write_json(output / "summary.json", summary)
        return summary

    monkeypatch.setattr(benchmark, "run_task", unknown)
    with pytest.raises(RuntimeError, match="usage is unknown"):
        benchmark.run_benchmark(path)
    saved = (output / "results.json").read_bytes()
    with pytest.raises(RuntimeError, match="usage is unknown"):
        benchmark.run_benchmark(path, resume=True)
    assert calls == [17] and (output / "results.json").read_bytes() == saved


def test_completed_orphan_is_recovered_but_incomplete_run_is_not_retried(suite, monkeypatch):
    path, output, calls, _, _ = suite
    save = benchmark._save
    monkeypatch.setattr(benchmark, "_save", lambda *args: (_ for _ in ()).throw(OSError("interrupted save")))
    with pytest.raises(OSError):
        benchmark.run_benchmark(path)
    assert calls == [17]
    monkeypatch.setattr(benchmark, "_save", save)
    benchmark.run_benchmark(path, resume=True)
    assert calls == [17, 42]
    assert read_json(output / "results.json")[0]["wall_seconds"] is None
    (output / "results.json").write_text("[]")
    (output / "run-0000" / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="interrupted run has no final summary"):
        benchmark.run_benchmark(path, resume=True)
    assert calls == [17, 42]


def test_suite_lock_rejects_concurrent_writers_and_releases(suite):
    _, output, _, _, _ = suite
    with benchmark._lock(output):
        with pytest.raises(RuntimeError, match="another benchmark"):
            with benchmark._lock(output):
                pass
    with benchmark._lock(output):
        pass


def test_dry_run_is_read_only_and_rejects_duplicate_seeds(suite):
    path, output, calls, _, _ = suite
    plan = benchmark.plan_benchmark(path)
    assert plan["runs"] == 2 and plan["new_api_requests"] == 0
    assert not output.exists() and calls == []
    values = yaml.safe_load(path.read_text())
    values["seeds"] = [17, 17]
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="without duplicates"):
        benchmark.plan_benchmark(path)


def test_checkpoint_tampering_is_rejected(suite):
    path, output, calls, _, _ = suite
    benchmark.run_benchmark(path)
    rows = read_json(output / "results.json")
    rows[0]["test_macro_f1"] = 1
    (output / "results.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="checkpoint row"):
        benchmark.run_benchmark(path, resume=True)
    assert calls == [17, 42]


def test_input_change_after_protocol_creation_is_rejected_before_execution(task, tmp_path):
    from tabular_lab.runner import run_task

    destination = tmp_path / "not-started"
    with pytest.raises(ValueError, match="changed since benchmark"):
        run_task(task, output=destination, expected_protocol={
            "data_hashes": {"data": "different"},
            "package_source_sha256": code_version()["package_source_sha256"], "image_id": None})
    assert not destination.exists()


@pytest.mark.parametrize("seed", [True, "42", -1])
def test_suite_seed_types_are_not_silently_coerced(suite, seed):
    path, _, _, _, _ = suite
    values = yaml.safe_load(path.read_text())
    values["seeds"] = [seed]
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="nonnegative integers"):
        benchmark.plan_benchmark(path)


def test_oversized_suite_is_rejected_before_loading_tasks(suite, monkeypatch):
    path, _, _, _, _ = suite
    values = yaml.safe_load(path.read_text())
    values["seeds"] = list(range(1001))
    path.write_text(yaml.safe_dump(values))
    monkeypatch.setattr(benchmark, "load_task", lambda path: pytest.fail("oversized suite was expanded"))
    with pytest.raises(ValueError, match="1000-run"):
        benchmark.plan_benchmark(path)
