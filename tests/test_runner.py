import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import joblib
import pandas as pd
import pytest

from tabular_lab.config import LLMConfig
from tabular_lab.data import load_dataset
from tabular_lab.io import read_json
from tabular_lab.llm import JSONClient
from tabular_lab.process import run_process, worker_environment
from tabular_lab.runner import predict, run_task
from tabular_lab.search import search_space


def test_export_and_prediction_in_fresh_process(task, tmp_path):
    task.strategy = "baseline"
    result = run_task(task)
    assert result["status"] == "completed" and result["trials"] == 2
    run = Path(result["path"])
    assert result["test_metrics"]["macro_f1"] > 0.5
    original = pd.read_csv(task.data_path).drop(columns="target").iloc[:12]
    original.loc[0, "category"] = "new-category"
    source = tmp_path / "new.csv"
    original.to_csv(source, index=False)
    output = tmp_path / "predictions.csv"
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    subprocess.run([sys.executable, "-m", "tabular_lab", "predict", "--model", str(run / "model"),
                    "--input", str(source), "--output", str(output)], env=env, check=True, timeout=40)
    data = pd.read_csv(output)
    assert len(data) == 12 and data.id.tolist() == original.id.tolist()
    model = joblib.load(run / "model" / "pipeline.joblib")
    schema = read_json(run / "model" / "metadata.json")
    expected = model.predict(original[schema["columns"]])
    assert data.prediction.tolist() == [schema["classes"][int(i)] for i in expected]
    assert all((run / name).exists() for name in
               ("config.json", "manifest.json", "splits.json", "events.jsonl", "report.md"))
    assert len(set(load_dataset(task).test.indices) & set(load_dataset(task).train.indices)) == 0


def test_feedback_loop_with_mocked_http_and_no_test_context(task, monkeypatch):
    import tabular_lab.runner as runner
    contexts = []
    candidate = next(s for s in search_space() if s.model == "extra_trees")

    def handler(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][1]["content"])
        contexts.append(context)
        content = {"action": "experiment", "experiment": candidate.model_dump()}
        return httpx.Response(200, json={"model": "fake", "usage": {"prompt_tokens": 100,
            "completion_tokens": 100}, "choices": [{"message": {"content": json.dumps(content)}}]})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-only")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(runner, "JSONClient", lambda config, budget, log: JSONClient(config, budget, log, client))
    task.llm = LLMConfig(provider="deepseek", timeout_seconds=5)
    result = run_task(task)
    assert result["trials"] == 3 and result["llm_calls"] == 1
    assert len(contexts[0]["history"]) == 2
    serialized = json.dumps(contexts[0])
    assert "test_metrics" not in serialized and "test_rows" not in serialized
    assert str(task.data_path) not in serialized and "class-0" not in serialized
    assert "fake-only" not in (Path(result["path"]) / "events.jsonl").read_text()


def test_missing_api_key_preserves_model_and_reports_partial(task, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    task.llm = LLMConfig(provider="deepseek", timeout_seconds=5)
    result = run_task(task)
    assert result["status"] == "partial"
    assert "DEEPSEEK_API_KEY" in result["stop_reason"]
    assert (Path(result["path"]) / "model" / "pipeline.joblib").exists()


@pytest.mark.parametrize("strategy", ["agent", "codegen"])
def test_short_search_window_stops_before_request_and_preserves_export(task, monkeypatch, strategy):
    import tabular_lab.llm as llm_module
    import tabular_lab.runner as runner
    import tabular_lab.sandbox as sandbox

    clock = [1000.0]
    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(runner, "time", timer)
    monkeypatch.setattr(llm_module, "time", timer)
    monkeypatch.setattr(sandbox, "preflight", lambda image, device: {
        "device": "cpu", "gpu_enabled": False, "gpu_devices": []})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-only")
    actual_trial = runner._trial
    trial_timeouts, requests = [], []

    def timed_trial(*args, **kwargs):
        trial_timeouts.append(args[4])
        result = actual_trial(*args, **kwargs)
        clock[0] += 0.5 if kwargs.get("final") else 43.5
        return result

    def handler(request):
        requests.append(request.extensions["timeout"])
        raise AssertionError("short search windows must not issue a request")

    monkeypatch.setattr(runner, "_trial", timed_trial)
    task.llm = LLMConfig(provider="deepseek")
    task.strategy = strategy
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = None

        def make_client(config, budget, log):
            nonlocal llm
            llm = JSONClient(config, budget, log, client)
            return llm

        monkeypatch.setattr(runner, "JSONClient", make_client)
        summary = run_task(task)
    folder = Path(summary["path"])
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    stopped = next(event for event in events if event["event"] == "planning_time_stop")
    assert requests == []
    assert stopped["available_seconds"] == 3
    assert stopped["required_seconds"] == 90 and stopped["trial_reserve_seconds"] == 30
    assert not any(event["event"] in {"llm_request", "llm_usage_unknown"} for event in events)
    assert summary["status"] == "completed" and not summary["usage_unknown"]
    assert summary["stop_reason"] == "planning_time_limit"
    assert summary["llm_calls"] == 0 and summary["trials"] == 2
    assert summary["cost_cny_estimate"] == 0
    assert summary["finalization"]["status"] == "ok"
    assert trial_timeouts[-1] == 30
    assert read_json(folder / "model" / "metadata.json")["refit_on_train_validation"]
    assert (folder / "model" / "pipeline.joblib").exists()
    assert "fake-only" not in (folder / "events.jsonl").read_text()


def test_failed_final_refit_preserves_train_only_model(task, monkeypatch, tmp_path):
    import tabular_lab.runner as runner
    actual = runner._trial

    def failing_final(*args, **kwargs):
        if kwargs.get("final"):
            return {"status": "timeout", "seconds": 0, "error": "injected timeout"}
        return actual(*args, **kwargs)

    monkeypatch.setattr(runner, "_trial", failing_final)
    task.strategy = "baseline"
    result = run_task(task)
    assert result["status"] == "partial" and "test_metrics" not in result
    model = Path(result["path"]) / "model"
    assert not read_json(model / "metadata.json")["refit_on_train_validation"]
    assert predict(model, task.data_path, tmp_path / "fallback.csv") == 180


def test_timeout_and_secret_free_worker_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("ANOTHER_ACCESS_TOKEN", "secret")
    environment = worker_environment(2)
    assert "DEEPSEEK_API_KEY" not in environment and "ANOTHER_ACCESS_TOKEN" not in environment
    result = run_process([sys.executable, "-c", "import time; time.sleep(10)"],
                         tmp_path / "timeout.log", timeout=0.3)
    assert result["status"] == "timeout"
