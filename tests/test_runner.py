import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import joblib
import pandas as pd

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
    task.llm = LLMConfig(provider="deepseek")
    result = run_task(task)
    assert result["trials"] == 3 and result["llm_calls"] == 1
    assert len(contexts[0]["history"]) == 2
    serialized = json.dumps(contexts[0])
    assert "test_metrics" not in serialized and "test_rows" not in serialized
    assert str(task.data_path) not in serialized and "class-0" not in serialized
    assert "fake-only" not in (Path(result["path"]) / "events.jsonl").read_text()


def test_missing_api_key_preserves_model_and_reports_partial(task, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    task.llm = LLMConfig(provider="deepseek")
    result = run_task(task)
    assert result["status"] == "partial"
    assert "DEEPSEEK_API_KEY" in result["stop_reason"]
    assert (Path(result["path"]) / "model" / "pipeline.joblib").exists()


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
