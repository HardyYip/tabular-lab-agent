import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from tabular_lab.config import ExperimentSpec, LLMConfig
from tabular_lab.io import read_json
from tabular_lab.llm import JSONClient
from tabular_lab.runner import run_task
from tabular_lab.search import baselines


def response(content):
    return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 50},
                                    "choices": [{"message": {"content": json.dumps(content)}}]})


def test_rejected_plans_get_feedback_then_canonical_id_choice_succeeds(task, monkeypatch):
    import tabular_lab.runner as runner

    contexts, selected = [], []

    def handler(request):
        context = json.loads(json.loads(request.content)["messages"][1]["content"])
        contexts.append(context)
        if len(contexts) == 1:
            return response({"action": "experiment", "experiment": ExperimentSpec(
                model="catboost", params={"depth": 9}, hypothesis="outside registry").model_dump()})
        if len(contexts) == 2:
            return response({"action": "experiment", "experiment": baselines()[0].model_dump()})
        if len(contexts) == 3:
            option = context["candidate_options"][0]
            selected.append(option)
            return response({"action": "experiment", "candidate_id": option["candidate_id"],
                             "hypothesis": "Use an offered untried option after reading rejection feedback"})
        return response({"action": "stop", "reason": "enough for offline verification"})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    task.llm = LLMConfig(provider="deepseek", timeout_seconds=5)
    task.limits.max_trials = 4
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(runner, "JSONClient", lambda config, budget, log: JSONClient(config, budget, log, client))
        summary = run_task(task)
    assert summary["status"] == "completed" and summary["trials"] == 3 and summary["llm_calls"] == 4
    assert "outside shared candidate_space" in contexts[1]["planning_feedback"][-1]["error"]
    assert contexts[2]["planning_feedback"][-1]["error"] == "duplicate experiment"
    assert contexts[3]["planning_feedback"] == []
    assert selected[0]["candidate_id"] not in {item["candidate_id"] for item in contexts[3]["candidate_options"]}
    history = read_json(Path(summary["path"]) / "experiments.json")
    expected = ExperimentSpec.model_validate(selected[0]["experiment"])
    assert history[-1]["fingerprint"] == expected.fingerprint()
    assert history[-1]["spec"]["params"] == expected.params
    assert all("candidate_space" not in context for context in contexts)
    assert "test_metrics" not in json.dumps(contexts) and str(task.data_path) not in json.dumps(contexts)


def test_three_invalid_ids_still_stop_without_expanding_search_space(task, monkeypatch):
    import tabular_lab.runner as runner

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    task.llm = LLMConfig(provider="deepseek", timeout_seconds=5)
    with httpx.Client(transport=httpx.MockTransport(lambda request: response({
        "action": "experiment", "candidate_id": "0" * 64, "hypothesis": "not offered"}))) as client:
        monkeypatch.setattr(runner, "JSONClient", lambda config, budget, log: JSONClient(config, budget, log, client))
        summary = run_task(task)
    assert summary["status"] == "partial" and summary["stop_reason"] == "three_invalid_plans"
    assert summary["llm_calls"] == 3 and summary["trials"] == 2
    assert summary["finalization"]["status"] == "ok"


def test_empty_candidate_space_stops_before_request(task, monkeypatch):
    import tabular_lab.runner as runner

    monkeypatch.setattr(runner, "search_space", lambda: [])
    task.llm = LLMConfig(provider="deepseek", timeout_seconds=5)
    summary = run_task(task)
    assert summary["status"] == "completed" and summary["stop_reason"] == "search_space_exhausted"
    assert summary["llm_calls"] == 0 and summary["trials"] == 2


def test_response_overrun_does_not_start_a_shortened_candidate(task, monkeypatch):
    import tabular_lab.llm as llm_module
    import tabular_lab.runner as runner

    clock = [1000.0]
    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(runner, "time", timer)
    monkeypatch.setattr(llm_module, "time", timer)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    actual_trial = runner._trial

    def trial(*args, **kwargs):
        result = actual_trial(*args, **kwargs)
        clock[0] += 0 if kwargs.get("final") else 30
        return result

    def handler(request):
        context = json.loads(json.loads(request.content)["messages"][1]["content"])
        clock[0] += 61
        return response({"action": "experiment", "candidate_id": context["candidate_options"][0]["candidate_id"],
                         "hypothesis": "a valid plan arriving too late for a full training window"})

    monkeypatch.setattr(runner, "_trial", trial)
    task.llm = LLMConfig(provider="deepseek")
    task.limits.max_seconds = 180
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(runner, "JSONClient", lambda config, budget, log: JSONClient(config, budget, log, client))
        summary = run_task(task)
    assert summary["status"] == "completed" and summary["stop_reason"] == "search_time_limit"
    assert summary["llm_calls"] == 1 and summary["trials"] == 2
    assert summary["finalization"]["status"] == "ok" and not summary["usage_unknown"]
    events = [json.loads(line) for line in (Path(summary["path"]) / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "experiment_start" for event in events) == 2
    assert next(event for event in events if event["event"] == "trial_time_stop")["available_seconds"] == 29
