
import json
from types import SimpleNamespace

import httpx
import pytest

from tabular_lab.config import LLMConfig
from tabular_lab.io import append_event
from tabular_lab.llm import JSONClient, PlanningTimeLimit, StopRun


def fake_client(body, status=200):
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)))


def body(content='{"action":"stop","reason":"done"}'):
    return {"model": "actual-model-version", "id": "fake-response", "usage": {
        "prompt_tokens": 100, "completion_tokens": 50, "prompt_cache_hit_tokens": 90},
        "choices": [{"message": {"content": content}}]}


def test_accounting_uses_actual_usage_and_no_cache_discount(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret")
    log = tmp_path / "calls.jsonl"
    llm = JSONClient(LLMConfig(provider="deepseek"), 20, log, fake_client(body()))
    assert llm.ask("json only", {})["action"] == "stop"
    assert llm.spent == pytest.approx(llm.cost(100, 50))
    text = log.read_text()
    assert "unit-test-secret" not in text
    assert "actual-model-version" in text


def test_zero_budget_does_not_issue_request(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    llm = JSONClient(LLMConfig(provider="deepseek"), 0, tmp_path / "log", fake_client(body()))
    with pytest.raises(StopRun, match="budget"):
        llm.ask("json", {})
    assert llm.calls == 0


@pytest.mark.parametrize("response,status", [({"choices": []}, 200), ({"error": "rate limited"}, 429),
                                            ({"error": "insufficient balance"}, 402)])
def test_unknown_usage_stops_further_calls(tmp_path, monkeypatch, response, status):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "log", fake_client(response, status))
    with pytest.raises(StopRun):
        llm.ask("json", {})
    assert llm.usage_unknown
    with pytest.raises(StopRun):
        llm.ask("json", {})
    assert llm.calls == 1
    events = [json.loads(line) for line in llm.log.read_text().splitlines()]
    assert events[-1]["http_status"] == status
    assert events[-1]["elapsed_seconds"] >= 0


@pytest.mark.parametrize("available,expected", [(None, 60), (3, 3), (90, 60)])
def test_request_timing_matches_transport_and_records_success(tmp_path, monkeypatch, available, expected):
    import tabular_lab.llm as llm_module

    clock = [100.0]
    monkeypatch.setattr(llm_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret")
    observed = []

    def handler(request):
        observed.append(request.extensions["timeout"])
        clock[0] += 0.25
        return httpx.Response(200, json=body())

    log = tmp_path / "calls.jsonl"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = JSONClient(LLMConfig(provider="deepseek"), 20, log, client)
        assert llm.ask("json", {}, timeout=available)["action"] == "stop"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    request, response = events
    assert observed == [{phase: expected for phase in ("connect", "read", "write", "pool")}]
    assert request["available_seconds"] == available
    assert request["configured_timeout_seconds"] == 60
    assert request["timeout_seconds"] == expected
    assert request["call"] == response["call"] == 1
    assert response["elapsed_seconds"] == pytest.approx(0.25)
    assert not llm.usage_unknown and llm.spent == pytest.approx(llm.cost(100, 50))
    assert "unit-test-secret" not in log.read_text()


def test_invalid_json_still_billed(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "log", fake_client(body("")))
    with pytest.raises(ValueError, match="invalid JSON"):
        llm.ask("json", {})
    assert llm.spent > 0 and not llm.usage_unknown


def test_replay_has_zero_new_cost(tmp_path):
    original = tmp_path / "original.jsonl"
    append_event(original, {"event": "llm_response", "content": '{"action":"stop"}',
                            "usage": {"prompt_tokens": 999}})
    llm = JSONClient(LLMConfig(provider="replay", replay_path=original), 0, tmp_path / "replay.jsonl")
    assert llm.ask("json", {}) == {"action": "stop"}
    assert llm.spent == 0 and llm.calls == 1
    with pytest.raises(StopRun, match="exhausted"):
        llm.ask("json", {})


@pytest.mark.parametrize("available,allowed", [(89.9, False), (90, True), (91, True)])
def test_deadline_requires_full_request_and_training_window(tmp_path, monkeypatch, available, allowed):
    import tabular_lab.llm as module

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=body())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "log", client)
        if allowed:
            llm.ask("json", {}, deadline=1000 + available, reserve_seconds=30)
            assert requests[0].extensions["timeout"]["read"] == 60
        else:
            with pytest.raises(PlanningTimeLimit):
                llm.ask("json", {}, deadline=1000 + available, reserve_seconds=30)
        assert len(requests) == llm.calls == int(allowed)
        assert not llm.usage_unknown


def test_deadline_rechecked_after_credential_lookup_without_issuing_request(tmp_path, monkeypatch):
    import tabular_lab.llm as module

    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def key():
        clock[0] += 2
        return "fake"

    monkeypatch.setattr(module, "get_api_key", key)
    with fake_client(body()) as client:
        llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "log", client)
        with pytest.raises(PlanningTimeLimit):
            llm.ask("json", {}, deadline=1091, reserve_seconds=30)
    assert llm.calls == 0 and llm.spent == 0 and not llm.usage_unknown
    events = [json.loads(line) for line in llm.log.read_text().splitlines()]
    assert len(events) == 1 and events[0]["event"] == "planning_time_stop"


@pytest.mark.parametrize("timeout", [0, -1])
def test_nonpositive_timeout_does_not_fall_back_to_configured_timeout(tmp_path, timeout):
    with fake_client(body()) as client:
        llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "log", client)
        with pytest.raises(PlanningTimeLimit):
            llm.ask("json", {}, timeout=timeout)
    assert llm.calls == 0 and not llm.usage_unknown


def test_read_timeout_records_elapsed_and_stops_subsequent_calls(tmp_path, monkeypatch):
    import tabular_lab.llm as module

    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    requests = []

    def handler(request):
        requests.append(request)
        clock[0] += 3.125
        raise httpx.ReadTimeout("simulated response delay", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "log", client)
        with pytest.raises(StopRun, match="ReadTimeout"):
            llm.ask("json", {}, timeout=3)
        with pytest.raises(StopRun, match="previous API call has unknown usage"):
            llm.ask("json", {})
    events = [json.loads(line) for line in llm.log.read_text().splitlines()]
    assert events[-1]["elapsed_seconds"] == 3.125 and events[-1]["http_status"] is None
    assert events[0]["timeout_seconds"] == requests[0].extensions["timeout"]["read"] == 3
    assert events[-1]["reserved_cny"] > 0 and llm.spent == 0 and llm.usage_unknown
    assert llm.calls == len(requests) == 1
