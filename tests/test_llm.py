
import httpx
import pytest

from tabular_lab.config import LLMConfig
from tabular_lab.io import append_event
from tabular_lab.llm import JSONClient, StopRun


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


@pytest.mark.parametrize("response,status", [({"choices": []}, 200), ({"error": "rate limited"}, 429)])
def test_unknown_usage_stops_further_calls(tmp_path, monkeypatch, response, status):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "log", fake_client(response, status))
    with pytest.raises(StopRun):
        llm.ask("json", {})
    assert llm.usage_unknown
    with pytest.raises(StopRun):
        llm.ask("json", {})
    assert llm.calls == 1


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
