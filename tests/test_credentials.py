import json

import httpx
import pytest
from typer.testing import CliRunner

from tabular_lab import credentials
from tabular_lab.cli import app
from tabular_lab.config import LLMConfig
from tabular_lab.llm import JSONClient, StopRun


def test_environment_override_does_not_access_store(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", " env-key ")
    monkeypatch.setattr(credentials, "read_saved_key", lambda: pytest.fail("store should not be accessed"))
    assert credentials.get_api_key() == "env-key"


def test_blank_environment_falls_back_to_saved_credential(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "  ")
    monkeypatch.setattr(credentials, "read_saved_key", lambda: "saved-test-key")
    assert credentials.get_api_key() == "saved-test-key"


def test_saved_credential_used_for_http_without_logging_or_exporting_it(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "read_saved_key", lambda: "saved-test-key")

    def handler(request):
        assert request.headers["Authorization"] == "Bearer saved-test-key"
        return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "choices": [{"message": {"content": json.dumps({"action": "stop"})}}]})

    log = tmp_path / "events.jsonl"
    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = JSONClient(LLMConfig(provider="deepseek"), 1, log, client)
    assert llm.ask("json only", {}) == {"action": "stop"}
    assert "saved-test-key" not in log.read_text()
    from tabular_lab.process import worker_environment
    assert "DEEPSEEK_API_KEY" not in worker_environment(1)


def test_store_failure_stops_before_http_or_accounting(tmp_path, monkeypatch):
    def failing_store():
        raise credentials.CredentialError("Windows credential read failed (error 1312).")

    monkeypatch.setattr(credentials, "read_saved_key", failing_store)
    llm = JSONClient(LLMConfig(provider="deepseek"), 1, tmp_path / "events.jsonl")
    with pytest.raises(StopRun, match="credential read failed"):
        llm.ask("json", {})
    assert llm.calls == 0 and llm.spent == 0


def test_auth_login_hides_input_and_status_never_reveals_it(monkeypatch):
    saved = []

    class FakeStore:
        def write(self, key):
            saved.append(key)

    monkeypatch.setattr(credentials, "WindowsCredentialStore", FakeStore)
    runner = CliRunner()
    result = runner.invoke(app, ["auth", "login"], input="one-time-test-key\n")
    assert result.exit_code == 0, result.output
    assert saved == ["one-time-test-key"] and "one-time-test-key" not in result.output
    monkeypatch.setattr(credentials, "read_saved_key", lambda: saved[0])
    status = runner.invoke(app, ["auth", "status"])
    assert status.exit_code == 0 and "one-time-test-key" not in status.output


def test_auth_logout_and_missing_status(monkeypatch):
    deleted = []
    monkeypatch.setattr(credentials, "delete_api_key", lambda: deleted.append(True) or True)
    runner = CliRunner()
    assert runner.invoke(app, ["auth", "logout"]).exit_code == 0
    assert deleted == [True]
    assert runner.invoke(app, ["auth", "status"]).exit_code == 1


@pytest.mark.parametrize("key", ["", "a\nb", "a\x00b", "a b", "a" * 2561, "密钥"])
def test_invalid_key_does_not_echo_secret(key):
    with pytest.raises(credentials.CredentialError) as error:
        credentials._validate_key(key)
    assert str(error.value) == "API key must be a nonempty ASCII token without whitespace (max 2560 bytes)."
