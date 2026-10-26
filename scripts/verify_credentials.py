"""Check that a locally saved DeepSeek key reaches the HTTP header; no network request is made."""
import os
import sys
import tempfile
from pathlib import Path

import httpx

from tabular_lab.config import LLMConfig
from tabular_lab.credentials import CredentialError, get_api_key
from tabular_lab.llm import JSONClient, StopRun


def main() -> int:
    try:
        key = get_api_key()
    except CredentialError as exc:
        print(f"FAIL: {exc}")
        return 1
    if not key:
        print("FAIL: no DeepSeek key configured; run python -m tabular_lab auth login")
        return 1
    source = "environment" if os.environ.get("DEEPSEEK_API_KEY", "").strip() else "credential-store"
    seen = []

    def handler(request):
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"message": {"content": '{"action": "stop"}'}}]})

    transport = httpx.Client(transport=httpx.MockTransport(handler))
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "events.jsonl"
        client = JSONClient(LLMConfig(provider="deepseek"), 10.0, log, transport)
        try:
            reply = client.ask("return json", {})
        except StopRun as exc:
            print(f"FAIL: {exc}")
            return 1
        log_text = log.read_text(encoding="utf-8")
    transport.close()
    checks = {
        "response_parsed": reply == {"action": "stop"},
        "header_uses_saved_key": seen == [f"Bearer {key}"],
        "log_does_not_contain_key": key not in log_text,
    }
    print(f"key source: {source}")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
