import json
import multiprocessing
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from tabular_lab.http_request import RequestDeadlineExceeded, post_json


@pytest.fixture
def server():
    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/slow":
                content = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content[:1])
                self.wfile.flush()
                stop.wait(10)
                return
            content = b"x" * 2048 if self.path == "/large" else json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    instance.daemon_threads = True
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}"
    finally:
        stop.set()
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def test_isolated_request_returns_bounded_response(server):
    response = post_json(server + "/ok", {}, "local-test-key", 10)
    assert response.status_code == 200 and response.json() == {"ok": True}


def test_total_deadline_kills_slow_request_without_leaving_child(server):
    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(RequestDeadlineExceeded):
        post_json(server + "/slow", {}, "local-test-key", 2)
    assert time.monotonic() - started < 5
    assert {child.pid for child in multiprocessing.active_children()} <= before


def test_response_byte_limit_is_enforced(server):
    with pytest.raises(httpx.TransportError):
        post_json(server + "/large", {}, "local-test-key", 10, max_bytes=1024)


def test_deadline_failure_is_unknown_usage_and_never_retried(tmp_path, monkeypatch):
    import tabular_lab.http_request as request_module
    from tabular_lab.config import LLMConfig
    from tabular_lab.llm import JSONClient, StopRun

    calls = []

    def expired(*args):
        calls.append(True)
        raise RequestDeadlineExceeded("simulated total timeout")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-local-key")
    monkeypatch.setattr(request_module, "post_json", expired)
    llm = JSONClient(LLMConfig(provider="deepseek"), 20, tmp_path / "events.jsonl")
    with pytest.raises(StopRun, match="RequestDeadlineExceeded"):
        llm.ask("json", {})
    with pytest.raises(StopRun, match="previous API call has unknown usage"):
        llm.ask("json", {})
    assert calls == [True] and llm.calls == 1 and llm.usage_unknown
    events = [json.loads(line) for line in llm.log.read_text().splitlines()]
    assert events[0]["total_deadline_enforced"] is True
    assert "fake-local-key" not in llm.log.read_text()
