from __future__ import annotations

import multiprocessing
import time

import httpx


class RequestDeadlineExceeded(httpx.TimeoutException):
    pass


class ResponseLimitExceeded(httpx.TransportError):
    pass


def _request(connection, url, payload, key, timeout, max_bytes):
    try:
        with httpx.Client(follow_redirects=False, timeout=timeout) as client:
            with client.stream("POST", url, json=payload, headers={"Authorization": f"Bearer {key}"}) as response:
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise ResponseLimitExceeded("response exceeds byte limit")
                    chunks.append(chunk)
                result = {"status": response.status_code, "body": b"".join(chunks)}
    except Exception as exc:
        result = {"error": type(exc).__name__}
    try:
        connection.send(result)
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def post_json(url, payload, key, timeout, max_bytes=1_000_000):
    if timeout <= 0:
        raise RequestDeadlineExceeded("request deadline exhausted")
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_request, args=(child, url, payload, key, timeout, max_bytes), daemon=True)
    try:
        process.start()
        child.close()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not parent.poll(remaining):
            raise RequestDeadlineExceeded("total request deadline exceeded; usage may be unknown")
        try:
            result = parent.recv()
        except (EOFError, OSError) as exc:
            raise httpx.TransportError("request process exited without a response") from exc
        if time.monotonic() > deadline:
            raise RequestDeadlineExceeded("total request deadline exceeded; usage may be unknown")
        if "error" in result:
            error = (ResponseLimitExceeded if result["error"] == "ResponseLimitExceeded" else
                     getattr(httpx, result["error"], httpx.TransportError))
            if not isinstance(error, type) or not issubclass(error, httpx.HTTPError):
                error = httpx.TransportError
            raise error("request failed; response unavailable")
        return httpx.Response(result["status"], content=result["body"], request=httpx.Request("POST", url))
    finally:
        parent.close()
        child.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if not process.is_alive():
                process.close()
