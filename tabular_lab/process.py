"""Bounded trusted worker processes. This is timeout isolation, not a security sandbox."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil


def worker_environment(workers: int) -> dict[str, str]:
    allowed = {"PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
               "COMSPEC", "PATHEXT", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "LANG"}
    result = {key: value for key, value in os.environ.items() if key in allowed}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result[name] = str(workers)
    result["PYTHONUTF8"] = "1"
    return result


def kill_tree(pid: int):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs(children + [parent], timeout=5)
    except psutil.NoSuchProcess:
        pass


def run_process(command: list[str], log: Path, timeout: float, workers=4, cwd=None):
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env=worker_environment(workers), cwd=cwd)
        try:
            code = process.wait(timeout=max(0.1, timeout))
        except subprocess.TimeoutExpired:
            kill_tree(process.pid)
            process.wait(timeout=5)
            return {"status": "timeout", "error": f"exceeded {timeout:.2f}s"}
    if code != 0:
        # Local logs remain private unless users explicitly curate them for publication.
        return {"status": "failed", "error": log.read_text(encoding="utf-8", errors="replace")[-4000:]}
    return {"status": "ok"}


def run_worker(request: Path, timeout: float, workers=4):
    return run_process([sys.executable, "-m", "tabular_lab.worker", str(request)],
                       request.parent / "worker.log", timeout, workers,
                       cwd=Path(__file__).resolve().parent.parent)
