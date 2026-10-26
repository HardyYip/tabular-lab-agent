from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not serializable: {type(value)}")


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_event(path: Path, event: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": datetime.now(UTC).isoformat(), **event},
                                default=json_default, ensure_ascii=False, allow_nan=False) + "\n")


def code_version() -> dict:
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted((root / "tabular_lab").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    source_hash = digest.hexdigest()
    try:
        cwd = root
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd,
                                      stderr=subprocess.DEVNULL, timeout=5, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd,
                                        timeout=5, text=True).strip()
        return {"git_commit": sha, "git_dirty": bool(dirty), "package_source_sha256": source_hash}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None, "package_source_sha256": source_hash}
