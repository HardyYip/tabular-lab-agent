"""Publish this repository using an existing Git Credential Manager login.

Explicit invocation is required. Credentials stay in memory; no token is printed or written.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OWNER, NAME = "HardyYip", "tabular-lab-agent"
REMOTE = f"https://github.com/{OWNER}/{NAME}.git"


def main():
    environment = {**os.environ, "GCM_INTERACTIVE": "never", "GIT_TERMINAL_PROMPT": "0"}
    credential = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                                capture_output=True, text=True, timeout=30, env=environment)
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if credential.returncode or not token:
        raise SystemExit("No cached GitHub login available. Authenticate Git Credential Manager, then rerun.")
    with httpx.Client(base_url="https://api.github.com", timeout=30,
                      headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"}) as client:
        profile = client.get("/user")
        if profile.status_code != 200 or profile.json().get("login", "").lower() != OWNER.lower():
            raise SystemExit("Cached GitHub login does not match HardyYip; no repository changed.")
        repo = client.get(f"/repos/{OWNER}/{NAME}")
        if repo.status_code == 404:
            repo = client.post("/user/repos", json={"name": NAME, "private": False,
                "description": "Reproducible tabular ML experiments with a budget-aware LLM agent and optional Docker code generation"})
        if repo.status_code not in (200, 201):
            raise SystemExit(f"GitHub repository creation/access failed (HTTP {repo.status_code}).")
        if repo.json().get("private"):
            raise SystemExit("Existing repository is private; its visibility was not changed.")
    if not (ROOT / ".git").exists():
        subprocess.run(["git", "init", "-b", "main"], cwd=ROOT, check=True)
    current = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
    if current.returncode:
        subprocess.run(["git", "remote", "add", "origin", REMOTE], cwd=ROOT, check=True)
    elif current.stdout.strip() != REMOTE:
        raise SystemExit("Existing origin differs from the intended repository; no push attempted.")
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=ROOT, env=environment, check=True)
    print(f"Published https://github.com/{OWNER}/{NAME}")


if __name__ == "__main__":
    main()
