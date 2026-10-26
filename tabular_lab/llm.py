"""Audited JSON calls with conservative reservations and explicit unknown usage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from .config import ExperimentSpec, LLMConfig, StrictModel
from .credentials import CredentialError, get_api_key
from .io import append_event


class StopRun(RuntimeError):
    pass


class Plan(StrictModel):
    action: Literal["experiment", "stop"]
    experiment: ExperimentSpec | None = None
    reason: str = ""


class CodePlan(StrictModel):
    action: Literal["code", "stop"]
    hypothesis: str = Field(default="", max_length=3000)
    code: str = Field(default="", max_length=50_000)
    reason: str = ""


class JSONClient:
    def __init__(self, config: LLMConfig, budget: float, log: Path, client=None):
        self.config, self.budget, self.log = config, budget, log
        self.spent = 0.0
        self.calls = 0
        self.usage_unknown = False
        self.client = client
        self.responses = []
        if config.provider == "replay":
            if not config.replay_path:
                raise ValueError("replay requires replay_path")
            for line in config.replay_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("event") == "llm_response" and "content" in row:
                    self.responses.append(row)

    def cost(self, input_tokens: int, output_tokens: int):
        prices = self.config.pricing
        return (input_tokens * prices.input_usd_per_million +
                output_tokens * prices.output_usd_per_million) * prices.usd_to_cny / 1_000_000

    def ask(self, system: str, context: dict, timeout: float | None = None) -> dict:
        if self.usage_unknown:
            raise StopRun("previous API call has unknown usage; no further calls allowed")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        if self.config.provider == "replay":
            if self.calls >= len(self.responses):
                raise StopRun("replay exhausted")
            original = self.responses[self.calls]
            self.calls += 1
            append_event(self.log, {"event": "llm_response", "provider": "replay", "cost_cny": 0,
                                   "original_usage": original.get("usage"), "content": original["content"]})
            return json.loads(original["content"])
        if self.config.provider != "deepseek":
            raise ValueError("JSONClient requires deepseek or replay")
        try:
            key = get_api_key()
        except CredentialError as exc:
            raise StopRun(str(exc)) from None
        if not key:
            raise StopRun("No DeepSeek key: set DEEPSEEK_API_KEY or run 'python -m tabular_lab auth login' "
                          "on Windows; use mock for an offline run")
        # UTF-8 bytes + overhead is deliberately conservative; no cache discounts assumed.
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 1024
        reservation = self.cost(input_bound, self.config.max_output_tokens)
        if self.spent + reservation > self.budget:
            append_event(self.log, {"event": "budget_stop", "spent_cny": self.spent,
                                   "reserved_cny": reservation, "budget_cny": self.budget})
            raise StopRun("LLM budget cannot cover next call's conservative reservation")
        self.calls += 1
        append_event(self.log, {"event": "llm_request", "call": self.calls,
                               "model": self.config.model, "reservation_cny": reservation,
                               "messages": messages})
        payload = {"model": self.config.model, "messages": messages,
                   "max_tokens": self.config.max_output_tokens,
                   "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}}
        own_client = self.client is None
        client = self.client or httpx.Client()
        try:
            response = client.post("https://api.deepseek.com/chat/completions", json=payload,
                                   headers={"Authorization": f"Bearer {key}"},
                                   timeout=min(self.config.timeout_seconds, timeout or self.config.timeout_seconds))
            # HTTP rejections do not expose a reliable usage record; stop conservatively.
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("API response must be an object")
            usage = body.get("usage", {})
            input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if (type(input_tokens) is not int or type(output_tokens) is not int or
                    input_tokens < 0 or output_tokens < 0):
                raise ValueError("API response has no valid token usage")
            charge = self.cost(input_tokens, output_tokens)
            self.spent += charge
            choices = body.get("choices")
            message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            append_event(self.log, {"event": "llm_response", "call": self.calls, "provider": "deepseek",
                                   "model": body.get("model"), "response_id": body.get("id"),
                                   "usage": usage, "cost_cny": charge, "spent_cny": self.spent,
                                   "pricing_basis": "configured peak/cache-miss estimate; not invoice",
                                   "content": content})
        except (httpx.HTTPError, ValueError, IndexError, TypeError) as exc:
            self.usage_unknown = True
            # Never log request headers, API keys or raw error response bodies.
            append_event(self.log, {"event": "llm_usage_unknown", "call": self.calls,
                                   "error_type": type(exc).__name__, "reserved_cny": reservation})
            raise StopRun(f"API call failed or usage unknown ({type(exc).__name__}); inspect logs") from exc
        finally:
            if own_client:
                client.close()
        if self.spent > self.budget:
            raise StopRun("actual recorded usage exceeded budget estimate; further calls stopped")
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("model returned empty or invalid JSON; usage was recorded") from exc


def planning_prompt() -> str:
    return (
        "You are a tabular classification experiment planner. Respond only with a JSON object. "
        "Use training profile and validation history, never dataset identity or test results. "
        "Choose exactly one candidate from candidate_space by copying its model, params and preprocessing; "
        "rewrite hypothesis to explain why this experiment follows from the evidence. Do not repeat attempts. "
        "Return {\"action\":\"experiment\",\"experiment\":{...}} or "
        "{\"action\":\"stop\",\"reason\":\"...\"}. "
        "Experiment JSON schema: " + json.dumps(ExperimentSpec.model_json_schema())
    )


def code_prompt() -> str:
    return (
        "You are a classification pipeline programmer. Reply with JSON only: "
        '{"action":"code","hypothesis":"...","code":"Python source"} or '
        '{"action":"stop","reason":"..."}. '
        "Implement build_pipeline(seed: int, feature_schema: dict) returning an sklearn-compatible pipeline "
        "with fit(X,y), predict(X), predict_proba(X), classes_. Inputs are pandas DataFrames. "
        "Only numpy, pandas, scipy, sklearn, catboost and standard-library modules are installed. "
        "Do not read files, access labels outside fit, run commands, use network or install packages. "
        "All preprocessing must be fitted inside the returned pipeline. Use at most 4 CPU threads. "
        "The current working directory is writable temporary scratch; the root filesystem and inputs are read-only. "
        "For CatBoost, set allow_writing_files=False to avoid unnecessary training files. "
        "Handle missing values and unseen categories. Labels are integer encoded. "
        "Use validation scores and error feedback to improve or repair your code. Never change the evaluator."
        " When best_generated is present, use its source and validation metrics as a starting point for improvement."
    )
