"""Audited JSON calls with conservative reservations and explicit unknown usage."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from .config import ExperimentSpec, LLMConfig, StrictModel
from .credentials import CredentialError, get_api_key
from .io import append_event


class StopRun(RuntimeError):
    pass


class PlanningTimeLimit(StopRun):
    pass


class Plan(StrictModel):
    action: Literal["experiment", "stop"]
    experiment: ExperimentSpec | None = None
    candidate_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    hypothesis: str = Field(default="", max_length=3000)
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

    def _window(self, timeout, deadline, reserve_seconds):
        available = deadline - time.monotonic() if deadline is not None else timeout
        required = reserve_seconds + (self.config.timeout_seconds if self.config.provider == "deepseek" else 0)
        if deadline is not None and available < required:
            append_event(self.log, {"event": "planning_time_stop", "available_seconds": available,
                                   "required_seconds": required, "trial_reserve_seconds": reserve_seconds})
            raise PlanningTimeLimit("planning_time_limit")
        if available is not None and available <= 0:
            raise PlanningTimeLimit("planning_time_limit")
        window = available - reserve_seconds if deadline is not None else available
        return available, min(self.config.timeout_seconds, window if window is not None
                              else self.config.timeout_seconds)

    def ask(self, system: str, context: dict, timeout: float | None = None, *,
            deadline: float | None = None, reserve_seconds: float = 0) -> dict:
        if self.usage_unknown:
            raise StopRun("previous API call has unknown usage; no further calls allowed")
        self._window(timeout, deadline, reserve_seconds)
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
        payload = {"model": self.config.model, "messages": messages,
                   "max_tokens": self.config.max_output_tokens,
                   "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}}
        own_client = self.client is None
        client = self.client or httpx.Client()
        started = time.monotonic()
        http_status = None
        try:
            available, request_timeout = self._window(timeout, deadline, reserve_seconds)
            self.calls += 1
            append_event(self.log, {"event": "llm_request", "call": self.calls,
                                   "model": self.config.model, "reservation_cny": reservation,
                                   "available_seconds": available,
                                   "trial_reserve_seconds": reserve_seconds,
                                   "configured_timeout_seconds": self.config.timeout_seconds,
                                   "timeout_seconds": request_timeout, "messages": messages})
            started = time.monotonic()
            response = client.post("https://api.deepseek.com/chat/completions", json=payload,
                                   headers={"Authorization": f"Bearer {key}"},
                                   timeout=request_timeout)
            http_status = response.status_code
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
                                   "elapsed_seconds": time.monotonic() - started,
                                   "pricing_basis": "configured peak/cache-miss estimate; not invoice",
                                   "content": content})
        except (httpx.HTTPError, ValueError, IndexError, TypeError) as exc:
            self.usage_unknown = True
            # Never log request headers, API keys or raw error response bodies.
            append_event(self.log, {"event": "llm_usage_unknown", "call": self.calls,
                                   "error_type": type(exc).__name__, "reserved_cny": reservation,
                                   "http_status": http_status, "elapsed_seconds": time.monotonic() - started})
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
        "Choose exactly one untried option from candidate_options using its candidate_id. "
        "Return {\"action\":\"experiment\",\"candidate_id\":\"...\",\"hypothesis\":\"...\"} or "
        "{\"action\":\"stop\",\"reason\":\"...\"}. "
        "Explain why the choice follows from the evidence. Read planning_feedback to correct rejected plans. "
        "Do not invent IDs or change candidate parameters. Do not repeat attempts. "
        "Legacy experiment objects must exactly match an offered option's experiment. "
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
        "All preprocessing must be fitted inside the returned pipeline. Respect execution_limits.cpus. "
        "GPU access is available only when execution_limits.gpu_enabled is true; CPU estimators remain allowed. "
        "When GPU is unavailable, use CPU settings. For CatBoost GPU training, set task_type='GPU', "
        "devices='0' and gpu_ram_part=0.25 to leave GPU memory headroom; this is not a hard VRAM quota. "
        "The container memory limit is host RAM, not GPU VRAM. Do not assume PyTorch is installed. "
        "The current working directory is writable temporary scratch; the root filesystem and inputs are read-only. "
        "For CatBoost, set allow_writing_files=False to avoid unnecessary training files. "
        "For CatBoost early stopping you MUST use the installed helper "
        "from tabular_lab.early_stopping import EarlyStoppingCatBoost. Return "
        "EarlyStoppingCatBoost(params={'iterations':600,'depth':6,'thread_count':4,...}, "
        "validation_fraction=0.2, patience=50, random_state=seed). Put GPU parameters in params only when enabled. "
        "The helper handles numeric/categorical missing values and unknown categories, splits fit data internally, "
        "fits preprocessing on the internal fitting fold, selects rounds on its internal monitoring fold, "
        "then refits preprocessing and CatBoost on all supplied fit data with the selected rounds. "
        "It accepts an optional sklearn preprocessor, fitted inside each stage. Do not wrap it in a learned "
        "preprocessor fitted before its internal split. Do not pass od_type, od_wait, early_stopping_rounds, "
        "use_best_model or cat_features in params. Direct CatBoost early stopping without this helper raises "
        "UnverifiedEarlyStoppingError. Never use the training data itself as eval_set. "
        "Fixed-iteration CatBoost without early stopping is still allowed. The executor records fitted-model "
        "device evidence and helper early-stopping metadata in runtime feedback; unknown means unverified. "
        "Handle missing values and unseen categories. Labels are integer encoded. "
        "Use validation scores and error feedback to improve or repair your code. Never change the evaluator."
        " When best_generated is present, use its source and validation metrics as a starting point for improvement."
    )
