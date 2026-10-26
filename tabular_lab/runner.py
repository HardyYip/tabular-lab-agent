"""Orchestrate experiments; keep test data out of all planning feedback."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentSpec, TaskConfig
from .data import load_dataset, prediction_frame
from .io import append_event, code_version, read_json, write_json
from .llm import CodePlan, JSONClient, Plan, StopRun, code_prompt, planning_prompt
from .process import run_worker
from .search import allowed_fingerprints, baselines, choose_mock, choose_random, search_space


def _trial(task, dataset, spec, folder, timeout, final=False, fitted_model=None):
    folder.mkdir(parents=True, exist_ok=True)
    X_train = dataset.train.X
    y_train = dataset.train.y
    evaluation = dataset.test if final else dataset.validation
    if final and not fitted_model:
        X_train = pd.concat([X_train, dataset.validation.X], ignore_index=True)
        y_train = np.concatenate([y_train, dataset.validation.y])
    joblib.dump({"X_train": X_train, "y_train": y_train, "X_eval": evaluation.X,
                 "y_eval": evaluation.y}, folder / "input.joblib")
    request = {"action": "predict" if fitted_model else "train", "spec": spec.model_dump(),
               "device": task.device,
               "seed": task.seed, "class_count": len(dataset.classes),
               "workers": task.limits.max_workers, "max_dense_cells": task.limits.max_dense_cells,
               "model_path": str(fitted_model) if fitted_model else None}
    write_json(folder / "request.json", request)
    start = time.monotonic()
    result = run_worker(folder / "request.json", timeout, task.limits.max_workers)
    if result["status"] == "ok":
        result = read_json(folder / "result.json")
    result["seconds"] = time.monotonic() - start
    return result


def _context(dataset, history, task):
    # No dataset names, file paths, raw rows, test summaries or target labels sent to LLM.
    return {"training_profile": dataset.profile, "metric": task.metric,
            "history": [{k: h[k] for k in ("trial", "spec", "status", "metrics", "fingerprint", "error_type")
                         if k in h} for h in history],
            "candidate_space": [s.model_dump() for s in search_space()]}


def run_task(task: TaskConfig, output: Path | None = None) -> dict:
    if task.strategy == "codegen":
        from .sandbox import preflight
        preflight(task.docker.image)
    start = time.monotonic()
    deadline = start + task.limits.max_seconds
    dataset = load_dataset(task)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = output or task.run_dir / run_id
    folder.mkdir(parents=True, exist_ok=False)
    log = folder / "events.jsonl"
    write_json(folder / "config.json", task.model_dump(mode="json"))
    write_json(folder / "profile.json", dataset.profile)
    write_json(folder / "splits.json", {name: getattr(dataset, name).indices
                                        for name in ("train", "validation", "test")})
    versions = {name: importlib.metadata.version(name) for name in
                ("tabular-lab-agent", "numpy", "pandas", "scikit-learn", "catboost", "pydantic")}
    write_json(folder / "manifest.json", {**code_version(), "versions": versions,
               "data_hashes": dataset.hashes, "seed": task.seed,
               "split_source": dataset.split_source,
               "provider": task.llm.provider, "model": task.llm.model,
               "pricing": task.llm.pricing.model_dump()})
    llm = JSONClient(task.llm, task.limits.budget_cny, log) if task.llm.provider != "mock" else None
    history, best = [], None
    stop_reason = "trial_limit"
    no_progress = 0
    pending_code = None
    repair_count = 0
    planning_failed = False
    while len(history) < task.limits.max_trials:
        remaining = deadline - time.monotonic()
        # Reserve finalization time rather than spend the entire run on search.
        available = remaining - min(task.limits.final_seconds, task.limits.max_seconds / 3)
        if available <= 1:
            stop_reason = "search_time_limit"
            break
        code = None
        try:
            if len(history) < 2:
                spec = baselines()[len(history)]
            elif task.strategy == "baseline":
                stop_reason = "baseline_complete"
                break
            elif task.strategy == "random":
                spec = choose_random(history, task.seed)
            elif task.strategy == "agent" and task.llm.provider == "mock":
                spec = choose_mock(history, task.seed)
            elif task.strategy == "codegen":
                context = _context(dataset, history, task)
                context.pop("candidate_space")
                context["feature_schema"] = {c: str(dataset.train.X[c].dtype) for c in dataset.columns}
                context["execution_limits"] = {"trial_seconds": task.limits.trial_seconds,
                                                "cpus": task.docker.cpus, "memory": task.docker.memory}
                successful_code = [h for h in history if h["spec"]["model"] == "generated" and h["status"] == "ok"]
                if successful_code:
                    previous = max(successful_code, key=lambda h: h["metrics"][task.metric])
                    source = folder / "trials" / f"{previous['trial']:03d}" / "container-input" / "candidate.py"
                    context["best_generated"] = {"trial": previous["trial"],
                                                 "code": source.read_text(encoding="utf-8"),
                                                 "metrics": previous["metrics"]}
                if pending_code:
                    context["repair"] = pending_code
                decision = CodePlan.model_validate(llm.ask(code_prompt(), context, timeout=available))
                if decision.action == "stop":
                    stop_reason = "agent_stop: " + decision.reason
                    break
                if not decision.code.strip() or not decision.hypothesis.strip():
                    raise ValueError("code candidate requires source and hypothesis")
                code = decision.code
                spec = {"model": "generated", "hypothesis": decision.hypothesis}
            else:
                decision = Plan.model_validate(llm.ask(planning_prompt(), _context(dataset, history, task),
                                                       timeout=available))
                if decision.action == "stop":
                    stop_reason = "agent_stop: " + decision.reason
                    break
                spec = decision.experiment
                if spec is None:
                    raise ValueError("experiment action requires experiment")
                if spec.fingerprint() not in allowed_fingerprints():
                    raise ValueError("experiment is outside shared candidate_space")
            if spec is None:
                stop_reason = "search_space_exhausted"
                break
            fingerprint = hashlib.sha256(code.encode()).hexdigest() if code else spec.fingerprint()
            if fingerprint in {h["fingerprint"] for h in history}:
                raise ValueError("duplicate experiment")
        except StopRun as exc:
            planning_failed = True
            stop_reason = str(exc)
            append_event(log, {"event": "planning_stopped", "reason": stop_reason})
            break
        except ValueError as exc:
            no_progress += 1
            append_event(log, {"event": "invalid_plan", "error": str(exc)[:1000]})
            if no_progress >= 3:
                planning_failed = True
                stop_reason = "three_invalid_plans"
                break
            continue
        no_progress = 0
        trial_id = len(history)
        trial_folder = folder / "trials" / f"{trial_id:03d}"
        spec_dict = spec if code else spec.model_dump()
        append_event(log, {"event": "experiment_start", "trial": trial_id, "spec": spec_dict,
                           "fingerprint": fingerprint})
        # Account for planning latency before launching a bounded experiment.
        timeout = min(task.limits.trial_seconds, deadline - time.monotonic() -
                      min(task.limits.final_seconds, task.limits.max_seconds / 3))
        if timeout <= 0:
            stop_reason = "search_time_limit"
            break
        if code:
            from .sandbox import evaluate_code
            result = evaluate_code(task, dataset, code, trial_folder, timeout)
            if result["status"] != "ok" and repair_count < task.docker.max_repairs:
                repair_count += 1
                pending_code = {"code": code, "error": result.get("safe_error", result["status"]),
                                "repair_attempt": repair_count}
            else:
                pending_code, repair_count = None, 0
        else:
            result = _trial(task, dataset, spec, trial_folder, timeout)
        row = {"trial": trial_id, "spec": spec_dict, "fingerprint": fingerprint, **result}
        if result["status"] != "ok":
            row["error_type"] = result["status"]
        history.append(row)
        write_json(folder / "experiments.json", history)
        append_event(log, {"event": "experiment_result", **row})
        if result["status"] == "ok" and (best is None or
                result["metrics"][task.metric] > best["metrics"][task.metric]):
            best = row
            append_event(log, {"event": "best_updated", "trial": trial_id,
                               "metric": task.metric, "value": result["metrics"][task.metric]})
    summary = {"run_id": run_id, "path": str(folder), "strategy": task.strategy,
               "provider": task.llm.provider, "metric": task.metric, "stop_reason": stop_reason,
               "trials": len(history), "failed_trials": sum(h["status"] != "ok" for h in history),
               "llm_calls": llm.calls if llm else 0, "cost_cny_estimate": llm.spent if llm else 0,
               "usage_unknown": llm.usage_unknown if llm else False,
               "best_trial": best["trial"] if best else None,
               "best_validation": best["metrics"] if best else None,
               "status": "no_model" if best is None else "partial" if planning_failed else "completed"}
    if best is not None:
        model_folder = folder / "model"
        model_folder.mkdir()
        final_folder = folder / "final"
        timeout = min(task.limits.final_seconds, max(0.1, deadline - time.monotonic()))
        if best["spec"]["model"] == "generated":
            from .sandbox import finalize_code
            final = finalize_code(task, dataset, folder / "trials" / f"{best['trial']:03d}",
                                  final_folder, model_folder, timeout)
            backend = "docker"
        else:
            best_spec = ExperimentSpec.model_validate(best["spec"])
            final = _trial(task, dataset, best_spec, final_folder, timeout, final=True)
            backend = "native"
            if final["status"] == "ok":
                shutil.copy2(final_folder / "pipeline.joblib", model_folder / "pipeline.joblib")
            else:
                # Preserve a usable train-only model even if final refitting timed out.
                saved = folder / "trials" / f"{best['trial']:03d}" / "pipeline.joblib"
                shutil.copy2(saved, model_folder / "pipeline.joblib")
        metadata = {"backend": backend, "columns": dataset.columns, "classes": dataset.classes,
                    "feature_schema": {c: str(dataset.train.X[c].dtype) for c in dataset.columns},
                    "target": task.target, "id_column": task.id_column, "seed": task.seed,
                    "spec": best["spec"], "refit_on_train_validation": final["status"] == "ok",
                    "docker_image": task.docker.image if backend == "docker" else None}
        write_json(model_folder / "metadata.json", metadata)
        summary["finalization"] = final
        if final["status"] == "ok":
            summary["test_metrics"] = final["metrics"]
            values = np.load(final_folder / "predictions.npz", allow_pickle=False)
            _prediction_csv(values["predictions"], values["probabilities"], metadata,
                            folder / "test_predictions.csv", dataset.test.ids)
        else:
            summary["status"] = "partial"
    summary["seconds"] = time.monotonic() - start
    write_json(folder / "summary.json", summary)
    append_event(log, {"event": "run_finished", "summary": summary})
    _report(folder, summary, history)
    return summary


def _prediction_csv(predictions, probabilities, metadata, output, ids=None):
    classes = metadata["classes"]
    result = pd.DataFrame({"prediction": [classes[int(i)] for i in predictions]})
    if ids is not None:
        result.insert(0, metadata["id_column"] or "id", list(ids))
    for index, label in enumerate(classes):
        result[f"probability_{label}"] = probabilities[:, index]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def predict(model: Path, input_path: Path, output: Path):
    metadata = read_json(model / "metadata.json")
    frame, ids = prediction_frame(input_path, metadata)
    if metadata["backend"] == "docker":
        from .sandbox import predict_code
        predictions, probabilities = predict_code(model, frame, metadata)
    else:
        # Native model files are executable pickle artifacts; load only your own trusted runs.
        pipeline = joblib.load(model / "pipeline.joblib")
        predictions = np.asarray(pipeline.predict(frame)).reshape(-1)
        probabilities = np.asarray(pipeline.predict_proba(frame))[:, np.argsort(pipeline.classes_)]
    _prediction_csv(predictions, probabilities, metadata, output, ids)
    return len(frame)


def _report(folder, summary, history):
    devices = sorted({h["device_used"] for h in history if h.get("device_used")})
    execution = ("Generated pipelines in CPU Docker containers" if summary["strategy"] == "codegen" else
                 "Finite search space; observed training devices: " + (", ".join(devices) or "not recorded"))
    lines = ["# Tabular Lab experiment report", "", f"Strategy: **{summary['strategy']}**; "
             f"provider: **{summary['provider']}**. Mock/replay results are not live LLM results.", "",
             f"Status: {summary['status']}; stop reason: {summary['stop_reason']}", "",
             f"Estimated model cost: CNY {summary['cost_cny_estimate']:.6f}; "
             f"calls: {summary['llm_calls']}; unknown usage: {summary['usage_unknown']}", "",
             "| Trial | Model | Status | Validation macro-F1 | Seconds |",
             "| --- | --- | --- | ---: | ---: |"]
    for h in history:
        score = h.get("metrics", {}).get("macro_f1")
        lines.append(f"| {h['trial']} | {h['spec']['model']} | {h['status']} | "
                     f"{score if score is not None else '—'} | {h['seconds']:.2f} |")
    lines.extend(["", "## Decisions", ""])
    for h in history:
        lines.append(f"- Trial {h['trial']}: {h['spec']['hypothesis']}")
    lines.extend(["", "## Final test (not used for selection)", "", "```json",
                  json.dumps(summary.get("test_metrics", {}), indent=2), "```", "",
                  "## Limitations", "", execution + "; one validation split per run. "
                  "Scores on these datasets do not establish broad generalization or LLM superiority. "
                  "Configured peak/cache-miss prices and FX yield conservative estimates, not provider invoices. "
                  "Unknown-usage calls stop subsequent requests. Run three seeds before drawing comparisons."])
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
