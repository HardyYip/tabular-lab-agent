from __future__ import annotations

from itertools import islice
from typing import Literal

from catboost import CatBoost
from pydantic import Field, model_validator
from sklearn.base import ClassifierMixin, RegressorMixin

from .config import StrictModel
from .early_stopping import EarlyStoppingCatBoost

Device = Literal["cpu", "gpu", "mixed", "unknown"]


class EstimatorRuntime(StrictModel):
    path: str = Field(max_length=200)
    estimator: str = Field(max_length=160)
    training_device: Literal["cpu", "gpu", "unknown"]
    evidence: Literal["catboost_fitted_params", "sklearn_cpu_estimator", "unsupported_estimator"]
    fitted_iterations: int | None = Field(default=None, ge=1)
    early_stopping_status: Literal["not_used", "verified_internal_holdout", "unverified"] = "not_used"


class EarlyStoppingRuntime(StrictModel):
    path: str = Field(max_length=200)
    method: Literal["internal_holdout_then_full_refit"]
    split_method: Literal["stratified", "stratified_feature_groups"]
    fit_rows: int = Field(ge=1)
    monitor_rows: int = Field(ge=1)
    refit_rows: int = Field(ge=2)
    validation_fraction: float = Field(gt=0, le=0.5)
    patience: int = Field(ge=1)
    max_iterations: int = Field(ge=1)
    selected_iterations: int = Field(ge=1)
    selection_iterations_run: int = Field(ge=1)
    stopped_early: bool
    selection_device: Literal["cpu", "gpu", "unknown"]
    monitor_metric: str = Field(max_length=160)
    fit_indices_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    monitor_indices_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection_seconds: float = Field(ge=0, allow_inf_nan=False)
    refit_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def coherent_counts(self):
        if self.fit_rows + self.monitor_rows != self.refit_rows:
            raise ValueError("internal folds must partition the refit rows")
        if not self.selected_iterations <= self.selection_iterations_run <= self.max_iterations:
            raise ValueError("inconsistent early-stopping iteration counts")
        return self


class RuntimeReport(StrictModel):
    schema_version: Literal[1] = 1
    device_used: Device = "unknown"
    complete: bool = False
    estimators: list[EstimatorRuntime] = Field(default_factory=list, max_length=64)
    early_stopping: list[EarlyStoppingRuntime] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def coherent_device(self):
        devices = {entry.training_device for entry in self.estimators}
        if self.complete:
            expected = "mixed" if len(devices) > 1 else next(iter(devices), "unknown")
            if "unknown" in devices or self.device_used != expected or not devices:
                raise ValueError("inconsistent fitted-device evidence")
        elif self.device_used != "unknown":
            raise ValueError("incomplete device evidence must be marked unknown")
        return self


def inspect_model(model):
    estimators, stopping, seen, active = [], [], set(), set()
    truncated = False

    def walk(node, path, verified_stopping=False, depth=0):
        nonlocal truncated
        if id(node) in active or depth > 16:
            truncated = True
            return
        if id(node) in seen:
            return
        if len(seen) >= 64:
            truncated = True
            return
        seen.add(id(node))
        active.add(id(node))
        kind = type(node)
        entry = {"path": path[:200], "estimator": (kind.__module__ + "." + kind.__name__)[:160]}
        try:
            helper = False
            if isinstance(node, EarlyStoppingCatBoost) and hasattr(node, "training_metadata_"):
                try:
                    metadata = EarlyStoppingRuntime.model_validate({"path": path[:200], **node.training_metadata_})
                except Exception:
                    truncated = True
                else:
                    helper = True
                    stopping.append(metadata.model_dump())
            if isinstance(node, CatBoost):
                params = node.get_all_params()
                device = params.get("task_type", "unknown").lower()
                entry.update(training_device=device if device in {"cpu", "gpu"} else "unknown",
                             evidence="catboost_fitted_params", fitted_iterations=int(node.tree_count_))
                entry["early_stopping_status"] = ("verified_internal_holdout" if verified_stopping else
                                                  "unverified" if params.get("od_type") else "not_used")
                estimators.append(entry)
                return
            children = []
            steps = getattr(node, "steps", None)
            if steps is not None:
                if type(steps) not in (list, tuple):
                    truncated = True
                else:
                    if len(steps) > 64:
                        truncated = True
                    for name, child in steps[:64]:
                        if hasattr(child, "predict"):
                            children.append((str(name), child))
            collections = ["members_"]
            if not kind.__module__.startswith("sklearn.") or kind.__name__ in {
                "VotingClassifier", "StackingClassifier", "VotingRegressor", "StackingRegressor"
            }:
                collections.append("estimators_")
            for attr in collections:
                values = getattr(node, attr, None)
                if values is None:
                    continue
                if type(values) not in (list, tuple, dict):
                    truncated = True
                    continue
                if len(values) > 64:
                    truncated = True
                if isinstance(values, dict):
                    values = list(islice(values.values(), 64))
                for index, child in enumerate(values[:64]):
                    if isinstance(child, tuple) and len(child) == 2 and isinstance(child[0], str):
                        child = child[1]
                    children.append((attr + "." + str(index), child))
            for attr in ("model_", "estimator_", "classifier_", "regressor_", "final_estimator_", "model"):
                child = getattr(node, attr, None)
                if child is not None and hasattr(child, "predict"):
                    children.append((attr, child))
            if children:
                for name, child in children:
                    walk(child, path + "." + name, helper and name == "model_", depth + 1)
                return
            if kind.__module__.startswith("sklearn.") and isinstance(node, (ClassifierMixin, RegressorMixin)):
                entry.update(training_device="cpu", evidence="sklearn_cpu_estimator")
            else:
                entry.update(training_device="unknown", evidence="unsupported_estimator")
            estimators.append(entry)
        except Exception:
            truncated = True
            entry.update(training_device="unknown", evidence="unsupported_estimator")
            estimators.append(entry)
        finally:
            active.remove(id(node))

    walk(model, "model")
    devices = {entry["training_device"] for entry in estimators}
    complete = bool(devices) and "unknown" not in devices and not truncated
    device = ("mixed" if len(devices) > 1 else next(iter(devices))) if complete else "unknown"
    return RuntimeReport(device_used=device, complete=complete, estimators=estimators,
                         early_stopping=stopping).model_dump()
