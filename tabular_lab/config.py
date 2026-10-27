"""Strict task and experiment contracts; LLM output never controls the evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_default=True)


class Preprocess(StrictModel):
    impute: Literal["median", "most_frequent"] = "median"
    scale: bool = True
    min_frequency: int = Field(default=1, ge=1, le=100)
    select_percentile: int = Field(default=100, ge=10, le=100)


class LogisticParams(StrictModel):
    C: float = Field(default=1, ge=0.0001, le=1000)
    class_weight: Literal["balanced"] | None = None
    max_iter: int = Field(default=1000, ge=100, le=3000)


class TreesParams(StrictModel):
    n_estimators: int = Field(default=200, ge=20, le=600)
    max_depth: int | None = Field(default=None, ge=2, le=40)
    min_samples_leaf: int = Field(default=1, ge=1, le=100)
    max_features: Literal["sqrt", "log2"] | float = "sqrt"
    class_weight: Literal["balanced"] | None = None

    @model_validator(mode="after")
    def valid_fraction(self):
        if isinstance(self.max_features, float) and not 0 < self.max_features <= 1:
            raise ValueError("max_features fraction must be in (0, 1]")
        return self


class HistParams(StrictModel):
    learning_rate: float = Field(default=0.1, ge=0.005, le=0.5)
    max_iter: int = Field(default=150, ge=20, le=600)
    max_leaf_nodes: int = Field(default=31, ge=3, le=127)
    l2_regularization: float = Field(default=0, ge=0, le=100)
    class_weight: Literal["balanced"] | None = None


class CatParams(StrictModel):
    iterations: int = Field(default=200, ge=20, le=600)
    depth: int = Field(default=6, ge=2, le=10)
    learning_rate: float = Field(default=0.1, ge=0.005, le=0.5)
    l2_leaf_reg: float = Field(default=3, ge=0, le=100)
    auto_class_weights: Literal["Balanced", "SqrtBalanced"] | None = None


PARAM_TYPES = {"dummy": StrictModel, "logistic": LogisticParams, "extra_trees": TreesParams,
               "hist_gradient_boosting": HistParams, "catboost": CatParams}


class ExperimentSpec(StrictModel):
    hypothesis: str = Field(min_length=1, max_length=3000)
    model: Literal["dummy", "logistic", "extra_trees", "hist_gradient_boosting", "catboost"]
    params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: Preprocess = Field(default_factory=Preprocess)

    @model_validator(mode="after")
    def validate_params(self):
        validated = PARAM_TYPES[self.model].model_validate(self.params).model_dump()
        object.__setattr__(self, "params", validated)
        return self

    def fingerprint(self) -> str:
        values = self.model_dump(exclude={"hypothesis"})
        if self.model in {"dummy", "catboost"}:
            values.pop("preprocessing")  # native CatBoost and dummy do not use these options
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class Pricing(StrictModel):
    # Conservative PEAK/cache-miss snapshot, not a live invoice or exchange-rate feed.
    input_usd_per_million: float = Field(default=0.30, ge=0)
    output_usd_per_million: float = Field(default=1.20, ge=0)
    usd_to_cny: float = Field(default=7.20, gt=0)
    snapshot_date: str = "2026-10-26"
    source: str = "https://api-docs.deepseek.com/quick_start/pricing/"


class LLMConfig(StrictModel):
    provider: Literal["mock", "deepseek", "replay"] = "mock"
    model: str = "deepseek-flash"
    max_output_tokens: int = Field(default=2048, ge=256, le=8192)
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    replay_path: Path | None = None
    pricing: Pricing = Field(default_factory=Pricing)


class Limits(StrictModel):
    budget_cny: float = Field(default=20, ge=0)
    max_trials: int = Field(default=8, ge=2, le=100)
    max_seconds: float = Field(default=1800, ge=5)
    trial_seconds: float = Field(default=120, ge=1)
    final_seconds: float = Field(default=120, ge=1)
    max_dense_cells: int = Field(default=20_000_000, ge=100)
    max_workers: int = Field(default=4, ge=1, le=16)


class DockerConfig(StrictModel):
    image: str = "tabular-lab-worker:0.1.0"
    memory: str = Field(default="8g", pattern=r"^[1-9][0-9]*[mg]$")
    cpus: int = Field(default=4, ge=1, le=16)
    max_repairs: int = Field(default=2, ge=0, le=2)


class Modalities(StrictModel):
    text_columns: list[str] = Field(default_factory=list, max_length=16)
    image_columns: list[str] = Field(default_factory=list, max_length=4)
    image_root: Path | None = None
    image_size: int = Field(default=16, ge=8, le=64)
    text_features: int = Field(default=256, ge=16, le=4096)
    max_text_chars: int = Field(default=20_000, ge=256, le=100_000)
    max_image_bytes: int = Field(default=10_000_000, ge=1024, le=50_000_000)
    max_image_pixels: int = Field(default=20_000_000, ge=64, le=50_000_000)

    @model_validator(mode="after")
    def distinct_columns(self):
        columns = self.text_columns + self.image_columns
        if any(not name.strip() for name in columns) or len(set(columns)) != len(columns):
            raise ValueError("modality columns must be nonempty, unique and disjoint")
        return self


class TaskConfig(StrictModel):
    data_path: Path
    target: str
    id_column: str | None = None
    group_column: str | None = None
    validation_path: Path | None = None
    test_path: Path | None = None
    metric: Literal["macro_f1", "accuracy", "roc_auc"] = "macro_f1"
    seed: int = Field(default=42, ge=0)
    device: Literal["cpu", "auto", "gpu"] = "cpu"
    strategy: Literal["baseline", "random", "agent", "codegen"] = "agent"
    run_dir: Path = Path("../runs")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    limits: Limits = Field(default_factory=Limits)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    modalities: Modalities = Field(default_factory=Modalities)

    @model_validator(mode="after")
    def split_contract(self):
        if bool(self.validation_path) != bool(self.test_path):
            raise ValueError("provide both validation_path and test_path, or neither")
        if self.id_column == self.target:
            raise ValueError("id_column cannot be target")
        if self.group_column and self.group_column in {self.target, self.id_column}:
            raise ValueError("group_column cannot be the target or ID column")
        if {self.target, self.id_column, self.group_column} & set(
                self.modalities.text_columns + self.modalities.image_columns):
            raise ValueError("modality columns cannot be target, ID or group columns")
        if self.strategy == "codegen" and self.llm.provider == "mock":
            raise ValueError("codegen requires deepseek or replay, not mock")
        return self


def load_task(path: Path) -> TaskConfig:
    path = path.resolve()
    task = TaskConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    for key in ("data_path", "validation_path", "test_path", "run_dir"):
        value = getattr(task, key)
        if value is not None:
            setattr(task, key, (path.parent / value).resolve())
    if task.llm.replay_path:
        task.llm.replay_path = (path.parent / task.llm.replay_path).resolve()
    if task.modalities.image_root is not None:
        task.modalities.image_root = (path.parent / task.modalities.image_root).resolve()
    return task
