"""Split before profiling/fitting. Test labels never enter the planning context."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import TaskConfig


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in frame:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            frame[col] = frame[col].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return frame


@dataclass
class Split:
    X: pd.DataFrame
    y: np.ndarray
    indices: list[int]
    ids: list | None


@dataclass
class Dataset:
    train: Split
    validation: Split
    test: Split
    columns: list[str]
    classes: list[str]
    hashes: dict[str, str]
    profile: dict
    split_source: str


def _read(path: Path, task: TaskConfig, encoder=None):
    frame = pd.read_csv(path)
    if frame.empty or task.target not in frame:
        raise ValueError("CSV must contain non-empty rows and the target column")
    if frame[task.target].isna().any():
        raise ValueError("target contains missing labels")
    ids = None
    if task.id_column:
        if task.id_column not in frame:
            raise ValueError("configured id_column is missing")
        ids = frame[task.id_column].tolist()
    labels = frame[task.target].astype(str)
    encoder = encoder or LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    X = clean_features(frame.drop(columns=[task.target] + ([task.id_column] if task.id_column else [])))
    if not len(X.columns) or X.columns.duplicated().any():
        raise ValueError("at least one unique feature column is required")
    return X, y, ids, encoder


def load_dataset(task: TaskConfig) -> Dataset:
    X, y, ids, encoder = _read(task.data_path, task)
    if len(encoder.classes_) < 2:
        raise ValueError("classification requires at least two classes")
    if task.metric == "roc_auc" and len(encoder.classes_) != 2:
        raise ValueError("roc_auc is supported only for binary tasks")
    hashes = {"data": hashlib.sha256(task.data_path.read_bytes()).hexdigest()}

    def subset(indices):
        return Split(X.iloc[indices].reset_index(drop=True), y[indices], indices.tolist(),
                     [ids[i] for i in indices] if ids is not None else None)

    if task.validation_path:
        split_source = "external CSV files"
        train = subset(np.arange(len(X)))
        external = []
        for key, path in (("validation", task.validation_path), ("test", task.test_path)):
            other_X, other_y, other_ids, _ = _read(path, task, encoder)
            if set(other_X.columns) != set(X.columns):
                raise ValueError(f"{key} feature columns differ from training columns")
            external.append(Split(other_X[X.columns], other_y, list(range(len(other_X))), other_ids))
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        validation, test = external
        feature_groups = [set(pd.util.hash_pandas_object(s.X, index=False).tolist())
                          for s in (train, validation, test)]
        if any(feature_groups[i] & feature_groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("external splits contain overlapping feature rows; prepare disjoint data first")
    else:
        try:
            groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
            if len(np.unique(groups)) != len(groups):
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=task.seed)
                folds = [held for _, held in splitter.split(X, y, groups)]
                test_idx, val_idx = folds[0], folds[1]
                train_idx = np.concatenate(folds[2:])
                split_source = "stratified feature-group folds, approximately 60/20/20"
            else:
                development, test_idx = train_test_split(np.arange(len(X)), test_size=0.2,
                                                         stratify=y, random_state=task.seed)
                train_idx, val_idx = train_test_split(development, test_size=0.25,
                                                      stratify=y[development], random_state=task.seed)
                split_source = "stratified 60/20/20"
        except ValueError as exc:
            raise ValueError(f"not enough rows per class for stratified 60/20/20 split: {exc}") from exc
        train, validation, test = map(subset, (train_idx, val_idx, test_idx))
    expected = set(range(len(encoder.classes_)))
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if set(split.y) != expected:
            raise ValueError(f"{name} must contain all target classes")
    numeric = train.X.select_dtypes(include="number").columns.tolist()
    columns = []
    for name in train.X.columns:
        values = train.X[name]
        entry = {"index": len(columns), "kind": "numeric" if name in numeric else "categorical",
                 "missing_fraction": float(values.isna().mean()), "unique_count": int(values.nunique())}
        if name in numeric and values.notna().any():
            entry.update({"mean": float(values.mean()), "std": float(values.std(ddof=0)),
                          "min": float(values.min()), "max": float(values.max())})
        columns.append(entry)
    profile = {"training_rows": len(train.X), "feature_count": len(X.columns),
               "class_count": len(encoder.classes_), "class_counts": np.bincount(train.y).tolist(),
               "columns": columns}
    return Dataset(train, validation, test, X.columns.tolist(), encoder.classes_.tolist(), hashes,
                   profile, split_source)


def prediction_frame(path: Path, metadata: dict):
    frame = pd.read_csv(path)
    ids = None
    if metadata.get("id_column"):
        id_col = metadata["id_column"]
        if id_col not in frame:
            raise ValueError(f"missing ID column: {id_col}")
        ids = frame[id_col].copy()
    features = frame.drop(columns=[c for c in (metadata.get("target"), metadata.get("id_column"))
                                  if c and c in frame])
    missing = set(metadata["columns"]) - set(features.columns)
    extra = set(features.columns) - set(metadata["columns"])
    if missing or extra:
        raise ValueError(f"feature schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return clean_features(features[metadata["columns"]]), ids
