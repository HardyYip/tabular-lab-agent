from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ExperimentSpec


class DenseGuard(TransformerMixin, BaseEstimator):
    def __init__(self, max_cells=20_000_000):
        self.max_cells = max_cells

    def fit(self, X, y=None):
        self._check(X)
        return self

    def _check(self, X):
        if X.shape[0] * X.shape[1] > self.max_cells:
            raise ValueError("encoded feature matrix exceeds max_dense_cells")

    def transform(self, X):
        self._check(X)
        return X.toarray() if hasattr(X, "toarray") else X


class CatFrame(TransformerMixin, BaseEstimator):
    """Native categorical features with training-only numeric imputation."""
    def fit(self, X, y=None):
        self.numeric_ = X.select_dtypes(include="number").columns.tolist()
        self.categorical_ = [c for c in X if c not in self.numeric_]
        self.medians_ = X[self.numeric_].median().fillna(0)
        return self

    def transform(self, X):
        result = X.copy()
        result[self.numeric_] = result[self.numeric_].fillna(self.medians_)
        for name in self.categorical_:
            result[name] = result[name].fillna("__MISSING__").astype(str)
        return result


def build_pipeline(spec: ExperimentSpec, X: pd.DataFrame, seed: int, max_cells: int, workers: int,
                   device: str = "cpu"):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    params = dict(spec.params)
    if spec.model == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**params, random_seed=seed, thread_count=workers,
                                   cat_features=categorical, verbose=False, allow_writing_files=False,
                                   task_type="GPU" if device == "gpu" else "CPU",
                                   **({"devices": "0"} if device == "gpu" else {}))
        return Pipeline([("preprocess", CatFrame()), ("model", model)])
    if spec.model == "dummy":
        return Pipeline([("preprocess", ColumnTransformer([
            ("numbers", SimpleImputer(keep_empty_features=True), numeric),
            ("categories", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                                     ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)
        ])), ("model", DummyClassifier(strategy="most_frequent", random_state=seed))])
    numeric_steps = [("impute", SimpleImputer(strategy=spec.preprocessing.impute, keep_empty_features=True))]
    if spec.preprocessing.scale:
        numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numbers", Pipeline(numeric_steps), numeric),
        ("categories", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                     min_frequency=spec.preprocessing.min_frequency))
        ]), categorical)
    ], sparse_threshold=1.0)
    models = {
        "logistic": lambda: LogisticRegression(**params, random_state=seed),
        "extra_trees": lambda: ExtraTreesClassifier(**params, random_state=seed, n_jobs=workers),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(**params, random_state=seed),
    }
    steps = [("preprocess", preprocess)]
    if spec.model == "hist_gradient_boosting":
        steps.append(("dense_guard", DenseGuard(max_cells)))
    if spec.preprocessing.select_percentile < 100:
        steps.append(("select", SelectPercentile(f_classif, percentile=spec.preprocessing.select_percentile)))
    steps.append(("model", models[spec.model]()))
    return Pipeline(steps)


def score_predictions(y, predictions, probabilities, class_count: int) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    predictions = np.asarray(predictions).reshape(-1)
    if len(predictions) != len(y) or not np.isin(predictions, np.arange(class_count)).all():
        raise ValueError("invalid predicted labels")
    result = {"accuracy": float(accuracy_score(y, predictions)),
              "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0))}
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(y), class_count) or not np.isfinite(probabilities).all():
            raise ValueError("invalid probability matrix shape or values")
        if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(
            probabilities.sum(axis=1), 1, atol=1e-5
        ):
            raise ValueError("probabilities must be in [0,1] and sum to one")
        if class_count == 2:
            result["roc_auc"] = float(roc_auc_score(y, probabilities[:, 1]))
    return result
