from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.validation import check_is_fitted

from .models import CatFrame


def internal_split(X, y, fraction, seed):
    if not 0 < fraction <= 0.5:
        raise ValueError("internal monitoring fraction must be in (0, 0.5]")
    indices = np.arange(len(y))
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    expected = set(np.unique(y))
    try:
        if len(np.unique(groups)) == len(groups):
            fit, monitor = train_test_split(indices, test_size=fraction, stratify=y, random_state=seed)
            method = "stratified"
        else:
            folds = max(2, round(1 / fraction))
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            pairs = ((a, b) for a, b in splitter.split(X, y, groups)
                     if set(y[a]) == expected and set(y[b]) == expected)
            fit, monitor = next(pairs)
            method = "stratified_feature_groups"
        if set(y[fit]) != expected or set(y[monitor]) != expected:
            raise ValueError("missing internal classes")
    except (ValueError, StopIteration) as exc:
        raise ValueError("training data cannot form disjoint internal monitoring folds with all classes") from exc
    return fit, monitor, method


class EarlyStoppingCatBoost(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, validation_fraction=0.2, patience=50, random_state=42,
                 preprocessor=None):
        self.params = params
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.preprocessor = preprocessor

    def _parameters(self):
        params = dict(self.params or {})
        forbidden = {"n_estimators", "num_boost_round", "num_trees", "early_stopping_rounds", "od_type",
                     "od_wait", "od_pval", "use_best_model", "cat_features", "text_features",
                     "embedding_features", "save_snapshot", "train_dir", "logging_level", "silent"}
        if forbidden & params.keys():
            raise ValueError("use iterations/patience; early stopping and feature types are managed by the helper")
        params.setdefault("iterations", 600)
        if type(params["iterations"]) is not int or params["iterations"] < 1:
            raise ValueError("iterations must be a positive integer")
        params.update(random_seed=self.random_state, allow_writing_files=False, verbose=False)
        return params

    def _model(self, params, transformed):
        categorical = (transformed.select_dtypes(exclude="number").columns.tolist()
                       if isinstance(transformed, pd.DataFrame) else [])
        return CatBoostClassifier(**params, cat_features=categorical)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or len(X) != len(y):
            raise ValueError("EarlyStoppingCatBoost requires a DataFrame and matching labels")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("patience must be a positive integer")
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("early stopping requires at least two classes")
        for attribute in ("model_", "preprocessor_", "training_metadata_", "classes_"):
            self.__dict__.pop(attribute, None)
        params = self._parameters()
        fit, monitor, method = internal_split(X, y, self.validation_fraction, self.random_state)
        template = self.preprocessor if self.preprocessor is not None else CatFrame()
        selection_preprocessor = clone(template)
        started = time.monotonic()
        X_fit = selection_preprocessor.fit_transform(X.iloc[fit], y[fit])
        X_monitor = selection_preprocessor.transform(X.iloc[monitor])
        selector = self._model(params, X_fit)
        selector.fit(X_fit, y[fit], eval_set=(X_monitor, y[monitor]),
                     early_stopping_rounds=self.patience, use_best_model=True)
        best = selector.get_best_iteration()
        selected = int(best) + 1 if best is not None and best >= 0 else int(selector.tree_count_)
        evaluation = selector.get_evals_result().get("validation", {})
        evaluated = max((len(values) for values in evaluation.values()), default=selected)
        selection_params = selector.get_all_params()
        selection_device = selection_params.get("task_type", "unknown").lower()
        if "learning_rate" not in params and "learning_rate" in selection_params:
            params["learning_rate"] = selection_params["learning_rate"]
        selection_seconds = time.monotonic() - started
        self.preprocessor_ = clone(template)
        started = time.monotonic()
        X_all = self.preprocessor_.fit_transform(X, y)
        self.model_ = self._model({**params, "iterations": selected}, X_all)
        self.model_.fit(X_all, y, use_best_model=False)
        self.classes_ = self.model_.classes_
        self.fit_indices_, self.monitor_indices_ = fit, monitor
        self.training_metadata_ = {
            "method": "internal_holdout_then_full_refit", "split_method": method,
            "fit_rows": len(fit), "monitor_rows": len(monitor), "refit_rows": len(y),
            "validation_fraction": self.validation_fraction, "patience": self.patience,
            "max_iterations": params["iterations"], "selected_iterations": selected,
            "selection_iterations_run": evaluated, "stopped_early": evaluated < params["iterations"],
            "selection_device": selection_device,
            "monitor_metric": selection_params.get("eval_metric", "unknown"),
            "fit_indices_sha256": hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest(),
            "monitor_indices_sha256": hashlib.sha256(monitor.astype("<i8").tobytes()).hexdigest(),
            "selection_seconds": selection_seconds, "refit_seconds": time.monotonic() - started,
        }
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        return np.asarray(self.model_.predict(self.preprocessor_.transform(X))).reshape(-1)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        return self.model_.predict_proba(self.preprocessor_.transform(X))
