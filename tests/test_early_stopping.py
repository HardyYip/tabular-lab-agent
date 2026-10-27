import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from tabular_lab.data import load_dataset
from tabular_lab.early_stopping import EarlyStoppingCatBoost, internal_split
from tabular_lab.models import CatFrame
from tabular_lab.runtime import inspect_model


def test_internal_monitoring_keeps_duplicate_features_together(task):
    data = load_dataset(task)
    X = pd.concat([data.train.X, data.train.X], ignore_index=True)
    y = np.tile(data.train.y, 2)
    fit, monitor, method = internal_split(X, y, 0.2, 42)
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    assert not set(groups[fit]) & set(groups[monitor])
    assert set(y[fit]) == set(y[monitor]) == set(y)
    assert len(fit) + len(monitor) == len(y)
    assert method == "stratified_feature_groups"
    assert np.array_equal(fit, internal_split(X, y, 0.2, 42)[0])


def test_internal_monitoring_rejects_unsplittable_classes():
    with pytest.raises(ValueError, match="internal monitoring folds"):
        internal_split(pd.DataFrame({"x": [1, 2, 3]}), np.array([0, 0, 1]), 0.2, 42)


def test_early_stopping_preprocessing_is_fitted_inside_each_stage(task, monkeypatch):
    import tabular_lab.early_stopping as module
    data = load_dataset(task)
    fits, transformations = [], []

    class SpyFrame(CatFrame):
        def fit(self, X, y=None):
            fits.append(set(X.index))
            return super().fit(X, y)

        def transform(self, X):
            transformations.append(set(X.index))
            return super().transform(X)

    trained = []

    class FakeCatBoost:
        def __init__(self, **params):
            self.params = params
            trained.append(self)

        def fit(self, X, y, **kwargs):
            self.X, self.y, self.options = X.copy(), np.asarray(y), kwargs
            self.classes_ = np.unique(y)
            self.tree_count_ = 3 if "eval_set" in kwargs else self.params["iterations"]
            return self

        def get_best_iteration(self):
            return 2

        def get_evals_result(self):
            return {"validation": {"MultiClass": [1.0, 0.9, 0.8, 0.85, 0.86]}}

        def get_all_params(self):
            return {"task_type": "CPU", "eval_metric": "MultiClass", "learning_rate": 0.07}

    monkeypatch.setattr(module, "CatBoostClassifier", FakeCatBoost)
    params = {"iterations": 20}
    model = EarlyStoppingCatBoost(params=params, patience=2, preprocessor=SpyFrame())
    assert clone(model).params == params
    model.fit(data.train.X, data.train.y)
    fit, monitor = set(model.fit_indices_), set(model.monitor_indices_)
    assert not fit & monitor and fit | monitor == set(range(len(data.train.y)))
    assert fits == [fit, fit | monitor]
    assert transformations[:3] == [fit, monitor, fit | monitor]
    assert trained[0].options["early_stopping_rounds"] == 2
    assert len(trained[0].X) == len(fit)
    assert len(trained[0].options["eval_set"][0]) == len(monitor)
    assert trained[1].params["iterations"] == 3
    assert trained[1].params["learning_rate"] == 0.07
    assert trained[1].options == {"use_best_model": False}
    assert len(trained[1].X) == len(data.train.y)
    assert model.training_metadata_["selection_iterations_run"] == 5
    assert model.training_metadata_["selected_iterations"] == 3
    assert model.training_metadata_["stopped_early"] is True
    assert params == {"iterations": 20}


def test_real_cpu_early_stopping_refits_and_audits_missing_mixed_data(task):
    data = load_dataset(task)
    model = EarlyStoppingCatBoost(params={"iterations": 25, "depth": 3, "thread_count": 2}, patience=4)
    model.fit(data.train.X, data.train.y)
    unseen = data.validation.X.copy()
    unseen["category"] = "unseen-category"
    assert model.predict_proba(unseen).shape == (len(unseen), 3)
    assert model.model_.tree_count_ == model.training_metadata_["selected_iterations"]
    assert model.training_metadata_["refit_rows"] == len(data.train.y)
    assert model.preprocessor_.medians_.equals(data.train.X[model.preprocessor_.numeric_].median().fillna(0))
    report = inspect_model(model)
    assert report["device_used"] == "cpu" and report["complete"] is True
    assert report["estimators"][0]["early_stopping_status"] == "verified_internal_holdout"
    assert report["early_stopping"][0]["refit_rows"] == len(data.train.y)


def test_device_audit_follows_custom_wrapper_and_does_not_guess_unknown():
    from types import SimpleNamespace
    X = [[0], [1], [2], [3]]
    model = LogisticRegression().fit(X, [0, 0, 1, 1])
    wrapped = inspect_model(SimpleNamespace(model_=model))
    assert wrapped["device_used"] == "cpu" and wrapped["complete"] is True
    assert wrapped["estimators"][0]["path"] == "model.model_"
    unknown = inspect_model(SimpleNamespace())
    assert unknown["device_used"] == "unknown" and unknown["complete"] is False


def test_direct_catboost_detector_is_reported_as_unverified():
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=4, depth=2, thread_count=2, allow_writing_files=False, verbose=False)
    X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]
    model.fit(X, y, eval_set=(X, y), early_stopping_rounds=2)
    assert inspect_model(model)["estimators"][0]["early_stopping_status"] == "unverified"


def test_runtime_metadata_is_bounded_and_strict(tmp_path):
    from pydantic import ValidationError

    from tabular_lab.io import write_json
    from tabular_lab.sandbox import _runtime
    assert _runtime(tmp_path)["device_used"] == "unknown"
    output = tmp_path / "container-output"
    output.mkdir()
    write_json(output / "runtime.json", {"device_used": "gpu", "complete": True, "estimators": []})
    with pytest.raises(ValidationError):
        _runtime(tmp_path)
    (output / "runtime.json").write_bytes(b"x" * 128_001)
    with pytest.raises(ValueError, match="metadata artifact"):
        _runtime(tmp_path)


def test_failed_final_refit_exports_candidate_device_evidence(task, tmp_path, monkeypatch):
    import tabular_lab.sandbox as sandbox
    from tabular_lab.io import read_json, write_json
    data = load_dataset(task)
    candidate = tmp_path / "candidate"
    inputs, outputs = candidate / "container-input", candidate / "container-output"
    inputs.mkdir(parents=True)
    outputs.mkdir()
    (inputs / "candidate.py").write_text("def build_pipeline(seed, feature_schema): pass\n", encoding="utf-8")
    (outputs / "pipeline.joblib").write_bytes(b"opaque model must not be loaded on host")
    write_json(candidate / "image.json", {"image_id": "sha256:recorded", "device": "cpu"})
    runtime = inspect_model(LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1]))
    write_json(outputs / "runtime.json", runtime)
    monkeypatch.setattr(sandbox, "_execute", lambda *args: {"status": "timeout", "seconds": 1})
    model = tmp_path / "model"
    model.mkdir()
    result = sandbox.finalize_code(task, data, candidate, tmp_path / "final", model, 1)
    assert result["status"] == "timeout"
    assert read_json(model / "runtime.json") == runtime
    assert (model / "pipeline.joblib").read_bytes() == (outputs / "pipeline.joblib").read_bytes()
