import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import sparse

from tabular_lab.config import ExperimentSpec, TaskConfig
from tabular_lab.data import load_dataset, prediction_frame
from tabular_lab.models import DenseGuard, build_pipeline, score_predictions


def test_splits_repeatable_disjoint_and_profile_training_only(task):
    data = load_dataset(task)
    again = load_dataset(task)
    assert data.train.indices == again.train.indices
    assert len(data.train.X) == 108
    assert len(data.validation.X) == len(data.test.X) == 36
    a, b, c = map(set, [data.train.indices, data.validation.indices, data.test.indices])
    assert not (a & b or a & c or b & c)
    assert "id" not in data.columns and "target" not in data.columns
    assert data.profile["training_rows"] == len(data.train.X)
    original = pd.read_csv(task.data_path)
    original.loc[data.test.indices, "x0"] = 999999
    original.to_csv(task.data_path, index=False)
    changed = load_dataset(task)
    assert changed.train.indices == data.train.indices
    for before, after in zip(data.profile["columns"], changed.profile["columns"], strict=True):
        assert before.keys() == after.keys()
        for key in before:
            if isinstance(before[key], float):
                assert before[key] == pytest.approx(after[key], abs=1e-12)
            else:
                assert before[key] == after[key]


def test_equal_feature_groups_cannot_cross_splits(task):
    frame = pd.read_csv(task.data_path)
    copy = frame.copy()
    copy["id"] += len(frame)
    pd.concat([frame, copy]).to_csv(task.data_path, index=False)
    data = load_dataset(task)
    groups = [set(pd.util.hash_pandas_object(s.X, index=False)) for s in
              (data.train, data.validation, data.test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert "group" in data.split_source


@pytest.mark.parametrize("model", ["dummy", "logistic", "extra_trees", "hist_gradient_boosting", "catboost"])
def test_models_handle_mixed_missing_and_unseen_categories(task, model):
    data = load_dataset(task)
    spec = ExperimentSpec(model=model, hypothesis="test")
    pipeline = build_pipeline(spec, data.train.X, 42, 100_000, 2)
    pipeline.fit(data.train.X, data.train.y)
    unseen = data.validation.X.copy()
    unseen["category"] = "never-seen-before"
    predictions = pipeline.predict(unseen)
    probabilities = pipeline.predict_proba(unseen)
    metrics = score_predictions(data.validation.y, predictions, probabilities, 3)
    assert 0 <= metrics["macro_f1"] <= 1
    assert "roc_auc" not in metrics


def test_imputer_fits_only_training_data(task):
    data = load_dataset(task)
    pipeline = build_pipeline(ExperimentSpec(model="logistic", hypothesis="test"),
                              data.train.X, 42, 100_000, 2)
    pipeline.fit(data.train.X, data.train.y)
    stats = pipeline.named_steps["preprocess"].named_transformers_["numbers"].named_steps["impute"].statistics_
    expected = data.train.X.select_dtypes(include="number").median().to_numpy()
    np.testing.assert_allclose(stats, expected)


def test_schema_reorders_columns_and_rejects_missing_and_extra(task, tmp_path):
    data = load_dataset(task)
    metadata = {"columns": data.columns, "target": "target", "id_column": "id"}
    original = pd.read_csv(task.data_path)
    original[original.columns[::-1]].to_csv(tmp_path / "reordered.csv", index=False)
    frame, ids = prediction_frame(tmp_path / "reordered.csv", metadata)
    assert frame.columns.tolist() == data.columns and ids is not None
    original.drop(columns="x0").to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        prediction_frame(tmp_path / "bad.csv", metadata)


def test_strict_experiment_contract_and_dedup():
    with pytest.raises(ValidationError):
        ExperimentSpec(model="logistic", hypothesis="bad", params={"arbitrary_code": "x"})
    with pytest.raises(ValidationError):
        ExperimentSpec(model="catboost", hypothesis="bad", params={"depth": 100})
    a = ExperimentSpec(model="logistic", hypothesis="a")
    b = ExperimentSpec(model="logistic", hypothesis="different explanation", params={"C": 1})
    assert a.fingerprint() == b.fingerprint()


def test_dense_guard_checks_before_allocating():
    X = sparse.csr_matrix((1000, 1000))
    with pytest.raises(ValueError, match="max_dense_cells"):
        DenseGuard(100).fit_transform(X)


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        score_predictions([0, 1], [0, 1], [[2, -1], [0, 1]], 2)


def test_reject_empty_target_and_overlapping_external_splits(task, tmp_path):
    task = TaskConfig.model_validate({**task.model_dump(), "validation_path": task.data_path,
                                     "test_path": task.data_path})
    with pytest.raises(ValueError, match="overlapping"):
        load_dataset(task)
