from types import SimpleNamespace

import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from tabular_lab.runtime import inspect_model


@pytest.fixture
def fitted():
    return LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])


@pytest.mark.parametrize("attr,collection", [
    ("members_", "list"), ("members_", "dict"), ("estimators_", "named")])
def test_custom_collections_and_shared_members_are_audited(fitted, attr, collection):
    members = {"a": fitted, "b": fitted} if collection == "dict" else (
        [("a", fitted), ("b", fitted)] if collection == "named" else [fitted, fitted])
    report = inspect_model(SimpleNamespace(**{attr: members}))
    assert report["complete"] and report["device_used"] == "cpu"
    assert len(report["estimators"]) == 1
    assert report["estimators"][0]["path"].startswith("model." + attr)


@pytest.mark.parametrize("members", ["unsupported", [None], iter([1])])
def test_unknown_members_never_become_complete_by_ignoring_them(fitted, members):
    report = inspect_model(SimpleNamespace(model_=fitted, members_=members))
    assert not report["complete"] and report["device_used"] == "unknown"


def test_cycles_and_member_count_are_bounded(fitted):
    node = SimpleNamespace(members_=[fitted])
    node.members_.append(node)
    cyclic = inspect_model(node)
    assert not cyclic["complete"] and cyclic["device_used"] == "unknown"
    large = inspect_model(SimpleNamespace(members_=[fitted] * 65))
    assert not large["complete"] and len(large["estimators"]) <= 64


def test_depth_and_unique_node_limits_are_bounded(fitted):
    node = fitted
    for _ in range(20):
        node = SimpleNamespace(members_=[node])
    assert not inspect_model(node)["complete"]
    wide = SimpleNamespace(members_=[SimpleNamespace() for _ in range(64)])
    report = inspect_model(wide)
    assert not report["complete"] and len(report["estimators"]) <= 64


def test_failing_property_does_not_hide_incomplete_audit(fitted):
    class Broken:
        model_ = fitted

        @property
        def members_(self):
            raise ValueError("private data must not be exposed")

    report = inspect_model(Broken())
    assert not report["complete"] and report["device_used"] == "unknown"
    assert "private data" not in str(report)


def test_sklearn_tree_ensemble_keeps_known_cpu_evidence():
    model = ExtraTreesClassifier(n_estimators=70, random_state=0).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    report = inspect_model(model)
    assert report["complete"] and report["device_used"] == "cpu"


def test_mixed_fitted_devices_are_combined_without_guessing(monkeypatch):
    import tabular_lab.runtime as module

    class FittedCatBoost:
        tree_count_ = 3

        def __init__(self, device):
            self.device = device

        def get_all_params(self):
            return {"task_type": self.device}

    monkeypatch.setattr(module, "CatBoost", FittedCatBoost)
    report = inspect_model(SimpleNamespace(members_=[FittedCatBoost("CPU"), FittedCatBoost("GPU")]))
    assert report["complete"] and report["device_used"] == "mixed"


def test_nested_helper_and_unverified_sibling_keep_separate_evidence(task):
    from catboost import CatBoostClassifier

    from tabular_lab.data import load_dataset
    from tabular_lab.early_stopping import EarlyStoppingCatBoost

    data = load_dataset(task)
    verified = EarlyStoppingCatBoost(params={"iterations": 8, "depth": 2, "thread_count": 2}, patience=2)
    verified.fit(data.train.X, data.train.y)
    direct = CatBoostClassifier(iterations=4, depth=2, thread_count=2, verbose=False, allow_writing_files=False)
    direct.fit([[0], [1], [2], [3]], [0, 0, 1, 1], eval_set=([[0], [1], [2], [3]], [0, 0, 1, 1]),
               early_stopping_rounds=2)
    verified.members_ = [direct]
    report = inspect_model(SimpleNamespace(members_=[verified]))
    assert report["complete"] and len(report["early_stopping"]) == 1
    assert {entry["early_stopping_status"] for entry in report["estimators"]} == {
        "unverified", "verified_internal_holdout"}
    malformed = EarlyStoppingCatBoost()
    malformed.model_ = direct
    malformed.training_metadata_ = {}
    incomplete = inspect_model(malformed)
    assert not incomplete["complete"] and incomplete["device_used"] == "unknown"
    assert incomplete["estimators"][0]["early_stopping_status"] == "unverified"
