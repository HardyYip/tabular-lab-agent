"""One shared finite search space for mock, random and LLM strategies."""
from __future__ import annotations

import random

from .config import ExperimentSpec, Preprocess


def baselines():
    return [ExperimentSpec(model="dummy", hypothesis="Majority-class reference."),
            ExperimentSpec(model="logistic", hypothesis="Linear baseline with training-only preprocessing.")]


def search_space():
    candidates = []
    for balanced in (False, True):
        weight = "balanced" if balanced else None
        for C in (0.1, 1.0, 10.0):
            candidates.append(ExperimentSpec(model="logistic", params={"C": C, "class_weight": weight},
                                              hypothesis="Test linear regularization and class weighting."))
        for depth in (8, None):
            candidates.append(ExperimentSpec(model="extra_trees", params={"max_depth": depth,
                                              "class_weight": weight}, hypothesis="Test nonlinear tree interactions."))
        for rate in (0.03, 0.1):
            candidates.append(ExperimentSpec(model="hist_gradient_boosting", params={"learning_rate": rate,
                                              "class_weight": weight}, hypothesis="Test boosted nonlinear interactions."))
        for depth in (4, 6):
            candidates.append(ExperimentSpec(model="catboost", params={"depth": depth,
                "auto_class_weights": "Balanced" if balanced else None},
                hypothesis="Test native categorical boosting."))
        for preprocessing in (
            Preprocess(scale=False), Preprocess(impute="most_frequent"),
            Preprocess(min_frequency=5), Preprocess(select_percentile=75),
        ):
            candidates.append(ExperimentSpec(model="logistic", params={"class_weight": weight},
                preprocessing=preprocessing, hypothesis="Test a preprocessing alternative to the linear baseline."))
    return candidates


def allowed_fingerprints():
    return {spec.fingerprint() for spec in search_space() + baselines()}


def choose_random(history: list[dict], seed: int):
    seen = {h.get("fingerprint") for h in history}
    options = [s for s in search_space() if s.fingerprint() not in seen]
    random.Random(seed + len(history)).shuffle(options)
    return options[0] if options else None


def choose_mock(history: list[dict], seed: int):
    # This is explicitly an OFFLINE policy, not an LLM or an efficacy claim.
    attempted = {h.get("spec", {}).get("model") for h in history}
    preferred = ["extra_trees", "hist_gradient_boosting", "catboost"]
    seen = {h.get("fingerprint") for h in history}
    for model in preferred:
        for candidate in search_space():
            if candidate.model == model and model not in attempted and candidate.fingerprint() not in seen:
                return candidate
    return choose_random(history, seed)
