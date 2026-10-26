import json
import os
import shutil
from pathlib import Path

import pytest

from tabular_lab.data import load_dataset
from tabular_lab.sandbox import _inputs, docker_command, preflight, validate_code


def require_docker(image):
    try:
        preflight(image)
    except RuntimeError as exc:
        if os.environ.get("CI"):
            pytest.fail(f"Docker CI prerequisite failed: {exc}")
        pytest.skip(str(exc))


def test_codegen_syntax_and_interface():
    validate_code("def build_pipeline(seed, feature_schema):\n    return None\n")
    with pytest.raises(ValueError):
        validate_code("def wrong(): pass")
    with pytest.raises(SyntaxError):
        validate_code("def build_pipeline(:")


def test_docker_boundary_has_no_api_secret_or_eval_labels(task, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-in-container")
    data = load_dataset(task)
    _inputs(tmp_path, data.train.X, data.train.y, data.validation.X,
            "def build_pipeline(seed, feature_schema): pass", 42, 3)
    command = docker_command(task, "sha256:fixed-image", tmp_path / "container-input",
                             tmp_path / "container-output", "test-container")
    text = " ".join(command)
    assert "--network=none" in command and "--read-only" in command
    assert "--cap-drop=ALL" in command and "--user=65534:65534" in command
    assert "never-in-container" not in text and "DEEPSEEK" not in text
    names = {p.name for p in (tmp_path / "container-input").iterdir()}
    assert names == {"train.csv", "train_labels.npy", "eval.csv", "candidate.py", "request.json"}
    assert not any("test" in name or "eval_label" in name for name in names)


def test_missing_docker_error_is_actionable(monkeypatch):
    import tabular_lab.sandbox as sandbox
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sandbox, "_docker_candidates", lambda: [])
    with pytest.raises(RuntimeError, match="V1 needs no Docker"):
        preflight("tabular-lab-worker:0.1.0")


def test_new_docker_install_is_found_without_refreshing_parent_path(monkeypatch, tmp_path):
    import tabular_lab.sandbox as sandbox
    executable = tmp_path / "docker.exe"
    executable.write_text("discovery test fixture", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sandbox, "_docker_candidates", lambda: [tmp_path / "missing.exe", executable])
    assert sandbox.docker_executable() == str(executable)


@pytest.mark.docker
def test_real_container_train_score_and_export(task, tmp_path):
    from tabular_lab.sandbox import evaluate_code, finalize_code, predict_code
    require_docker(task.docker.image)
    code = '''from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
def build_pipeline(seed, feature_schema):
    numeric = make_pipeline(SimpleImputer(), StandardScaler())
    category = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    preprocess = make_column_transformer((numeric, make_column_selector(dtype_include="number")),
                                        (category, make_column_selector(dtype_exclude="number")))
    return make_pipeline(preprocess, LogisticRegression(max_iter=1000, random_state=seed))
'''
    data = load_dataset(task)
    trial = tmp_path / "trial"
    result = evaluate_code(task, data, code, trial, 60)
    assert result["status"] == "ok", result
    model = tmp_path / "model"
    model.mkdir()
    final = finalize_code(task, data, trial, tmp_path / "final", model, 60)
    assert final["status"] == "ok", final
    predictions, probabilities = predict_code(model, data.test.X,
        {"docker_image": task.docker.image, "seed": 42, "target": task.target, "classes": data.classes,
         "feature_schema": {c: str(data.train.X[c].dtype) for c in data.columns}})
    assert len(predictions) == len(data.test.y) and probabilities.shape == (len(data.test.y), 3)


@pytest.mark.docker
def test_codegen_controller_repairs_failed_candidate_with_replay(task, tmp_path):
    from tabular_lab.config import LLMConfig
    from tabular_lab.io import append_event, read_json
    from tabular_lab.runner import run_task
    require_docker(task.docker.image)
    responses = tmp_path / "responses.jsonl"
    broken = "def build_pipeline(seed, feature_schema):\n    raise ValueError('intentional test failure')\n"
    repaired = '''from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
def build_pipeline(seed, feature_schema):
    numeric = SimpleImputer()
    category = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    preprocess = make_column_transformer((numeric, make_column_selector(dtype_include="number")),
                                        (category, make_column_selector(dtype_exclude="number")))
    return make_pipeline(preprocess, ExtraTreesClassifier(n_estimators=40, random_state=seed, n_jobs=2))
'''
    for code in (broken, repaired):
        append_event(responses, {"event": "llm_response", "content": json.dumps(
            {"action": "code", "hypothesis": "Exercise controlled repair", "code": code})})
    task.llm = LLMConfig(provider="replay", replay_path=responses)
    task.strategy = "codegen"
    task.limits.max_trials = 4
    result = run_task(task)
    assert result["status"] == "completed", result
    assert result["trials"] == 4 and result["failed_trials"] == 1
    assert result["llm_calls"] == 2 and result["cost_cny_estimate"] == 0
    history = read_json(Path(result["path"]) / "experiments.json")
    assert history[2]["status"] == "failed" and history[3]["status"] == "ok"
    assert history[2]["spec"]["model"] == history[3]["spec"]["model"] == "generated"
