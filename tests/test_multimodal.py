import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from tabular_lab.config import Limits, Modalities, TaskConfig
from tabular_lab.data import load_dataset
from tabular_lab.features import encode_features
from tabular_lab.io import read_json, write_json
from tabular_lab.multimodal_demo import prepare
from tabular_lab.runner import predict, run_task


def demo_task(tmp_path, mixed=True):
    name = "multimodal-demo" if mixed else "image-demo"
    result = prepare(name, tmp_path, rows=90)
    return TaskConfig(data_path=Path(result["path"]), target="target", strategy="baseline",
                      modalities=Modalities(image_columns=["image"], text_columns=["caption"] if mixed else []),
                      run_dir=tmp_path / "runs", limits=Limits(max_trials=2, max_seconds=180,
                                                              trial_seconds=30, final_seconds=30))


@pytest.mark.parametrize("mixed", [False, True])
def test_image_and_multimodal_export_predict_in_fresh_process(tmp_path, mixed):
    task = demo_task(tmp_path, mixed)
    summary = run_task(task)
    assert summary["status"] == "completed" and summary["llm_calls"] == 0
    assert summary["test_metrics"]["macro_f1"] > 0.5
    folder = Path(summary["path"])
    metadata = read_json(folder / "model" / "metadata.json")
    assert "image" in metadata["input_columns"] and "image" not in metadata["columns"]
    original = pd.read_csv(task.data_path).drop(columns="target").iloc[:9]
    source, output = tmp_path / "new.csv", tmp_path / "new-predictions.csv"
    original.to_csv(source, index=False)
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    subprocess.run([sys.executable, "-m", "tabular_lab", "predict", "--model", str(folder / "model"),
                    "--input", str(source), "--output", str(output)], env=env, check=True, timeout=40)
    assert len(pd.read_csv(output)) == 9
    moved = tmp_path / "moved"
    moved.mkdir()
    moved_source = moved / "new.csv"
    original.to_csv(moved_source, index=False)
    with pytest.raises(ValueError, match="outside the image root|missing"):
        predict(folder / "model", moved_source, moved / "failed.csv")
    predict(folder / "model", moved_source, moved / "predictions.csv", image_root=tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(output), pd.read_csv(moved / "predictions.csv"))
    assert "caption" not in metadata["columns"]
    profile = json.dumps(read_json(folder / "profile.json"))
    assert "curved closed boundary" not in profile and str(tmp_path) not in profile


def test_stateless_text_encoding_has_no_validation_vocabulary_or_batch_statistics():
    config = Modalities(text_columns=["caption"], text_features=32)
    train = pd.DataFrame({"caption": ["中文图形描述", "rounded boundary"]})
    encoded, _, _ = encode_features(train, config, Path("."))
    together, _, _ = encode_features(pd.concat([train, pd.DataFrame({"caption": ["validation-only term"]})],
                                              ignore_index=True), config, Path("."))
    np.testing.assert_array_equal(encoded.to_numpy(), together.iloc[:2].to_numpy())
    missing, _, _ = encode_features(pd.DataFrame({"caption": [None, ""]}), config, Path("."))
    assert not missing.to_numpy().any()


@pytest.mark.parametrize("value", ["../outside.png", "https://example.com/a.png", "C:\\private\\a.png"])
def test_image_paths_cannot_escape_root_or_fetch_urls(tmp_path, value):
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "outside.png")
    with pytest.raises(ValueError, match="relative local|outside the image root"):
        encode_features(pd.DataFrame({"image": [value]}), Modalities(image_columns=["image"]), root)


def test_image_pixel_byte_and_feature_limits_are_checked(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "small.png")
    frame = pd.DataFrame({"image": ["small.png"]})
    with pytest.raises(ValueError, match="pixel limit"):
        encode_features(frame, Modalities(image_columns=["image"], max_image_pixels=64), tmp_path)
    noise = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(noise).save(tmp_path / "large.png")
    with pytest.raises(ValueError, match="byte limit"):
        encode_features(pd.DataFrame({"image": ["large.png"]}),
                        Modalities(image_columns=["image"], max_image_bytes=1024), tmp_path)
    with pytest.raises(ValueError, match="max_dense_cells"):
        encode_features(frame, Modalities(image_columns=["image"]), tmp_path, max_cells=100)
    with pytest.raises(ValueError, match="max_text_chars"):
        encode_features(pd.DataFrame({"text": ["a" * 257]}),
                        Modalities(text_columns=["text"], max_text_chars=256), tmp_path)


def test_missing_and_invalid_images_have_distinct_behavior(tmp_path):
    values, assets, _ = encode_features(pd.DataFrame({"image": [None]}),
                                         Modalities(image_columns=["image"]), tmp_path)
    assert values.iloc[0, 0] == 1 and assets == [[]]
    (tmp_path / "invalid.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalid"):
        encode_features(pd.DataFrame({"image": ["invalid.png"]}), Modalities(image_columns=["image"]), tmp_path)


def test_same_image_pixels_with_different_names_and_text_stay_in_one_outer_fold(tmp_path):
    task = demo_task(tmp_path)
    frame = pd.read_csv(task.data_path)
    duplicates = frame.iloc[:9].copy()
    for index in duplicates.index:
        source = tmp_path / duplicates.loc[index, "image"]
        destination = tmp_path / f"duplicate-{index}.png"
        with Image.open(source) as image:
            image.save(destination, compress_level=0)
        duplicates.loc[index, "image"] = destination.name
        duplicates.loc[index, "caption"] = "different accompanying text"
    frame = pd.concat([frame, duplicates], ignore_index=True)
    frame.to_csv(task.data_path, index=False)
    data = load_dataset(task)
    folds = [set(split.indices) for split in (data.train, data.validation, data.test)]
    assert all(any(index in fold and index + 90 in fold for fold in folds) for index in range(9))


def test_external_image_overlap_is_rejected_even_with_different_text(tmp_path):
    task = demo_task(tmp_path)
    frame = pd.read_csv(task.data_path)
    train, validation, test = frame.iloc[:54].copy(), frame.iloc[54:72].copy(), frame.iloc[72:].copy()
    test.iloc[0, test.columns.get_loc("image")] = train.iloc[0]["image"]
    test.iloc[0, test.columns.get_loc("caption")] = "unseen accompanying text"
    paths = [tmp_path / (name + ".csv") for name in ("train", "validation", "test")]
    for values, path in zip((train, validation, test), paths, strict=True):
        values.to_csv(path, index=False)
    task = TaskConfig.model_validate({**task.model_dump(), "data_path": paths[0],
                                      "validation_path": paths[1], "test_path": paths[2]})
    with pytest.raises(ValueError, match="overlapping image content"):
        load_dataset(task)


def test_image_asset_changes_are_fingerprinted(tmp_path):
    task = demo_task(tmp_path, mixed=False)
    first = load_dataset(task).hashes
    path = tmp_path / pd.read_csv(task.data_path).iloc[0]["image"]
    with Image.open(path) as image:
        values = np.asarray(image).copy()
    values[0, 0] = [0, 0, 0]
    Image.fromarray(values).save(path)
    second = load_dataset(task).hashes
    assert first["data"] == second["data"] and first["data_assets"] != second["data_assets"]


def test_subject_group_column_is_not_a_feature_and_stays_disjoint(task):
    frame = pd.read_csv(task.data_path)
    frame["subject"] = np.arange(len(frame)) // 3
    frame.to_csv(task.data_path, index=False)
    task.group_column = "subject"
    data = load_dataset(task)
    assert "subject" not in data.columns and "subject" not in data.input_columns
    folds = [set(frame.iloc[split.indices].subject) for split in (data.train, data.validation, data.test)]
    assert not folds[0] & folds[1] and not folds[0] & folds[2] and not folds[1] & folds[2]


def test_modality_configuration_rejects_target_and_overlapping_columns(tmp_path):
    with pytest.raises(ValidationError):
        Modalities(text_columns=["same"], image_columns=["same"])
    with pytest.raises(ValidationError):
        TaskConfig(data_path=tmp_path / "input.csv", target="label", modalities=Modalities(text_columns=["label"]))


def test_inspect_command_uses_training_profile_without_credentials(tmp_path):
    import yaml

    from tabular_lab.cli import app

    task = demo_task(tmp_path)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    result = CliRunner().invoke(app, ["inspect", "--task", str(path)])
    assert result.exit_code == 0, result.output
    values = json.loads(result.output)
    assert values["new_api_requests"] == 0 and values["encoded_feature_count"] > len(values["input_columns"])
    assert values["training_profile"]["training_rows"] == values["split_rows"]["train"]
    assert "test_metrics" not in values and "curved closed boundary" not in result.output


@pytest.mark.docker
def test_multimodal_features_work_with_generated_container_export(task, tmp_path):
    from tabular_lab.sandbox import evaluate_code, finalize_code, preflight

    try:
        preflight(task.docker.image)
    except RuntimeError as exc:
        if os.environ.get("CI"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    image_task = demo_task(tmp_path, mixed=False)
    image_task.docker = task.docker
    data = load_dataset(image_task)
    code = '''from sklearn.ensemble import ExtraTreesClassifier
def build_pipeline(seed, feature_schema):
    return ExtraTreesClassifier(n_estimators=30, random_state=seed, n_jobs=2)
'''
    trial = tmp_path / "image-trial"
    result = evaluate_code(image_task, data, code, trial, 60)
    assert result["status"] == "ok", result
    files = {path.name for path in (trial / "container-input").iterdir()}
    assert not any(name.endswith(".png") for name in files)
    assert set(pd.read_csv(trial / "container-input" / "train.csv").columns) == set(data.columns)
    model = tmp_path / "image-model"
    model.mkdir()
    final_folder = tmp_path / "image-final"
    final = finalize_code(image_task, data, trial, final_folder, model, 60)
    assert final["status"] == "ok", final
    metadata = {"backend": "docker", "columns": data.columns, "input_columns": data.input_columns,
                "classes": data.classes, "target": image_task.target, "seed": image_task.seed,
                "modalities": image_task.modalities.model_dump(mode="json"), "docker_image": task.docker.image}
    write_json(model / "metadata.json", metadata)
    source = tmp_path / "image-test.csv"
    pd.read_csv(image_task.data_path).iloc[data.test.indices].drop(columns="target").to_csv(source, index=False)
    output = tmp_path / "image-predictions.csv"
    predict(model, source, output)
    with np.load(final_folder / "predictions.npz", allow_pickle=False) as saved:
        assert pd.read_csv(output).prediction.tolist() == [data.classes[int(value)] for value in saved["predictions"]]
