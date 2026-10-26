"""Trusted training worker. Generated V2 code is never imported by this process."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

from .config import ExperimentSpec
from .io import read_json, write_json
from .models import build_pipeline, score_predictions


def execute(request: Path):
    config = read_json(request)
    payload = joblib.load(request.parent / "input.joblib")
    if config["action"] == "predict":
        model = joblib.load(config["model_path"])
    else:
        spec = ExperimentSpec.model_validate(config["spec"])
        requested = config.get("device", "cpu")
        device = "gpu" if spec.model == "catboost" and (requested == "gpu" or
            requested == "auto" and shutil.which("nvidia-smi")) else "cpu"
        model = build_pipeline(spec, payload["X_train"], config["seed"], config["max_dense_cells"],
                               config["workers"], device)
        try:
            model.fit(payload["X_train"], payload["y_train"])
        except Exception:
            if requested != "auto" or device != "gpu":
                raise
            # Auto mode preserves a CPU path on unsupported drivers or small-memory devices.
            print("GPU training failed in auto mode; retrying the same candidate on CPU.")
            device = "cpu"
            model = build_pipeline(spec, payload["X_train"], config["seed"], config["max_dense_cells"],
                                   config["workers"], device)
            model.fit(payload["X_train"], payload["y_train"])
        joblib.dump(model, request.parent / "pipeline.joblib")
    predictions = np.asarray(model.predict(payload["X_eval"])).reshape(-1)
    probabilities = np.asarray(model.predict_proba(payload["X_eval"]))
    classes = np.asarray(model.classes_).astype(int)
    if sorted(classes.tolist()) != list(range(config["class_count"])):
        raise ValueError("model class mapping does not match task labels")
    probabilities = probabilities[:, np.argsort(classes)]
    np.savez(request.parent / "predictions.npz", predictions=predictions, probabilities=probabilities)
    result = {"status": "ok", "device_used": device if config["action"] != "predict" else "saved_model"}
    if "y_eval" in payload:
        result["metrics"] = score_predictions(payload["y_eval"], predictions, probabilities,
                                              config["class_count"])
    write_json(request.parent / "result.json", result)


if __name__ == "__main__":
    execute(Path(sys.argv[1]))
