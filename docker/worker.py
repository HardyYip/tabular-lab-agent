"""Fixed evaluator inside the image. Candidate code gets no evaluation labels."""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

INPUT, OUTPUT = Path("/input"), Path("/output")


def main():
    config = json.loads((INPUT / "request.json").read_text())
    module_spec = importlib.util.spec_from_file_location("candidate", INPUT / "candidate.py")
    candidate = importlib.util.module_from_spec(module_spec)
    sys.modules["candidate"] = candidate
    module_spec.loader.exec_module(candidate)
    if config["action"] == "train":
        model = candidate.build_pipeline(config["seed"], config["feature_schema"])
        X = pd.read_csv(INPUT / "train.csv")
        for col, dtype in config["feature_schema"].items():
            if dtype == "object":
                X[col] = X[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
        model.fit(X, np.load(INPUT / "train_labels.npy", allow_pickle=False))
        joblib.dump(model, OUTPUT / "pipeline.joblib")
    else:
        model = joblib.load(INPUT / "pipeline.joblib")
    X_eval = pd.read_csv(INPUT / "eval.csv")
    for col, dtype in config.get("feature_schema", {}).items():
        if dtype == "object":
            X_eval[col] = X_eval[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    predictions = np.asarray(model.predict(X_eval)).reshape(-1)
    probabilities = np.asarray(model.predict_proba(X_eval), dtype=float)
    classes = np.asarray(model.classes_).astype(int)
    if sorted(classes.tolist()) != list(range(config["class_count"])):
        raise ValueError("invalid model classes")
    probabilities = probabilities[:, np.argsort(classes)]
    np.savez(OUTPUT / "predictions.npz", predictions=predictions, probabilities=probabilities)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Stack locations assist repair without forwarding messages that might contain raw data values.
        frames = [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                  for f in traceback.extract_tb(exc.__traceback__)[-8:]]
        (OUTPUT / "error.json").write_text(json.dumps({"error_type": type(exc).__name__, "frames": frames}))
