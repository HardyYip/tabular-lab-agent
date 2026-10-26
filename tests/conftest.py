from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tabular_lab.config import Limits, TaskConfig


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    import tabular_lab.credentials as credentials

    class BlockedStore:
        def __init__(self, *args, **kwargs):
            raise credentials.CredentialError("saved credentials are unavailable during tests")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "read_saved_key", lambda: None)
    monkeypatch.setattr(credentials, "WindowsCredentialStore", BlockedStore)


@pytest.fixture
def task(tmp_path: Path):
    X, y = make_classification(n_samples=180, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, random_state=7)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    frame["category"] = np.where(frame.x0 > 0, "alpha", "beta")
    frame.loc[::11, "category"] = None
    frame.loc[::13, "x1"] = np.nan
    frame["id"] = range(len(frame))
    frame["target"] = [f"class-{v}" for v in y]
    source = tmp_path / "dataset.csv"
    frame.to_csv(source, index=False)
    return TaskConfig(data_path=source, target="target", id_column="id", run_dir=tmp_path / "runs",
                      limits=Limits(max_trials=3, max_seconds=120, trial_seconds=30, final_seconds=30))
