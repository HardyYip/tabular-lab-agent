"""Reproducible public-data preparation; dataset-specific parsing never chooses models."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split

from .io import write_json

ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                 "hours_per_week", "native_country", "income"]


def download_zip(url: str, destination: Path) -> tuple[zipfile.ZipFile, str]:
    if destination.exists():
        content = destination.read_bytes()
    else:
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.content
        # Validate before committing a cached download; never extract untrusted paths.
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.testzip() is not None:
                raise ValueError("downloaded archive failed CRC verification")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return zipfile.ZipFile(io.BytesIO(content)), hashlib.sha256(content).hexdigest()


def prepare(name: str, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if name in {"demo", "breast-cancer", "wine"}:
        if name == "demo":
            X, y = make_classification(n_samples=360, n_features=8, n_informative=5,
                                       weights=[0.75, 0.25], random_state=42)
            frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
            frame["category"] = pd.Series(["a", "b", "c"] * 120)
            frame.loc[::17, "feature_0"] = float("nan")
            frame.loc[::19, "category"] = None
            frame["target"] = y
        else:
            bunch = (load_breast_cancer if name == "breast-cancer" else load_wine)(as_frame=True)
            frame = bunch.frame
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        return {"dataset": name, "rows": len(frame), "path": str(path), "source": "scikit-learn"}
    if name == "wine-quality":
        source = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
        archive, checksum = download_zip(source, destination / "raw" / "wine-quality.zip")
        frames = []
        with archive:
            for kind in ("red", "white"):
                frame = pd.read_csv(archive.open(f"winequality-{kind}.csv"), sep=";")
                frame["wine_type"] = kind
                frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        raw_rows = len(frame)
        frame = frame.drop_duplicates().reset_index(drop=True)
        # Task definition is fixed BEFORE experimentation. Never leave original quality as a feature.
        frame["quality_class"] = frame["quality"].map(lambda x: "low" if x <= 5 else "medium" if x == 6 else "high")
        frame = frame.drop(columns="quality")
        frame.to_csv(destination / "wine-quality.csv", index=False)
        result = {"dataset": name, "raw_rows": raw_rows, "rows": len(frame), "archive_sha256": checksum,
                  "source": source, "license": "CC BY 4.0",
                  "citation": "Cortez et al. (2009), https://doi.org/10.24432/C56S3T",
                  "task": "quality<=5: low; quality==6: medium; quality>=7: high; raw quality removed",
                  "duplicate_policy": "drop exact original rows; group equal feature rows during splitting"}
    elif name == "adult":
        source = "https://archive.ics.uci.edu/static/public/2/adult.zip"
        archive, checksum = download_zip(source, destination / "raw" / "adult.zip")
        with archive:
            train = pd.read_csv(archive.open("adult.data"), names=ADULT_COLUMNS,
                                skipinitialspace=True, na_values=["?"])
            test = pd.read_csv(archive.open("adult.test"), names=ADULT_COLUMNS,
                               skipinitialspace=True, na_values=["?"], skiprows=1)
        for frame in (train, test):
            frame["income"] = frame["income"].str.rstrip(".")
        # Keep the official test intact; remove train/test feature overlaps from development only.
        features = [c for c in train if c != "income"]
        test_hashes = set(pd.util.hash_pandas_object(test[features], index=False).tolist())
        development = train.loc[~pd.util.hash_pandas_object(train[features], index=False).isin(test_hashes)]
        development = development.drop_duplicates(subset=features).reset_index(drop=True)
        fit, validation = train_test_split(development, test_size=0.2,
                                           stratify=development.income, random_state=42)
        fit.to_csv(destination / "adult-train.csv", index=False)
        validation.to_csv(destination / "adult-validation.csv", index=False)
        test.to_csv(destination / "adult-test.csv", index=False)
        result = {"dataset": name, "official_train_rows": len(train), "train_rows": len(fit),
                  "validation_rows": len(validation), "official_test_rows": len(test),
                  "source": source, "archive_sha256": checksum, "license": "CC BY 4.0",
                  "citation": "Becker & Kohavi (1996), https://doi.org/10.24432/C5XW20",
                  "duplicate_policy": "official test unchanged; remove overlapping and repeated development features",
                  "validation_seed": 42, "validation_fraction_of_development": 0.2}
    else:
        raise ValueError(f"unknown dataset: {name}")
    write_json(destination / f"{name}-provenance.json", result)
    return result
