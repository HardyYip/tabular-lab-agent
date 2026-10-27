from __future__ import annotations

import hashlib
import io
import json
import warnings
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from .config import Modalities


def _image(value, root, config):
    from PIL import Image, ImageOps

    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).is_absolute() or "://" in value:
        raise ValueError("image paths must be relative local paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("image path is missing or outside the image root")
    if path.stat().st_size > config.max_image_bytes:
        raise ValueError("image exceeds byte limit")
    raw = path.read_bytes()
    if len(raw) > config.max_image_bytes:
        raise ValueError("image exceeds byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw), formats=["PNG", "JPEG", "BMP", "WEBP"]) as source:
                if source.width * source.height > config.max_image_pixels:
                    raise ValueError("image exceeds pixel limit")
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                content_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                pixels = np.asarray(rgb.resize((config.image_size, config.image_size)), dtype=np.float32) / 255
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("invalid or oversized image") from exc
    gray = pixels @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    features = [np.asarray([0]), gray.reshape(-1)]
    features.extend(np.histogram(pixels[:, :, channel], bins=8, range=(0, 1))[0] / gray.size
                    for channel in range(3))
    gy, gx = np.gradient(gray)
    angles, magnitude = np.arctan2(gy, gx), np.hypot(gx, gy)
    for row_ids in np.array_split(np.arange(config.image_size), 4):
        for col_ids in np.array_split(np.arange(config.image_size), 4):
            cell = np.ix_(row_ids, col_ids)
            histogram = np.histogram(angles[cell], bins=8, range=(-np.pi, np.pi), weights=magnitude[cell])[0]
            features.append(histogram / max(float(histogram.sum()), 1e-8))
    return np.concatenate(features).astype(np.float32), content_hash, hashlib.sha256(raw).hexdigest()


def encode_features(frame, config: Modalities, base: Path, max_cells=20_000_000):
    columns = config.text_columns + config.image_columns
    if not columns:
        return frame.copy(), [[] for _ in range(len(frame))], None
    if set(columns) - set(frame.columns):
        raise ValueError("configured modality columns are missing")
    if any(str(name).startswith("__modal_") for name in frame.columns):
        raise ValueError("feature names beginning with __modal_ are reserved")
    image_dimensions = 1 + config.image_size ** 2 + 24 + 128
    dimensions = len(frame.columns) - len(columns) + len(config.text_columns) * config.text_features + (
        len(config.image_columns) * image_dimensions)
    if len(frame) * dimensions > max_cells:
        raise ValueError("multimodal features exceed max_dense_cells")
    result = frame.drop(columns=columns).copy()
    assets = [[] for _ in range(len(frame))]
    chunks, names, checksums, cache = [], [], {}, {}
    for index, name in enumerate(config.text_columns):
        text = frame[name].fillna("").astype(str)
        if text.str.len().gt(config.max_text_chars).any():
            raise ValueError("text exceeds max_text_chars")
        encoded = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=config.text_features,
                                   alternate_sign=False, dtype=np.float32).transform(text).toarray()
        chunks.append(encoded)
        names.extend(f"__modal_text_{index}_{i:04d}" for i in range(config.text_features))
    root = (config.image_root or base).resolve()
    for index, name in enumerate(config.image_columns):
        values = np.zeros((len(frame), image_dimensions), dtype=np.float32)
        for row, value in enumerate(frame[name]):
            if pd.isna(value) or not str(value).strip():
                values[row, 0] = 1
                continue
            value = str(value)
            if value not in cache:
                cache[value] = _image(value, root, config)
            values[row], content_hash, checksum = cache[value]
            assets[row].append(content_hash)
            checksums[value] = checksum
        chunks.append(values)
        names.extend(f"__modal_image_{index}_{i:04d}" for i in range(image_dimensions))
    if chunks:
        result = pd.concat([result, pd.DataFrame(np.concatenate(chunks, axis=1), columns=names, index=frame.index)],
                           axis=1)
    asset_hash = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest() if checksums else None
    return result, assets, asset_hash


def split_groups(frame, group_values, assets):
    parents = list(range(len(frame)))
    owners = {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    hashes = pd.util.hash_pandas_object(frame, index=False).tolist()
    for index in range(len(frame)):
        tokens = [("features", hashes[index]), *(("image", value) for value in assets[index])]
        if group_values is not None:
            tokens.append(("group", str(group_values[index])))
        for token in tokens:
            if token in owners:
                parents[find(index)] = find(owners[token])
            else:
                owners[token] = index
    return np.asarray([find(index) for index in range(len(frame))])
