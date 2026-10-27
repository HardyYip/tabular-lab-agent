from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def prepare(name, destination: Path, rows=180):
    mixed = name == "multimodal-demo"
    images = destination / name / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    records = []
    for index in range(rows):
        label = index % 3
        image = Image.new("RGB", (40, 40), (240, 240, 240))
        draw = ImageDraw.Draw(image)
        x, y = rng.randint(6, 10), rng.randint(6, 10)
        size = rng.randint(21, 25)
        color = tuple(rng.randint(20, 140) for _ in range(3))
        if label == 0:
            draw.ellipse((x, y, x + size, y + size), fill=color)
        elif label == 1:
            draw.rectangle((x, y, x + size, y + size), fill=color)
        else:
            draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=color)
        relative = Path(name) / "images" / f"{index:04d}.png"
        image.save(destination / relative)
        row = {"image": relative.as_posix(), "target": ("circle", "square", "triangle")[label]}
        if mixed:
            row.update(caption=("curved closed boundary", "four straight boundaries", "three pointed corners")[label],
                       size=float(size), sensor=rng.random(), source=("a", "b")[index % 2])
        records.append(row)
    path = destination / f"{name}.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return {"dataset": name, "rows": rows, "path": str(path), "source": "deterministic synthetic shapes",
            "scope": "functional classification demo; not evidence of general image understanding"}
