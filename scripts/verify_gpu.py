"""Small GPU runtime check; training accuracy here is not a benchmark result."""
from time import perf_counter

import numpy as np
from catboost import CatBoostClassifier

rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 12))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
start = perf_counter()
model = CatBoostClassifier(iterations=40, depth=4, task_type="GPU", devices="0",
                          verbose=False, allow_writing_files=False)
model.fit(X, y)
print({"device": model.get_all_params()["task_type"], "seconds": round(perf_counter() - start, 3)})
