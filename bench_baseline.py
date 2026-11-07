"""Time the official ONNX baseline per step on this machine, for calibration."""
import sys
import time

import numpy as np

from common import DATA_ROOT

sys.path.insert(0, str(DATA_ROOT / "baseline"))
sys.path.insert(0, str(DATA_ROOT))
from solution import PredictionModel  # noqa: E402  (baseline's solution.py)
from utils import DataPoint  # noqa: E402

rng = np.random.default_rng(0)
x = rng.normal(size=(3000, 112)).astype(np.float32)
pm = PredictionModel()
t = time.perf_counter()
for i in range(3000):
    pm.predict(DataPoint(0, i, i >= 99, x[i]))
us = (time.perf_counter() - t) / 3000 * 1e6
print(f"baseline: {us:.1f} us/step -> ~{us * 39.4:.0f} s = {us * 39.4 / 60:.1f} min for test set")
