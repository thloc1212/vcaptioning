"""PELT segmentation of frame features. Intervals are half-open [start, end)."""

import numpy as np


def segment(features, penalty=10.0, min_size=2, model="rbf"):
    import ruptures as rpt

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("features must have shape [T, D] with T > 0")
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if len(x) < 2 * min_size:
        return [(0, len(x))]
    boundaries = rpt.Pelt(model=model, min_size=min_size).fit(x).predict(pen=penalty)
    starts = [0] + boundaries[:-1]
    return [(start, end) for start, end in zip(starts, boundaries) if end > start]
