from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitSlices:
    train: slice
    val: slice
    test: slice


def sensor_dropout(x: np.ndarray, rate: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mask observations completely at random for a controlled robustness test.

    Returns a float copy with missing values represented by NaN and a boolean
    observation mask (True = observed). Structured outages are implemented as
    separate shift families later; this function intentionally does not mix them.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1]; got {rate}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    arr = np.asarray(x)
    out = arr.astype(float, copy=True)
    rng = np.random.default_rng(seed)
    mask = rng.random(arr.shape) >= rate
    out[~mask] = np.nan
    return out, mask


def scale_der(profile: np.ndarray, factor: float) -> np.ndarray:
    if factor < 0:
        raise ValueError("DER scale factor must be non-negative")
    return np.asarray(profile, dtype=float) * factor


def chronological_split(
    n: int, train: float = 0.6, val: float = 0.2, *, min_partition_size: int = 1
) -> SplitSlices:
    """Return strictly chronological train/validation/test slices.

    No shuffling is permitted here; leakage-sensitive windowing is handled later.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if min_partition_size < 1:
        raise ValueError("min_partition_size must be >= 1")
    if not (0.0 < train < 1.0 and 0.0 < val < 1.0 and train + val < 1.0):
        raise ValueError("train and val must be in (0,1) and train + val < 1")
    i = int(n * train)
    j = int(n * (train + val))
    if i < min_partition_size or j - i < min_partition_size or n - j < min_partition_size:
        raise ValueError("split would create a partition smaller than min_partition_size")
    return SplitSlices(slice(0, i), slice(i, j), slice(j, n))
