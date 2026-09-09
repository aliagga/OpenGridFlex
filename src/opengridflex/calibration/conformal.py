from __future__ import annotations

import numpy as np


def conformal_radius(y_true, y_pred, alpha: float = 0.1) -> float:
    """Finite-sample split-conformal absolute-residual radius.

    Uses the standard ceil((n+1)(1-alpha))/n empirical quantile with the
    conservative ``higher`` quantile rule.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    if y.shape != yhat.shape or y.size == 0:
        raise ValueError("y_true and y_pred must have identical non-empty shapes")
    if not (np.isfinite(y).all() and np.isfinite(yhat).all()):
        raise ValueError("calibration inputs must be finite")
    r = np.abs(y - yhat).reshape(-1)
    q_level = min(1.0, np.ceil((len(r) + 1) * (1.0 - alpha)) / len(r))
    return float(np.quantile(r, q_level, method="higher"))


def interval(y_pred, radius: float) -> tuple[np.ndarray, np.ndarray]:
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be a finite non-negative number")
    y = np.asarray(y_pred, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("predictions must be finite")
    return y - radius, y + radius


def coverage(y_true, lo, hi) -> float:
    y = np.asarray(y_true, dtype=float)
    lower = np.asarray(lo, dtype=float)
    upper = np.asarray(hi, dtype=float)
    if not (y.shape == lower.shape == upper.shape) or y.size == 0:
        raise ValueError("y_true, lo and hi must have identical non-empty shapes")
    if np.any(lower > upper):
        raise ValueError("lo exceeds hi")
    return float(np.mean((y >= lower) & (y <= upper)))
