from __future__ import annotations

import numpy as np


def _paired(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y, dtype=float)
    b = np.asarray(yhat, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y={a.shape}, yhat={b.shape}")
    if a.size == 0:
        raise ValueError("metric inputs must be non-empty")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("metric inputs must contain only finite values")
    return a, b


def mae(y, yhat) -> float:
    a, b = _paired(y, yhat)
    return float(np.mean(np.abs(a - b)))


def rmse(y, yhat) -> float:
    a, b = _paired(y, yhat)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pinball_loss(y, qhat, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    a, b = _paired(y, qhat)
    e = a - b
    return float(np.mean(np.maximum(quantile * e, (quantile - 1.0) * e)))


def interval_coverage(y, lower, upper) -> float:
    a = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError("y, lower and upper must have identical shapes")
    if np.any(lo > hi):
        raise ValueError("lower interval bound exceeds upper bound")
    if not (np.isfinite(a).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("interval inputs must contain only finite values")
    return float(np.mean((a >= lo) & (a <= hi)))


def mean_interval_width(lower, upper) -> float:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.shape != hi.shape or lo.size == 0:
        raise ValueError("lower and upper must have the same non-empty shape")
    if np.any(lo > hi):
        raise ValueError("lower interval bound exceeds upper bound")
    return float(np.mean(hi - lo))


def interval_score(y, lower, upper, alpha: float) -> float:
    """Mean Winkler interval score for a central 1-alpha prediction interval."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    a = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (a.shape == lo.shape == hi.shape) or a.size == 0:
        raise ValueError("y, lower and upper must have identical non-empty shapes")
    if np.any(lo > hi):
        raise ValueError("lower interval bound exceeds upper bound")
    score = (hi - lo) + (2.0 / alpha) * (lo - a) * (a < lo) + (2.0 / alpha) * (a - hi) * (a > hi)
    return float(np.mean(score))


def decision_regret(cost: float, oracle_cost: float, *, atol: float = 1e-9) -> float:
    regret = float(cost - oracle_cost)
    if regret < -atol:
        raise ValueError("cost is lower than oracle_cost; verify oracle definition or solver tolerances")
    return max(0.0, regret)


def violation_rate(values, lower, upper) -> float:
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("values must be non-empty and finite")
    return float(np.mean((v < lower) | (v > upper)))
