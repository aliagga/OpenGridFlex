import numpy as np
import pytest

from opengridflex.metrics.core import (
    decision_regret,
    interval_coverage,
    interval_score,
    mae,
    mean_interval_width,
    pinball_loss,
    rmse,
    violation_rate,
)


def test_point_metrics_known_values():
    y = np.array([0.0, 2.0])
    p = np.array([1.0, 2.0])
    assert mae(y, p) == pytest.approx(0.5)
    assert rmse(y, p) == pytest.approx(np.sqrt(0.5))


def test_probabilistic_metrics_known_values():
    y = np.array([0.0, 2.0])
    lo = np.array([-1.0, 1.0])
    hi = np.array([1.0, 3.0])
    assert interval_coverage(y, lo, hi) == 1.0
    assert mean_interval_width(lo, hi) == 2.0
    assert interval_score(y, lo, hi, alpha=0.1) == 2.0
    assert pinball_loss(y, y, 0.5) == 0.0


def test_metrics_fail_on_nonfinite_input():
    with pytest.raises(ValueError):
        mae([1.0, np.nan], [1.0, 2.0])


def test_negative_regret_raises_beyond_tolerance():
    with pytest.raises(ValueError):
        decision_regret(9.0, 10.0)


def test_metric_validation_paths() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mae([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-empty"):
        mae([], [])
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss([1.0], [1.0], 1.0)

    with pytest.raises(ValueError, match="identical shapes"):
        interval_coverage([1.0], [0.0, 0.0], [2.0])
    with pytest.raises(ValueError, match="lower interval"):
        interval_coverage([1.0], [2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        interval_coverage([np.nan], [0.0], [1.0])

    with pytest.raises(ValueError, match="same non-empty shape"):
        mean_interval_width([], [])
    with pytest.raises(ValueError, match="lower interval"):
        mean_interval_width([2.0], [1.0])

    with pytest.raises(ValueError, match="alpha"):
        interval_score([1.0], [0.0], [2.0], alpha=0.0)
    with pytest.raises(ValueError, match="identical non-empty"):
        interval_score([], [], [], alpha=0.1)
    with pytest.raises(ValueError, match="lower interval"):
        interval_score([1.0], [2.0], [1.0], alpha=0.1)


def test_regret_and_violation_rate_edge_cases() -> None:
    assert decision_regret(10.0 - 1e-10, 10.0) == 0.0
    assert violation_rate([0.0, 2.0, 4.0], 1.0, 3.0) == pytest.approx(2.0 / 3.0)

    with pytest.raises(ValueError, match="non-empty and finite"):
        violation_rate([], 0.0, 1.0)
    with pytest.raises(ValueError, match="non-empty and finite"):
        violation_rate([np.nan], 0.0, 1.0)
