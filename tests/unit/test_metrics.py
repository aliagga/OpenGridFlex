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
