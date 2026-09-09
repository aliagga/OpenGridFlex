import numpy as np
import pytest

from opengridflex.calibration.conformal import conformal_radius, coverage, interval


def test_conformal_radius_known_residuals():
    y = np.arange(10.0)
    pred = y + np.arange(10.0) / 10.0
    r = conformal_radius(y, pred, alpha=0.2)
    assert r == pytest.approx(0.9)


def test_interval_and_coverage():
    pred = np.array([0.0, 1.0])
    lo, hi = interval(pred, 0.5)
    assert coverage(np.array([0.25, 1.5]), lo, hi) == 1.0


def test_invalid_alpha_fails():
    with pytest.raises(ValueError):
        conformal_radius([1], [1], alpha=0)
