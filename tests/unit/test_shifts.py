import numpy as np
import pytest

from opengridflex.shifts.core import chronological_split, scale_der, sensor_dropout


def test_sensor_dropout_is_reproducible_and_does_not_mutate_input():
    x = np.arange(100).reshape(10, 10)
    original = x.copy()
    a, ma = sensor_dropout(x, 0.25, seed=7)
    b, mb = sensor_dropout(x, 0.25, seed=7)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(ma, mb)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[ma], b[mb])


def test_sensor_dropout_validates_rate():
    with pytest.raises(ValueError):
        sensor_dropout(np.ones(5), 1.01)


def test_scale_der_rejects_negative_scale():
    with pytest.raises(ValueError):
        scale_der(np.ones(3), -0.1)


def test_chronological_split_has_no_overlap_and_full_coverage():
    s = chronological_split(100, train=0.6, val=0.2)
    assert (s.train.start, s.train.stop) == (0, 60)
    assert (s.val.start, s.val.stop) == (60, 80)
    assert (s.test.start, s.test.stop) == (80, 100)
