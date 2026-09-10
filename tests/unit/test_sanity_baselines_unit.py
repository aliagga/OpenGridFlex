from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opengridflex.baselines.persistence import (
    DAILY_SEASONAL,
    LAST_VALUE,
    WEEKLY_SEASONAL,
    predict_window,
    source_indices_for_window,
)
from opengridflex.data.leakproof_dataset import CanonicalGridSeries, ForecastWindow
from opengridflex.evaluation.deterministic import StreamingDeterministicMetrics
from opengridflex.grids.simbench_adapter import GridIntegrityError


def _series(n_steps: int = 1000) -> CanonicalGridSeries:
    values = np.arange(n_steps * 2, dtype=np.float32).reshape(n_steps, 2)
    return CanonicalGridSeries(
        grid_code="synthetic",
        time_utc=pd.date_range(
            "2026-01-01",
            periods=n_steps,
            freq="15min",
            tz="UTC",
        ),
        bus_index=pd.Index([0, 1]),
        values={"net_demand_p_mw": values},
        fingerprint="synthetic",
    )


def _window(origin: int, horizon: int = 4) -> ForecastWindow:
    return ForecastWindow(
        split="validation",
        history_start=origin - 9,
        history_end=origin,
        origin=origin,
        target_start=origin + 1,
        target_end=origin + horizon,
    )


def test_last_value_uses_only_forecast_origin() -> None:
    window = _window(700)
    source = source_indices_for_window(window, LAST_VALUE)

    assert np.array_equal(source, np.array([700, 700, 700, 700]))


def test_daily_seasonal_uses_target_minus_96() -> None:
    window = _window(700)
    source = source_indices_for_window(window, DAILY_SEASONAL)

    assert np.array_equal(source, np.array([605, 606, 607, 608]))
    assert np.all(source <= window.origin)


def test_weekly_seasonal_uses_target_minus_672() -> None:
    window = _window(700)
    source = source_indices_for_window(window, WEEKLY_SEASONAL)

    assert np.array_equal(source, np.array([29, 30, 31, 32]))


def test_seasonal_baseline_fails_if_horizon_would_read_future() -> None:
    window = _window(700, horizon=100)

    with pytest.raises(GridIntegrityError, match="future measured"):
        source_indices_for_window(window, DAILY_SEASONAL)


def test_mutating_future_values_cannot_change_persistence_prediction() -> None:
    series = _series()
    window = _window(700)

    before = predict_window(
        series,
        "net_demand_p_mw",
        window,
        DAILY_SEASONAL,
    ).copy()

    series.values["net_demand_p_mw"][window.target_start :] += 1_000_000.0

    after = predict_window(
        series,
        "net_demand_p_mw",
        window,
        DAILY_SEASONAL,
    )

    assert np.array_equal(before, after)


def test_streaming_metrics_are_zero_for_perfect_forecast() -> None:
    accumulator = StreamingDeterministicMetrics(
        n_buses=2,
        horizon_steps=3,
        active_bus_mask=np.array([True, True]),
    )
    target = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )
    accumulator.update(target, target.copy())
    result = accumulator.finalize()

    assert result["bus_micro"]["mae"] == pytest.approx(0.0)
    assert result["bus_micro"]["rmse"] == pytest.approx(0.0)
    assert result["bus_micro"]["wape"] == pytest.approx(0.0)
    assert result["bus_macro_active"]["mae"] == pytest.approx(0.0)
    assert result["system_aggregate"]["mae"] == pytest.approx(0.0)


def test_macro_active_excludes_inactive_bus() -> None:
    accumulator = StreamingDeterministicMetrics(
        n_buses=2,
        horizon_steps=1,
        active_bus_mask=np.array([True, False]),
    )
    target = np.array([[1.0, 0.0]])
    prediction = np.array([[2.0, 100.0]])
    accumulator.update(target, prediction)
    result = accumulator.finalize()

    assert result["bus_macro_active"]["active_buses"] == 1
    assert result["bus_macro_active"]["mae"] == pytest.approx(1.0)
