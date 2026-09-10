from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    GridIntegrityError,
    TrainOnlyStandardizer,
    WindowSpec,
    audit_window_plan,
    build_window_plan,
    known_future_calendar_features,
    make_chronological_split,
    materialize_window,
)


def _synthetic_series(n_steps: int = 100) -> CanonicalGridSeries:
    time = pd.date_range(
        "2026-01-01",
        periods=n_steps,
        freq="15min",
        tz="UTC",
    )
    buses = pd.Index([10, 20])
    base = np.arange(n_steps * 2, dtype=np.float32).reshape(n_steps, 2)
    values = {
        "load_p_mw": base.copy(),
        "net_demand_p_mw": base.copy(),
    }
    return CanonicalGridSeries(
        grid_code="synthetic",
        time_utc=time,
        bus_index=buses,
        values=values,
        fingerprint="synthetic-fingerprint",
    )


def test_chronological_split_is_contiguous() -> None:
    split = make_chronological_split(
        100,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    assert split.bounds("train") == (0, 70)
    assert split.bounds("validation") == (70, 85)
    assert split.bounds("test") == (85, 100)


def test_windows_with_targets_crossing_boundaries_are_dropped() -> None:
    split = make_chronological_split(
        100,
        train_fraction=0.70,
        validation_fraction=0.15,
    )
    plan = build_window_plan(
        100,
        split,
        WindowSpec(
            history_steps=8,
            horizon_steps=4,
            lead_steps=1,
        ),
    )
    audit = audit_window_plan(split, plan)

    assert audit.passed
    assert audit.dropped_cross_boundary > 0

    for window in plan.for_split("train"):
        assert window.target_end < split.train_end

    for window in plan.for_split("validation"):
        assert split.validation_start <= window.target_start
        assert window.target_end < split.validation_end

    for window in plan.for_split("test"):
        assert split.test_start <= window.target_start
        assert window.target_end < split.test_end


def test_standardizer_is_fitted_only_on_train_rows() -> None:
    series = _synthetic_series()
    split = make_chronological_split(100)
    scaler = TrainOnlyStandardizer.fit(
        series,
        split,
        channels=("load_p_mw",),
    )

    direct_mean = series.channel("load_p_mw")[: split.train_end].mean(axis=0)
    assert np.allclose(
        scaler.means["load_p_mw"],
        direct_mean,
        rtol=0.0,
        atol=1e-12,
    )

    mutated_values = {
        name: value.copy() for name, value in series.values.items()
    }
    mutated_values["load_p_mw"][split.train_end :] += 1_000_000.0

    mutated = CanonicalGridSeries(
        grid_code=series.grid_code,
        time_utc=series.time_utc,
        bus_index=series.bus_index,
        values=mutated_values,
        fingerprint="mutated",
    )
    mutated_scaler = TrainOnlyStandardizer.fit(
        mutated,
        split,
        channels=("load_p_mw",),
    )

    assert np.array_equal(
        scaler.means["load_p_mw"],
        mutated_scaler.means["load_p_mw"],
    )
    assert np.array_equal(
        scaler.scales["load_p_mw"],
        mutated_scaler.scales["load_p_mw"],
    )


def test_materialized_window_has_strict_history_target_order() -> None:
    series = _synthetic_series()
    split = make_chronological_split(100)
    plan = build_window_plan(
        100,
        split,
        WindowSpec(
            history_steps=8,
            horizon_steps=4,
            lead_steps=1,
        ),
    )
    window = plan.for_split("train")[0]

    history, target, calendar = materialize_window(
        series,
        window,
        history_channels=("load_p_mw",),
        target_channels=("net_demand_p_mw",),
    )

    assert history.shape == (8, 2, 1)
    assert target.shape == (4, 2, 1)
    assert calendar.shape == (4, 8)
    assert window.history_end < window.target_start


def test_known_future_calendar_features_distinguish_dst_occurrences() -> None:
    time = pd.DatetimeIndex(
        [
            "2016-10-30 00:00:00+00:00",
            "2016-10-30 01:00:00+00:00",
        ]
    )
    frame = known_future_calendar_features(time)

    assert frame.iloc[0]["is_dst"] == pytest.approx(1.0)
    assert frame.iloc[1]["is_dst"] == pytest.approx(0.0)
    assert frame.iloc[0]["utc_offset_hours"] == pytest.approx(2.0)
    assert frame.iloc[1]["utc_offset_hours"] == pytest.approx(1.0)


def test_invalid_split_fractions_fail_closed() -> None:
    with pytest.raises(GridIntegrityError):
        make_chronological_split(
            100,
            train_fraction=0.9,
            validation_fraction=0.2,
        )
