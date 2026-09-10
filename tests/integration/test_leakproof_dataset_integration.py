from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opengridflex.data.leakproof_dataset import (
    CANONICAL_CHANNELS,
    TrainOnlyStandardizer,
    WindowSpec,
    audit_window_plan,
    build_canonical_grid_series,
    build_window_plan,
    make_chronological_split,
)

GRID_CODE = "1-MV-urban--1-no_sw"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def series():
    return build_canonical_grid_series(GRID_CODE)


def test_canonical_bus_series_is_complete_and_finite(series) -> None:
    assert series.n_steps == 35136
    assert series.n_buses == 136
    assert tuple(series.values) == CANONICAL_CHANNELS
    assert str(series.time_utc.tz) == "UTC"
    assert not series.time_utc.has_duplicates
    assert series.time_utc.is_monotonic_increasing

    deltas = series.time_utc[1:] - series.time_utc[:-1]
    assert set(deltas) == {pd.Timedelta(minutes=15)}

    for array in series.values.values():
        assert array.shape == (35136, 136)
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_net_demand_identity_holds_at_every_bus_and_timestep(series) -> None:
    expected = (
        series.channel("load_p_mw")
        + series.channel("storage_p_mw")
        - series.channel("sgen_p_mw")
        - series.channel("gen_p_mw")
    )

    assert np.allclose(
        series.channel("net_demand_p_mw"),
        expected,
        rtol=0.0,
        atol=2e-6,
    )
    assert np.array_equal(
        series.channel("net_demand_q_mvar"),
        series.channel("load_q_mvar"),
    )


def test_real_window_plan_passes_leakage_audit(series) -> None:
    split = make_chronological_split(series.n_steps)
    plan = build_window_plan(
        series.n_steps,
        split,
        WindowSpec(
            history_steps=96,
            horizon_steps=16,
            lead_steps=1,
        ),
    )
    audit = audit_window_plan(split, plan)

    assert audit.passed
    assert audit.train_windows > 0
    assert audit.validation_windows > 0
    assert audit.test_windows > 0
    assert audit.dropped_cross_boundary > 0


def test_day_ahead_window_plan_is_also_leak_free(series) -> None:
    split = make_chronological_split(series.n_steps)
    plan = build_window_plan(
        series.n_steps,
        split,
        WindowSpec(
            history_steps=672,
            horizon_steps=96,
            lead_steps=1,
        ),
    )
    audit = audit_window_plan(split, plan)

    assert audit.passed


def test_real_standardizer_uses_train_rows_only(series) -> None:
    split = make_chronological_split(series.n_steps)
    scaler = TrainOnlyStandardizer.fit(
        series,
        split,
        channels=(
            "net_demand_p_mw",
            "net_demand_q_mvar",
        ),
    )

    direct = series.channel("net_demand_p_mw")[: split.train_end].astype(np.float64)
    assert np.allclose(
        scaler.means["net_demand_p_mw"],
        direct.mean(axis=0),
        rtol=0.0,
        atol=1e-12,
    )
    assert scaler.fit_end == split.train_end
    assert scaler.source_fingerprint == series.fingerprint
