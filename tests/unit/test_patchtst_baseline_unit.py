from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opengridflex.baselines.patchtst_global import (
    build_channel_panel,
    validation_request,
)
from opengridflex.data.leakproof_dataset import CanonicalGridSeries
from opengridflex.grids.simbench_adapter import GridIntegrityError

GRID_CODE = "1-MV-urban--1-no_sw"
N_STEPS = 35136


def _series(n_buses: int = 2) -> CanonicalGridSeries:
    time = pd.date_range(
        "2015-12-31 23:00:00",
        periods=N_STEPS,
        freq="15min",
        tz="UTC",
    )
    step = np.arange(N_STEPS, dtype=np.float32)
    values = np.column_stack(
        [0.1 * (bus + 1) + np.sin(step / 96.0 + bus) for bus in range(n_buses)]
    ).astype(np.float32)
    return CanonicalGridSeries(
        grid_code=GRID_CODE,
        time_utc=time,
        bus_index=pd.Index(range(n_buses)),
        values={
            "net_demand_p_mw": values,
            "net_demand_q_mvar": 0.25 * values,
        },
        fingerprint="synthetic-m3-3",
    )


def test_full_validation_request_matches_frozen_primary_geometry() -> None:
    request = validation_request(_series())

    assert request.validation_start == 24572
    assert request.validation_end == 29856
    assert request.panel_end == 29856
    assert request.test_size == 5284
    assert request.n_windows == 5269


def test_smoke_validation_request_preserves_training_boundary() -> None:
    request = validation_request(
        _series(),
        max_validation_windows=3,
    )

    assert request.validation_start == 24572
    assert request.test_size == 18
    assert request.panel_end == request.validation_start + request.test_size
    assert request.n_windows == 3
    assert request.panel_end < request.validation_end


def test_validation_request_rejects_invalid_window_limit() -> None:
    with pytest.raises(GridIntegrityError, match="must be >= 1"):
        validation_request(
            _series(),
            max_validation_windows=0,
        )


def test_channel_panel_is_series_major_and_truncated() -> None:
    series = _series()
    panel = build_channel_panel(
        series,
        "net_demand_p_mw",
        end_exclusive=100,
    )

    assert len(panel) == 200
    assert panel["unique_id"].iloc[:100].eq(0).all()
    assert panel["unique_id"].iloc[100:].eq(1).all()
    assert panel["ds"].iloc[0] == series.time_utc[0].tz_localize(None)
    assert panel["ds"].iloc[99] == series.time_utc[99].tz_localize(None)
    np.testing.assert_allclose(
        panel["y"].iloc[:100].to_numpy(),
        series.channel("net_demand_p_mw")[:100, 0],
    )


def test_channel_panel_rejects_out_of_range_end() -> None:
    series = _series()
    with pytest.raises(GridIntegrityError, match="outside canonical series"):
        build_channel_panel(
            series,
            "net_demand_p_mw",
            end_exclusive=series.n_steps + 1,
        )
