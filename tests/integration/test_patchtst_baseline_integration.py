from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from opengridflex.baselines.patchtst_global import (
    NEURALFORECAST_VERSION,
    PatchTSTDevelopmentConfig,
    evaluate_patchtst_channel,
    require_neuralforecast_version,
)
from opengridflex.data.leakproof_dataset import CanonicalGridSeries

GRID_CODE = "1-MV-urban--1-no_sw"
N_STEPS = 35136

pytestmark = pytest.mark.integration


def _series() -> CanonicalGridSeries:
    time = pd.date_range(
        "2015-12-31 23:00:00",
        periods=N_STEPS,
        freq="15min",
        tz="UTC",
    )
    step = np.arange(N_STEPS, dtype=np.float32)
    p = np.column_stack(
        (
            0.6 + 0.15 * np.sin(step / 96.0),
            0.9 + 0.20 * np.cos(step / 96.0),
        )
    ).astype(np.float32)
    return CanonicalGridSeries(
        grid_code=GRID_CODE,
        time_utc=time,
        bus_index=pd.Index([0, 1]),
        values={
            "net_demand_p_mw": p,
            "net_demand_q_mvar": 0.3 * p,
        },
        fingerprint="synthetic-m3-3-integration",
    )


def test_required_neuralforecast_version_is_installed() -> None:
    assert require_neuralforecast_version() == NEURALFORECAST_VERSION


def test_patchtst_two_window_smoke_is_finite() -> None:
    config = PatchTSTDevelopmentConfig(
        max_steps=1,
        encoder_layers=1,
        n_heads=2,
        hidden_size=16,
        linear_hidden_size=32,
        dropout=0.0,
        fc_dropout=0.0,
        batch_size=2,
        windows_batch_size=8,
        inference_windows_batch_size=32,
    )

    result = evaluate_patchtst_channel(
        _series(),
        "net_demand_p_mw",
        config=config,
        max_validation_windows=2,
    )

    assert result["test_split_sealed"] is True
    assert result["request"]["n_windows"] == 2
    assert result["metrics"]["windows"] == 2
    assert result["parameter_count"] > 0
    assert math.isfinite(result["metrics"]["bus_micro"]["mae"])
    assert math.isfinite(result["fit_predict_seconds"])
