from __future__ import annotations

import numpy as np
import pandas as pd

from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    SECONDARY_TASK_ID,
    default_benchmark_contract,
    make_contract_split,
    training_activity_mask,
)
from opengridflex.data.leakproof_dataset import CanonicalGridSeries


def _full_year_stub() -> CanonicalGridSeries:
    time = pd.date_range(
        "2015-12-31 23:00:00",
        periods=35136,
        freq="15min",
        tz="UTC",
    )
    buses = pd.Index([0, 1, 2])
    values = {
        "net_demand_p_mw": np.ones((35136, 3), dtype=np.float32),
        "net_demand_q_mvar": np.ones((35136, 3), dtype=np.float32),
    }
    return CanonicalGridSeries(
        grid_code="1-MV-urban--1-no_sw",
        time_utc=time,
        bus_index=buses,
        values=values,
        fingerprint="stub",
    )


def test_default_contract_has_one_primary_and_one_secondary() -> None:
    contract = default_benchmark_contract()

    assert contract.task(PRIMARY_TASK_ID).role == "primary"
    assert contract.task(SECONDARY_TASK_ID).role == "secondary"

    primary = contract.task(PRIMARY_TASK_ID)
    assert primary.history_steps == 96
    assert primary.horizon_steps == 16
    assert primary.target_channels == (
        "net_demand_p_mw",
        "net_demand_q_mvar",
    )

    secondary = contract.task(SECONDARY_TASK_ID)
    assert secondary.history_steps == 672
    assert secondary.horizon_steps == 96


def test_calendar_aligned_split_has_expected_exact_boundaries() -> None:
    series = _full_year_stub()
    split = make_contract_split(series)

    assert split.train_end == 24572
    assert split.validation_start == 24572
    assert split.validation_end == 29856
    assert split.test_start == 29856
    assert split.test_end == 35136

    local = series.time_utc.tz_convert("Europe/Berlin")
    assert local[split.train_start] == pd.Timestamp(
        "2016-01-01 00:00:00",
        tz="Europe/Berlin",
    )
    assert local[split.validation_start] == pd.Timestamp(
        "2016-09-13 00:00:00",
        tz="Europe/Berlin",
    )
    assert local[split.test_start] == pd.Timestamp(
        "2016-11-07 00:00:00",
        tz="Europe/Berlin",
    )


def test_activity_mask_excludes_training_zero_bus() -> None:
    series = _full_year_stub()
    series.values["net_demand_p_mw"][:, 2] = 0.0
    split = make_contract_split(series)

    mask = training_activity_mask(
        series,
        split,
        "net_demand_p_mw",
    )

    assert np.array_equal(mask, np.array([True, True, False]))


def test_contract_quantiles_are_ordered_and_include_median() -> None:
    contract = default_benchmark_contract()
    quantiles = contract.evaluation.quantiles

    assert quantiles == tuple(sorted(quantiles))
    assert 0.50 in quantiles
    assert contract.evaluation.primary_interval == (0.05, 0.95)
