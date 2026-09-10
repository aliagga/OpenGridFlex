from __future__ import annotations

import math

import pytest

from opengridflex.baselines.persistence import SANITY_BASELINES
from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    default_benchmark_contract,
    make_contract_split,
)
from opengridflex.data.leakproof_dataset import (
    build_canonical_grid_series,
    build_window_plan,
)
from opengridflex.evaluation.baseline_runner import (
    audit_baseline_window_sources,
    evaluate_persistence_baseline,
    run_sanity_baselines,
)
from opengridflex.grids.simbench_adapter import GridIntegrityError

GRID_CODE = "1-MV-urban--1-no_sw"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def series():
    return build_canonical_grid_series(GRID_CODE)


def test_all_primary_validation_sources_are_strictly_observed(series) -> None:
    contract = default_benchmark_contract()
    task = contract.task(PRIMARY_TASK_ID)
    split = make_contract_split(series, contract)
    plan = build_window_plan(series.n_steps, split, task.spec)
    windows = plan.for_split("validation")

    for baseline in SANITY_BASELINES:
        audit_baseline_window_sources(windows, baseline)


def test_one_real_baseline_evaluation_is_finite(series) -> None:
    contract = default_benchmark_contract()
    task = contract.task(PRIMARY_TASK_ID)

    result = evaluate_persistence_baseline(
        series,
        task=task,
        split_name="validation",
        baseline=SANITY_BASELINES[0],
        zero_activity_epsilon=contract.evaluation.zero_activity_epsilon,
    )

    for channel_result in result["channels"].values():
        for level in (
            "bus_micro",
            "bus_macro_active",
            "system_aggregate",
        ):
            for metric in ("mae", "rmse"):
                assert math.isfinite(channel_result[level][metric])


def test_m3_1_runner_refuses_to_open_test_split(series) -> None:
    with pytest.raises(GridIntegrityError, match="test split sealed"):
        run_sanity_baselines(
            series,
            splits=("validation", "test"),
        )
