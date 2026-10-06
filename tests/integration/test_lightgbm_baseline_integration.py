from __future__ import annotations

import math

import pytest

from opengridflex.baselines.lightgbm_global import (
    LIGHTGBM_VERSION,
    LightGBMDevelopmentConfig,
    fit_global_lightgbm,
    require_lightgbm_version,
)
from opengridflex.data.benchmark_contract import PRIMARY_TASK_ID
from opengridflex.data.leakproof_dataset import build_canonical_grid_series
from opengridflex.evaluation.lightgbm_runner import (
    evaluate_global_lightgbm,
)
from opengridflex.grids.simbench_adapter import GridIntegrityError

GRID_CODE = "1-MV-urban--1-no_sw"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fitted():
    series = build_canonical_grid_series(GRID_CODE)
    config = LightGBMDevelopmentConfig(
        max_train_rows=2_000,
        n_estimators=5,
        n_jobs=1,
    )
    bundle, context = fit_global_lightgbm(
        series,
        config=config,
    )
    return series, bundle, context


def test_required_lightgbm_version_is_installed() -> None:
    assert require_lightgbm_version() == LIGHTGBM_VERSION


def test_smoke_fit_has_expected_primary_task(fitted) -> None:
    series, bundle, _ = fitted

    assert bundle.task_id == PRIMARY_TASK_ID
    assert bundle.source_fingerprint == series.fingerprint
    assert set(bundle.channels) == {
        "net_demand_p_mw",
        "net_demand_q_mvar",
    }
    assert len(bundle.quantiles) == 7


def test_validation_evaluation_is_finite(fitted) -> None:
    series, bundle, context = fitted
    result = evaluate_global_lightgbm(
        series,
        bundle,
        context,
        batch_windows=8,
        max_windows=8,
    )

    assert result["test_split_sealed"] is True
    for channel in result["channels"].values():
        assert math.isfinite(channel["point"]["bus_micro"]["mae"])
        assert math.isfinite(channel["probabilistic"]["mean_pinball_loss"])


def test_test_split_cannot_be_opened(fitted) -> None:
    series, bundle, context = fitted

    with pytest.raises(GridIntegrityError, match="test sealed"):
        evaluate_global_lightgbm(
            series,
            bundle,
            context,
            split_name="test",
        )
