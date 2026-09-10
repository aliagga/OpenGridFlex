from __future__ import annotations

from pathlib import Path

import pytest

from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    SECONDARY_TASK_ID,
    build_contract_window_plans,
    validate_contract,
)
from opengridflex.data.benchmark_serialization import (
    verify_benchmark_artifact,
    write_benchmark_artifact,
)
from opengridflex.data.leakproof_dataset import build_canonical_grid_series
from opengridflex.grids.simbench_adapter import GridIntegrityError

GRID_CODE = "1-MV-urban--1-no_sw"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def series():
    return build_canonical_grid_series(GRID_CODE)


def test_contract_validates_on_real_canonical_series(series) -> None:
    audit = validate_contract(series)

    assert audit["contract_id"] == "gridshiftbench-v1"
    assert audit["split"]["train_end"] == 24572
    assert audit["split"]["validation_end"] == 29856
    assert audit["split"]["test_end"] == 35136
    assert audit["audits"][PRIMARY_TASK_ID]["passed"]
    assert audit["audits"][SECONDARY_TASK_ID]["passed"]


def test_real_task_windows_are_nonempty_and_leak_free(series) -> None:
    plans = build_contract_window_plans(series)

    for task_id in (PRIMARY_TASK_ID, SECONDARY_TASK_ID):
        plan = plans[task_id]
        assert len(plan.for_split("train")) > 0
        assert len(plan.for_split("validation")) > 0
        assert len(plan.for_split("test")) > 0
        assert plan.dropped_cross_boundary > 0


def test_serialization_round_trip_and_tamper_detection(
    series,
    tmp_path: Path,
) -> None:
    artifact = write_benchmark_artifact(
        series,
        tmp_path / "gridshiftbench-v1",
    )
    manifest = verify_benchmark_artifact(artifact)

    assert manifest["source_fingerprint"] == series.fingerprint
    assert manifest["grid_code"] == GRID_CODE
    assert manifest["contract"]["contract_id"] == "gridshiftbench-v1"
    assert manifest["split"]["train_end"] == 24572

    payload = artifact / "channels" / "net_demand_p_mw.npy"
    original = payload.read_bytes()
    corrupted = bytearray(original)
    corrupted[-1] ^= 1
    payload.write_bytes(bytes(corrupted))

    with pytest.raises(GridIntegrityError, match="SHA-256 mismatch"):
        verify_benchmark_artifact(artifact)
