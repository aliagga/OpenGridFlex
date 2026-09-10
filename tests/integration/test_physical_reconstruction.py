from __future__ import annotations

import pytest

from opengridflex.grids.physical_reconstruction import (
    BALANCE_TOLERANCE_MVAR,
    BALANCE_TOLERANCE_MW,
    build_absolute_profile_bundle,
    reconstruct_snapshot,
    verify_manual_profile_formula,
)
from opengridflex.grids.simbench_adapter import load_validated_simbench_grid

GRID_CODE = "1-MV-urban--1-no_sw"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def validated():
    return load_validated_simbench_grid(GRID_CODE)


@pytest.fixture(scope="module")
def bundle(validated):
    return build_absolute_profile_bundle(validated.net)


def test_absolute_profiles_cover_full_year(validated, bundle) -> None:
    assert bundle.row_count == 35136

    for key, frame in bundle.values.items():
        assert len(frame) == 35136
        element, _ = key
        assert frame.columns.equals(validated.net[element].index)


def test_independent_formula_matches_simbench_at_representative_step(
    validated,
    bundle,
) -> None:
    verify_manual_profile_formula(
        validated.net,
        bundle,
        position=12345,
    )


def test_reconstructed_snapshot_converges_and_balances() -> None:
    snapshot = reconstruct_snapshot(
        GRID_CODE,
        position=12345,
    )
    summary = snapshot.summary

    assert summary.converged
    assert 0.5 < summary.min_vm_pu < 1.5
    assert 0.5 < summary.max_vm_pu < 1.5
    assert abs(summary.active_balance_residual_mw) <= BALANCE_TOLERANCE_MW
    assert abs(summary.reactive_balance_residual_mvar) <= BALANCE_TOLERANCE_MVAR
