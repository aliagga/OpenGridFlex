from __future__ import annotations

from importlib.metadata import version

import pytest

from opengridflex.grids.simbench_adapter import (
    GridIntegrityError,
    load_validated_simbench_grid,
)

GRID_CODE = "1-MV-urban--1-no_sw"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def validated_grid():
    return load_validated_simbench_grid(GRID_CODE)


def test_grid_stack_versions_are_frozen() -> None:
    assert version("pandapower") == "3.5.4"
    assert version("simbench") == "1.6.2"


def test_known_tap_compatibility_issue_is_repaired(validated_grid) -> None:
    report = validated_grid.report

    assert len(report.tap_repairs) == 2
    assert {repair.assigned_type for repair in report.tap_repairs} == {"Ratio"}
    assert {repair.tap_pos for repair in report.tap_repairs} == {-1.0}
    assert {repair.tap_neutral for repair in report.tap_repairs} == {0.0}

    assert report.raw_fingerprint != report.validated_fingerprint
    assert report.max_voltage_delta_vs_raw_pu == pytest.approx(
        0.01649815,
        abs=5e-5,
    )


def test_validated_ac_solution_matches_regression_reference(validated_grid) -> None:
    report = validated_grid.report

    assert report.raw_power_flow.converged
    assert report.validated_power_flow.converged

    assert report.raw_power_flow.min_vm_pu == pytest.approx(0.974230, abs=1e-4)
    assert report.validated_power_flow.min_vm_pu == pytest.approx(0.990728, abs=1e-4)
    assert report.validated_power_flow.max_vm_pu == pytest.approx(1.025000, abs=1e-4)


def test_profiles_have_common_full_year_length(validated_grid) -> None:
    shapes = dict(validated_grid.report.profile_shapes)

    assert set(shapes) == {"load", "powerplants", "renewables", "storage"}
    assert {shape[0] for shape in shapes.values()} == {35136}


def test_unsupported_grid_fails_closed() -> None:
    with pytest.raises(GridIntegrityError, match="not yet in the validated"):
        load_validated_simbench_grid("not-a-real-grid")
