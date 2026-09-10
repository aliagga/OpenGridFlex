from __future__ import annotations

import json

from opengridflex.grids.physical_reconstruction import (
    build_absolute_profile_bundle,
    reconstruct_snapshot,
    select_validation_positions,
)
from opengridflex.grids.simbench_adapter import load_validated_simbench_grid

GRID_CODE = "1-MV-urban--1-no_sw"


def main() -> int:
    validated = load_validated_simbench_grid(GRID_CODE)
    bundle = build_absolute_profile_bundle(validated.net)
    positions = select_validation_positions(validated.net, bundle)

    print("=" * 78)
    print("OpenGridFlex — M1.3 Physical Reconstruction Validation")
    print("=" * 78)
    print(f"Grid: {GRID_CODE}")
    print(f"Timesteps: {bundle.row_count}")
    print(f"Validation positions: {positions}")
    print()
    print("Storage sign convention: +P = charging, -P = discharging")
    print()

    summaries = []
    for position in positions:
        snapshot = reconstruct_snapshot(
            GRID_CODE,
            position,
            verify_manual_formula=True,
        )
        summary = snapshot.summary.to_dict()
        summaries.append(summary)

        print(
            f"[{position:5d}] {summary['local_time']} | "
            f"V=[{summary['min_vm_pu']:.5f}, {summary['max_vm_pu']:.5f}] pu | "
            f"line={summary['max_line_loading_percent']:.2f}% | "
            f"trafo={summary['max_trafo_loading_percent']:.2f}% | "
            f"dP={summary['active_balance_residual_mw']:.3e} MW | "
            f"dQ={summary['reactive_balance_residual_mvar']:.3e} Mvar"
        )

    print("\n--- Machine-readable summaries ---")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print("\nM1.3 PHYSICS VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
