from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandapower as pp
import simbench as sb

GRID_CODE = "1-MV-urban--1-no_sw"


def main() -> None:
    print("=" * 72)
    print("OpenGridFlex — M1 Grid Stack Diagnostic")
    print("=" * 72)

    print(f"\npandapower version : {pp.__version__}")
    print(f"SimBench version   : {sb.__version__}")
    print(f"Grid               : {GRID_CODE}")

    net = sb.get_simbench_net(GRID_CODE)

    print("\n--- Network summary ---")
    print(f"Buses        : {len(net.bus)}")
    print(f"Lines        : {len(net.line)}")
    print(f"Loads        : {len(net.load)}")
    print(f"Static gens  : {len(net.sgen)}")
    print(f"Transformers : {len(net.trafo)}")

    if not hasattr(net, "profiles") or not net.profiles:
        raise RuntimeError("SimBench network contains no profiles.")

    print("\n--- Profiles ---")
    for name, frame in net.profiles.items():
        shape = getattr(frame, "shape", None)
        print(f"{name:20s} shape={shape}")

    required_trafo_columns = {
        "tap_pos",
        "tap_neutral",
        "tap_step_percent",
        "tap_changer_type",
    }

    missing_columns = required_trafo_columns - set(net.trafo.columns)

    if missing_columns:
        raise RuntimeError(f"Missing required transformer columns: {sorted(missing_columns)}")

    print("\n--- Transformer tap data ---")

    columns = [
        "tap_pos",
        "tap_neutral",
        "tap_min",
        "tap_max",
        "tap_step_percent",
        "tap_step_degree",
        "tap_side",
        "tap_changer_type",
    ]

    existing_columns = [c for c in columns if c in net.trafo.columns]
    print(net.trafo[existing_columns].to_string())

    tappable = (
        net.trafo["tap_pos"].notna()
        & net.trafo["tap_step_percent"].notna()
        & (net.trafo["tap_step_percent"].abs() > 0)
    )

    print("\nTappable transformers:", int(tappable.sum()))

    missing_type = tappable & net.trafo["tap_changer_type"].isna()

    print(
        "Tappable transformers without tap_changer_type:",
        int(missing_type.sum()),
    )

    print("\n--- Original SimBench power flow ---")

    original = deepcopy(net)

    pp.runpp(
        original,
        calculate_voltage_angles=True,
        init="auto",
    )

    if not original.converged:
        raise RuntimeError("Original SimBench power flow did not converge.")

    original_vm = original.res_bus["vm_pu"].copy()

    print("Converged: YES")
    print(f"Minimum voltage: {original_vm.min():.6f} pu")
    print(f"Maximum voltage: {original_vm.max():.6f} pu")

    print("\n--- Diagnostic tap correction ---")

    corrected = deepcopy(net)

    corrected.trafo.loc[
        missing_type,
        "tap_changer_type",
    ] = "Ratio"

    pp.runpp(
        corrected,
        calculate_voltage_angles=True,
        init="auto",
    )

    if not corrected.converged:
        raise RuntimeError("Power flow after diagnostic tap correction did not converge.")

    corrected_vm = corrected.res_bus["vm_pu"].copy()

    voltage_delta = (corrected_vm - original_vm).abs()

    print("Converged: YES")
    print(f"Minimum voltage: {corrected_vm.min():.6f} pu")
    print(f"Maximum voltage: {corrected_vm.max():.6f} pu")

    print("\n--- Tap compatibility diagnostic ---")

    max_delta = float(voltage_delta.max())
    mean_delta = float(voltage_delta.mean())

    print(f"Maximum |ΔV| : {max_delta:.8f} pu")
    print(f"Mean |ΔV|    : {mean_delta:.8f} pu")

    if missing_type.sum() > 0 and max_delta > 1e-8:
        print()
        print("KNOWN COMPATIBILITY ISSUE DETECTED")
        print(
            "Transformer tap positions affect the power-flow solution "
            "when tap_changer_type='Ratio' is explicitly assigned."
        )
        print(
            "OpenGridFlex must therefore apply and validate a controlled "
            "compatibility correction before using these networks."
        )

    elif missing_type.sum() == 0:
        print()
        print("No missing tap_changer_type values detected. The upstream behavior may have changed.")

    else:
        print()
        print("Missing tap changer types were found, but no material voltage difference was detected.")

    if not np.isfinite(original_vm.to_numpy()).all():
        raise RuntimeError("Original power flow contains non-finite voltages.")

    if not np.isfinite(corrected_vm.to_numpy()).all():
        raise RuntimeError("Corrected power flow contains non-finite voltages.")

    print("\nM1.1 DIAGNOSTIC: COMPLETE")


if __name__ == "__main__":
    main()
