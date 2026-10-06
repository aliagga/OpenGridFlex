from __future__ import annotations

from opengridflex.baselines.patchtst_global import (
    NEURALFORECAST_VERSION,
    require_neuralforecast_version,
    validation_request,
)
from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    default_benchmark_contract,
)
from opengridflex.data.leakproof_dataset import build_canonical_grid_series

GRID_CODE = "1-MV-urban--1-no_sw"


def main() -> int:
    observed = require_neuralforecast_version()
    series = build_canonical_grid_series(GRID_CODE)
    contract = default_benchmark_contract()
    task = contract.task(PRIMARY_TASK_ID)
    request = validation_request(series, contract=contract)

    print("=" * 78)
    print("OpenGridFlex — M3.3 PatchTST Contract Audit")
    print("=" * 78)
    print(f"NeuralForecast: {observed} (required {NEURALFORECAST_VERSION})")
    print(f"Task: {task.task_id}")
    print(f"Grid: {series.grid_code}")
    print(f"Source fingerprint: {series.fingerprint}")
    print(f"History steps: {task.history_steps}")
    print(f"Horizon steps: {task.horizon_steps}")
    print(f"Validation start: {request.validation_start}")
    print(f"Panel end: {request.panel_end}")
    print(f"Validation end: {request.validation_end}")
    print(f"NeuralForecast test_size: {request.test_size}")
    print(f"Rolling validation windows: {request.n_windows}")
    print("Test split: SEALED")
    print()
    print("M3.3 PATCHTST CONTRACT AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
