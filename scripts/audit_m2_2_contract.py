from __future__ import annotations

import json
import tempfile
from pathlib import Path

from opengridflex.data.benchmark_contract import (
    build_contract_window_plans,
    default_benchmark_contract,
    make_contract_split,
    validate_contract,
)
from opengridflex.data.benchmark_serialization import (
    verify_benchmark_artifact,
    write_benchmark_artifact,
)
from opengridflex.data.leakproof_dataset import build_canonical_grid_series

GRID_CODE = "1-MV-urban--1-no_sw"


def main() -> int:
    series = build_canonical_grid_series(GRID_CODE)
    contract = default_benchmark_contract()
    audit = validate_contract(series, contract)
    split = make_contract_split(series, contract)
    plans = build_contract_window_plans(series, contract)

    print("=" * 78)
    print("OpenGridFlex — M2.2 Benchmark Contract Audit")
    print("=" * 78)
    print(f"Contract: {contract.contract_id}")
    print(f"Grid: {series.grid_code}")
    print(f"Source fingerprint: {series.fingerprint}")
    print()
    print("Frozen split:")
    print(json.dumps(split.to_dict(), indent=2, sort_keys=True))

    print("\nFrozen tasks:")
    for task in contract.tasks:
        plan = plans[task.task_id]
        print(
            f"- {task.task_id} [{task.role}] | "
            f"history={task.history_steps} | "
            f"horizon={task.horizon_steps} | "
            f"targets={task.target_channels} | "
            f"train={len(plan.for_split('train'))} | "
            f"val={len(plan.for_split('validation'))} | "
            f"test={len(plan.for_split('test'))}"
        )

    print("\nEvaluation contract:")
    print(
        json.dumps(
            contract.evaluation.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )

    print("\nContract validation:")
    print(json.dumps(audit, indent=2, sort_keys=True))

    with tempfile.TemporaryDirectory() as temporary:
        artifact = write_benchmark_artifact(
            series,
            Path(temporary) / contract.contract_id,
        )
        manifest = verify_benchmark_artifact(artifact)
        print("\nSerialization round-trip:")
        print(f"PASS — {len(manifest['files'])} hashed payload files verified.")

    print("\nM2.2 BENCHMARK CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
