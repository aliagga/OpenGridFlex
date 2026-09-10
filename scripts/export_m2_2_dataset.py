from __future__ import annotations

import argparse
from pathlib import Path

from opengridflex.data.benchmark_contract import default_benchmark_contract
from opengridflex.data.benchmark_serialization import write_benchmark_artifact
from opengridflex.data.leakproof_dataset import build_canonical_grid_series

GRID_CODE = "1-MV-urban--1-no_sw"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/datasets/gridshiftbench-v1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    series = build_canonical_grid_series(GRID_CODE)
    contract = default_benchmark_contract()
    destination = Path(args.output) / GRID_CODE

    artifact = write_benchmark_artifact(
        series,
        destination,
        contract=contract,
        overwrite=args.overwrite,
    )
    print(f"Benchmark artifact written and verified: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
