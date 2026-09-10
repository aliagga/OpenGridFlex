from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from opengridflex.data.benchmark_contract import default_benchmark_contract
from opengridflex.data.leakproof_dataset import build_canonical_grid_series
from opengridflex.evaluation.baseline_runner import run_sanity_baselines

GRID_CODE = "1-MV-urban--1-no_sw"


def _json_text(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


def _format_metric(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"


def _summary(results: dict[str, Any]) -> None:
    print("=" * 78)
    print("OpenGridFlex — M3.1 Deterministic Sanity Baselines")
    print("=" * 78)
    print(f"Contract: {results['contract_id']}")
    print(f"Source fingerprint: {results['source_fingerprint']}")
    print("Test split: SEALED")
    print()

    for task_id, task_result in results["tasks"].items():
        validation = task_result["validation"]
        print(task_id)
        for baseline_id, baseline_result in validation.items():
            print(f"  {baseline_id}")
            for channel, metrics in baseline_result["channels"].items():
                micro = metrics["bus_micro"]
                macro = metrics["bus_macro_active"]
                system = metrics["system_aggregate"]
                print(
                    f"    {channel}: "
                    f"micro MAE={_format_metric(micro['mae'])}, "
                    f"RMSE={_format_metric(micro['rmse'])}, "
                    f"WAPE={_format_metric(micro['wape'])} | "
                    f"macro-active MAE={_format_metric(macro['mae'])} | "
                    f"system MAE={_format_metric(system['mae'])}"
                )
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the validation-only JSON result artifact.",
    )
    args = parser.parse_args()

    series = build_canonical_grid_series(GRID_CODE)
    contract = default_benchmark_contract()
    results = run_sanity_baselines(
        series,
        contract=contract,
        splits=("validation",),
    )

    _summary(results)

    text = _json_text(results)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"Results SHA-256: {digest}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
        print(f"Validation results written: {output}")

    print("\nM3.1 SANITY BASELINES: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
