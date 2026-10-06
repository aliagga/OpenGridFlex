from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

from opengridflex.baselines.patchtst_global import (
    NEURALFORECAST_VERSION,
    PatchTSTDevelopmentConfig,
    evaluate_patchtst_channel,
)
from opengridflex.baselines.persistence import SANITY_BASELINES
from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    default_benchmark_contract,
)
from opengridflex.data.leakproof_dataset import build_canonical_grid_series
from opengridflex.evaluation.baseline_runner import evaluate_persistence_baseline
from opengridflex.reproducibility import (
    build_run_manifest,
    sha256_file,
    write_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
GRID_CODE = "1-MV-urban--1-no_sw"

CHANNELS = {
    "p": "net_demand_p_mw",
    "q": "net_demand_q_mvar",
}


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _best_naive_mae(
    baseline_results: dict[str, Any],
    channel: str,
) -> tuple[str, float]:
    candidates = []
    for baseline_id, result in baseline_results.items():
        mae = result["channels"][channel]["bus_micro"]["mae"]
        candidates.append((baseline_id, float(mae)))
    return min(candidates, key=lambda item: item[1])


def _print_summary(
    results: dict[str, Any],
    naive: dict[str, Any] | None,
) -> None:
    print("=" * 78)
    print("OpenGridFlex — M3.3 PatchTST Baseline")
    print("=" * 78)
    print(f"NeuralForecast: {NEURALFORECAST_VERSION}")
    print("Split: VALIDATION")
    print("Test split: SEALED")
    print()

    for channel, result in results.items():
        point = result["metrics"]["bus_micro"]
        request = result["request"]
        print(channel)
        print(
            f"  windows={request['n_windows']} | "
            f"MAE={point['mae']:.6g} | RMSE={point['rmse']:.6g} | WAPE={point['wape']:.6g}"
        )
        print(
            f"  parameters={result['parameter_count']:,} | "
            f"fit+predict={result['fit_predict_seconds']:.1f}s"
        )
        if naive is not None:
            best_id, best_mae = _best_naive_mae(naive, channel)
            skill = 1.0 - float(point["mae"]) / best_mae if best_mae > 0.0 else None
            if skill is not None:
                print(f"  best naive MAE={best_mae:.6g} ({best_id}) | MAE skill={skill:.3%}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for M3.3 metrics + reproducibility artifacts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--channel",
        choices=("both", "p", "q"),
        default="both",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="PatchTST optimizer steps. Frozen M3.3 default is 500.",
    )
    parser.add_argument(
        "--max-validation-windows",
        type=int,
        default=None,
        help="Development-only limit. Omit for the full frozen validation set.",
    )
    args = parser.parse_args()

    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1.")

    series = build_canonical_grid_series(GRID_CODE)
    contract = default_benchmark_contract()
    task = contract.task(PRIMARY_TASK_ID)
    config = PatchTSTDevelopmentConfig(max_steps=args.max_steps)

    if args.channel == "both":
        channels = tuple(CHANNELS.values())
    else:
        channels = (CHANNELS[args.channel],)

    results: dict[str, Any] = {}
    for channel in channels:
        results[channel] = evaluate_patchtst_channel(
            series,
            channel,
            contract=contract,
            config=config,
            max_validation_windows=args.max_validation_windows,
        )
        gc.collect()

    naive = None
    if args.max_validation_windows is None:
        naive = {
            baseline.baseline_id: evaluate_persistence_baseline(
                series,
                task=task,
                split_name="validation",
                baseline=baseline,
                zero_activity_epsilon=contract.evaluation.zero_activity_epsilon,
            )
            for baseline in SANITY_BASELINES
        }

    _print_summary(results, naive)

    payload: dict[str, Any] = {
        "stage": "M3.3",
        "status": (
            "full_validation_complete"
            if args.max_validation_windows is None and args.channel == "both"
            else "development_run"
        ),
        "test_split_sealed": True,
        "library": "neuralforecast",
        "neuralforecast_version": NEURALFORECAST_VERSION,
        "grid_code": GRID_CODE,
        "contract_id": contract.contract_id,
        "task_id": task.task_id,
        "source_fingerprint": series.fingerprint,
        "channels": list(channels),
        "max_validation_windows": args.max_validation_windows,
        "config": config.to_dict(),
        "results": results,
        "naive_validation": naive,
    }

    if args.output_dir:
        output = Path(args.output_dir)
        if output.exists():
            if not args.overwrite and any(output.iterdir()):
                raise RuntimeError(f"Output directory {output} is not empty. Use --overwrite to replace it.")
            if args.overwrite:
                shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

        effective_config = {
            "stage": "M3.3",
            "grid_code": GRID_CODE,
            "contract_id": contract.contract_id,
            "task_id": task.task_id,
            "test_split_sealed": True,
            "library": "neuralforecast",
            "neuralforecast_version": NEURALFORECAST_VERSION,
            "channels": list(channels),
            "max_validation_windows": args.max_validation_windows,
            "patchtst_config": config.to_dict(),
        }
        config_path = output / "effective_run_config.json"
        config_path.write_text(
            _json_text(effective_config),
            encoding="utf-8",
            newline="\n",
        )

        run_manifest = build_run_manifest(
            experiment="m3_3_patchtst",
            seed=config.seed,
            config_path=config_path,
            repo_root=ROOT,
        )
        run_manifest_path = write_run_manifest(
            run_manifest,
            output / "run_manifest.json",
        )

        payload["effective_run_config"] = {
            "path": config_path.relative_to(output).as_posix(),
            "sha256": sha256_file(config_path),
        }
        payload["run_manifest"] = {
            "path": run_manifest_path.relative_to(output).as_posix(),
            "sha256": sha256_file(run_manifest_path),
        }

        validation_manifest = output / "validation_manifest.json"
        validation_manifest.write_text(
            _json_text(payload),
            encoding="utf-8",
            newline="\n",
        )
        print(f"Artifact written: {output}")
        print(f"Run manifest: {run_manifest_path}")
        print(f"Validation manifest SHA-256: {sha256_file(validation_manifest)}")

    if args.max_validation_windows is None and args.channel == "both":
        print("\nM3.3 PATCHTST FULL VALIDATION: PASS")
    else:
        print("\nM3.3 PATCHTST DEVELOPMENT RUN: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
