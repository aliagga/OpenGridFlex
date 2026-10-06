from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

from opengridflex.baselines.lightgbm_global import (
    LightGBMDevelopmentConfig,
    fit_global_lightgbm,
    save_global_lightgbm,
)
from opengridflex.baselines.persistence import SANITY_BASELINES
from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    default_benchmark_contract,
)
from opengridflex.data.leakproof_dataset import build_canonical_grid_series
from opengridflex.evaluation.baseline_runner import (
    evaluate_persistence_baseline,
)
from opengridflex.evaluation.lightgbm_runner import (
    evaluate_global_lightgbm,
)

GRID_CODE = "1-MV-urban--1-no_sw"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
    learned: dict[str, Any],
    baseline_results: dict[str, Any],
) -> None:
    print("=" * 78)
    print("OpenGridFlex — M3.2 Global LightGBM Baseline")
    print("=" * 78)
    print(f"Task: {learned['task_id']}")
    print(f"Source fingerprint: {learned['source_fingerprint']}")
    print(f"LightGBM: {version('lightgbm')}")
    print("Split: VALIDATION")
    print("Test split: SEALED")
    print()

    for channel, result in learned["channels"].items():
        point = result["point"]["bus_micro"]
        prob = result["probabilistic"]
        best_id, best_mae = _best_naive_mae(
            baseline_results,
            channel,
        )
        learned_mae = float(point["mae"])
        skill = 1.0 - learned_mae / best_mae if best_mae > 0.0 else None

        print(channel)
        print(f"  point MAE={point['mae']:.6g} | RMSE={point['rmse']:.6g} | WAPE={point['wape']:.6g}")
        print(
            f"  probabilistic mean pinball={prob['mean_pinball_loss']:.6g} | "
            f"90% coverage={prob['interval_coverage_90']:.4f} | "
            f"90% width={prob['mean_interval_width_90']:.6g} | "
            f"crossing={prob['adjacent_quantile_crossing_rate']:.6g}"
        )
        print(
            f"  best naive MAE={best_mae:.6g} ({best_id}) | MAE skill={skill:.3%}"
            if skill is not None
            else f"  best naive MAE={best_mae:.6g} ({best_id})"
        )
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for models + validation result manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=150_000,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
    )
    args = parser.parse_args()

    series = build_canonical_grid_series(GRID_CODE)
    contract = default_benchmark_contract()
    task = contract.task(PRIMARY_TASK_ID)

    config = LightGBMDevelopmentConfig(
        max_train_rows=args.max_train_rows,
        n_estimators=args.n_estimators,
    )
    bundle, context = fit_global_lightgbm(
        series,
        contract=contract,
        config=config,
    )
    learned = evaluate_global_lightgbm(
        series,
        bundle,
        context,
        contract=contract,
        batch_windows=64,
    )

    naive = {
        baseline.baseline_id: evaluate_persistence_baseline(
            series,
            task=task,
            split_name="validation",
            baseline=baseline,
            zero_activity_epsilon=(contract.evaluation.zero_activity_epsilon),
        )
        for baseline in SANITY_BASELINES
    }

    _print_summary(learned, naive)

    point_importance = {}
    for channel, channel_bundle in bundle.channels.items():
        gain = channel_bundle.point_model.booster_.feature_importance(importance_type="gain")
        total = float(gain.sum())
        if total > 0.0:
            gain = gain / total
        point_importance[channel] = {
            name: float(value)
            for name, value in sorted(
                zip(bundle.feature_names, gain, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    payload: dict[str, Any] = {
        "stage": "M3.2",
        "test_split_sealed": True,
        "lightgbm_version": version("lightgbm"),
        "contract_id": contract.contract_id,
        "task_id": task.task_id,
        "source_fingerprint": series.fingerprint,
        "config": config.to_dict(),
        "feature_names": list(bundle.feature_names),
        "learned_validation": learned,
        "naive_validation": naive,
        "point_feature_importance_gain_fraction": point_importance,
    }

    if args.output_dir:
        output = Path(args.output_dir)
        if output.exists():
            if not args.overwrite and any(output.iterdir()):
                raise RuntimeError(f"Output directory {output} is not empty. Use --overwrite to replace it.")
            if args.overwrite:
                shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

        model_dir = output / "models"
        model_paths = save_global_lightgbm(bundle, model_dir)
        payload["model_files"] = {
            path.relative_to(output).as_posix(): {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(model_paths)
        }

        manifest = output / "validation_manifest.json"
        manifest.write_text(
            _json_text(payload),
            encoding="utf-8",
            newline="\n",
        )
        print(f"Artifact written: {output}")
        print(f"Manifest SHA-256: {_sha256_file(manifest)}")

    print("\nM3.2 LIGHTGBM BASELINE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
