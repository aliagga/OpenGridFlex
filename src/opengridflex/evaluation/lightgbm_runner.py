from __future__ import annotations

from typing import Any

import numpy as np

from opengridflex.baselines.lightgbm_global import (
    GlobalLightGBMBundle,
)
from opengridflex.baselines.tree_features import (
    FeatureContext,
    build_feature_matrix,
    exhaustive_cartesian_rows,
    target_vector,
)
from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    BenchmarkContract,
    default_benchmark_contract,
    make_contract_split,
    training_activity_mask,
)
from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    build_window_plan,
)
from opengridflex.evaluation.deterministic import (
    StreamingDeterministicMetrics,
)
from opengridflex.evaluation.probabilistic import (
    StreamingProbabilisticMetrics,
)
from opengridflex.grids.simbench_adapter import GridIntegrityError


def _reshape_flat(
    values: np.ndarray,
    *,
    n_windows: int,
    n_buses: int,
    horizon_steps: int,
) -> np.ndarray:
    array = np.asarray(values)
    expected = n_windows * n_buses * horizon_steps
    if array.shape != (expected,):
        raise GridIntegrityError(f"Flat prediction shape {array.shape} != {(expected,)}.")

    return array.reshape(
        n_windows,
        n_buses,
        horizon_steps,
    ).transpose(0, 2, 1)


def evaluate_global_lightgbm(
    series: CanonicalGridSeries,
    bundle: GlobalLightGBMBundle,
    context: FeatureContext,
    *,
    contract: BenchmarkContract | None = None,
    split_name: str = "validation",
    batch_windows: int = 64,
    max_windows: int | None = None,
) -> dict[str, Any]:
    if split_name != "validation":
        raise GridIntegrityError("M3.2 keeps test sealed and evaluates learned models on validation only.")
    if batch_windows < 1:
        raise GridIntegrityError("batch_windows must be >= 1.")

    benchmark = contract or default_benchmark_contract()
    task = benchmark.task(PRIMARY_TASK_ID)

    if bundle.contract_id != benchmark.contract_id:
        raise GridIntegrityError("Model/benchmark contract ID mismatch.")
    if bundle.task_id != task.task_id:
        raise GridIntegrityError("Model task ID mismatch.")
    if bundle.source_fingerprint != series.fingerprint:
        raise GridIntegrityError("Model/source dataset fingerprint mismatch.")

    split = make_contract_split(series, benchmark)
    plan = build_window_plan(series.n_steps, split, task.spec)
    windows = plan.for_split(split_name)
    if max_windows is not None:
        if max_windows < 1:
            raise GridIntegrityError("max_windows must be >= 1 when provided.")
        windows = windows[:max_windows]

    deterministic: dict[str, StreamingDeterministicMetrics] = {}
    probabilistic: dict[str, StreamingProbabilisticMetrics] = {}

    for channel in task.target_channels:
        mask = training_activity_mask(
            series,
            split,
            channel,
            epsilon=benchmark.evaluation.zero_activity_epsilon,
        )
        deterministic[channel] = StreamingDeterministicMetrics(
            n_buses=series.n_buses,
            horizon_steps=task.horizon_steps,
            active_bus_mask=mask,
            zero_activity_epsilon=(benchmark.evaluation.zero_activity_epsilon),
        )
        probabilistic[channel] = StreamingProbabilisticMetrics(
            quantiles=benchmark.evaluation.quantiles,
            lower_quantile=benchmark.evaluation.primary_interval[0],
            upper_quantile=benchmark.evaluation.primary_interval[1],
        )

    for start in range(0, len(windows), batch_windows):
        chunk = windows[start : start + batch_windows]
        sample = exhaustive_cartesian_rows(
            len(chunk),
            series.n_buses,
            task.horizon_steps,
        )
        features, target_time = build_feature_matrix(
            series,
            chunk,
            sample,
            context,
        )

        for channel in task.target_channels:
            truth_flat = target_vector(
                series,
                channel,
                target_time,
                sample.bus_position,
            ).astype(np.float64)

            channel_bundle = bundle.channels[channel]
            point_flat = channel_bundle.point_model.predict(features).astype(np.float64)

            quantile_flat = np.column_stack(
                [channel_bundle.quantile_models[quantile].predict(features) for quantile in bundle.quantiles]
            ).astype(np.float64)

            truth = _reshape_flat(
                truth_flat,
                n_windows=len(chunk),
                n_buses=series.n_buses,
                horizon_steps=task.horizon_steps,
            )
            point = _reshape_flat(
                point_flat,
                n_windows=len(chunk),
                n_buses=series.n_buses,
                horizon_steps=task.horizon_steps,
            )

            expected_rows = len(chunk) * series.n_buses * task.horizon_steps
            if quantile_flat.shape != (
                expected_rows,
                len(bundle.quantiles),
            ):
                raise GridIntegrityError("Unexpected quantile prediction matrix shape.")

            quantile = quantile_flat.reshape(
                len(chunk),
                series.n_buses,
                task.horizon_steps,
                len(bundle.quantiles),
            ).transpose(0, 2, 1, 3)

            for index in range(len(chunk)):
                deterministic[channel].update(
                    truth[index],
                    point[index],
                )
                probabilistic[channel].update(
                    truth[index],
                    quantile[index],
                )

    return {
        "stage": "M3.2",
        "split": split_name,
        "test_split_sealed": True,
        "contract_id": benchmark.contract_id,
        "task_id": task.task_id,
        "source_fingerprint": series.fingerprint,
        "lightgbm_config": bundle.config.to_dict(),
        "quantiles": list(bundle.quantiles),
        "feature_names": list(bundle.feature_names),
        "channels": {
            channel: {
                "point": deterministic[channel].finalize(),
                "probabilistic": probabilistic[channel].finalize(),
            }
            for channel in task.target_channels
        },
    }
