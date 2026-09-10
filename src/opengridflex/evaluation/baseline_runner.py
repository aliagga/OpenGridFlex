from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from opengridflex.baselines.persistence import (
    SANITY_BASELINES,
    PersistenceBaseline,
    predict_window,
    source_indices_for_window,
)
from opengridflex.data.benchmark_contract import (
    BenchmarkContract,
    default_benchmark_contract,
    make_contract_split,
    training_activity_mask,
)
from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    ForecastWindow,
    build_window_plan,
)
from opengridflex.evaluation.deterministic import StreamingDeterministicMetrics
from opengridflex.grids.simbench_adapter import GridIntegrityError

DEVELOPMENT_SPLITS = ("validation",)


def audit_baseline_window_sources(
    windows: Iterable[ForecastWindow],
    baseline: PersistenceBaseline,
) -> None:
    count = 0
    for window in windows:
        source = source_indices_for_window(window, baseline)
        if np.any(source > window.origin):
            raise GridIntegrityError(f"Baseline {baseline.baseline_id!r} reads future values.")
        count += 1

    if count == 0:
        raise GridIntegrityError(f"No windows were available to audit for {baseline.baseline_id!r}.")


def evaluate_persistence_baseline(
    series: CanonicalGridSeries,
    *,
    task: Any,
    split_name: str,
    baseline: PersistenceBaseline,
    zero_activity_epsilon: float,
) -> dict[str, Any]:
    if split_name not in {"train", "validation", "test"}:
        raise GridIntegrityError(f"Unknown split {split_name!r}.")

    split = make_contract_split(series)
    plan = build_window_plan(series.n_steps, split, task.spec)
    windows = plan.for_split(split_name)
    audit_baseline_window_sources(windows, baseline)

    output: dict[str, Any] = {
        "baseline": baseline.to_dict(),
        "split": split_name,
        "channels": {},
    }

    for channel in task.target_channels:
        active_mask = training_activity_mask(
            series,
            split,
            channel,
            epsilon=zero_activity_epsilon,
        )
        accumulator = StreamingDeterministicMetrics(
            n_buses=series.n_buses,
            horizon_steps=task.horizon_steps,
            active_bus_mask=active_mask,
            zero_activity_epsilon=zero_activity_epsilon,
        )

        values = series.channel(channel)
        for window in windows:
            target = values[window.target_start : window.target_end + 1].astype(np.float64, copy=False)
            prediction = predict_window(
                series,
                channel,
                window,
                baseline,
            )
            accumulator.update(target, prediction)

        output["channels"][channel] = accumulator.finalize()

    return output


def run_sanity_baselines(
    series: CanonicalGridSeries,
    *,
    contract: BenchmarkContract | None = None,
    splits: tuple[str, ...] = DEVELOPMENT_SPLITS,
) -> dict[str, Any]:
    benchmark = contract or default_benchmark_contract()

    if "test" in splits:
        raise GridIntegrityError(
            "M3.1 keeps the test split sealed. Test evaluation is reserved for the final benchmark stage."
        )

    invalid = set(splits) - {"train", "validation"}
    if invalid:
        raise GridIntegrityError(f"Unsupported development splits: {sorted(invalid)}.")

    results: dict[str, Any] = {
        "stage": "M3.1",
        "contract_id": benchmark.contract_id,
        "source_fingerprint": series.fingerprint,
        "test_split_sealed": True,
        "baseline_definitions": [baseline.to_dict() for baseline in SANITY_BASELINES],
        "tasks": {},
    }

    for task in benchmark.tasks:
        task_results: dict[str, Any] = {}
        for split_name in splits:
            split_results: dict[str, Any] = {}
            for baseline in SANITY_BASELINES:
                split_results[baseline.baseline_id] = evaluate_persistence_baseline(
                    series,
                    task=task,
                    split_name=split_name,
                    baseline=baseline,
                    zero_activity_epsilon=(benchmark.evaluation.zero_activity_epsilon),
                )
            task_results[split_name] = split_results
        results["tasks"][task.task_id] = task_results

    return results
