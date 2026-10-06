from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opengridflex.baselines.tree_features import (
    FEATURE_NAMES,
    FeatureContext,
    SampleIndex,
    StaticBusFeatures,
    build_feature_matrix,
    sample_cartesian_rows,
    target_vector,
)
from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    ForecastWindow,
)
from opengridflex.evaluation.probabilistic import (
    StreamingProbabilisticMetrics,
)


def _series(n_steps: int = 1000) -> CanonicalGridSeries:
    time = pd.date_range(
        "2026-01-01",
        periods=n_steps,
        freq="15min",
        tz="UTC",
    )
    p = np.arange(n_steps * 2, dtype=np.float32).reshape(n_steps, 2)
    q = 0.1 * p
    return CanonicalGridSeries(
        grid_code="synthetic",
        time_utc=time,
        bus_index=pd.Index([0, 1]),
        values={
            "net_demand_p_mw": p,
            "net_demand_q_mvar": q,
        },
        fingerprint="synthetic",
    )


def _context(series: CanonicalGridSeries) -> FeatureContext:
    p = series.channel("net_demand_p_mw")
    q = series.channel("net_demand_q_mvar")
    return FeatureContext(
        p=p,
        q=q,
        p_prefix=np.vstack(
            (
                np.zeros((1, 2)),
                np.cumsum(p, axis=0, dtype=np.float64),
            )
        ),
        q_prefix=np.vstack(
            (
                np.zeros((1, 2)),
                np.cumsum(q, axis=0, dtype=np.float64),
            )
        ),
        calendar=np.zeros((series.n_steps, 8), dtype=np.float32),
        static=StaticBusFeatures(
            bus_index=series.bus_index,
            values=np.zeros((2, 9), dtype=np.float32),
            names=(
                "vn_kv",
                "line_degree",
                "trafo_degree",
                "n_load",
                "n_sgen",
                "n_storage",
                "installed_load_p_mw",
                "installed_sgen_p_mw",
                "installed_storage_abs_p_mw",
            ),
        ),
    )


def test_sample_cartesian_rows_is_reproducible() -> None:
    first = sample_cartesian_rows(
        100,
        4,
        16,
        max_rows=500,
        seed=42,
    )
    second = sample_cartesian_rows(
        100,
        4,
        16,
        max_rows=500,
        seed=42,
    )

    assert np.array_equal(first.window_index, second.window_index)
    assert np.array_equal(first.bus_position, second.bus_position)
    assert np.array_equal(first.horizon_position, second.horizon_position)


def test_feature_matrix_uses_only_observed_or_known_future_information() -> None:
    series = _series()
    context = _context(series)
    windows = (
        ForecastWindow(
            split="train",
            history_start=600,
            history_end=700,
            origin=700,
            target_start=701,
            target_end=704,
        ),
    )
    sample = SampleIndex(
        window_index=np.array([0, 0], dtype=np.int64),
        bus_position=np.array([0, 1], dtype=np.int64),
        horizon_position=np.array([0, 3], dtype=np.int64),
    )

    features, target_time = build_feature_matrix(
        series,
        windows,
        sample,
        context,
    )

    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.array_equal(target_time, np.array([701, 704]))
    assert features[0, FEATURE_NAMES.index("origin_p")] == pytest.approx(
        series.channel("net_demand_p_mw")[700, 0]
    )
    assert features[1, FEATURE_NAMES.index("target_lag_24h_p")] == pytest.approx(
        series.channel("net_demand_p_mw")[704 - 96, 1]
    )


def test_target_vector_reads_exact_target_bus_pairs() -> None:
    series = _series()
    target = target_vector(
        series,
        "net_demand_p_mw",
        np.array([10, 20], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    )

    assert target[0] == pytest.approx(series.channel("net_demand_p_mw")[10, 0])
    assert target[1] == pytest.approx(series.channel("net_demand_p_mw")[20, 1])


def test_probabilistic_metrics_are_perfect_for_centered_interval() -> None:
    metric = StreamingProbabilisticMetrics(
        quantiles=(0.05, 0.50, 0.95),
    )
    truth = np.array([[1.0, 2.0]])
    prediction = np.array(
        [
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        ]
    )
    metric.update(truth, prediction)
    result = metric.finalize()

    assert result["interval_coverage_90"] == pytest.approx(1.0)
    assert result["mean_interval_width_90"] == pytest.approx(2.0)
    assert result["adjacent_quantile_crossing_rate"] == pytest.approx(0.0)


def test_probabilistic_metrics_report_quantile_crossing() -> None:
    metric = StreamingProbabilisticMetrics(
        quantiles=(0.05, 0.50, 0.95),
    )
    truth = np.array([[1.0]])
    prediction = np.array([[[2.0, 1.0, 0.0]]])
    metric.update(truth, prediction)
    result = metric.finalize()

    assert result["adjacent_quantile_crossing_rate"] == pytest.approx(1.0)
