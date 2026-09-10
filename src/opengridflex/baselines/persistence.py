from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from opengridflex.data.leakproof_dataset import CanonicalGridSeries, ForecastWindow
from opengridflex.grids.simbench_adapter import GridIntegrityError


@dataclass(frozen=True)
class PersistenceBaseline:
    baseline_id: str
    label: str
    mode: str
    lag_steps: int
    resolution_minutes: int = 15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LAST_VALUE = PersistenceBaseline(
    baseline_id="last_value_v1",
    label="Last-value persistence",
    mode="origin_constant",
    lag_steps=0,
)

DAILY_SEASONAL = PersistenceBaseline(
    baseline_id="seasonal_24h_v1",
    label="24-hour physical seasonal persistence",
    mode="target_lag",
    lag_steps=96,
)

WEEKLY_SEASONAL = PersistenceBaseline(
    baseline_id="seasonal_7d_v1",
    label="7-day physical seasonal persistence",
    mode="target_lag",
    lag_steps=672,
)

SANITY_BASELINES = (
    LAST_VALUE,
    DAILY_SEASONAL,
    WEEKLY_SEASONAL,
)


def source_indices_for_window(
    window: ForecastWindow,
    baseline: PersistenceBaseline,
) -> np.ndarray:
    horizon = window.target_end - window.target_start + 1
    target_indices = np.arange(
        window.target_start,
        window.target_end + 1,
        dtype=np.int64,
    )

    if baseline.mode == "origin_constant":
        source = np.full(horizon, window.origin, dtype=np.int64)
    elif baseline.mode == "target_lag":
        if baseline.lag_steps < 1:
            raise GridIntegrityError(f"Seasonal baseline {baseline.baseline_id!r} must have lag_steps >= 1.")
        source = target_indices - baseline.lag_steps
    else:
        raise GridIntegrityError(f"Unknown baseline mode {baseline.mode!r}.")

    if np.any(source < 0):
        raise GridIntegrityError(
            f"Baseline {baseline.baseline_id!r} requires observations "
            "before the beginning of the canonical dataset."
        )

    if np.any(source > window.origin):
        raise GridIntegrityError(
            f"Baseline {baseline.baseline_id!r} would read future measured values for origin {window.origin}."
        )

    return source


def predict_window(
    series: CanonicalGridSeries,
    channel: str,
    window: ForecastWindow,
    baseline: PersistenceBaseline,
) -> np.ndarray:
    values = series.channel(channel)
    source = source_indices_for_window(window, baseline)
    prediction = values[source]

    if prediction.ndim != 2:
        raise GridIntegrityError(
            f"Expected 2-D [horizon, bus] prediction for {channel!r}; got shape {prediction.shape}."
        )
    if not np.isfinite(prediction).all():
        raise GridIntegrityError(f"Baseline {baseline.baseline_id!r} produced non-finite values.")

    return prediction.astype(np.float64, copy=False)
