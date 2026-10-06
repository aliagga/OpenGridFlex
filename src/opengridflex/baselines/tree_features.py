from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    ForecastWindow,
    known_future_calendar_features,
)
from opengridflex.grids.simbench_adapter import (
    GridIntegrityError,
    load_validated_simbench_grid,
)

FEATURE_NAMES = (
    "bus_pos",
    "horizon_step",
    "origin_p",
    "origin_q",
    "lag_1h_p",
    "lag_1h_q",
    "lag_4h_p",
    "lag_4h_q",
    "lag_24h_p",
    "lag_24h_q",
    "lag_7d_p",
    "lag_7d_q",
    "target_lag_24h_p",
    "target_lag_24h_q",
    "target_lag_7d_p",
    "target_lag_7d_q",
    "mean_4h_p",
    "mean_4h_q",
    "mean_24h_p",
    "mean_24h_q",
    "mean_7d_p",
    "mean_7d_q",
    "trend_4h_p",
    "trend_4h_q",
    "vn_kv",
    "line_degree",
    "trafo_degree",
    "n_load",
    "n_sgen",
    "n_storage",
    "installed_load_p_mw",
    "installed_sgen_p_mw",
    "installed_storage_abs_p_mw",
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
    "utc_offset_hours",
    "is_dst",
)

CATEGORICAL_FEATURE_INDICES = (0,)


@dataclass(frozen=True)
class StaticBusFeatures:
    bus_index: pd.Index
    values: np.ndarray
    names: tuple[str, ...]


@dataclass(frozen=True)
class FeatureContext:
    p: np.ndarray
    q: np.ndarray
    p_prefix: np.ndarray
    q_prefix: np.ndarray
    calendar: np.ndarray
    static: StaticBusFeatures


@dataclass(frozen=True)
class SampleIndex:
    window_index: np.ndarray
    bus_position: np.ndarray
    horizon_position: np.ndarray

    @property
    def size(self) -> int:
        return len(self.window_index)


def _network_table(net: Any, name: str) -> pd.DataFrame:
    table = getattr(net, name, None)
    if isinstance(table, pd.DataFrame):
        return table
    raise GridIntegrityError(f"Network has no table {name!r}.")


def _counts_by_bus(
    table: pd.DataFrame,
    bus_to_pos: dict[Any, int],
    n_buses: int,
) -> np.ndarray:
    result = np.zeros(n_buses, dtype=np.float32)
    if table.empty:
        return result
    if "bus" not in table.columns:
        raise GridIntegrityError("Element table has no bus column.")

    for bus in table["bus"]:
        if bus not in bus_to_pos:
            raise GridIntegrityError(f"Element refers to unknown bus {bus!r}.")
        result[bus_to_pos[bus]] += 1.0
    return result


def _sum_abs_by_bus(
    table: pd.DataFrame,
    column: str,
    bus_to_pos: dict[Any, int],
    n_buses: int,
) -> np.ndarray:
    result = np.zeros(n_buses, dtype=np.float32)
    if table.empty:
        return result
    if "bus" not in table.columns or column not in table.columns:
        raise GridIntegrityError(f"Element table lacks bus/{column} required for static features.")

    for _, row in table.iterrows():
        bus = row["bus"]
        if bus not in bus_to_pos:
            raise GridIntegrityError(f"Element refers to unknown bus {bus!r}.")
        result[bus_to_pos[bus]] += abs(float(row[column]))
    return result


def build_static_bus_features(
    series: CanonicalGridSeries,
) -> StaticBusFeatures:
    validated = load_validated_simbench_grid(series.grid_code)
    net = validated.net

    if not pd.Index(net.bus.index).equals(series.bus_index):
        raise GridIntegrityError("Static network bus index differs from canonical series bus index.")

    n_buses = series.n_buses
    bus_to_pos = {bus: position for position, bus in enumerate(series.bus_index)}

    vn_kv = net.bus.loc[series.bus_index, "vn_kv"].to_numpy(dtype=np.float32)

    line_degree = np.zeros(n_buses, dtype=np.float32)
    for _, row in _network_table(net, "line").iterrows():
        for endpoint in ("from_bus", "to_bus"):
            bus = row[endpoint]
            if bus in bus_to_pos:
                line_degree[bus_to_pos[bus]] += 1.0

    trafo_degree = np.zeros(n_buses, dtype=np.float32)
    for _, row in _network_table(net, "trafo").iterrows():
        for endpoint in ("hv_bus", "lv_bus"):
            bus = row[endpoint]
            if bus in bus_to_pos:
                trafo_degree[bus_to_pos[bus]] += 1.0

    load = _network_table(net, "load")
    sgen = _network_table(net, "sgen")
    storage = _network_table(net, "storage")

    names = (
        "vn_kv",
        "line_degree",
        "trafo_degree",
        "n_load",
        "n_sgen",
        "n_storage",
        "installed_load_p_mw",
        "installed_sgen_p_mw",
        "installed_storage_abs_p_mw",
    )

    values = np.column_stack(
        (
            vn_kv,
            line_degree,
            trafo_degree,
            _counts_by_bus(load, bus_to_pos, n_buses),
            _counts_by_bus(sgen, bus_to_pos, n_buses),
            _counts_by_bus(storage, bus_to_pos, n_buses),
            _sum_abs_by_bus(load, "p_mw", bus_to_pos, n_buses),
            _sum_abs_by_bus(sgen, "p_mw", bus_to_pos, n_buses),
            _sum_abs_by_bus(storage, "p_mw", bus_to_pos, n_buses),
        )
    ).astype(np.float32)

    if not np.isfinite(values).all():
        raise GridIntegrityError("Static bus features contain non-finite values.")

    return StaticBusFeatures(
        bus_index=series.bus_index,
        values=values,
        names=names,
    )


def build_feature_context(series: CanonicalGridSeries) -> FeatureContext:
    p = series.channel("net_demand_p_mw").astype(np.float32, copy=False)
    q = series.channel("net_demand_q_mvar").astype(np.float32, copy=False)

    if p.shape != q.shape:
        raise GridIntegrityError("P/Q canonical channel shapes differ.")

    p_prefix = np.vstack(
        (
            np.zeros((1, p.shape[1]), dtype=np.float64),
            np.cumsum(p, axis=0, dtype=np.float64),
        )
    )
    q_prefix = np.vstack(
        (
            np.zeros((1, q.shape[1]), dtype=np.float64),
            np.cumsum(q, axis=0, dtype=np.float64),
        )
    )
    calendar = known_future_calendar_features(series.time_utc).to_numpy(dtype=np.float32)
    static = build_static_bus_features(series)

    return FeatureContext(
        p=p,
        q=q,
        p_prefix=p_prefix,
        q_prefix=q_prefix,
        calendar=calendar,
        static=static,
    )


def sample_cartesian_rows(
    n_windows: int,
    n_buses: int,
    horizon_steps: int,
    *,
    max_rows: int,
    seed: int,
) -> SampleIndex:
    if min(n_windows, n_buses, horizon_steps, max_rows) < 1:
        raise GridIntegrityError("Sample dimensions and max_rows must be positive.")

    total = n_windows * n_buses * horizon_steps
    size = min(total, max_rows)

    if size == total:
        flat = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        flat = rng.integers(
            0,
            total,
            size=size,
            dtype=np.int64,
        )

    per_window = n_buses * horizon_steps
    window_index = flat // per_window
    remainder = flat % per_window
    bus_position = remainder // horizon_steps
    horizon_position = remainder % horizon_steps

    return SampleIndex(
        window_index=window_index,
        bus_position=bus_position,
        horizon_position=horizon_position,
    )


def exhaustive_cartesian_rows(
    n_windows: int,
    n_buses: int,
    horizon_steps: int,
) -> SampleIndex:
    total = n_windows * n_buses * horizon_steps
    return sample_cartesian_rows(
        n_windows,
        n_buses,
        horizon_steps,
        max_rows=total,
        seed=0,
    )


def _safe_take(
    values: np.ndarray,
    time_index: np.ndarray,
    bus_position: np.ndarray,
) -> np.ndarray:
    result = np.full(len(time_index), np.nan, dtype=np.float32)
    valid = time_index >= 0
    result[valid] = values[
        time_index[valid],
        bus_position[valid],
    ]
    return result


def _rolling_mean_from_prefix(
    prefix: np.ndarray,
    origin: np.ndarray,
    bus_position: np.ndarray,
    steps: int,
) -> np.ndarray:
    result = np.full(len(origin), np.nan, dtype=np.float32)
    valid = origin - steps + 1 >= 0
    if not valid.any():
        return result

    end = origin[valid] + 1
    start = end - steps
    buses = bus_position[valid]
    sums = prefix[end, buses] - prefix[start, buses]
    result[valid] = (sums / steps).astype(np.float32)
    return result


def audit_feature_causality(task: Any) -> None:
    """Prove that all measured target-aligned lags are observable at origin."""
    max_target_offset = task.lead_steps + task.horizon_steps - 1

    if max_target_offset > 96:
        raise GridIntegrityError(
            "M3.2 target-aligned 24 h lag would become future information "
            "for this task. The current feature contract is valid only when "
            "lead_steps + horizon_steps - 1 <= 96."
        )

    if task.lead_steps < 1:
        raise GridIntegrityError("Forecast lead must be at least one step.")


def build_feature_matrix(
    series: CanonicalGridSeries,
    windows: tuple[ForecastWindow, ...],
    sample: SampleIndex,
    context: FeatureContext,
) -> tuple[np.ndarray, np.ndarray]:
    if sample.size == 0:
        raise GridIntegrityError("Cannot build features for zero rows.")
    if not context.static.bus_index.equals(series.bus_index):
        raise GridIntegrityError("Static feature bus index differs from canonical series.")

    origins = np.fromiter(
        (windows[index].origin for index in sample.window_index),
        dtype=np.int64,
        count=sample.size,
    )
    target_time = np.fromiter(
        (
            windows[index].target_start + horizon
            for index, horizon in zip(
                sample.window_index,
                sample.horizon_position,
                strict=True,
            )
        ),
        dtype=np.int64,
        count=sample.size,
    )
    buses = sample.bus_position

    if np.any(target_time >= series.n_steps):
        raise GridIntegrityError("Feature sample target lies outside dataset.")
    if np.any(target_time <= origins):
        raise GridIntegrityError("Feature sample target is not in the future.")

    mean_4h_p = _rolling_mean_from_prefix(context.p_prefix, origins, buses, 16)
    mean_4h_q = _rolling_mean_from_prefix(context.q_prefix, origins, buses, 16)
    mean_24h_p = _rolling_mean_from_prefix(context.p_prefix, origins, buses, 96)
    mean_24h_q = _rolling_mean_from_prefix(context.q_prefix, origins, buses, 96)
    mean_7d_p = _rolling_mean_from_prefix(context.p_prefix, origins, buses, 672)
    mean_7d_q = _rolling_mean_from_prefix(context.q_prefix, origins, buses, 672)

    origin_p = _safe_take(context.p, origins, buses)
    origin_q = _safe_take(context.q, origins, buses)

    feature_parts = [
        buses.astype(np.float32),
        (sample.horizon_position + 1).astype(np.float32),
        origin_p,
        origin_q,
        _safe_take(context.p, origins - 4, buses),
        _safe_take(context.q, origins - 4, buses),
        _safe_take(context.p, origins - 16, buses),
        _safe_take(context.q, origins - 16, buses),
        _safe_take(context.p, origins - 96, buses),
        _safe_take(context.q, origins - 96, buses),
        _safe_take(context.p, origins - 672, buses),
        _safe_take(context.q, origins - 672, buses),
        _safe_take(context.p, target_time - 96, buses),
        _safe_take(context.q, target_time - 96, buses),
        _safe_take(context.p, target_time - 672, buses),
        _safe_take(context.q, target_time - 672, buses),
        mean_4h_p,
        mean_4h_q,
        mean_24h_p,
        mean_24h_q,
        mean_7d_p,
        mean_7d_q,
        origin_p - mean_4h_p,
        origin_q - mean_4h_q,
    ]

    for column in range(context.static.values.shape[1]):
        feature_parts.append(context.static.values[buses, column])

    for column in range(context.calendar.shape[1]):
        feature_parts.append(context.calendar[target_time, column])

    matrix = np.column_stack(feature_parts).astype(np.float32)

    if matrix.shape[1] != len(FEATURE_NAMES):
        raise GridIntegrityError(f"Feature width {matrix.shape[1]} != schema width {len(FEATURE_NAMES)}.")
    if np.isinf(matrix).any():
        raise GridIntegrityError("Feature matrix contains infinite values.")

    return matrix, target_time


def target_vector(
    series: CanonicalGridSeries,
    channel: str,
    target_time: np.ndarray,
    bus_position: np.ndarray,
) -> np.ndarray:
    values = series.channel(channel)
    target = values[target_time, bus_position].astype(np.float32)
    if not np.isfinite(target).all():
        raise GridIntegrityError("Training target contains non-finite values.")
    return target
