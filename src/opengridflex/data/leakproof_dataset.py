from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from opengridflex.grids.physical_reconstruction import (
    AbsoluteProfileBundle,
    build_absolute_profile_bundle,
)
from opengridflex.grids.simbench_adapter import (
    GridIntegrityError,
    _parse_profile_time,
    load_validated_simbench_grid,
)

CANONICAL_CHANNELS = (
    "load_p_mw",
    "load_q_mvar",
    "sgen_p_mw",
    "gen_p_mw",
    "storage_p_mw",
    "net_demand_p_mw",
    "net_demand_q_mvar",
)

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class CanonicalGridSeries:
    grid_code: str
    time_utc: pd.DatetimeIndex
    bus_index: pd.Index
    values: dict[str, np.ndarray]
    fingerprint: str

    @property
    def n_steps(self) -> int:
        return len(self.time_utc)

    @property
    def n_buses(self) -> int:
        return len(self.bus_index)

    def channel(self, name: str) -> np.ndarray:
        try:
            return self.values[name]
        except KeyError as exc:
            raise GridIntegrityError(
                f"Unknown canonical channel {name!r}. Available: {sorted(self.values)}"
            ) from exc


@dataclass(frozen=True)
class ChronologicalSplit:
    n_steps: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def bounds(self, split: str) -> tuple[int, int]:
        if split == "train":
            return self.train_start, self.train_end
        if split == "validation":
            return self.validation_start, self.validation_end
        if split == "test":
            return self.test_start, self.test_end
        raise GridIntegrityError(f"Unknown split {split!r}.")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WindowSpec:
    history_steps: int
    horizon_steps: int
    lead_steps: int = 1
    stride: int = 1

    def validate(self) -> None:
        if self.history_steps < 1:
            raise GridIntegrityError("history_steps must be >= 1.")
        if self.horizon_steps < 1:
            raise GridIntegrityError("horizon_steps must be >= 1.")
        if self.lead_steps < 1:
            raise GridIntegrityError("lead_steps must be >= 1.")
        if self.stride < 1:
            raise GridIntegrityError("stride must be >= 1.")


@dataclass(frozen=True)
class ForecastWindow:
    split: str
    history_start: int
    history_end: int
    origin: int
    target_start: int
    target_end: int


@dataclass(frozen=True)
class WindowPlan:
    spec: WindowSpec
    windows: tuple[ForecastWindow, ...]
    dropped_cross_boundary: int

    def for_split(self, split: str) -> tuple[ForecastWindow, ...]:
        if split not in SPLIT_NAMES:
            raise GridIntegrityError(f"Unknown split {split!r}.")
        return tuple(window for window in self.windows if window.split == split)


@dataclass(frozen=True)
class LeakageAudit:
    total_windows: int
    train_windows: int
    validation_windows: int
    test_windows: int
    dropped_cross_boundary: int
    history_target_order_ok: bool
    target_containment_ok: bool
    cross_split_target_overlap_ok: bool
    chronological_split_ok: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.history_target_order_ok,
                self.target_containment_ok,
                self.cross_split_target_overlap_ok,
                self.chronological_split_ok,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    channels: tuple[str, ...]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    constant_mask: dict[str, np.ndarray]
    fit_start: int
    fit_end: int
    source_fingerprint: str

    @classmethod
    def fit(
        cls,
        series: CanonicalGridSeries,
        split: ChronologicalSplit,
        channels: Iterable[str],
        *,
        eps: float = 1e-12,
    ) -> TrainOnlyStandardizer:
        chosen = tuple(channels)
        if not chosen:
            raise GridIntegrityError("At least one channel is required.")

        start, end = split.bounds("train")
        if start != 0:
            raise GridIntegrityError("The current M2 contract requires training to begin at row 0.")
        if end <= start:
            raise GridIntegrityError("Training split is empty.")

        means: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        constant_mask: dict[str, np.ndarray] = {}

        for channel in chosen:
            train_values = series.channel(channel)[start:end].astype(
                np.float64,
                copy=False,
            )
            mean = train_values.mean(axis=0)
            std = train_values.std(axis=0)
            is_constant = std <= eps
            safe_std = std.copy()
            safe_std[is_constant] = 1.0

            if not np.isfinite(mean).all() or not np.isfinite(safe_std).all():
                raise GridIntegrityError(f"Non-finite train-only scaling statistics for {channel!r}.")

            means[channel] = mean
            scales[channel] = safe_std
            constant_mask[channel] = is_constant

        return cls(
            channels=chosen,
            means=means,
            scales=scales,
            constant_mask=constant_mask,
            fit_start=start,
            fit_end=end,
            source_fingerprint=series.fingerprint,
        )

    def transform(self, channel: str, values: np.ndarray) -> np.ndarray:
        if channel not in self.means:
            raise GridIntegrityError(f"Standardizer was not fitted for channel {channel!r}.")

        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.means[channel].shape[0]:
            raise GridIntegrityError(
                f"Channel {channel!r} bus dimension mismatch: "
                f"{array.shape[-1]} vs {self.means[channel].shape[0]}."
            )

        return (array - self.means[channel]) / self.scales[channel]


def _table(net: Any, element: str) -> pd.DataFrame:
    table = getattr(net, element, None)
    if isinstance(table, pd.DataFrame):
        return table

    try:
        table = net[element]
    except (KeyError, TypeError):
        table = None

    if not isinstance(table, pd.DataFrame):
        raise GridIntegrityError(f"Network has no table {element!r}.")
    return table


def _aggregate_element_to_bus(
    net: Any,
    bundle: AbsoluteProfileBundle,
    key: tuple[str, str],
    bus_index: pd.Index,
) -> np.ndarray:
    element, _ = key
    frame = bundle.values[key]
    table = _table(net, element)

    if "bus" not in table.columns:
        if len(table) == 0 and frame.shape[1] == 0:
            return np.zeros(
                (bundle.row_count, len(bus_index)),
                dtype=np.float64,
            )
        raise GridIntegrityError(f"Element table {element!r} has no 'bus' column.")

    if not frame.columns.equals(table.index):
        raise GridIntegrityError(f"Absolute profile columns do not align with net.{element}.index.")

    bus_to_position = {bus: pos for pos, bus in enumerate(bus_index)}
    output = np.zeros(
        (bundle.row_count, len(bus_index)),
        dtype=np.float64,
    )

    for element_index in frame.columns:
        bus = table.at[element_index, "bus"]
        if bus not in bus_to_position:
            raise GridIntegrityError(f"{element}[{element_index}] refers to unknown bus {bus!r}.")

        values = frame[element_index].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise GridIntegrityError(f"Non-finite absolute values in {key!r}, element {element_index!r}.")
        output[:, bus_to_position[bus]] += values

    element_total = frame.to_numpy(dtype=np.float64).sum(axis=1)
    bus_total = output.sum(axis=1)
    if not np.allclose(
        element_total,
        bus_total,
        rtol=0.0,
        atol=1e-10,
    ):
        raise GridIntegrityError(f"Bus aggregation does not conserve total power for {key!r}.")

    return output


def _series_fingerprint(
    grid_code: str,
    time_utc: pd.DatetimeIndex,
    bus_index: pd.Index,
    values: dict[str, np.ndarray],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(grid_code.encode("utf-8"))
    hasher.update(np.asarray(time_utc.asi8, dtype=np.int64).tobytes(order="C"))
    hasher.update("\n".join(str(value) for value in bus_index).encode("utf-8"))

    for channel in sorted(values):
        array = np.ascontiguousarray(values[channel])
        hasher.update(channel.encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes(order="C"))

    return hasher.hexdigest()


def build_canonical_grid_series(grid_code: str) -> CanonicalGridSeries:
    validated = load_validated_simbench_grid(grid_code)
    net = validated.net
    bundle = build_absolute_profile_bundle(net)

    time_utc = _parse_profile_time("load", net.profiles["load"])
    if len(time_utc) != bundle.row_count:
        raise GridIntegrityError("Profile timeline length does not match absolute-profile row count.")

    if time_utc.tz is None or str(time_utc.tz) != "UTC":
        raise GridIntegrityError("Canonical dataset timeline must be UTC.")
    if time_utc.has_duplicates or not time_utc.is_monotonic_increasing:
        raise GridIntegrityError("Canonical dataset timeline must be unique and chronological.")

    bus_index = pd.Index(net.bus.index.copy())

    load_p = _aggregate_element_to_bus(
        net,
        bundle,
        ("load", "p_mw"),
        bus_index,
    )
    load_q = _aggregate_element_to_bus(
        net,
        bundle,
        ("load", "q_mvar"),
        bus_index,
    )
    sgen_p = _aggregate_element_to_bus(
        net,
        bundle,
        ("sgen", "p_mw"),
        bus_index,
    )
    gen_p = _aggregate_element_to_bus(
        net,
        bundle,
        ("gen", "p_mw"),
        bus_index,
    )
    storage_p = _aggregate_element_to_bus(
        net,
        bundle,
        ("storage", "p_mw"),
        bus_index,
    )

    net_demand_p = load_p + storage_p - sgen_p - gen_p
    net_demand_q = load_q.copy()

    arrays_64 = {
        "load_p_mw": load_p,
        "load_q_mvar": load_q,
        "sgen_p_mw": sgen_p,
        "gen_p_mw": gen_p,
        "storage_p_mw": storage_p,
        "net_demand_p_mw": net_demand_p,
        "net_demand_q_mvar": net_demand_q,
    }

    if tuple(arrays_64) != CANONICAL_CHANNELS:
        raise GridIntegrityError("Canonical channel order changed unexpectedly.")

    values: dict[str, np.ndarray] = {}
    for channel, array in arrays_64.items():
        if array.shape != (bundle.row_count, len(bus_index)):
            raise GridIntegrityError(
                f"Canonical channel {channel!r} has shape {array.shape}, "
                f"expected {(bundle.row_count, len(bus_index))}."
            )
        if not np.isfinite(array).all():
            raise GridIntegrityError(f"Canonical channel {channel!r} contains non-finite values.")
        values[channel] = array.astype(np.float32)

    fingerprint = _series_fingerprint(
        grid_code,
        time_utc,
        bus_index,
        values,
    )

    return CanonicalGridSeries(
        grid_code=grid_code,
        time_utc=time_utc,
        bus_index=bus_index,
        values=values,
        fingerprint=fingerprint,
    )


def make_chronological_split(
    n_steps: int,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    if n_steps < 3:
        raise GridIntegrityError("At least three timesteps are required.")

    if not 0.0 < train_fraction < 1.0:
        raise GridIntegrityError("train_fraction must lie in (0, 1).")
    if not 0.0 < validation_fraction < 1.0:
        raise GridIntegrityError("validation_fraction must lie in (0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise GridIntegrityError("train_fraction + validation_fraction must be < 1.")

    train_end = int(np.floor(n_steps * train_fraction))
    validation_end = int(np.floor(n_steps * (train_fraction + validation_fraction)))

    if train_end <= 0:
        raise GridIntegrityError("Training split would be empty.")
    if validation_end <= train_end:
        raise GridIntegrityError("Validation split would be empty.")
    if validation_end >= n_steps:
        raise GridIntegrityError("Test split would be empty.")

    return ChronologicalSplit(
        n_steps=n_steps,
        train_start=0,
        train_end=train_end,
        validation_start=train_end,
        validation_end=validation_end,
        test_start=validation_end,
        test_end=n_steps,
    )


def _target_split(
    split: ChronologicalSplit,
    target_start: int,
    target_end: int,
) -> str | None:
    for name in SPLIT_NAMES:
        start, end = split.bounds(name)
        if target_start >= start and target_end < end:
            return name
    return None


def build_window_plan(
    n_steps: int,
    split: ChronologicalSplit,
    spec: WindowSpec,
) -> WindowPlan:
    spec.validate()

    if split.n_steps != n_steps:
        raise GridIntegrityError(f"Split length {split.n_steps} does not match dataset length {n_steps}.")

    first_origin = spec.history_steps - 1
    last_origin = n_steps - spec.lead_steps - spec.horizon_steps

    if last_origin < first_origin:
        raise GridIntegrityError("Dataset is too short for the requested history/lead/horizon.")

    windows: list[ForecastWindow] = []
    dropped = 0

    for origin in range(
        first_origin,
        last_origin + 1,
        spec.stride,
    ):
        history_start = origin - spec.history_steps + 1
        target_start = origin + spec.lead_steps
        target_end = target_start + spec.horizon_steps - 1

        split_name = _target_split(
            split,
            target_start,
            target_end,
        )

        if split_name is None:
            dropped += 1
            continue

        windows.append(
            ForecastWindow(
                split=split_name,
                history_start=history_start,
                history_end=origin,
                origin=origin,
                target_start=target_start,
                target_end=target_end,
            )
        )

    if not windows:
        raise GridIntegrityError("Window plan contains no usable samples.")

    return WindowPlan(
        spec=spec,
        windows=tuple(windows),
        dropped_cross_boundary=dropped,
    )


def audit_window_plan(
    split: ChronologicalSplit,
    plan: WindowPlan,
) -> LeakageAudit:
    history_target_order_ok = True
    target_containment_ok = True
    chronological_split_ok = (
        split.train_start
        == 0
        < split.train_end
        == split.validation_start
        < split.validation_end
        == split.test_start
        < split.test_end
        == split.n_steps
    )

    targets_by_split: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}

    for window in plan.windows:
        counts[window.split] += 1

        if not (
            window.history_start
            <= window.history_end
            == window.origin
            < window.target_start
            <= window.target_end
        ):
            history_target_order_ok = False

        start, end = split.bounds(window.split)
        if not (start <= window.target_start <= window.target_end < end):
            target_containment_ok = False

        targets_by_split[window.split].update(range(window.target_start, window.target_end + 1))

    cross_split_target_overlap_ok = (
        targets_by_split["train"].isdisjoint(targets_by_split["validation"])
        and targets_by_split["train"].isdisjoint(targets_by_split["test"])
        and targets_by_split["validation"].isdisjoint(targets_by_split["test"])
    )

    audit = LeakageAudit(
        total_windows=len(plan.windows),
        train_windows=counts["train"],
        validation_windows=counts["validation"],
        test_windows=counts["test"],
        dropped_cross_boundary=plan.dropped_cross_boundary,
        history_target_order_ok=history_target_order_ok,
        target_containment_ok=target_containment_ok,
        cross_split_target_overlap_ok=cross_split_target_overlap_ok,
        chronological_split_ok=chronological_split_ok,
    )

    if not audit.passed:
        raise GridIntegrityError(f"Leakage audit failed: {audit.to_dict()}")

    return audit


def known_future_calendar_features(
    time_utc: pd.DatetimeIndex,
    *,
    local_timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    if time_utc.tz is None:
        raise GridIntegrityError("Calendar features require timezone-aware canonical timestamps.")

    local = time_utc.tz_convert(local_timezone)

    minute_of_day = local.hour * 60 + local.minute
    day_of_week = local.dayofweek
    day_of_year = local.dayofyear

    minute_angle = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    weekday_angle = 2.0 * np.pi * day_of_week / 7.0
    year_angle = 2.0 * np.pi * (day_of_year - 1) / 366.0

    utc_offset_hours = np.array(
        [timestamp.utcoffset().total_seconds() / 3600.0 for timestamp in local],
        dtype=np.float64,
    )
    is_dst = np.array(
        [float(timestamp.dst().total_seconds() != 0.0) for timestamp in local],
        dtype=np.float64,
    )

    frame = pd.DataFrame(
        {
            "minute_sin": np.sin(minute_angle),
            "minute_cos": np.cos(minute_angle),
            "weekday_sin": np.sin(weekday_angle),
            "weekday_cos": np.cos(weekday_angle),
            "year_sin": np.sin(year_angle),
            "year_cos": np.cos(year_angle),
            "utc_offset_hours": utc_offset_hours,
            "is_dst": is_dst,
        },
        index=time_utc,
    )

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise GridIntegrityError("Known-future calendar features contain non-finite values.")

    return frame


def materialize_window(
    series: CanonicalGridSeries,
    window: ForecastWindow,
    *,
    history_channels: Iterable[str],
    target_channels: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_names = tuple(history_channels)
    target_names = tuple(target_channels)

    if not history_names or not target_names:
        raise GridIntegrityError("History and target channel lists must both be non-empty.")

    history = np.stack(
        [series.channel(name)[window.history_start : window.history_end + 1] for name in history_names],
        axis=-1,
    )

    target = np.stack(
        [series.channel(name)[window.target_start : window.target_end + 1] for name in target_names],
        axis=-1,
    )

    calendar = known_future_calendar_features(
        series.time_utc[window.target_start : window.target_end + 1]
    ).to_numpy(dtype=np.float32)

    return (
        history.astype(np.float32, copy=False),
        target.astype(np.float32, copy=False),
        calendar,
    )
