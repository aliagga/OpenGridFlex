from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
import pandapower as pp
import pandas as pd
import simbench as sb

SUPPORTED_GRID_CODES = frozenset(
    {
        "1-MV-urban--1-no_sw",
    }
)

REQUIRED_PROFILE_KEYS = frozenset(
    {
        "load",
        "powerplants",
        "renewables",
        "storage",
    }
)

FINGERPRINT_TABLES = (
    "bus",
    "line",
    "trafo",
    "load",
    "sgen",
    "storage",
    "ext_grid",
    "switch",
)

SIMBENCH_TIME_COLUMN = "time"
SIMBENCH_TIME_FORMAT = "%d.%m.%Y %H:%M"
SIMBENCH_TIMEZONE = "Europe/Berlin"
SIMBENCH_EXPECTED_STEP = pd.Timedelta(minutes=15)


class GridIntegrityError(RuntimeError):
    """Raised when a grid cannot satisfy the OpenGridFlex integrity contract."""


@dataclass(frozen=True)
class TapRepair:
    transformer_index: int
    previous_type: str | None
    assigned_type: str
    tap_pos: float
    tap_neutral: float
    tap_step_percent: float
    tap_step_degree: float
    reason: str


@dataclass(frozen=True)
class PowerFlowSummary:
    converged: bool
    min_vm_pu: float
    max_vm_pu: float


@dataclass(frozen=True)
class ProfileGroupSummary:
    profile_type: str
    signal_columns: int
    applied_profiles: tuple[str, ...]
    available_p_profiles: tuple[str, ...]
    available_q_profiles: tuple[str, ...]


@dataclass(frozen=True)
class GridIntegrityReport:
    grid_code: str
    pandapower_version: str
    simbench_version: str
    raw_fingerprint: str
    validated_fingerprint: str
    profile_fingerprints: tuple[tuple[str, str], ...]
    profile_shapes: tuple[tuple[str, tuple[int, int]], ...]
    profile_groups: tuple[ProfileGroupSummary, ...]
    profile_timezone: str
    profile_start_utc: str
    profile_end_utc: str
    profile_step_minutes: int
    buses: int
    lines: int
    loads: int
    static_generators: int
    storage_units: int
    transformers: int
    tap_repairs: tuple[TapRepair, ...]
    raw_power_flow: PowerFlowSummary
    validated_power_flow: PowerFlowSummary
    max_voltage_delta_vs_raw_pu: float
    mean_voltage_delta_vs_raw_pu: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedGrid:
    net: Any
    report: GridIntegrityReport


def _hash_dataframe(frame: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    schema = "\n".join(f"{column}:{dtype}" for column, dtype in frame.dtypes.items())
    hasher.update(schema.encode("utf-8"))
    payload = frame.to_json(
        orient="split",
        date_format="iso",
        double_precision=15,
        default_handler=str,
    )
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()


def _fingerprint_network(net: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"sn_mva={getattr(net, 'sn_mva', None)}".encode())
    hasher.update(f"f_hz={getattr(net, 'f_hz', None)}".encode())

    for table_name in FINGERPRINT_TABLES:
        frame = getattr(net, table_name, None)
        if not isinstance(frame, pd.DataFrame):
            continue
        hasher.update(table_name.encode())
        hasher.update(_hash_dataframe(frame).encode())

    profiles = getattr(net, "profiles", {})
    for profile_name in sorted(profiles):
        frame = profiles[profile_name]
        if not isinstance(frame, pd.DataFrame):
            raise GridIntegrityError(f"Profile {profile_name!r} is not a pandas DataFrame.")
        hasher.update(f"profile:{profile_name}".encode())
        hasher.update(_hash_dataframe(frame).encode())

    return hasher.hexdigest()


def _parse_profile_time(name: str, frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Validate SimBench local wall-clock timestamps and return canonical UTC."""
    if SIMBENCH_TIME_COLUMN not in frame.columns:
        raise GridIntegrityError(
            f"Profile {name!r} is missing required time column {SIMBENCH_TIME_COLUMN!r}."
        )

    raw_time = frame[SIMBENCH_TIME_COLUMN]

    try:
        naive = pd.DatetimeIndex(
            pd.to_datetime(
                raw_time,
                format=SIMBENCH_TIME_FORMAT,
                errors="raise",
            )
        )
    except (TypeError, ValueError) as exc:
        examples = raw_time.astype(str).drop_duplicates().head(5).tolist()
        raise GridIntegrityError(
            f"Profile {name!r} contains timestamps that do not match "
            f"{SIMBENCH_TIME_FORMAT!r}. Examples: {examples}"
        ) from exc

    try:
        localized = naive.tz_localize(
            SIMBENCH_TIMEZONE,
            ambiguous="infer",
            nonexistent="raise",
        )
    except (TypeError, ValueError) as exc:
        duplicate_examples = (
            pd.Series(naive[naive.duplicated()]).astype(str).drop_duplicates().head(8).tolist()
        )
        raise GridIntegrityError(
            f"Profile {name!r} cannot be localized safely to "
            f"{SIMBENCH_TIMEZONE!r}. Repeated local times must correspond "
            "to the ordered autumn DST fallback hour and nonexistent spring "
            "timestamps must not be present. "
            f"Repeated examples: {duplicate_examples}"
        ) from exc

    utc_index = localized.tz_convert("UTC")

    if utc_index.has_duplicates:
        duplicate_values = (
            pd.Series(utc_index[utc_index.duplicated()]).astype(str).drop_duplicates().head(5).tolist()
        )
        raise GridIntegrityError(
            f"Profile {name!r} contains duplicate absolute timestamps after "
            f"timezone localization. Examples: {duplicate_values}"
        )

    if not utc_index.is_monotonic_increasing:
        raise GridIntegrityError(
            f"Profile {name!r} is not strictly chronological after timezone localization."
        )

    if len(utc_index) > 1:
        deltas = pd.Series(utc_index[1:] - utc_index[:-1])
        bad = deltas != SIMBENCH_EXPECTED_STEP
        if bad.any():
            positions = bad[bad].index[:5].tolist()
            observed = [str(deltas.iloc[pos]) for pos in positions]
            raise GridIntegrityError(
                f"Profile {name!r} does not have a constant physical "
                f"{SIMBENCH_EXPECTED_STEP} cadence after DST normalization. "
                f"Bad interval positions: {positions}; observed: {observed}"
            )

    return utc_index


def _numeric_profile_view(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric signal columns without mutating the raw SimBench frame."""
    numeric_columns = [column for column in frame.columns if column != SIMBENCH_TIME_COLUMN]

    if not numeric_columns:
        return pd.DataFrame(index=frame.index)

    converted: dict[Any, pd.Series] = {}

    for column in numeric_columns:
        series = frame[column]
        try:
            numeric = pd.to_numeric(series, errors="raise")
        except (TypeError, ValueError) as exc:
            examples = series.loc[series.notna()].astype(str).drop_duplicates().head(5).tolist()
            raise GridIntegrityError(
                f"Profile {name!r}, column {column!r} contains values that "
                f"cannot be interpreted as numeric. Examples: {examples}"
            ) from exc

        values = np.asarray(numeric, dtype=float)
        if not np.isfinite(values).all():
            bad_positions = np.flatnonzero(~np.isfinite(values))[:5].tolist()
            raise GridIntegrityError(
                f"Profile {name!r}, column {column!r} contains non-finite "
                f"numeric values at row positions {bad_positions}."
            )

        converted[column] = pd.Series(
            values,
            index=series.index,
            name=column,
            dtype=float,
        )

    return pd.DataFrame(converted, index=frame.index)


def _safe_available_profiles(
    net: Any,
    profile_type: str,
    p_or_q: str,
) -> tuple[str, ...]:
    """Return normalized available profile IDs for diagnostic reporting."""
    try:
        values = sb.get_available_profiles(
            net,
            profile_type,
            p_or_q=p_or_q,
            continue_on_missing=True,
        )
    except Exception:
        return ()

    return tuple(sorted(str(value) for value in values))


def _profile_group_summary(
    net: Any,
    profile_type: str,
    numeric_view: pd.DataFrame,
) -> ProfileGroupSummary:
    try:
        applied = tuple(sorted(str(value) for value in sb.get_applied_profiles(net, profile_type)))
    except Exception as exc:
        raise GridIntegrityError(
            f"SimBench could not inspect applied profiles for {profile_type!r}: {exc}"
        ) from exc

    signal_columns = int(numeric_view.shape[1])

    # A time-only profile table is a legitimate structural placeholder when
    # this particular grid applies no profile of that family.
    if applied and signal_columns == 0:
        raise GridIntegrityError(
            f"Profile group {profile_type!r} is used by network elements "
            f"({list(applied)}) but contains no signal columns."
        )

    return ProfileGroupSummary(
        profile_type=profile_type,
        signal_columns=signal_columns,
        applied_profiles=applied,
        available_p_profiles=_safe_available_profiles(net, profile_type, "p"),
        available_q_profiles=_safe_available_profiles(net, profile_type, "q"),
    )


def _validate_profile_mapping_integrity(net: Any) -> None:
    """Use SimBench's own high-level mapping checker as the authority."""
    try:
        missing = sb.profiles_are_missing(net, return_as_bool=True)
    except Exception as exc:
        raise GridIntegrityError(f"SimBench profile-integrity check failed to execute: {exc}") from exc

    if bool(missing):
        raise GridIntegrityError(
            "SimBench reports that at least one profile used by a network "
            "element is missing from net.profiles."
        )


def _validate_profiles(
    net: Any,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, tuple[int, int]], ...],
    tuple[ProfileGroupSummary, ...],
    pd.DatetimeIndex,
]:
    profiles = getattr(net, "profiles", None)
    if not isinstance(profiles, dict) or not profiles:
        raise GridIntegrityError("SimBench network contains no profile dictionary.")

    missing_groups = REQUIRED_PROFILE_KEYS - set(profiles)
    if missing_groups:
        raise GridIntegrityError(f"Required profile groups are missing: {sorted(missing_groups)}")

    # This is the authoritative semantic mapping check. It understands
    # SimBench's profile naming/suffix conventions internally.
    _validate_profile_mapping_integrity(net)

    lengths: set[int] = set()
    fingerprints: list[tuple[str, str]] = []
    shapes: list[tuple[str, tuple[int, int]]] = []
    groups: list[ProfileGroupSummary] = []
    reference_time: pd.DatetimeIndex | None = None

    for name in sorted(profiles):
        frame = profiles[name]
        if not isinstance(frame, pd.DataFrame):
            raise GridIntegrityError(f"Profile {name!r} is not a DataFrame.")
        if frame.empty:
            raise GridIntegrityError(f"Profile {name!r} is empty.")
        if frame.index.has_duplicates:
            raise GridIntegrityError(f"Profile {name!r} has duplicate row indices.")
        if frame.isna().any().any():
            raise GridIntegrityError(f"Profile {name!r} contains missing values.")

        utc_time = _parse_profile_time(name, frame)
        numeric_view = _numeric_profile_view(name, frame)
        group = _profile_group_summary(net, name, numeric_view)

        if len(numeric_view) != len(frame):
            raise GridIntegrityError(
                f"Numeric validation changed the row count of profile {name!r}: "
                f"{len(frame)} -> {len(numeric_view)}"
            )

        if reference_time is None:
            reference_time = utc_time
        elif not reference_time.equals(utc_time):
            raise GridIntegrityError(
                f"Profile {name!r} uses a different absolute timestamp sequence "
                "from the other SimBench profile tables."
            )

        lengths.add(len(frame))
        fingerprints.append((name, _hash_dataframe(frame)))
        shapes.append((name, frame.shape))
        groups.append(group)

    if len(lengths) != 1:
        raise GridIntegrityError(f"Profile tables do not share one common time length: {sorted(lengths)}")

    if reference_time is None:
        raise GridIntegrityError("No profile timestamp sequence was validated.")

    return (
        tuple(fingerprints),
        tuple(shapes),
        tuple(groups),
        reference_time,
    )


def _validate_tap_bounds(net: Any) -> None:
    required = {
        "tap_pos",
        "tap_neutral",
        "tap_min",
        "tap_max",
        "tap_step_percent",
        "tap_step_degree",
        "tap_side",
        "tap_changer_type",
    }
    missing = required - set(net.trafo.columns)
    if missing:
        raise GridIntegrityError(f"Transformer table is missing required fields: {sorted(missing)}")

    active = net.trafo["tap_pos"].notna()
    below_min = active & net.trafo["tap_min"].notna() & (net.trafo["tap_pos"] < net.trafo["tap_min"])
    above_max = active & net.trafo["tap_max"].notna() & (net.trafo["tap_pos"] > net.trafo["tap_max"])

    invalid = net.trafo.index[below_min | above_max].tolist()
    if invalid:
        raise GridIntegrityError(f"Transformer tap_pos falls outside tap_min/tap_max at indices {invalid}.")


def _repair_missing_tap_types(net: Any) -> tuple[TapRepair, ...]:
    _validate_tap_bounds(net)

    repairs: list[TapRepair] = []
    trafo = net.trafo

    tap_pos = trafo["tap_pos"]
    step_percent = trafo["tap_step_percent"]
    step_degree = trafo["tap_step_degree"]

    has_position = tap_pos.notna()
    has_magnitude_step = step_percent.notna() & (step_percent.abs() > 1e-12)
    has_angle_step = step_degree.notna() & (step_degree.abs() > 1e-12)
    missing_type = trafo["tap_changer_type"].isna()

    unresolved_angle_only = has_position & ~has_magnitude_step & has_angle_step & missing_type
    if unresolved_angle_only.any():
        indices = trafo.index[unresolved_angle_only].tolist()
        raise GridIntegrityError(
            "Found angle-only transformer taps without tap_changer_type at "
            f"indices {indices}. OpenGridFlex refuses to guess their semantics."
        )

    candidates = has_position & has_magnitude_step & missing_type

    for idx in trafo.index[candidates]:
        degree_raw = trafo.at[idx, "tap_step_degree"]
        degree = 0.0 if pd.isna(degree_raw) else float(degree_raw)

        assigned_type = "Symmetrical" if np.isclose(abs(degree), 90.0, rtol=0.0, atol=1e-9) else "Ratio"

        neutral_raw = trafo.at[idx, "tap_neutral"]
        neutral = 0.0 if pd.isna(neutral_raw) else float(neutral_raw)

        repair = TapRepair(
            transformer_index=int(idx),
            previous_type=None,
            assigned_type=assigned_type,
            tap_pos=float(trafo.at[idx, "tap_pos"]),
            tap_neutral=neutral,
            tap_step_percent=float(trafo.at[idx, "tap_step_percent"]),
            tap_step_degree=degree,
            reason=(
                "SimBench 1.6.2 may leave tap_changer_type unset; pandapower >=3 "
                "then ignores tap_pos. Type classified from tap-step semantics."
            ),
        )
        trafo.at[idx, "tap_changer_type"] = assigned_type
        repairs.append(repair)

    return tuple(repairs)


def _in_service_bus_mask(net: Any) -> pd.Series:
    if "in_service" not in net.bus.columns:
        return pd.Series(True, index=net.bus.index)
    return net.bus["in_service"].fillna(True).astype(bool)


def _run_power_flow(net: Any) -> tuple[PowerFlowSummary, pd.Series]:
    pp.runpp(
        net,
        calculate_voltage_angles=True,
        init="auto",
        numba=True,
    )

    if not bool(net.converged):
        raise GridIntegrityError("AC power flow did not converge.")

    mask = _in_service_bus_mask(net)
    vm = net.res_bus.loc[mask, "vm_pu"].astype(float)

    if vm.empty:
        raise GridIntegrityError("Power flow produced no in-service bus voltages.")
    if not np.isfinite(vm.to_numpy()).all():
        raise GridIntegrityError("Power flow produced non-finite in-service voltages.")

    summary = PowerFlowSummary(
        converged=True,
        min_vm_pu=float(vm.min()),
        max_vm_pu=float(vm.max()),
    )
    return summary, vm


def load_validated_simbench_grid(grid_code: str) -> ValidatedGrid:
    if grid_code not in SUPPORTED_GRID_CODES:
        raise GridIntegrityError(
            f"Grid {grid_code!r} is not yet in the validated OpenGridFlex set. "
            f"Supported: {sorted(SUPPORTED_GRID_CODES)}"
        )

    raw = sb.get_simbench_net(grid_code)
    raw_fingerprint = _fingerprint_network(raw)
    (
        profile_fingerprints,
        profile_shapes,
        profile_groups,
        profile_time,
    ) = _validate_profiles(raw)

    raw_pf_net = deepcopy(raw)
    raw_power_flow, raw_vm = _run_power_flow(raw_pf_net)

    validated = deepcopy(raw)
    repairs = _repair_missing_tap_types(validated)
    validated_fingerprint = _fingerprint_network(validated)
    validated_power_flow, validated_vm = _run_power_flow(validated)

    common = raw_vm.index.intersection(validated_vm.index)
    delta = (validated_vm.loc[common] - raw_vm.loc[common]).abs()

    if delta.empty:
        raise GridIntegrityError("Cannot compare raw and validated bus voltages.")

    non_neutral_repairs = [
        repair
        for repair in repairs
        if not np.isclose(
            repair.tap_pos,
            repair.tap_neutral,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    max_delta = float(delta.max())
    mean_delta = float(delta.mean())

    if non_neutral_repairs and max_delta <= 1e-8:
        raise GridIntegrityError(
            "Non-neutral transformer taps were repaired, but the AC solution did "
            "not respond. Tap semantics remain unverified."
        )

    report = GridIntegrityReport(
        grid_code=grid_code,
        pandapower_version=version("pandapower"),
        simbench_version=version("simbench"),
        raw_fingerprint=raw_fingerprint,
        validated_fingerprint=validated_fingerprint,
        profile_fingerprints=profile_fingerprints,
        profile_shapes=profile_shapes,
        profile_groups=profile_groups,
        profile_timezone=SIMBENCH_TIMEZONE,
        profile_start_utc=profile_time[0].isoformat(),
        profile_end_utc=profile_time[-1].isoformat(),
        profile_step_minutes=int(SIMBENCH_EXPECTED_STEP.total_seconds() // 60),
        buses=len(validated.bus),
        lines=len(validated.line),
        loads=len(validated.load),
        static_generators=len(validated.sgen),
        storage_units=len(validated.storage),
        transformers=len(validated.trafo),
        tap_repairs=repairs,
        raw_power_flow=raw_power_flow,
        validated_power_flow=validated_power_flow,
        max_voltage_delta_vs_raw_pu=max_delta,
        mean_voltage_delta_vs_raw_pu=mean_delta,
    )

    return ValidatedGrid(net=validated, report=report)
