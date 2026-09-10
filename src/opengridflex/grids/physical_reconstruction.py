from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandapower as pp
import pandas as pd
import simbench as sb

from opengridflex.grids.simbench_adapter import (
    GridIntegrityError,
    ValidatedGrid,
    _parse_profile_time,
    load_validated_simbench_grid,
)

EXPECTED_ABSOLUTE_KEYS = frozenset(
    {
        ("load", "p_mw"),
        ("load", "q_mvar"),
        ("sgen", "p_mw"),
        ("gen", "p_mw"),
        ("storage", "p_mw"),
    }
)

BRANCH_RESULT_TABLES = (
    "res_line",
    "res_trafo",
    "res_trafo3w",
    "res_impedance",
    "res_dcline",
)

BALANCE_TOLERANCE_MW = 1e-6
BALANCE_TOLERANCE_MVAR = 1e-6


@dataclass(frozen=True)
class AbsoluteProfileBundle:
    values: dict[tuple[str, str], pd.DataFrame]
    row_count: int


@dataclass(frozen=True)
class SnapshotSummary:
    grid_code: str
    position: int
    local_time: str
    utc_time: str
    converged: bool
    min_vm_pu: float
    max_vm_pu: float
    max_line_loading_percent: float
    max_trafo_loading_percent: float
    total_load_p_mw: float
    total_load_q_mvar: float
    total_sgen_p_mw: float
    total_gen_p_mw: float
    total_storage_p_mw: float
    total_ext_grid_p_mw: float
    total_ext_grid_q_mvar: float
    active_branch_loss_mw: float
    reactive_branch_loss_mvar: float
    active_balance_residual_mw: float
    reactive_balance_residual_mvar: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalSnapshot:
    net: Any
    summary: SnapshotSummary


def _network_table(net: Any, element: str) -> pd.DataFrame:
    """Return a network table from pandapowerNet or a lightweight test double."""
    table = getattr(net, element, None)
    if isinstance(table, pd.DataFrame):
        return table

    try:
        table = net[element]
    except (KeyError, TypeError):
        table = None

    if not isinstance(table, pd.DataFrame):
        raise GridIntegrityError(f"Network has no DataFrame table {element!r}.")

    return table


def _normalize_absolute_key(key: Any) -> tuple[str, str]:
    if isinstance(key, tuple) and len(key) == 2:
        return str(key[0]), str(key[1])

    if isinstance(key, str) and "." in key:
        element, column = key.split(".", maxsplit=1)
        return element, column

    raise GridIntegrityError(
        "Unexpected SimBench absolute-profile key. Expected "
        f"(element, column) or 'element.column', got {key!r}."
    )


def build_absolute_profile_bundle(net: Any) -> AbsoluteProfileBundle:
    if sb.profiles_are_missing(net, return_as_bool=True):
        raise GridIntegrityError(
            "Cannot construct physical profiles because SimBench reports missing applied profiles."
        )

    raw_values = sb.get_absolute_values(
        net,
        profiles_instead_of_study_cases=True,
    )

    values: dict[tuple[str, str], pd.DataFrame] = {}
    for raw_key, frame in raw_values.items():
        key = _normalize_absolute_key(raw_key)

        if key in values:
            raise GridIntegrityError(f"Duplicate absolute-profile key {key!r}.")

        if not isinstance(frame, pd.DataFrame):
            raise GridIntegrityError(f"Absolute profile {key!r} is not a pandas DataFrame.")

        values[key] = frame

    missing_keys = EXPECTED_ABSOLUTE_KEYS - set(values)
    if missing_keys:
        raise GridIntegrityError(
            f"SimBench absolute-value output is missing required keys: {sorted(missing_keys)}"
        )

    row_counts = {len(frame) for frame in values.values()}
    if len(row_counts) != 1:
        raise GridIntegrityError(
            f"Absolute profile tables have inconsistent row counts: {sorted(row_counts)}"
        )

    row_count = row_counts.pop()
    if row_count <= 0:
        raise GridIntegrityError("Absolute profile bundle contains no timesteps.")

    for (element, column), frame in values.items():
        table = _network_table(net, element)

        if column not in table.columns:
            raise GridIntegrityError(
                f"Absolute profile refers to missing column net[{element!r}][{column!r}]."
            )

        expected_columns = pd.Index(table.index)
        actual_columns = pd.Index(frame.columns)

        if not actual_columns.equals(expected_columns):
            raise GridIntegrityError(
                f"Absolute profile {(element, column)!r} columns do not exactly "
                "match the network element index."
            )

        if frame.shape[1] == 0:
            continue

        numeric = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise GridIntegrityError(f"Absolute profile {(element, column)!r} contains non-finite values.")

    return AbsoluteProfileBundle(values=values, row_count=row_count)


def _profile_sources(net: Any, element: str) -> tuple[pd.DataFrame, ...]:
    profiles = getattr(net, "profiles", None)
    if not isinstance(profiles, dict):
        try:
            profiles = net["profiles"]
        except (KeyError, TypeError) as exc:
            raise GridIntegrityError("Network contains no profile dictionary.") from exc

    if element == "load":
        names = ("load",)
    elif element == "storage":
        names = ("storage",)
    elif element in {"sgen", "gen"}:
        names = ("powerplants", "renewables")
    else:
        raise GridIntegrityError(f"Unsupported profiled element {element!r}.")

    frames: list[pd.DataFrame] = []
    for name in names:
        frame = profiles.get(name)
        if not isinstance(frame, pd.DataFrame):
            raise GridIntegrityError(f"Required profile table {name!r} for element {element!r} is missing.")
        frames.append(frame)

    return tuple(frames)


def _profile_suffix(element: str, column: str) -> str:
    if element == "load" and column == "p_mw":
        return "_pload"
    if element == "load" and column == "q_mvar":
        return "_qload"
    return ""


def _resolve_profile_factor(
    net: Any,
    element: str,
    column: str,
    profile: str,
    position: int,
) -> float:
    signal_name = f"{profile}{_profile_suffix(element, column)}"
    sources = _profile_sources(net, element)

    matches = [frame for frame in sources if signal_name in frame.columns]

    if not matches:
        raise GridIntegrityError(
            f"Element {element!r} requests profile signal {signal_name!r}, "
            "but it is unavailable in the expected SimBench profile families."
        )

    if len(matches) > 1:
        raise GridIntegrityError(
            f"Profile signal {signal_name!r} for element {element!r} is "
            "ambiguous because it occurs in multiple SimBench profile families."
        )

    frame = matches[0]

    if not 0 <= position < len(frame):
        raise GridIntegrityError(
            f"Profile position {position} is outside profile {signal_name!r} with {len(frame)} rows."
        )

    try:
        factor = float(frame.iloc[position][signal_name])
    except (TypeError, ValueError) as exc:
        raise GridIntegrityError(
            f"Profile signal {signal_name!r} at position {position} cannot be interpreted as a float."
        ) from exc

    if not np.isfinite(factor):
        raise GridIntegrityError(f"Profile signal {signal_name!r} at position {position} is non-finite.")

    return factor


def manual_absolute_row(
    net: Any,
    element: str,
    column: str,
    position: int,
) -> pd.Series:
    table = _network_table(net, element)

    if column not in table.columns:
        raise GridIntegrityError(f"Network table {element!r} has no column {column!r}.")

    sources = _profile_sources(net, element)
    reference_length = len(sources[0])

    if position < 0:
        position += reference_length
    if not 0 <= position < reference_length:
        raise GridIntegrityError(f"Profile position {position} is out of range.")

    for frame in sources[1:]:
        if len(frame) != reference_length:
            raise GridIntegrityError(f"Profile families used by {element!r} do not share one row count.")

    expected: dict[Any, float] = {}

    for idx in table.index:
        base_value = float(table.at[idx, column])

        if "profile" not in table.columns:
            factor = 1.0
        else:
            raw_profile = table.at[idx, "profile"]

            if pd.isna(raw_profile):
                factor = 1.0
            else:
                factor = _resolve_profile_factor(
                    net,
                    element,
                    column,
                    str(raw_profile),
                    position,
                )

        expected[idx] = base_value * factor

    return pd.Series(expected, dtype=float)


def verify_manual_profile_formula(
    net: Any,
    bundle: AbsoluteProfileBundle,
    position: int,
    *,
    atol: float = 1e-12,
) -> None:
    for key in EXPECTED_ABSOLUTE_KEYS:
        element, column = key
        official = bundle.values[key].iloc[position].astype(float)
        manual = manual_absolute_row(net, element, column, position)

        if not official.index.equals(manual.index):
            raise GridIntegrityError(f"Manual/official profile indices differ for {key!r}.")

        if not np.allclose(
            official.to_numpy(),
            manual.to_numpy(),
            rtol=0.0,
            atol=atol,
        ):
            delta = np.abs(official.to_numpy() - manual.to_numpy())
            worst_pos = int(np.argmax(delta))
            worst_idx = official.index[worst_pos]
            raise GridIntegrityError(
                f"Manual reconstruction disagrees with SimBench for {key!r} "
                f"at element {worst_idx!r}: "
                f"official={official.iloc[worst_pos]!r}, "
                f"manual={manual.iloc[worst_pos]!r}, "
                f"abs_delta={float(delta[worst_pos]):.3e}."
            )


def apply_absolute_profile_row(
    net: Any,
    bundle: AbsoluteProfileBundle,
    position: int,
) -> None:
    if position < 0:
        position += bundle.row_count
    if not 0 <= position < bundle.row_count:
        raise GridIntegrityError(f"Profile position {position} is outside [0, {bundle.row_count}).")

    for (element, column), frame in bundle.values.items():
        if frame.shape[1] == 0:
            continue

        table = _network_table(net, element)
        row = frame.iloc[position].astype(float)
        table.loc[row.index, column] = row.to_numpy(dtype=float)

        assigned = table.loc[row.index, column].to_numpy(dtype=float)
        if not np.allclose(
            assigned,
            row.to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise GridIntegrityError(
                f"Failed to apply absolute profile {(element, column)!r} at position {position} exactly."
            )


def _safe_sum_result(net: Any, table_name: str, column: str) -> float:
    table = getattr(net, table_name, None)
    if not isinstance(table, pd.DataFrame):
        return 0.0
    if table.empty or column not in table.columns:
        return 0.0

    values = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise GridIntegrityError(f"{table_name}.{column} contains non-finite power-flow results.")
    return float(values.sum())


def _branch_loss(net: Any, kind: str) -> float:
    if kind not in {"p", "q"}:
        raise ValueError("kind must be 'p' or 'q'.")

    loss_column = "pl_mw" if kind == "p" else "ql_mvar"
    from_column = "p_from_mw" if kind == "p" else "q_from_mvar"
    to_column = "p_to_mw" if kind == "p" else "q_to_mvar"

    total = 0.0

    for table_name in BRANCH_RESULT_TABLES:
        table = getattr(net, table_name, None)
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue

        if loss_column in table.columns:
            total += _safe_sum_result(net, table_name, loss_column)
        elif from_column in table.columns and to_column in table.columns:
            total += _safe_sum_result(net, table_name, from_column)
            total += _safe_sum_result(net, table_name, to_column)

    return float(total)


def _max_result(net: Any, table_name: str, column: str) -> float:
    table = getattr(net, table_name, None)
    if not isinstance(table, pd.DataFrame):
        return 0.0
    if table.empty or column not in table.columns:
        return 0.0

    values = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise GridIntegrityError(f"{table_name}.{column} contains non-finite values.")
    return float(np.max(values))


def _snapshot_times(
    validated: ValidatedGrid,
    position: int,
) -> tuple[str, str]:
    raw_time = validated.net.profiles["load"]["time"]
    utc_time = _parse_profile_time("load", validated.net.profiles["load"])

    return str(raw_time.iloc[position]), utc_time[position].isoformat()


def reconstruct_snapshot(
    grid_code: str,
    position: int,
    *,
    verify_manual_formula: bool = True,
) -> PhysicalSnapshot:
    validated = load_validated_simbench_grid(grid_code)
    bundle = build_absolute_profile_bundle(validated.net)

    if position < 0:
        position += bundle.row_count
    if not 0 <= position < bundle.row_count:
        raise GridIntegrityError(f"Profile position {position} is outside [0, {bundle.row_count}).")

    if verify_manual_formula:
        verify_manual_profile_formula(validated.net, bundle, position)

    net = deepcopy(validated.net)
    apply_absolute_profile_row(net, bundle, position)

    pp.runpp(
        net,
        calculate_voltage_angles=True,
        init="auto",
        numba=True,
    )

    if not bool(net.converged):
        raise GridIntegrityError(f"AC power flow did not converge at profile position {position}.")

    vm = net.res_bus["vm_pu"].astype(float)
    if vm.empty or not np.isfinite(vm.to_numpy()).all():
        raise GridIntegrityError(f"Invalid bus voltages at profile position {position}.")

    if float(vm.min()) <= 0.5 or float(vm.max()) >= 1.5:
        raise GridIntegrityError(
            f"Physically implausible voltage magnitude at position {position}: "
            f"[{float(vm.min()):.6f}, {float(vm.max()):.6f}] pu."
        )

    active_loss = _branch_loss(net, "p")
    reactive_loss = _branch_loss(net, "q")

    bus_p = _safe_sum_result(net, "res_bus", "p_mw")
    bus_q = _safe_sum_result(net, "res_bus", "q_mvar")

    active_residual = bus_p + active_loss
    reactive_residual = bus_q + reactive_loss

    if abs(active_residual) > BALANCE_TOLERANCE_MW:
        raise GridIntegrityError(
            f"Active-power balance failed at position {position}: residual={active_residual:.3e} MW."
        )

    if abs(reactive_residual) > BALANCE_TOLERANCE_MVAR:
        raise GridIntegrityError(
            f"Reactive-power balance failed at position {position}: residual={reactive_residual:.3e} Mvar."
        )

    local_time, utc_time = _snapshot_times(validated, position)

    summary = SnapshotSummary(
        grid_code=grid_code,
        position=position,
        local_time=local_time,
        utc_time=utc_time,
        converged=True,
        min_vm_pu=float(vm.min()),
        max_vm_pu=float(vm.max()),
        max_line_loading_percent=_max_result(
            net,
            "res_line",
            "loading_percent",
        ),
        max_trafo_loading_percent=_max_result(
            net,
            "res_trafo",
            "loading_percent",
        ),
        total_load_p_mw=_safe_sum_result(net, "res_load", "p_mw"),
        total_load_q_mvar=_safe_sum_result(net, "res_load", "q_mvar"),
        total_sgen_p_mw=_safe_sum_result(net, "res_sgen", "p_mw"),
        total_gen_p_mw=_safe_sum_result(net, "res_gen", "p_mw"),
        total_storage_p_mw=_safe_sum_result(net, "res_storage", "p_mw"),
        total_ext_grid_p_mw=_safe_sum_result(net, "res_ext_grid", "p_mw"),
        total_ext_grid_q_mvar=_safe_sum_result(net, "res_ext_grid", "q_mvar"),
        active_branch_loss_mw=active_loss,
        reactive_branch_loss_mvar=reactive_loss,
        active_balance_residual_mw=float(active_residual),
        reactive_balance_residual_mvar=float(reactive_residual),
    )

    return PhysicalSnapshot(net=net, summary=summary)


def _argmax_row_sum(frame: pd.DataFrame) -> int | None:
    if frame.shape[1] == 0:
        return None

    values = frame.to_numpy(dtype=float)
    return int(np.argmax(values.sum(axis=1)))


def select_validation_positions(
    net: Any,
    bundle: AbsoluteProfileBundle,
) -> tuple[int, ...]:
    positions: set[int] = {
        0,
        bundle.row_count // 4,
        bundle.row_count // 2,
        (3 * bundle.row_count) // 4,
        bundle.row_count - 1,
    }

    peak_load = _argmax_row_sum(bundle.values[("load", "p_mw")])
    if peak_load is not None:
        positions.add(peak_load)

    generation_parts = [
        bundle.values[("sgen", "p_mw")],
        bundle.values[("gen", "p_mw")],
    ]
    generation_parts = [frame for frame in generation_parts if frame.shape[1] > 0]
    if generation_parts:
        generation = np.zeros(bundle.row_count, dtype=float)
        for frame in generation_parts:
            generation += frame.to_numpy(dtype=float).sum(axis=1)
        positions.add(int(np.argmax(generation)))

    naive = pd.DatetimeIndex(
        pd.to_datetime(
            net.profiles["load"]["time"],
            format="%d.%m.%Y %H:%M",
            errors="raise",
        )
    )
    if len(naive) > 1:
        deltas = naive[1:] - naive[:-1]
        transition_positions = np.flatnonzero(deltas != pd.Timedelta(minutes=15))
        for pos in transition_positions:
            positions.add(int(pos))
            positions.add(int(pos + 1))

    return tuple(sorted(pos for pos in positions if 0 <= pos < bundle.row_count))
