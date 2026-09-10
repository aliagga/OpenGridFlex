from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    ChronologicalSplit,
    TrainOnlyStandardizer,
    WindowPlan,
    WindowSpec,
    audit_window_plan,
    build_window_plan,
    known_future_calendar_features,
)
from opengridflex.grids.simbench_adapter import GridIntegrityError

CONTRACT_ID = "gridshiftbench-v1"
CONTRACT_SCHEMA_VERSION = 1
LOCAL_TIMEZONE = "Europe/Berlin"
GRID_CODE = "1-MV-urban--1-no_sw"

TRAIN_START_LOCAL = "2016-01-01 00:00:00"
VALIDATION_START_LOCAL = "2016-09-13 00:00:00"
TEST_START_LOCAL = "2016-11-07 00:00:00"
DATASET_END_LOCAL = "2017-01-01 00:00:00"

CALENDAR_FEATURES = (
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
    "utc_offset_hours",
    "is_dst",
)

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

PRIMARY_TASK_ID = "gsb_bus_pq_4h_v1"
SECONDARY_TASK_ID = "gsb_bus_pq_24h_v1"


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    role: str
    description: str
    history_channels: tuple[str, ...]
    target_channels: tuple[str, ...]
    known_future_features: tuple[str, ...]
    history_steps: int
    horizon_steps: int
    lead_steps: int
    stride: int
    resolution_minutes: int

    @property
    def spec(self) -> WindowSpec:
        return WindowSpec(
            history_steps=self.history_steps,
            horizon_steps=self.horizon_steps,
            lead_steps=self.lead_steps,
            stride=self.stride,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationContract:
    deterministic_metrics: tuple[str, ...]
    probabilistic_metrics: tuple[str, ...]
    quantiles: tuple[float, ...]
    central_quantile: float
    primary_interval: tuple[float, float]
    aggregation_levels: tuple[str, ...]
    zero_activity_epsilon: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkContract:
    contract_id: str
    schema_version: int
    grid_code: str
    local_timezone: str
    train_start_local: str
    validation_start_local: str
    test_start_local: str
    dataset_end_local: str
    tasks: tuple[TaskContract, ...]
    evaluation: EvaluationContract
    sign_convention: str
    split_policy: str
    scaling_policy: str
    future_covariate_policy: str

    def task(self, task_id: str) -> TaskContract:
        matches = [task for task in self.tasks if task.task_id == task_id]
        if len(matches) != 1:
            raise GridIntegrityError(f"Expected exactly one task {task_id!r}; found {len(matches)}.")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_benchmark_contract() -> BenchmarkContract:
    primary = TaskContract(
        task_id=PRIMARY_TASK_ID,
        role="primary",
        description=(
            "Bus-level net active/reactive demand forecasting: previous 24 h "
            "to next 4 h at 15-minute resolution."
        ),
        history_channels=(
            "net_demand_p_mw",
            "net_demand_q_mvar",
        ),
        target_channels=(
            "net_demand_p_mw",
            "net_demand_q_mvar",
        ),
        known_future_features=CALENDAR_FEATURES,
        history_steps=96,
        horizon_steps=16,
        lead_steps=1,
        stride=1,
        resolution_minutes=15,
    )

    secondary = TaskContract(
        task_id=SECONDARY_TASK_ID,
        role="secondary",
        description=(
            "Bus-level net active/reactive demand forecasting: previous 7 "
            "physical days to next 24 h at 15-minute resolution."
        ),
        history_channels=(
            "net_demand_p_mw",
            "net_demand_q_mvar",
        ),
        target_channels=(
            "net_demand_p_mw",
            "net_demand_q_mvar",
        ),
        known_future_features=CALENDAR_FEATURES,
        history_steps=672,
        horizon_steps=96,
        lead_steps=1,
        stride=1,
        resolution_minutes=15,
    )

    evaluation = EvaluationContract(
        deterministic_metrics=(
            "mae",
            "rmse",
            "wape",
        ),
        probabilistic_metrics=(
            "mean_pinball_loss",
            "interval_coverage_90",
            "mean_interval_width_90",
            "absolute_coverage_error_90",
        ),
        quantiles=QUANTILES,
        central_quantile=0.50,
        primary_interval=(0.05, 0.95),
        aggregation_levels=(
            "bus_micro",
            "bus_macro_active",
            "system_aggregate",
            "horizon_step",
        ),
        zero_activity_epsilon=1e-8,
        notes=(
            "Report P and Q metrics separately; never average MW and Mvar into one dimensional score.",
            "Macro metrics exclude buses with effectively zero training "
            "activity for the evaluated target channel.",
            "WAPE is reported only when the corresponding absolute target "
            "denominator is above zero_activity_epsilon.",
            "The q=0.50 forecast is the canonical point prediction for probabilistic models.",
        ),
    )

    return BenchmarkContract(
        contract_id=CONTRACT_ID,
        schema_version=CONTRACT_SCHEMA_VERSION,
        grid_code=GRID_CODE,
        local_timezone=LOCAL_TIMEZONE,
        train_start_local=TRAIN_START_LOCAL,
        validation_start_local=VALIDATION_START_LOCAL,
        test_start_local=TEST_START_LOCAL,
        dataset_end_local=DATASET_END_LOCAL,
        tasks=(primary, secondary),
        evaluation=evaluation,
        sign_convention=("consumer-positive net demand: load + storage - sgen - gen"),
        split_policy=(
            "Calendar-aligned contiguous local-day split: 256 training days, "
            "55 validation days, 55 test days; all target horizons must be "
            "fully contained in one split."
        ),
        scaling_policy=(
            "Fit normalization statistics on training rows only. Validation "
            "and test values never influence fitted preprocessing."
        ),
        future_covariate_policy=(
            "Only variables known at forecast issuance are allowed in the "
            "future-covariate tensor. Base v1 uses calendar/DST features only."
        ),
    )


def _local_boundary_to_utc(
    value: str,
    timezone: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        raise GridIntegrityError(f"Contract boundary {value!r} must be local-naive.")
    return timestamp.tz_localize(
        timezone,
        ambiguous="raise",
        nonexistent="raise",
    ).tz_convert("UTC")


def _exact_position(
    time_utc: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
    *,
    allow_end: bool = False,
) -> int:
    position = int(time_utc.searchsorted(timestamp, side="left"))

    if allow_end and position == len(time_utc):
        expected_end = time_utc[-1] + pd.Timedelta(minutes=15)
        if timestamp != expected_end:
            raise GridIntegrityError(
                f"Dataset end {timestamp.isoformat()} does not equal the "
                f"physical next step {expected_end.isoformat()}."
            )
        return position

    if position >= len(time_utc) or time_utc[position] != timestamp:
        raise GridIntegrityError(f"Split boundary {timestamp.isoformat()} is not an exact dataset timestamp.")

    return position


def make_contract_split(
    series: CanonicalGridSeries,
    contract: BenchmarkContract | None = None,
) -> ChronologicalSplit:
    benchmark = contract or default_benchmark_contract()

    if series.grid_code != benchmark.grid_code:
        raise GridIntegrityError(
            f"Contract grid {benchmark.grid_code!r} does not match series grid {series.grid_code!r}."
        )

    start_utc = _local_boundary_to_utc(
        benchmark.train_start_local,
        benchmark.local_timezone,
    )
    validation_utc = _local_boundary_to_utc(
        benchmark.validation_start_local,
        benchmark.local_timezone,
    )
    test_utc = _local_boundary_to_utc(
        benchmark.test_start_local,
        benchmark.local_timezone,
    )
    end_utc = _local_boundary_to_utc(
        benchmark.dataset_end_local,
        benchmark.local_timezone,
    )

    if series.time_utc[0] != start_utc:
        raise GridIntegrityError(
            f"Dataset begins at {series.time_utc[0].isoformat()}, expected {start_utc.isoformat()}."
        )

    train_end = _exact_position(series.time_utc, validation_utc)
    validation_end = _exact_position(series.time_utc, test_utc)
    test_end = _exact_position(
        series.time_utc,
        end_utc,
        allow_end=True,
    )

    split = ChronologicalSplit(
        n_steps=series.n_steps,
        train_start=0,
        train_end=train_end,
        validation_start=train_end,
        validation_end=validation_end,
        test_start=validation_end,
        test_end=test_end,
    )

    if split.test_end != series.n_steps:
        raise GridIntegrityError("Contract split does not consume the full canonical dataset.")

    return split


def build_contract_window_plans(
    series: CanonicalGridSeries,
    contract: BenchmarkContract | None = None,
) -> dict[str, WindowPlan]:
    benchmark = contract or default_benchmark_contract()
    split = make_contract_split(series, benchmark)
    plans: dict[str, WindowPlan] = {}

    for task in benchmark.tasks:
        plan = build_window_plan(
            series.n_steps,
            split,
            task.spec,
        )
        audit_window_plan(split, plan)
        plans[task.task_id] = plan

    return plans


def fit_contract_standardizer(
    series: CanonicalGridSeries,
    contract: BenchmarkContract | None = None,
) -> TrainOnlyStandardizer:
    benchmark = contract or default_benchmark_contract()
    split = make_contract_split(series, benchmark)

    channels = tuple(
        dict.fromkeys(
            channel
            for task in benchmark.tasks
            for channel in (
                *task.history_channels,
                *task.target_channels,
            )
        )
    )

    return TrainOnlyStandardizer.fit(
        series,
        split,
        channels=channels,
    )


def training_activity_mask(
    series: CanonicalGridSeries,
    split: ChronologicalSplit,
    channel: str,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    start, end = split.bounds("train")
    values = series.channel(channel)[start:end].astype(
        np.float64,
        copy=False,
    )
    activity = np.mean(np.abs(values), axis=0)
    return activity > epsilon


def validate_contract(
    series: CanonicalGridSeries,
    contract: BenchmarkContract | None = None,
) -> dict[str, Any]:
    benchmark = contract or default_benchmark_contract()

    if benchmark.contract_id != CONTRACT_ID:
        raise GridIntegrityError("Unexpected benchmark contract ID.")
    if benchmark.schema_version != CONTRACT_SCHEMA_VERSION:
        raise GridIntegrityError("Unexpected benchmark schema version.")

    roles = [task.role for task in benchmark.tasks]
    if roles.count("primary") != 1:
        raise GridIntegrityError("Benchmark contract must contain exactly one primary task.")

    task_ids = [task.task_id for task in benchmark.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise GridIntegrityError("Benchmark task IDs must be unique.")

    quantiles = benchmark.evaluation.quantiles
    if tuple(sorted(quantiles)) != quantiles:
        raise GridIntegrityError("Quantiles must be strictly ordered.")
    if len(set(quantiles)) != len(quantiles):
        raise GridIntegrityError("Quantiles must be unique.")
    if any(not 0.0 < quantile < 1.0 for quantile in quantiles):
        raise GridIntegrityError("All quantiles must lie in (0, 1).")
    if benchmark.evaluation.central_quantile not in quantiles:
        raise GridIntegrityError("Central quantile must be present in the quantile grid.")

    calendar = known_future_calendar_features(series.time_utc)
    if tuple(calendar.columns) != CALENDAR_FEATURES:
        raise GridIntegrityError("Runtime calendar feature schema differs from contract.")

    split = make_contract_split(series, benchmark)
    plans = build_contract_window_plans(series, benchmark)
    scaler = fit_contract_standardizer(series, benchmark)

    audits = {task_id: audit_window_plan(split, plan).to_dict() for task_id, plan in plans.items()}

    active_masks = {
        channel: int(
            training_activity_mask(
                series,
                split,
                channel,
                epsilon=benchmark.evaluation.zero_activity_epsilon,
            ).sum()
        )
        for channel in (
            "net_demand_p_mw",
            "net_demand_q_mvar",
        )
    }

    return {
        "contract_id": benchmark.contract_id,
        "schema_version": benchmark.schema_version,
        "grid_code": benchmark.grid_code,
        "source_fingerprint": series.fingerprint,
        "split": split.to_dict(),
        "tasks": {task.task_id: task.to_dict() for task in benchmark.tasks},
        "audits": audits,
        "scaler_fit_rows": [
            scaler.fit_start,
            scaler.fit_end,
        ],
        "active_bus_counts": active_masks,
    }
