from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

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
from opengridflex.evaluation.deterministic import StreamingDeterministicMetrics
from opengridflex.grids.simbench_adapter import GridIntegrityError

NEURALFORECAST_VERSION = "3.2.2"
PATCHTST_ALIAS = "PatchTST"


@dataclass(frozen=True)
class PatchTSTDevelopmentConfig:
    seed: int = 42
    max_steps: int = 500
    encoder_layers: int = 3
    n_heads: int = 8
    hidden_size: int = 128
    linear_hidden_size: int = 256
    dropout: float = 0.20
    fc_dropout: float = 0.20
    head_dropout: float = 0.0
    attn_dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 32
    windows_batch_size: int = 512
    inference_windows_batch_size: int = 1024
    accelerator: str = "cpu"
    devices: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationRequest:
    panel_end: int
    test_size: int
    n_windows: int
    validation_start: int
    validation_end: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def require_neuralforecast_version() -> str:
    try:
        observed = version("neuralforecast")
    except PackageNotFoundError as exc:
        raise GridIntegrityError(
            f"M3.3 requires neuralforecast=={NEURALFORECAST_VERSION}; package is not installed."
        ) from exc

    if observed != NEURALFORECAST_VERSION:
        raise GridIntegrityError(
            f"M3.3 requires neuralforecast=={NEURALFORECAST_VERSION}; found {observed}."
        )
    return observed


def validation_request(
    series: CanonicalGridSeries,
    *,
    contract: BenchmarkContract | None = None,
    max_validation_windows: int | None = None,
) -> ValidationRequest:
    benchmark = contract or default_benchmark_contract()
    task = benchmark.task(PRIMARY_TASK_ID)
    split = make_contract_split(series, benchmark)
    plan = build_window_plan(series.n_steps, split, task.spec)
    validation_windows = plan.for_split("validation")

    if not validation_windows:
        raise GridIntegrityError("M3.3 validation contains no windows.")

    n_windows = len(validation_windows)
    if max_validation_windows is not None:
        if max_validation_windows < 1:
            raise GridIntegrityError("max_validation_windows must be >= 1.")
        n_windows = min(n_windows, max_validation_windows)

    test_size = task.horizon_steps + task.stride * (n_windows - 1)
    panel_end = split.validation_start + test_size

    if panel_end > split.validation_end:
        raise GridIntegrityError("M3.3 development request would cross the validation boundary.")
    if panel_end > split.test_start:
        raise GridIntegrityError("M3.3 request would expose the sealed test split.")

    expected_windows = (test_size - task.horizon_steps) // task.stride + 1
    if expected_windows != n_windows:
        raise GridIntegrityError(
            f"NeuralForecast window geometry mismatch: expected {n_windows}, got {expected_windows}."
        )

    if max_validation_windows is None and n_windows != len(validation_windows):
        raise GridIntegrityError("Full M3.3 validation must use every frozen validation window.")
    if max_validation_windows is None and panel_end != split.validation_end:
        raise GridIntegrityError("Full M3.3 validation must terminate exactly at validation end.")

    return ValidationRequest(
        panel_end=panel_end,
        test_size=test_size,
        n_windows=n_windows,
        validation_start=split.validation_start,
        validation_end=split.validation_end,
    )


def build_channel_panel(
    series: CanonicalGridSeries,
    channel: str,
    *,
    end_exclusive: int,
) -> pd.DataFrame:
    if end_exclusive < 1 or end_exclusive > series.n_steps:
        raise GridIntegrityError(
            f"Panel end {end_exclusive} is outside canonical series length {series.n_steps}."
        )

    values = series.channel(channel)[:end_exclusive]
    if values.shape != (end_exclusive, series.n_buses):
        raise GridIntegrityError("Canonical channel shape changed while building PatchTST panel.")
    if not np.isfinite(values).all():
        raise GridIntegrityError("PatchTST panel source contains non-finite values.")

    times = series.time_utc[:end_exclusive]
    if times.tz is None:
        raise GridIntegrityError("PatchTST panel requires timezone-aware canonical timestamps.")

    # NeuralForecast receives a regular UTC-naive timeline. The instant ordering
    # remains unchanged and no local/DST transformation is performed here.
    ds = times.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    bus_position = np.arange(series.n_buses, dtype=np.int32)

    frame = pd.DataFrame(
        {
            "unique_id": np.repeat(bus_position, end_exclusive),
            "ds": np.tile(ds, series.n_buses),
            "y": values.T.reshape(-1).astype(np.float32, copy=False),
        }
    )

    expected_rows = end_exclusive * series.n_buses
    if len(frame) != expected_rows:
        raise GridIntegrityError(f"PatchTST panel has {len(frame)} rows; expected {expected_rows}.")
    return frame


def _build_patchtst_model(task: Any, config: PatchTSTDevelopmentConfig) -> Any:
    require_neuralforecast_version()
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import PatchTST

    return PatchTST(
        h=task.horizon_steps,
        input_size=task.history_steps,
        encoder_layers=config.encoder_layers,
        n_heads=config.n_heads,
        hidden_size=config.hidden_size,
        linear_hidden_size=config.linear_hidden_size,
        dropout=config.dropout,
        fc_dropout=config.fc_dropout,
        head_dropout=config.head_dropout,
        attn_dropout=config.attn_dropout,
        patch_len=config.patch_len,
        stride=config.stride,
        revin=True,
        revin_affine=False,
        revin_subtract_last=True,
        activation="gelu",
        loss=MAE(),
        valid_loss=MAE(),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        early_stop_patience_steps=-1,
        val_check_steps=max(1, min(100, config.max_steps)),
        batch_size=config.batch_size,
        valid_batch_size=config.batch_size,
        windows_batch_size=config.windows_batch_size,
        inference_windows_batch_size=config.inference_windows_batch_size,
        scaler_type="identity",
        random_seed=config.seed,
        alias=PATCHTST_ALIAS,
        accelerator=config.accelerator,
        devices=config.devices,
        deterministic=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        num_sanity_val_steps=0,
    )


def _validate_cv_layout(
    cv: pd.DataFrame,
    *,
    n_buses: int,
    n_windows: int,
    horizon_steps: int,
) -> None:
    required = {"unique_id", "ds", "cutoff", "y", PATCHTST_ALIAS}
    missing = required - set(cv.columns)
    if missing:
        raise GridIntegrityError(f"PatchTST cross-validation output is missing columns {sorted(missing)}.")

    block = n_windows * horizon_steps
    expected_rows = n_buses * block
    if len(cv) != expected_rows:
        raise GridIntegrityError(f"PatchTST produced {len(cv)} rows; expected {expected_rows}.")

    ids = cv["unique_id"].to_numpy(copy=False)
    for bus_position in range(n_buses):
        start = bus_position * block
        end = start + block
        if not np.all(ids[start:end] == bus_position):
            raise GridIntegrityError("PatchTST cross-validation row ordering changed unexpectedly.")

    prediction = cv[PATCHTST_ALIAS].to_numpy(copy=False)
    truth = cv["y"].to_numpy(copy=False)
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise GridIntegrityError("PatchTST cross-validation returned non-finite values.")


def evaluate_patchtst_channel(
    series: CanonicalGridSeries,
    channel: str,
    *,
    contract: BenchmarkContract | None = None,
    config: PatchTSTDevelopmentConfig | None = None,
    max_validation_windows: int | None = None,
) -> dict[str, Any]:
    require_neuralforecast_version()
    from neuralforecast import NeuralForecast

    benchmark = contract or default_benchmark_contract()
    task = benchmark.task(PRIMARY_TASK_ID)
    development = config or PatchTSTDevelopmentConfig()
    request = validation_request(
        series,
        contract=benchmark,
        max_validation_windows=max_validation_windows,
    )

    if channel not in task.target_channels:
        raise GridIntegrityError(f"M3.3 channel {channel!r} is not part of primary task targets.")

    panel = build_channel_panel(
        series,
        channel,
        end_exclusive=request.panel_end,
    )
    model = _build_patchtst_model(task, development)
    neural = NeuralForecast(models=[model], freq=f"{task.resolution_minutes}min")

    started = time.perf_counter()
    cv = neural.cross_validation(
        df=panel,
        n_windows=None,
        test_size=request.test_size,
        step_size=task.stride,
        val_size=0,
        refit=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - started

    if not isinstance(cv, pd.DataFrame):
        cv = cv.to_pandas()

    _validate_cv_layout(
        cv,
        n_buses=series.n_buses,
        n_windows=request.n_windows,
        horizon_steps=task.horizon_steps,
    )

    prediction = cv[PATCHTST_ALIAS].to_numpy(copy=False).reshape(
        series.n_buses,
        request.n_windows,
        task.horizon_steps,
    )
    truth = cv["y"].to_numpy(copy=False).reshape(
        series.n_buses,
        request.n_windows,
        task.horizon_steps,
    )

    prediction = prediction.transpose(1, 2, 0)
    truth = truth.transpose(1, 2, 0)

    split = make_contract_split(series, benchmark)
    active_mask = training_activity_mask(
        series,
        split,
        channel,
        epsilon=benchmark.evaluation.zero_activity_epsilon,
    )
    metric = StreamingDeterministicMetrics(
        n_buses=series.n_buses,
        horizon_steps=task.horizon_steps,
        active_bus_mask=active_mask,
        zero_activity_epsilon=benchmark.evaluation.zero_activity_epsilon,
    )

    for index in range(request.n_windows):
        metric.update(truth[index], prediction[index])

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    return {
        "stage": "M3.3",
        "model": "PatchTST",
        "library": "neuralforecast",
        "neuralforecast_version": version("neuralforecast"),
        "split": "validation",
        "test_split_sealed": True,
        "contract_id": benchmark.contract_id,
        "task_id": task.task_id,
        "source_fingerprint": series.fingerprint,
        "channel": channel,
        "request": request.to_dict(),
        "config": development.to_dict(),
        "parameter_count": parameter_count,
        "fit_predict_seconds": float(elapsed),
        "metrics": metric.finalize(),
    }
