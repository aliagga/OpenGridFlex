from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from opengridflex.baselines.tree_features import (
    CATEGORICAL_FEATURE_INDICES,
    FEATURE_NAMES,
    FeatureContext,
    SampleIndex,
    audit_feature_causality,
    build_feature_context,
    build_feature_matrix,
    sample_cartesian_rows,
    target_vector,
)
from opengridflex.data.benchmark_contract import (
    PRIMARY_TASK_ID,
    BenchmarkContract,
    default_benchmark_contract,
    make_contract_split,
)
from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    build_window_plan,
)
from opengridflex.grids.simbench_adapter import GridIntegrityError

LIGHTGBM_VERSION = "4.7.0"


@dataclass(frozen=True)
class LightGBMDevelopmentConfig:
    seed: int = 42
    max_train_rows: int = 150_000
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 100
    subsample: float = 0.80
    subsample_freq: int = 1
    colsample_bytree: float = 0.90
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    max_bin: int = 255
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChannelModelBundle:
    channel: str
    point_model: lgb.LGBMRegressor
    quantile_models: dict[float, lgb.LGBMRegressor]


@dataclass
class GlobalLightGBMBundle:
    contract_id: str
    task_id: str
    source_fingerprint: str
    feature_names: tuple[str, ...]
    quantiles: tuple[float, ...]
    config: LightGBMDevelopmentConfig
    channels: dict[str, ChannelModelBundle]


def require_lightgbm_version() -> str:
    observed = version("lightgbm")
    if observed != LIGHTGBM_VERSION:
        raise GridIntegrityError(f"M3.2 requires lightgbm=={LIGHTGBM_VERSION}; found {observed}.")
    return observed


def _base_parameters(
    config: LightGBMDevelopmentConfig,
) -> dict[str, Any]:
    return {
        "boosting_type": "gbdt",
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "subsample_freq": config.subsample_freq,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "max_bin": config.max_bin,
        "random_state": config.seed,
        "n_jobs": config.n_jobs,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def _fit_one_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: LightGBMDevelopmentConfig,
    objective: str,
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    parameters = _base_parameters(config)
    parameters["objective"] = objective
    if alpha is not None:
        parameters["alpha"] = alpha

    model = lgb.LGBMRegressor(**parameters)
    model.fit(
        x_train,
        y_train,
        categorical_feature=list(CATEGORICAL_FEATURE_INDICES),
    )

    if model.n_features_in_ != len(FEATURE_NAMES):
        raise GridIntegrityError("LightGBM fitted with an unexpected feature count.")

    return model


def training_sample(
    series: CanonicalGridSeries,
    context: FeatureContext,
    *,
    contract: BenchmarkContract,
    config: LightGBMDevelopmentConfig,
) -> tuple[np.ndarray, SampleIndex, np.ndarray]:
    task = contract.task(PRIMARY_TASK_ID)
    split = make_contract_split(series, contract)
    plan = build_window_plan(series.n_steps, split, task.spec)
    windows = plan.for_split("train")

    sample = sample_cartesian_rows(
        len(windows),
        series.n_buses,
        task.horizon_steps,
        max_rows=config.max_train_rows,
        seed=config.seed,
    )
    x_train, target_time = build_feature_matrix(
        series,
        windows,
        sample,
        context,
    )

    if x_train.shape != (sample.size, len(FEATURE_NAMES)):
        raise GridIntegrityError(f"Unexpected LightGBM training matrix shape {x_train.shape}.")

    return x_train, sample, target_time


def fit_global_lightgbm(
    series: CanonicalGridSeries,
    *,
    contract: BenchmarkContract | None = None,
    config: LightGBMDevelopmentConfig | None = None,
) -> tuple[GlobalLightGBMBundle, FeatureContext]:
    require_lightgbm_version()

    benchmark = contract or default_benchmark_contract()
    development = config or LightGBMDevelopmentConfig()
    task = benchmark.task(PRIMARY_TASK_ID)
    audit_feature_causality(task)

    context = build_feature_context(series)
    x_train, sample, target_time = training_sample(
        series,
        context,
        contract=benchmark,
        config=development,
    )

    channels: dict[str, ChannelModelBundle] = {}

    for channel in task.target_channels:
        y_train = target_vector(
            series,
            channel,
            target_time,
            sample.bus_position,
        )

        point_model = _fit_one_model(
            x_train,
            y_train,
            config=development,
            objective="regression",
        )

        quantile_models = {
            quantile: _fit_one_model(
                x_train,
                y_train,
                config=development,
                objective="quantile",
                alpha=quantile,
            )
            for quantile in benchmark.evaluation.quantiles
        }

        channels[channel] = ChannelModelBundle(
            channel=channel,
            point_model=point_model,
            quantile_models=quantile_models,
        )

    bundle = GlobalLightGBMBundle(
        contract_id=benchmark.contract_id,
        task_id=task.task_id,
        source_fingerprint=series.fingerprint,
        feature_names=FEATURE_NAMES,
        quantiles=benchmark.evaluation.quantiles,
        config=development,
        channels=channels,
    )
    return bundle, context


def save_global_lightgbm(
    bundle: GlobalLightGBMBundle,
    output_dir: str | Path,
) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for channel, channel_bundle in bundle.channels.items():
        point_path = root / f"{channel}__point.txt"
        channel_bundle.point_model.booster_.save_model(str(point_path))
        written.append(point_path)

        for quantile, model in channel_bundle.quantile_models.items():
            q_label = f"{quantile:.2f}".replace(".", "p")
            path = root / f"{channel}__q{q_label}.txt"
            model.booster_.save_model(str(path))
            written.append(path)

    return written
