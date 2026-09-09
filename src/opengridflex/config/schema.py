from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a research configuration is invalid or ambiguous."""


def _require_fraction(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ConfigError(f"{name} must be strictly between 0 and 1; got {value!r}")


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    grids: tuple[str, ...]
    resolution_minutes: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int

    def validate(self) -> DataConfig:
        if not self.dataset.strip():
            raise ConfigError("dataset must be non-empty")
        if not self.grids or any(not g.strip() for g in self.grids):
            raise ConfigError("at least one non-empty grid identifier is required")
        if self.resolution_minutes <= 0 or 1440 % self.resolution_minutes != 0:
            raise ConfigError(
                "resolution_minutes must be a positive divisor of 1440 so day boundaries are exact"
            )
        for name, value in (
            ("train_fraction", self.train_fraction),
            ("val_fraction", self.val_fraction),
            ("test_fraction", self.test_fraction),
        ):
            _require_fraction(name, value)
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-12:
            raise ConfigError(f"train/val/test fractions must sum to 1.0; got {total:.12f}")
        if self.seed < 0:
            raise ConfigError("seed must be non-negative")
        return self

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DataConfig:
        try:
            obj = cls(
                dataset=str(raw["dataset"]),
                grids=tuple(str(x) for x in raw["grids"]),
                resolution_minutes=int(raw["resolution_minutes"]),
                train_fraction=float(raw["train_fraction"]),
                val_fraction=float(raw["val_fraction"]),
                test_fraction=float(raw["test_fraction"]),
                seed=int(raw["seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid data configuration: {exc}") from exc
        return obj.validate()


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    data_config: Path
    models: tuple[str, ...]
    shifts: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: tuple[str, ...]

    def validate(self) -> ExperimentConfig:
        if not self.experiment.strip():
            raise ConfigError("experiment must be non-empty")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ConfigError("models must be non-empty and unique")
        if len(set(self.shifts)) != len(self.shifts):
            raise ConfigError("shifts must be unique")
        if not self.seeds or any(s < 0 for s in self.seeds):
            raise ConfigError("seeds must be non-empty and non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("seeds must be unique")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ConfigError("metrics must be non-empty and unique")
        return self

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, base_dir: Path) -> ExperimentConfig:
        try:
            data_path = Path(str(raw["data_config"]))
            if not data_path.is_absolute():
                # Config paths are repository-root relative by contract.
                data_path = (base_dir / data_path).resolve()
            obj = cls(
                experiment=str(raw["experiment"]),
                data_config=data_path,
                models=tuple(str(x) for x in raw["models"]),
                shifts=tuple(str(x) for x in raw.get("shifts", [])),
                seeds=tuple(int(x) for x in raw["seeds"]),
                metrics=tuple(str(x) for x in raw["metrics"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid experiment configuration: {exc}") from exc
        return obj.validate()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return raw


def load_experiment_config(
    path: str | Path, *, repo_root: str | Path | None = None
) -> tuple[ExperimentConfig, DataConfig]:
    exp_path = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else exp_path.parents[2]
    exp = ExperimentConfig.from_mapping(_read_yaml(exp_path), base_dir=root)
    if not exp.data_config.exists():
        raise ConfigError(f"referenced data_config does not exist: {exp.data_config}")
    data = DataConfig.from_mapping(_read_yaml(exp.data_config))
    return exp, data
