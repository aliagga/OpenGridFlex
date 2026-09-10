from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opengridflex.grids.simbench_adapter import GridIntegrityError


def _safe_wape(
    absolute_error_sum: float,
    absolute_target_sum: float,
    epsilon: float,
) -> float | None:
    if absolute_target_sum <= epsilon:
        return None
    return absolute_error_sum / absolute_target_sum


@dataclass
class StreamingDeterministicMetrics:
    n_buses: int
    horizon_steps: int
    active_bus_mask: np.ndarray
    zero_activity_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        mask = np.asarray(self.active_bus_mask, dtype=bool)
        if mask.shape != (self.n_buses,):
            raise GridIntegrityError("active_bus_mask has incompatible shape.")
        if not mask.any():
            raise GridIntegrityError("At least one active bus is required for macro metrics.")
        self.active_bus_mask = mask

        self._micro_abs = 0.0
        self._micro_sq = 0.0
        self._micro_true_abs = 0.0
        self._micro_count = 0

        self._bus_abs = np.zeros(self.n_buses, dtype=np.float64)
        self._bus_sq = np.zeros(self.n_buses, dtype=np.float64)
        self._bus_true_abs = np.zeros(self.n_buses, dtype=np.float64)
        self._bus_count = np.zeros(self.n_buses, dtype=np.int64)

        self._system_abs = 0.0
        self._system_sq = 0.0
        self._system_true_abs = 0.0
        self._system_count = 0

        self._horizon_abs = np.zeros(self.horizon_steps, dtype=np.float64)
        self._horizon_sq = np.zeros(self.horizon_steps, dtype=np.float64)
        self._horizon_true_abs = np.zeros(self.horizon_steps, dtype=np.float64)
        self._horizon_count = np.zeros(self.horizon_steps, dtype=np.int64)

        self._windows = 0

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        truth = np.asarray(target, dtype=np.float64)
        forecast = np.asarray(prediction, dtype=np.float64)

        expected = (self.horizon_steps, self.n_buses)
        if truth.shape != expected or forecast.shape != expected:
            raise GridIntegrityError(
                f"Metric update expects shape {expected}; "
                f"got target={truth.shape}, prediction={forecast.shape}."
            )
        if not np.isfinite(truth).all() or not np.isfinite(forecast).all():
            raise GridIntegrityError("Deterministic metric update received non-finite values.")

        error = forecast - truth
        abs_error = np.abs(error)
        sq_error = np.square(error)
        abs_truth = np.abs(truth)

        self._micro_abs += float(abs_error.sum())
        self._micro_sq += float(sq_error.sum())
        self._micro_true_abs += float(abs_truth.sum())
        self._micro_count += int(truth.size)

        self._bus_abs += abs_error.sum(axis=0)
        self._bus_sq += sq_error.sum(axis=0)
        self._bus_true_abs += abs_truth.sum(axis=0)
        self._bus_count += self.horizon_steps

        system_truth = truth.sum(axis=1)
        system_forecast = forecast.sum(axis=1)
        system_error = system_forecast - system_truth
        self._system_abs += float(np.abs(system_error).sum())
        self._system_sq += float(np.square(system_error).sum())
        self._system_true_abs += float(np.abs(system_truth).sum())
        self._system_count += self.horizon_steps

        self._horizon_abs += abs_error.sum(axis=1)
        self._horizon_sq += sq_error.sum(axis=1)
        self._horizon_true_abs += abs_truth.sum(axis=1)
        self._horizon_count += self.n_buses

        self._windows += 1

    def finalize(self) -> dict[str, Any]:
        if self._windows == 0:
            raise GridIntegrityError("Cannot finalize deterministic metrics with zero windows.")

        micro = {
            "mae": self._micro_abs / self._micro_count,
            "rmse": float(np.sqrt(self._micro_sq / self._micro_count)),
            "wape": _safe_wape(
                self._micro_abs,
                self._micro_true_abs,
                self.zero_activity_epsilon,
            ),
            "points": self._micro_count,
        }

        active = self.active_bus_mask
        bus_mae = self._bus_abs[active] / self._bus_count[active]
        bus_rmse = np.sqrt(self._bus_sq[active] / self._bus_count[active])

        bus_wape_values = []
        for absolute_error, absolute_truth in zip(
            self._bus_abs[active],
            self._bus_true_abs[active],
            strict=True,
        ):
            value = _safe_wape(
                float(absolute_error),
                float(absolute_truth),
                self.zero_activity_epsilon,
            )
            if value is not None:
                bus_wape_values.append(value)

        macro_active = {
            "mae": float(bus_mae.mean()),
            "rmse": float(bus_rmse.mean()),
            "wape": (float(np.mean(bus_wape_values)) if bus_wape_values else None),
            "active_buses": int(active.sum()),
        }

        system_aggregate = {
            "mae": self._system_abs / self._system_count,
            "rmse": float(np.sqrt(self._system_sq / self._system_count)),
            "wape": _safe_wape(
                self._system_abs,
                self._system_true_abs,
                self.zero_activity_epsilon,
            ),
            "points": self._system_count,
        }

        horizon_mae = self._horizon_abs / self._horizon_count
        horizon_rmse = np.sqrt(self._horizon_sq / self._horizon_count)
        horizon_wape = [
            _safe_wape(
                float(self._horizon_abs[index]),
                float(self._horizon_true_abs[index]),
                self.zero_activity_epsilon,
            )
            for index in range(self.horizon_steps)
        ]

        return {
            "windows": self._windows,
            "bus_micro": micro,
            "bus_macro_active": macro_active,
            "system_aggregate": system_aggregate,
            "horizon_step": {
                "mae": [float(value) for value in horizon_mae],
                "rmse": [float(value) for value in horizon_rmse],
                "wape": horizon_wape,
            },
        }
