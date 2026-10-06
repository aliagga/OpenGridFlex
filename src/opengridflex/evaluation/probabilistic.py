from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opengridflex.grids.simbench_adapter import GridIntegrityError


@dataclass
class StreamingProbabilisticMetrics:
    quantiles: tuple[float, ...]
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95

    def __post_init__(self) -> None:
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise GridIntegrityError("Quantiles must be sorted.")
        if len(set(self.quantiles)) != len(self.quantiles):
            raise GridIntegrityError("Quantiles must be unique.")
        if self.lower_quantile not in self.quantiles:
            raise GridIntegrityError("Lower interval quantile is missing.")
        if self.upper_quantile not in self.quantiles:
            raise GridIntegrityError("Upper interval quantile is missing.")

        self._pinball_sum = np.zeros(len(self.quantiles), dtype=np.float64)
        self._pinball_count = 0
        self._covered = 0
        self._interval_points = 0
        self._width_sum = 0.0
        self._crossings = 0
        self._crossing_pairs = 0

    def update(
        self,
        target: np.ndarray,
        quantile_prediction: np.ndarray,
    ) -> None:
        truth = np.asarray(target, dtype=np.float64)
        pred = np.asarray(quantile_prediction, dtype=np.float64)

        if pred.shape[:-1] != truth.shape:
            raise GridIntegrityError("Probabilistic prediction shape does not match target.")
        if pred.shape[-1] != len(self.quantiles):
            raise GridIntegrityError("Probabilistic prediction quantile dimension is wrong.")
        if not np.isfinite(truth).all() or not np.isfinite(pred).all():
            raise GridIntegrityError("Probabilistic metrics received non-finite values.")

        for index, quantile in enumerate(self.quantiles):
            error = truth - pred[..., index]
            loss = np.maximum(
                quantile * error,
                (quantile - 1.0) * error,
            )
            self._pinball_sum[index] += float(loss.sum())

        self._pinball_count += int(truth.size)

        lower_index = self.quantiles.index(self.lower_quantile)
        upper_index = self.quantiles.index(self.upper_quantile)
        lower = pred[..., lower_index]
        upper = pred[..., upper_index]

        covered = (truth >= lower) & (truth <= upper)
        self._covered += int(covered.sum())
        self._interval_points += int(truth.size)
        self._width_sum += float((upper - lower).sum())

        adjacent_crossing = pred[..., :-1] > pred[..., 1:]
        self._crossings += int(adjacent_crossing.sum())
        self._crossing_pairs += int(adjacent_crossing.size)

    def finalize(self) -> dict[str, Any]:
        if self._pinball_count == 0:
            raise GridIntegrityError("Cannot finalize probabilistic metrics with zero points.")

        per_quantile = {
            f"{quantile:.2f}": float(self._pinball_sum[index] / self._pinball_count)
            for index, quantile in enumerate(self.quantiles)
        }
        mean_pinball = float(self._pinball_sum.sum() / (self._pinball_count * len(self.quantiles)))

        coverage = self._covered / self._interval_points
        nominal = self.upper_quantile - self.lower_quantile
        width = self._width_sum / self._interval_points
        crossing_rate = self._crossings / self._crossing_pairs if self._crossing_pairs else 0.0

        return {
            "mean_pinball_loss": mean_pinball,
            "pinball_by_quantile": per_quantile,
            "interval_coverage_90": float(coverage),
            "mean_interval_width_90": float(width),
            "absolute_coverage_error_90": float(abs(coverage - nominal)),
            "adjacent_quantile_crossing_rate": float(crossing_rate),
            "points": self._interval_points,
        }
