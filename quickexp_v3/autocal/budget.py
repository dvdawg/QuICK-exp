"""Predictive and hard accounting for automated calibration acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class BudgetExceeded(RuntimeError):
    """An acquisition would exceed the reviewed session budget."""


@dataclass
class BudgetModel:
    """Small conservative duration model, expressed in elapsed seconds."""

    fixed_overhead_seconds: float = 20.0
    point_overhead_seconds: float = 2.0e-5

    def estimate(self, plan: Any) -> float:
        variables = dict(getattr(plan, "variables", {}))
        axes = tuple(getattr(plan, "axes", ()))
        points = 1
        for name in axes:
            value = np.asarray(variables.get(name, [0.0]))
            points *= max(int(value.size), 1)
        hard_avg = max(int(variables.get("hard_avg", 1)), 1)
        soft_avg = max(int(variables.get("soft_avg", 1)), 1)
        rep = max(int(variables.get("rep", 1)), 1)
        r_relax = float(variables.get("r_relax", 0.0))
        r_length = float(variables.get("r_length", 0.0))
        drive_extent = 0.0
        for key in ("q_length", "time"):
            if key in variables:
                values = np.asarray(variables[key], dtype=float)
                if values.size:
                    drive_extent = max(drive_extent, float(np.max(values)))
        shot_scale = rep if not axes else 1
        acquisition_seconds = (
            points
            * hard_avg
            * soft_avg
            * shot_scale
            * max(r_relax + r_length + drive_extent, 1.0)
            * 1.0e-6
        )
        return max(
            0.0,
            float(self.fixed_overhead_seconds)
            + acquisition_seconds
            + float(self.point_overhead_seconds) * points,
        )

    def observe(self, predicted_seconds: float, measured_seconds: float) -> None:
        """Slowly adapt fixed overhead without allowing negative predictions."""
        predicted = max(float(predicted_seconds), 0.0)
        measured = max(float(measured_seconds), 0.0)
        residual = measured - predicted
        self.fixed_overhead_seconds = max(
            0.0,
            float(self.fixed_overhead_seconds) + 0.1 * residual,
        )


@dataclass
class BudgetTracker:
    max_wall_clock_seconds: float
    max_total_runs: int
    spent_seconds: float = 0.0
    total_runs: int = 0

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        max_wall_clock_hours: float,
        max_total_runs: int,
    ) -> "BudgetTracker":
        raw = state.get("budget", {})
        raw = raw if isinstance(raw, Mapping) else {}
        return cls(
            max_wall_clock_seconds=max(float(max_wall_clock_hours), 0.0) * 3600.0,
            max_total_runs=max(int(max_total_runs), 0),
            spent_seconds=max(float(raw.get("spent_seconds", 0.0)), 0.0),
            total_runs=max(int(raw.get("total_runs", 0)), 0),
        )

    def check(self, predicted_seconds: float) -> None:
        estimate = max(float(predicted_seconds), 0.0)
        if self.total_runs + 1 > self.max_total_runs:
            raise BudgetExceeded(
                f"run cap {self.max_total_runs} would be exceeded"
            )
        if self.spent_seconds + estimate > self.max_wall_clock_seconds:
            raise BudgetExceeded(
                "predicted acquisition would exceed the wall-clock budget "
                f"({self.spent_seconds + estimate:.1f}s > "
                f"{self.max_wall_clock_seconds:.1f}s)"
            )

    def record(self, measured_seconds: float) -> None:
        self.total_runs += 1
        self.spent_seconds += max(float(measured_seconds), 0.0)

    def as_dict(self) -> dict:
        return {
            "spent_seconds": float(self.spent_seconds),
            "total_runs": int(self.total_runs),
            "max_wall_clock_seconds": float(self.max_wall_clock_seconds),
            "max_total_runs": int(self.max_total_runs),
        }
