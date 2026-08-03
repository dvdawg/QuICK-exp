"""Deterministic adaptive row scheduling for expensive two-dimensional maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


def spanning_rows(bounds: Sequence[float], count: int = 5) -> np.ndarray:
    lower, upper = sorted(float(value) for value in bounds)
    rows = int(count)
    if not np.all(np.isfinite([lower, upper])) or lower >= upper:
        raise ValueError("adaptive row bounds must be finite and increasing")
    if rows < 3:
        raise ValueError("adaptive acquisition requires at least three initial rows")
    return np.linspace(lower, upper, rows)


def tracked_frequency_axis(
    previous_center_mhz: float,
    span_mhz: float,
    points: int,
    bounds: Sequence[float],
) -> np.ndarray:
    """Build a fixed-width row window, shifting it inside hardware bounds."""
    lower, upper = sorted(float(value) for value in bounds)
    span = abs(float(span_mhz))
    count = int(points)
    if count < 2 or span <= 0.0 or span > upper - lower:
        raise ValueError("tracked frequency window is incompatible with its bounds")
    start = float(previous_center_mhz) - span / 2.0
    stop = float(previous_center_mhz) + span / 2.0
    if start < lower:
        stop += lower - start
        start = lower
    if stop > upper:
        start -= stop - upper
        stop = upper
    return np.linspace(start, stop, count)


@dataclass(frozen=True)
class AdaptiveRow:
    value: float
    center_mhz: Optional[float]
    trackable: bool
    uncertainty_mhz: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "value": float(self.value),
            "center_mhz": (
                None if self.center_mhz is None else float(self.center_mhz)
            ),
            "trackable": bool(self.trackable),
            "uncertainty_mhz": (
                None
                if self.uncertainty_mhz is None
                else float(self.uncertainty_mhz)
            ),
        }


class AdaptiveRowScheduler:
    def __init__(
        self,
        bounds: Sequence[float],
        initial_rows: int = 5,
        max_rows: int = 7,
        abort_after_rows: int = 5,
    ):
        lower, upper = sorted(float(value) for value in bounds)
        self.bounds = (lower, upper)
        self.initial_rows = int(initial_rows)
        self.max_rows = int(max_rows)
        self.abort_after_rows = int(abort_after_rows)
        if self.initial_rows < 3:
            raise ValueError("initial_rows must be at least three")
        if self.max_rows < self.initial_rows:
            raise ValueError("max_rows cannot be smaller than initial_rows")
        if not 3 <= self.abort_after_rows <= self.max_rows:
            raise ValueError("abort_after_rows must be between three and max_rows")
        spanning_rows(self.bounds, self.initial_rows)
        self.rows = []
        self.aborted = False
        self.abort_reason = ""

    @property
    def done(self) -> bool:
        return bool(self.aborted or len(self.rows) >= self.max_rows)

    def _already_measured(self, value: float) -> bool:
        return any(
            np.isclose(float(row.value), float(value), rtol=0.0, atol=1.0e-12)
            for row in self.rows
        )

    def _adaptive_candidate(self) -> float:
        measured = sorted(float(row.value) for row in self.rows)
        boundaries = sorted(set([self.bounds[0], self.bounds[1]] + measured))
        candidates = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right - left <= np.finfo(float).eps:
                continue
            midpoint = 0.5 * (left + right)
            if not self._already_measured(midpoint):
                candidates.append((midpoint, right - left))
        if not candidates:
            grid = np.linspace(self.bounds[0], self.bounds[1], 1001)
            available = [value for value in grid if not self._already_measured(float(value))]
            if not available:
                raise RuntimeError("adaptive scheduler has no unmeasured row")
            return float(available[0])

        trackable = [
            row
            for row in self.rows
            if row.trackable and row.center_mhz is not None
        ]
        scale = max(self.bounds[1] - self.bounds[0], np.finfo(float).eps)
        if len(trackable) >= 3:
            x = np.asarray(
                [2.0 * (row.value - self.bounds[0]) / scale - 1.0 for row in trackable]
            )
            design = np.column_stack((np.ones(x.size), x, x ** 2))
            covariance = np.linalg.pinv(design.T @ design)
        else:
            covariance = np.eye(3)

        scored = []
        for value, gap in candidates:
            normalized = 2.0 * (value - self.bounds[0]) / scale - 1.0
            vector = np.asarray([1.0, normalized, normalized ** 2])
            predictive = float(np.sqrt(max(vector @ covariance @ vector, 0.0)))
            scored.append((gap * (1.0 + predictive), -value, value))
        return float(max(scored)[2])

    def next_row(self) -> float:
        if self.done:
            raise RuntimeError("adaptive row schedule is complete")
        for value in spanning_rows(self.bounds, self.initial_rows):
            if not self._already_measured(float(value)):
                return float(value)
        return self._adaptive_candidate()

    def record(
        self,
        value: float,
        *,
        center_mhz: Optional[float],
        trackable: bool,
        uncertainty_mhz: Optional[float] = None,
    ) -> None:
        measured = float(value)
        if self.done:
            raise RuntimeError("cannot record a completed adaptive schedule")
        if self._already_measured(measured):
            raise ValueError("adaptive row was already recorded")
        if not self.bounds[0] <= measured <= self.bounds[1]:
            raise ValueError("adaptive row lies outside configured bounds")
        self.rows.append(
            AdaptiveRow(
                measured,
                None if center_mhz is None else float(center_mhz),
                bool(trackable),
                None if uncertainty_mhz is None else float(uncertainty_mhz),
            )
        )
        if len(self.rows) >= self.abort_after_rows:
            trackable_count = sum(row.trackable for row in self.rows)
            if trackable_count < 3:
                self.aborted = True
                self.abort_reason = (
                    "fewer than three trackable features after "
                    + str(len(self.rows))
                    + " rows"
                )

    def as_dict(self) -> dict:
        return {
            "bounds": list(self.bounds),
            "initial_rows": int(self.initial_rows),
            "max_rows": int(self.max_rows),
            "abort_after_rows": int(self.abort_after_rows),
            "rows": [row.as_dict() for row in self.rows],
            "aborted": bool(self.aborted),
            "abort_reason": str(self.abort_reason),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AdaptiveRowScheduler":
        scheduler = cls(
            raw.get("bounds", (-0.3, 0.3)),
            int(raw.get("initial_rows", 5)),
            int(raw.get("max_rows", 7)),
            int(raw.get("abort_after_rows", 5)),
        )
        scheduler.rows = [
            AdaptiveRow(
                float(row["value"]),
                None if row.get("center_mhz") is None else float(row["center_mhz"]),
                bool(row.get("trackable", False)),
                (
                    None
                    if row.get("uncertainty_mhz") is None
                    else float(row["uncertainty_mhz"])
                ),
            )
            for row in raw.get("rows", ())
            if isinstance(row, Mapping)
        ]
        scheduler.aborted = bool(raw.get("aborted", False))
        scheduler.abort_reason = str(raw.get("abort_reason", ""))
        return scheduler

