"""Deterministic search-window and averaging helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import ConfigError


def centered_sweep(
    center: float,
    span: float,
    points: int,
    *,
    bounds: Sequence[float],
) -> np.ndarray:
    """Create a centered finite sweep, shifted to remain inside bounds."""
    center_value = float(center)
    span_value = float(span)
    count = int(points)
    lower, upper = map(float, bounds)
    if not np.all(np.isfinite([center_value, span_value, lower, upper])):
        raise ConfigError("autocal search bounds must be finite")
    if lower >= upper or span_value <= 0 or count < 3:
        raise ConfigError("autocal sweep needs ordered bounds, positive span, and 3 points")
    width = min(span_value, upper - lower)
    start = np.clip(center_value - width / 2.0, lower, upper - width)
    return np.linspace(float(start), float(start + width), count)


def expected_center(
    hardware: Mapping[str, Any],
    key: str,
    fallback: float,
) -> float:
    """Use the midpoint of a hardware expected range, else a scalar fallback."""
    raw = hardware.get("expected", {}).get(key)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        values = np.asarray(raw, dtype=float)
        if np.all(np.isfinite(values)) and values[0] < values[1]:
            return float(np.mean(values))
    return float(fallback)


def averaging_ladder(initial: int, *, maximum_factor: int = 4) -> tuple:
    """Return the plan's start, ×2, ×4 SNR ladder."""
    first = max(int(initial), 1)
    factor = max(int(maximum_factor), 1)
    values = []
    multiplier = 1
    while multiplier <= factor:
        values.append(first * multiplier)
        multiplier *= 2
    return tuple(values)


def search_attempt(base_span: float, attempt: int, *, maximum_multiplier: float = 3.0) -> float:
    """Widen once, then leave the window fixed while averaging escalates."""
    if int(attempt) < 1:
        raise ConfigError("autocal attempt numbers start at one")
    multiplier = min(float(maximum_multiplier), 3.0 if int(attempt) > 1 else 1.0)
    return float(base_span) * multiplier
