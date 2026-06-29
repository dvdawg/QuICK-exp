"""Adaptive coarse-to-fine sweep-window logic.

Pure functions with no hardware or I/O: given a fitted feature (center +
characteristic width) they propose the next scan window, the step needed for a
target point count, and a convergence test. The ``steps`` layer uses these to
suggest the operator's next scan and to drive opt-in auto-narrowing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def next_window(
    center: float,
    width: float,
    span_factor: float = 8.0,
    min_span: float = 0.0,
    bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Window centered on ``center`` spanning ``span_factor * width``.

    The span is clamped to at least ``min_span`` and, if ``bounds`` is given,
    the window is clipped to lie within them.
    """
    span = max(span_factor * abs(width), min_span)
    lo = center - span / 2.0
    hi = center + span / 2.0
    if bounds is not None:
        blo, bhi = bounds
        lo = max(lo, blo)
        hi = min(hi, bhi)
    return float(lo), float(hi)


def step_for_points(span: float, points: int) -> float:
    """Step size that covers ``span`` with ``points`` samples (inclusive)."""
    if points <= 1:
        return float(span)
    return float(span) / (points - 1)


def scan_array(
    center: float,
    width: float,
    points: int,
    span_factor: float = 8.0,
    min_span: float = 0.0,
    bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Inclusive linspace over the next window -- ready to hand to an experiment."""
    lo, hi = next_window(center, width, span_factor, min_span, bounds)
    return np.linspace(lo, hi, points)


def converged(width: float, step: float, factor: float = 1.0) -> bool:
    """True when the feature is too narrow to resolve further at this step."""
    return abs(width) <= factor * abs(step)
