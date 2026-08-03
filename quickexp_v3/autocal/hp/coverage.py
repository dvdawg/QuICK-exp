"""Assess whether an acquisition was capable of answering its question."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CoverageAssessment:
    sufficient: bool
    reasons: Tuple[str, ...]
    prior_coverage: float
    points_per_fwhm: float
    detectable_contrast: float
    edge_margin_fwhm: float


def assess_coverage(
    candidates: Sequence,
    prior_window: Tuple[float, float],
    scan_window: Tuple[float, float],
    points: int,
    expected_fwhm_mhz: float,
    expected_contrast: float,
    min_prior_coverage: float = 0.9,
    min_points_per_fwhm: float = 5.0,
    edge_margin_fwhm: float = 2.0,
) -> CoverageAssessment:
    """Return measurement sufficiency and every class-A failure reason."""
    prior_low, prior_high = sorted(float(value) for value in prior_window)
    scan_low, scan_high = sorted(float(value) for value in scan_window)
    epsilon = np.finfo(float).eps

    prior_span = max(prior_high - prior_low, epsilon)
    overlap = max(
        0.0,
        min(prior_high, scan_high) - max(prior_low, scan_low),
    )
    prior_coverage = float(overlap / prior_span)

    scan_span = max(scan_high - scan_low, epsilon)
    spacing = scan_span / max(int(points) - 1, 1)
    points_per_fwhm = float(
        abs(float(expected_fwhm_mhz)) / max(spacing, epsilon)
    )

    real = [item for item in candidates if not item.is_null]
    detectable_contrast = float("nan")
    for item in candidates:
        if item.is_null:
            value = item.statistics.get("detectable_contrast")
            if value is not None and np.isfinite(float(value)):
                detectable_contrast = float(value)
                break
    if not np.isfinite(detectable_contrast):
        for item in candidates:
            value = item.statistics.get("rmse")
            if value is not None and np.isfinite(float(value)):
                detectable_contrast = 3.0 * abs(float(value))
                break

    margins = []
    for item in real:
        width = max(abs(float(item.fwhm_mhz)), epsilon)
        distance = min(
            float(item.center_mhz) - scan_low,
            scan_high - float(item.center_mhz),
        )
        margins.append(float(distance / width))
    observed_edge_margin = min(margins) if margins else float("inf")

    reasons = []
    if prior_coverage < float(min_prior_coverage):
        reasons.append("prior_coverage")
    if points_per_fwhm < float(min_points_per_fwhm):
        reasons.append("resolution")
    if (
        np.isfinite(detectable_contrast)
        and abs(float(expected_contrast)) < detectable_contrast
    ):
        reasons.append("detectability")
    if observed_edge_margin < float(edge_margin_fwhm):
        reasons.append("edge_proximity")

    return CoverageAssessment(
        sufficient=not reasons,
        reasons=tuple(reasons),
        prior_coverage=prior_coverage,
        points_per_fwhm=points_per_fwhm,
        detectable_contrast=detectable_contrast,
        edge_margin_fwhm=observed_edge_margin,
    )
