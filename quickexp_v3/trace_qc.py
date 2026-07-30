"""Per-trace quality diagnostics shared by offline fitters and autocal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import median_filter

from .fit_stats import oriented_rotate_iq


@dataclass(frozen=True)
class TraceQC:
    noise_mad: float
    spike_count: int
    baseline_drift: float
    axis_uniform: bool
    clipping_suspected: bool
    snr_estimate: float

    def as_dict(self) -> dict:
        return asdict(self)


def _component_clipped(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size < 20:
        return False
    span = float(np.ptp(finite))
    if span <= 0:
        return True
    tolerance = 0.005 * span
    near_minimum = np.count_nonzero(finite <= float(np.min(finite)) + tolerance)
    near_maximum = np.count_nonzero(finite >= float(np.max(finite)) - tolerance)
    return max(near_minimum, near_maximum) / finite.size > 0.01


def qc_trace(x, y_complex) -> TraceQC:
    axis = np.asarray(x, dtype=float).ravel()
    values = np.asarray(y_complex, dtype=complex).ravel()
    finite = (
        np.isfinite(axis)
        & np.isfinite(values.real)
        & np.isfinite(values.imag)
    )
    axis, values = axis[finite], values[finite]
    if axis.size < 2:
        return TraceQC(
            noise_mad=float("nan"),
            spike_count=0,
            baseline_drift=float("nan"),
            axis_uniform=False,
            clipping_suspected=False,
            snr_estimate=float("nan"),
        )
    order = np.argsort(axis)
    axis, values = axis[order], values[order]
    projected, _ = oriented_rotate_iq(values)
    projected = np.asarray(projected, dtype=float)
    differences = np.diff(projected)
    noise = float(
        np.median(np.abs(differences - np.median(differences)))
        / (0.6744897501960817 * np.sqrt(2.0))
    )
    local = median_filter(projected, size=7, mode="nearest")
    residual = projected - local
    robust = float(
        np.median(np.abs(residual - np.median(residual))) / 0.6744897501960817
    )
    spike_count = int(
        np.count_nonzero(np.abs(residual - np.median(residual)) > 6.0 * max(robust, np.finfo(float).eps))
    )
    decile = max(1, axis.size // 10)
    signal_span = float(
        np.percentile(projected, 98) - np.percentile(projected, 2)
    )
    baseline_drift = abs(
        float(np.median(projected[-decile:]))
        - float(np.median(projected[:decile]))
    ) / max(signal_span, np.finfo(float).eps)
    steps = np.diff(axis)
    typical_step = float(np.median(steps))
    uniform = bool(
        typical_step > 0
        and np.allclose(steps, typical_step, rtol=0.05, atol=0.0)
    )
    clipping = _component_clipped(values.real) or _component_clipped(values.imag)
    percentile_span = float(np.percentile(projected, 98) - np.percentile(projected, 2))
    snr = percentile_span / max(noise, np.finfo(float).eps)
    return TraceQC(
        noise_mad=noise,
        spike_count=spike_count,
        baseline_drift=float(baseline_drift),
        axis_uniform=uniform,
        clipping_suspected=bool(clipping),
        snr_estimate=float(snr),
    )


def qc_map(native_map) -> Mapping[float, TraceQC]:
    complex_values = native_map.complex_signal
    return {
        float(outer): qc_trace(native_map.inner, complex_values[index])
        for index, outer in enumerate(native_map.outer)
    }
