"""Pure helpers for loopback bring-up calibration.

Two things dominated the 2026-06-29 lab session: finding the readout-window
delay (``r_offset``, which should start at 0 and be measured, not guessed) and
choosing a drive/receive level that sits above the noise floor but below ADC
over-range (the ~30 ``level-test-*`` loopback files). These pure functions
automate both; the hardware-driving orchestration lives in
``steps.Session.loopback_calibrate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

# A level whose peak never reaches this fraction of full scale is "weak":
# probably a path/connection problem rather than a usable signal.
WEAK_FRACTION = 0.02


def find_offset(time: np.ndarray, amplitude: np.ndarray) -> float:
    """Readout-window delay = onset of the returned pulse in the time trace.

    Returns the time at which the amplitude first crosses halfway between its
    baseline (trace minimum) and peak. A flat trace (no pulse) returns 0.0 so the
    caller keeps r_offset at its starting value.
    """
    time = np.asarray(time, float)
    amplitude = np.abs(np.asarray(amplitude, float))
    if time.size == 0:
        return 0.0
    baseline = float(np.min(amplitude))
    peak = float(np.max(amplitude))
    if peak - baseline <= 0:
        return 0.0
    half = baseline + 0.5 * (peak - baseline)
    above = np.flatnonzero(amplitude >= half)
    if above.size == 0:
        return 0.0
    return float(time[above[0]])


def select_power(
    powers: Sequence[float],
    peaks: Sequence[float],
    fullscale: float,
    headroom: float = 0.8,
) -> Tuple[float, str]:
    """Pick the highest power whose peak stays within ``headroom`` of full scale.

    Returns ``(best_power, status)`` where status is:
      * ``"ok"``             -- found a safe level above the weak floor;
      * ``"all_over_range"`` -- every level over-ranges; backs off to the lowest;
      * ``"weak"``           -- the best safe level is still barely above noise.
    """
    powers = list(powers)
    peaks = np.asarray(peaks, float)
    safe_ceiling = headroom * fullscale

    safe = peaks <= safe_ceiling
    if not np.any(safe):
        return float(min(powers)), "all_over_range"

    safe_idx = np.flatnonzero(safe)
    # Among safe levels, the highest power gives the strongest (best-SNR) signal.
    best_idx = safe_idx[int(np.argmax([powers[i] for i in safe_idx]))]
    status = "weak" if peaks[best_idx] < WEAK_FRACTION * fullscale else "ok"
    return float(powers[best_idx]), status


@dataclass
class LoopbackCalibration:
    """Result of a loopback bring-up: calibrated r_offset + a safe r_power."""

    r_offset: float
    r_power: float
    peak: float
    fullscale: float
    powers: List[float]
    peaks: List[float]
    valid: bool
    status: str = "ok"
    time: np.ndarray = field(default=None, repr=False)
    amplitude: np.ndarray = field(default=None, repr=False)

    def suggestion(self) -> str:
        headroom = 100.0 * self.peak / self.fullscale if self.fullscale else float("nan")
        return (
            f"loopback: r_offset={self.r_offset:.3f} us, r_power={self.r_power:g} dB "
            f"(peak {self.peak:.0f} = {headroom:.0f}% of full scale); "
            f"accept() to commit both, or override before the first scan"
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(10, 4))
        else:
            ax_p = None
        if self.time is not None and self.amplitude is not None:
            ax.plot(self.time, self.amplitude, lw=1.0)
            ax.axvline(self.r_offset, color="k", ls="--", lw=0.8, label="r_offset")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Amplitude (counts)")
            ax.legend()
        if ax_p is not None:
            ax_p.plot(self.powers, self.peaks, "o-")
            ax_p.axhline(self.fullscale, color="r", ls="--", lw=0.8, label="full scale")
            ax_p.axvline(self.r_power, color="k", ls="--", lw=0.8, label="chosen")
            ax_p.set_xlabel("r_power (dB)")
            ax_p.set_ylabel("peak (counts)")
            ax_p.legend()
        return ax
