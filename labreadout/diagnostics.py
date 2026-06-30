"""Advisory result diagnosis -- "does this measurement make sense?"

A pure layer that looks at a measurement's raw signal plus its fit and renders a
short health report: is the output noise-dominated, is the ADC railing, is the
fitted value physically plausible? A small decision tree turns the combination
of checks into a single most-likely cause and a concrete next action.

It is **advisory only** -- it never blocks a step (the separate ``accept()``
invalid-fit guard does that). Everything here is pure and unit-tested offline
against synthetic signals with known SNR/over-range and against real lab CSVs.

Known values are hybrid: structural checks (over-range, SNR, in-band) are
hardcoded physics; an optional ``expected_range`` supplies operator-known bounds
(resonator from VNA, qubit band) that tighten the sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Thresholds. Tuned to be forgiving: the report nudges, it does not gate.
SNR_MIN = 3.0          # below this the feature is in the noise
GOF_MIN = 0.5          # r-squared below this means no clear fit
OVER_RANGE_FRAC = 0.01  # >1% of samples at/over full scale = railing


@dataclass
class Check:
    name: str
    status: str   # 'pass' | 'warn' | 'fail'
    message: str


@dataclass
class Diagnosis:
    status: str                 # 'ok' | 'warn' | 'bad'
    snr: float
    checks: List[Check]
    likely_cause: str
    suggested_action: str
    recommendations: Dict[str, Any] = field(default_factory=dict)
    metric: str = ""   # headline metric override; defaults to "SNR=<snr>"

    def report(self) -> str:
        icon = {"ok": "OK", "warn": "WARN", "bad": "BAD"}[self.status]
        metric = self.metric or f"SNR={self.snr:.1f}"
        lines = [f"[{icon}] diagnosis ({metric}): {self.likely_cause}"]
        for c in self.checks:
            lines.append(f"    - {c.status:>4}: {c.name} -- {c.message}")
        lines.append(f"  -> {self.suggested_action}")
        if self.recommendations:
            recs = ", ".join(f"{k}={v}" for k, v in self.recommendations.items())
            lines.append(f"  recommended: {recs}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Signal primitives
# --------------------------------------------------------------------------- #
def estimate_noise(signal: np.ndarray) -> float:
    """Robust high-frequency noise estimate from successive differences.

    Uses the median absolute deviation of the point-to-point difference, which
    ignores the slow spectroscopic feature and the few large steps at its edges.
    """
    signal = np.asarray(signal, float)
    if signal.size < 3:
        return 0.0
    d = np.diff(signal)
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def _feature_amplitude(signal: np.ndarray) -> float:
    """Peak-to-peak of the de-noised signal -- the real feature size.

    Uses a median filter (edge-preserving) so a sharp narrow notch survives while
    single-sample noise spikes are removed; a box average would blur a one-point
    deep resonance and under-report its size.
    """
    from scipy.signal import medfilt

    signal = np.asarray(signal, float)
    n = signal.size
    if n < 5:
        return float(np.ptp(signal)) if n else 0.0
    w = max(3, n // 100)
    if w % 2 == 0:
        w += 1
    smooth = medfilt(signal, w)
    edge = w  # the filter is biased within a window of the ends
    core = smooth[edge : n - edge] if n > 2 * edge else smooth
    return float(np.ptp(core))


def estimate_snr(signal: np.ndarray) -> float:
    """Feature-amplitude / noise. High for a clean resonance, ~1 for pure noise."""
    noise = estimate_noise(signal)
    if noise == 0.0:
        return float("inf")
    return _feature_amplitude(signal) / noise


def over_range_fraction(values: np.ndarray, fullscale: float) -> float:
    """Fraction of samples within 1% of +/- ``fullscale`` (ADC railing).

    Real ADC clipping plateaus just under the nominal maximum, so anything at or
    beyond 99% of full scale counts as railed.
    """
    values = np.abs(np.asarray(values, float))
    if fullscale <= 0:
        return 0.0
    return float(np.mean(values >= 0.99 * fullscale))


# --------------------------------------------------------------------------- #
# Spectroscopy decision tree
# --------------------------------------------------------------------------- #
def diagnose_spectroscopy(
    freq: np.ndarray,
    signal: np.ndarray,
    fit: Any,
    expected_range: Optional[Tuple[float, float]] = None,
    fullscale: Optional[float] = None,
    recommendations: Optional[Dict[str, Any]] = None,
) -> Diagnosis:
    """Diagnose a resonator/qubit spectroscopy sweep from its signal + fit.

    ``signal`` is the magnitude (resonator) or rotated-IQ (qubit). ``fit`` is any
    object exposing ``f0`` and ``gof`` (and optionally ``valid``).
    """
    freq = np.asarray(freq, float)
    snr = estimate_snr(signal)
    gof = float(getattr(fit, "gof", float("nan")))
    valid = bool(getattr(fit, "valid", True))
    f0 = float(getattr(fit, "f0", float("nan")))

    checks: List[Check] = []

    railing = False
    if fullscale is not None:
        frac = over_range_fraction(signal, fullscale)
        railing = frac > OVER_RANGE_FRAC
        checks.append(Check(
            "over-range", "fail" if railing else "pass",
            f"{frac*100:.1f}% of samples at/over full scale ({fullscale:g})",
        ))

    noise_dominated = snr < SNR_MIN
    checks.append(Check(
        "snr", "fail" if noise_dominated else "pass",
        f"feature/noise = {snr:.1f} (min {SNR_MIN:g})",
    ))

    poor_fit = (not valid) or (np.isfinite(gof) and gof < GOF_MIN)
    checks.append(Check(
        "fit", "warn" if poor_fit else "pass",
        f"valid={valid}, R^2={gof:.3f}" if np.isfinite(gof) else f"valid={valid}",
    ))

    out_of_range = False
    if expected_range is not None and np.isfinite(f0):
        lo, hi = expected_range
        out_of_range = not (lo <= f0 <= hi)
        checks.append(Check(
            "known-value", "warn" if out_of_range else "pass",
            f"f0={f0:.3f} vs expected [{lo:g}, {hi:g}]",
        ))

    # Decision tree -- first matching cause wins (most fundamental first).
    if railing:
        status, cause = "bad", "ADC over-range / railing on part of the sweep"
        action = "Reduce readout power or add input attenuation, then rescan."
    elif noise_dominated:
        status, cause = "bad", "Output is noise-dominated (feature lost in noise)"
        action = ("Increase averaging, raise drive power, or check the port path / "
                  "RF connections (run check_ports).")
    elif poor_fit:
        status, cause = "warn", "No clear feature / poor fit in this window"
        action = "Narrow the scan to one isolated feature and rescan."
    elif out_of_range:
        status, cause = "warn", "Fitted value is outside the expected range (implausible)"
        action = ("Verify you are on the right resonance/qubit and the right port; "
                  "the fit may have locked onto a spurious feature.")
    else:
        status, cause = "ok", "Healthy: clear feature, good fit, plausible value"
        action = "Proceed (review the plot, then .accept())."

    return Diagnosis(
        status=status, snr=float(snr), checks=checks,
        likely_cause=cause, suggested_action=action,
        recommendations=dict(recommendations or {}),
    )


def diagnose_loopback(status: str, peak: float, fullscale: float) -> Diagnosis:
    """Turn a loopback level-selection ``status`` into a health verdict."""
    frac = (peak / fullscale) if fullscale else float("nan")
    metric = f"peak {frac*100:.0f}% FS" if np.isfinite(frac) else "peak n/a"
    check = Check("level", "pass" if status == "ok" else "fail",
                  f"peak {peak:.0f} = {frac*100:.0f}% of full scale ({status})")
    if status == "all_over_range":
        return Diagnosis("bad", float(frac), [check],
                         "Even the lowest swept power over-ranges the ADC",
                         "Reduce drive power / add input attenuation before calibrating.",
                         metric=metric)
    if status == "weak":
        return Diagnosis("warn", float(frac), [check],
                         "Loopback signal stays near the noise floor at every level",
                         "Check the loopback path / RF connections and run check_ports.",
                         metric=metric)
    return Diagnosis("ok", float(frac), [check],
                     "Loopback level is above noise and below over-range",
                     "Proceed; accept() to commit r_offset and r_power.",
                     metric=metric)
