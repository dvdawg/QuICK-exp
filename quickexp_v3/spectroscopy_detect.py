"""Locate spectroscopy features before fitting them.

Fitting a Lorentzian straight to the largest raw deviation reliably latches
onto a single noisy sample or onto the curvature of the background. This
module performs the detection step explicitly, and it treats the two
one-dimensional spectroscopies as the different measurements they are.

In **resonator** spectroscopy the readout tone itself sweeps, so the trace
carries a large electrical delay -- tens of full turns in the I/Q plane across
a wide sweep -- and the resonance shows up in the transmitted magnitude, in the
residual group delay, and as a circular arc once that delay is removed.

In **qubit** spectroscopy the readout tone never moves; the drive does. There
is no delay along the swept axis, and the measured point is a fixed
combination of the two readout states, so it slides along a *line* rather than
around a circle. The component across that line carries no signal at all,
which makes it a control channel: a real line shows up along the displacement
direction and not across it, whereas noise shows up in both.

The detection itself is scale-free. Each channel is compared against a rolling
baseline at several widths, because a baseline narrow enough to follow a curved
background will swallow a narrow resonance and one wide enough to preserve a
narrow resonance will not follow the background. Candidates found at the same
frequency in different channels or at different scales are merged and ranked by
pooled evidence.

The result seeds the fitters in :mod:`quickexp_v3.native_fit` and
:mod:`quickexp_v3.notch_fit`, and reports when a trace holds no credible
feature rather than returning a confident but meaningless centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .errors import AnalysisError, ConfigError


CHANNEL_LABELS = {
    "amplitude": "|I + iQ|",
    "group_delay": "Residual group delay (rad/MHz)",
    "projection": "I/Q displacement (a.u.)",
    "quadrature": "I/Q across displacement (control)",
}

#: Per-kind measurement model. ``control`` names a channel that carries no
#: signal by construction and therefore votes against a candidate instead of
#: for it.
DETECTION_KINDS = {
    "resonator": {
        "remove_delay": True,
        "channels": ("amplitude", "group_delay", "projection"),
        "primary": "amplitude",
        "control": None,
        "geometry": "arc",
    },
    "qubit": {
        "remove_delay": False,
        "channels": ("amplitude", "projection", "quadrature"),
        "primary": "projection",
        "control": "quadrature",
        "geometry": "displacement",
    },
}

#: Minimum resultant length of the per-step phase increments before a measured
#: slope is treated as a real cable delay.
MINIMUM_DELAY_COHERENCE = 0.30

#: Candidates narrower than this are single-sample spikes, not resonances.
MINIMUM_WIDTH_SAMPLES = 3.0

#: Candidates wider than this fraction of the sweep are background curvature.
MAXIMUM_WIDTH_FRACTION = 0.25

#: I/Q geometry agreement required before a single channel's marginal
#: detection is accepted. Traces holding a real line score 0.72 to 0.99 on the
#: sample data; noise-only traces score 0.21 to 0.27.
MINIMUM_GEOMETRY_SCORE = 0.50

#: Rolling-baseline widths, as divisors of the point count. A resonance that
#: survives any one of these is kept; see the module docstring. The range has
#: to span narrow lines a few samples across and broad ones covering a tenth
#: of the sweep, because a baseline comparable to the feature absorbs it.
BASELINE_DIVISORS = (25.0, 12.0, 6.0, 3.0)


def robust_sigma(values: Any) -> float:
    """Median-absolute-deviation estimate of the standard deviation."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(array - np.median(array))))


def normalize_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized not in DETECTION_KINDS:
        raise ConfigError("detection kind must be resonator or qubit")
    return normalized


@dataclass(frozen=True)
class DelayEstimate:
    """Electrical delay measured from the per-sample phase advance."""

    slope_rad_per_mhz: float
    coherence: float
    applied: bool
    turns: float
    aliased: bool

    @property
    def delay_ns(self) -> float:
        if not np.isfinite(self.slope_rad_per_mhz):
            return float("nan")
        return float(-self.slope_rad_per_mhz / (2.0 * np.pi) * 1e3)

    def as_dict(self) -> dict:
        return {
            "slope_rad_per_mhz": self.slope_rad_per_mhz,
            "delay_ns": self.delay_ns,
            "coherence": self.coherence,
            "applied": self.applied,
            "turns": self.turns,
            "aliased": self.aliased,
        }


def estimate_electrical_delay(x: Any, iq: Any) -> DelayEstimate:
    """Measure the cable delay without unwrapping the phase.

    Each step contributes ``iq[k + 1] * conj(iq[k])``, whose argument is the
    phase advance over that step. Averaging those increments as vectors,
    weighted by magnitude, wraps correctly by construction and lets bright
    samples dominate. The length of the resultant is the coherence: near one
    for a genuine delay, near zero when the phase is noise. Unwrapping instead
    manufactures a large slope out of a noise-dominated trace.
    """
    frequency = np.asarray(x, dtype=float).ravel()
    values = np.asarray(iq, dtype=complex).ravel()
    if frequency.size != values.size:
        raise AnalysisError("frequency and I/Q arrays must have equal lengths")
    if frequency.size < 3:
        return DelayEstimate(0.0, 0.0, False, 0.0, False)
    step = np.diff(frequency)
    increment = values[1:] * np.conj(values[:-1])
    usable = (step > 0) & (np.abs(increment) > 0) & np.isfinite(increment)
    if np.count_nonzero(usable) < 2:
        return DelayEstimate(0.0, 0.0, False, 0.0, False)
    increment = increment[usable]
    magnitude = np.abs(increment)
    resultant = np.sum(increment) / np.sum(magnitude)
    coherence = float(np.abs(resultant))
    # The resonance contributes its own rapid phase swing over the steps that
    # cross it, which drags the vector mean and leaves a residual rotation
    # across the whole de-delayed trace. The vector mean is wrap-safe but not
    # robust; stepping it onto the circular median of the per-step advances
    # is both, because the feature is a minority of the steps.
    if increment.size >= 8 and coherence > 0:
        for _ in range(3):
            unit = resultant / np.abs(resultant)
            offset = float(np.median(np.angle(increment * np.conj(unit))))
            if abs(offset) < 1e-12:
                break
            resultant = resultant * np.exp(1j * offset)
    advance = float(np.angle(resultant))
    median_step = float(np.median(step[usable]))
    slope = advance / median_step if median_step > 0 else 0.0
    span = float(np.ptp(frequency))
    return DelayEstimate(
        slope_rad_per_mhz=float(slope),
        coherence=coherence,
        applied=bool(coherence >= MINIMUM_DELAY_COHERENCE),
        turns=float(abs(slope) * span / (2.0 * np.pi)),
        # An advance approaching pi per sample is indistinguishable from its
        # alias, so the recovered delay cannot be trusted.
        aliased=bool(abs(advance) > 0.8 * np.pi),
    )


def remove_electrical_delay(
    x: Any,
    iq: Any,
    delay: Optional[DelayEstimate] = None,
) -> tuple[np.ndarray, DelayEstimate]:
    """Undo the cable delay and park the off-resonant point on the real axis."""
    frequency = np.asarray(x, dtype=float).ravel()
    values = np.asarray(iq, dtype=complex).ravel()
    estimate = (
        estimate_electrical_delay(frequency, values) if delay is None else delay
    )
    slope = estimate.slope_rad_per_mhz if estimate.applied else 0.0
    reference = float(np.mean(frequency)) if frequency.size else 0.0
    corrected = values * np.exp(-1j * slope * (frequency - reference))
    anchor = complex(np.median(corrected.real), np.median(corrected.imag))
    if abs(anchor) > 0:
        corrected = corrected * np.exp(-1j * np.angle(anchor))
    return corrected, estimate


def residual_group_delay(x: Any, corrected: Any) -> np.ndarray:
    """Group delay of a de-delayed trace: a peak at the resonance centre.

    A resonance's phase is dispersive, so the phase deviation peaks on either
    side of the centre rather than on it -- seeding from it pulls the centre
    off by roughly a linewidth. Its derivative does peak on the centre.
    """
    frequency = np.asarray(x, dtype=float).ravel()
    values = np.asarray(corrected, dtype=complex).ravel()
    if frequency.size < 2:
        return np.zeros(frequency.size, dtype=float)
    step = np.diff(frequency)
    increment = values[1:] * np.conj(values[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        advance = np.angle(increment) / np.where(step > 0, step, np.nan)
    advance = np.concatenate((advance[:1], advance))
    return np.nan_to_num(advance, nan=0.0, posinf=0.0, neginf=0.0)


def displacement_axes(values: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Split an I/Q cloud into its dominant direction and the one across it.

    Returns ``(along, across, anisotropy)``. For a qubit sweep the readout
    point is a fixed combination of the two readout states, so it moves along
    one direction and the across component is signal-free.
    """
    complex_values = np.asarray(values, dtype=complex).ravel()
    matrix = np.column_stack((complex_values.real, complex_values.imag))
    if matrix.shape[0] < 2:
        zeros = np.zeros(matrix.shape[0], dtype=float)
        return zeros, zeros.copy(), 1.0
    centered = matrix - np.mean(matrix, axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    along = np.asarray(centered @ right[0], dtype=float)
    across = np.asarray(centered @ right[1], dtype=float)
    anisotropy = float(
        singular[0] / singular[1]
        if singular.size > 1 and singular[1] > 0
        else np.inf
    )
    return along, across, anisotropy


def resonance_projection(
    x: Any,
    corrected: Any,
    center_mhz: float,
) -> tuple[np.ndarray, float]:
    """Project an I/Q trace onto the direction the resonance displaces it.

    The maximum-variance axis is the wrong one for a notch: the resonance
    sweeps a circle, so that axis lands somewhere between the absorptive and
    dispersive quadratures and the projection comes out as an asymmetric Fano
    shape that no symmetric Lorentzian fits -- the width runs to its bound
    instead. The line joining the off-resonant point to the point at
    resonance is the absorptive quadrature, and along it the response is
    Lorentzian.

    Returns the projection and the angle of the direction chosen.
    """
    frequency = np.asarray(x, dtype=float).ravel()
    values = np.asarray(corrected, dtype=complex).ravel()
    if frequency.size == 0:
        return np.zeros(0, dtype=float), 0.0
    off_resonance = complex(np.median(values.real), np.median(values.imag))
    index = int(np.argmin(np.abs(frequency - float(center_mhz))))
    direction = values[index] - off_resonance
    if not np.isfinite(direction) or abs(direction) <= 0:
        along, _across, _ = displacement_axes(values)
        return np.asarray(along, dtype=float), 0.0
    unit = direction / abs(direction)
    projected = np.real((values - off_resonance) * np.conj(unit))
    return np.asarray(projected, dtype=float), float(np.angle(unit))


def prominence_cut(n_points: int, fine_sigma: float, *, margin: float = 1.5) -> float:
    """Prominence a lone channel must reach, in units of deviation noise.

    Prominence spans a trough-to-peak excursion, so on pure noise its maximum
    over ``m`` independent resolution elements approaches ``2 sqrt(2 ln m)``,
    twice the usual single-sided extreme value. Simulated white noise gives a
    maximum prominence-to-noise ratio of 6.3 at 100 points rising to 8.2 at
    10000, which this reproduces to a few percent; the margin holds the cut
    above that observed maximum.
    """
    effective = max(float(n_points) / max(2.0 * float(fine_sigma), 1.0), 2.0)
    return float(2.0 * np.sqrt(2.0 * np.log(effective)) + margin)


def deviation_cut(
    n_points: int,
    fine_sigma: float,
    *,
    margin: float = 1.0,
) -> float:
    """Prominence each channel must reach when two or more channels agree.

    Independent channels rarely peak at the same frequency by chance, so
    demanding agreement buys back roughly a squared false-alarm rate and
    justifies the single-sided extreme value rather than the trough-to-peak
    one.
    """
    effective = max(float(n_points) / max(2.0 * float(fine_sigma), 1.0), 2.0)
    return float(np.sqrt(2.0 * np.log(effective)) + margin)


@dataclass(frozen=True)
class ChannelScale:
    """One channel viewed against a rolling baseline of one width."""

    name: str
    wide_sigma: float
    smoothed: np.ndarray
    baseline: np.ndarray
    deviation: np.ndarray
    noise: float

    def snr(self) -> np.ndarray:
        return self.deviation / max(self.noise, np.finfo(float).eps)


@dataclass(frozen=True)
class ChannelDecomposition:
    """One measurement channel, decomposed at every baseline width."""

    name: str
    values: np.ndarray
    scales: tuple
    sample_noise: float
    fine_sigma: float
    prominence_cut: float
    deviation_cut: float
    is_control: bool = False

    @property
    def label(self) -> str:
        return CHANNEL_LABELS.get(self.name, self.name)

    @property
    def primary(self) -> ChannelScale:
        """The middle baseline width, used for display."""
        return self.scales[min(1, len(self.scales) - 1)]

    def best_snr_near(self, x: np.ndarray, center_mhz: float, samples: int = 2) -> float:
        """Largest deviation-to-noise ratio near a frequency, over all scales."""
        index = int(np.argmin(np.abs(x - float(center_mhz))))
        window = slice(max(0, index - samples), min(x.size, index + samples + 1))
        return max(
            float(np.max(np.abs(scale.deviation[window])) / scale.noise)
            for scale in self.scales
        )

    def as_dict(self) -> dict:
        return {
            "sample_noise": self.sample_noise,
            "fine_sigma": self.fine_sigma,
            "prominence_cut": self.prominence_cut,
            "deviation_cut": self.deviation_cut,
            "is_control": self.is_control,
            "noise_by_scale": {
                f"{scale.wide_sigma:.1f}": scale.noise for scale in self.scales
            },
        }


def decompose_channel(
    x: Any,
    values: Any,
    name: str,
    *,
    fine_sigma: Optional[float] = None,
    baseline_divisors: Sequence[float] = BASELINE_DIVISORS,
    is_control: bool = False,
) -> ChannelDecomposition:
    """Smooth a channel and subtract a rolling baseline at several widths.

    The baseline is a wide Gaussian rather than a straight line because real
    transmission backgrounds curve and ripple; a line leaves that structure in
    the residual, where a Lorentzian promptly absorbs it and reports a
    linewidth as wide as the sweep. Several widths are used because one width
    cannot both follow a curved background and preserve a narrow resonance.
    """
    frequency = np.asarray(x, dtype=float).ravel()
    signal = np.asarray(values, dtype=float).ravel()
    if frequency.size != signal.size:
        raise AnalysisError("channel and frequency arrays must have equal lengths")
    count = signal.size
    fine = (
        float(np.clip(count / 400.0, 1.0, 3.0))
        if fine_sigma is None
        else float(fine_sigma)
    )
    if fine <= 0:
        raise ConfigError("smoothing width must be positive")
    smoothed = gaussian_filter1d(signal, fine, mode="nearest")

    scales = []
    seen = set()
    for divisor in baseline_divisors:
        if divisor <= 0:
            raise ConfigError("baseline divisors must be positive")
        wide = float(
            np.clip(count / divisor, 4.0 * fine, max(count / 2.5, 4.0 * fine))
        )
        key = round(wide, 3)
        if key in seen:
            continue
        seen.add(key)
        baseline = gaussian_filter1d(smoothed, wide, mode="nearest")
        deviation = smoothed - baseline
        # Measured from the deviation itself. Propagating per-sample noise
        # through an analytic smoothing gain understated it twofold, doubling
        # every reported signal-to-noise ratio; the median absolute deviation
        # tracks the true scatter to within two percent for white noise from
        # 100 to 10000 points, because features occupy a small minority of
        # the samples.
        scales.append(
            ChannelScale(
                name=name,
                wide_sigma=wide,
                smoothed=smoothed,
                baseline=baseline,
                deviation=deviation,
                noise=max(robust_sigma(deviation), np.finfo(float).eps),
            )
        )
    if not scales:
        raise ConfigError("at least one baseline width is required")

    return ChannelDecomposition(
        name=name,
        values=signal,
        scales=tuple(scales),
        sample_noise=(
            float(robust_sigma(np.diff(signal)) / np.sqrt(2.0))
            if signal.size > 1
            else float("nan")
        ),
        fine_sigma=fine,
        prominence_cut=prominence_cut(count, fine),
        deviation_cut=deviation_cut(count, fine),
        is_control=bool(is_control),
    )


def _fit_circle(points: np.ndarray) -> Optional[tuple]:
    """Algebraic circle fit; returns centre, radius, relative RMS residual."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] < 4:
        return None
    design = np.column_stack(
        (2.0 * array[:, 0], 2.0 * array[:, 1], np.ones(array.shape[0]))
    )
    target = array[:, 0] ** 2 + array[:, 1] ** 2
    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center_x, center_y, offset = solution
    squared = offset + center_x**2 + center_y**2
    if not np.isfinite(squared) or squared <= 0:
        return None
    radius = float(np.sqrt(squared))
    radii = np.hypot(array[:, 0] - center_x, array[:, 1] - center_y)
    relative = float(
        np.sqrt(np.mean((radii - radius) ** 2)) / max(radius, np.finfo(float).tiny)
    )
    return float(center_x), float(center_y), radius, relative


def _window(x: np.ndarray, center_mhz: float, hwhm_mhz: float, step_mhz: float):
    reach = max(3.0 * abs(float(hwhm_mhz)), 3.0 * abs(float(step_mhz)))
    return np.abs(x - float(center_mhz)) <= reach


def arc_geometry(
    x: Any,
    values: Any,
    center_mhz: float,
    hwhm_mhz: float,
    step_mhz: float,
) -> dict:
    """How cleanly the de-delayed I/Q path sweeps a circular arc.

    Crossing a resonance drags the point smoothly around a circle, so the
    angular increments about the circle centre share a sign. Noise turns the
    same measurement into a random walk whose increments cancel, making the
    ratio of net to total rotation a discriminator that neither magnitude nor
    phase provides alone.
    """
    frequency = np.asarray(x, dtype=float).ravel()
    complex_values = np.asarray(values, dtype=complex).ravel()
    empty = {"geometry_score": 0.0, "arc_rad": 0.0, "circle_residual": float("nan")}
    inside = _window(frequency, center_mhz, hwhm_mhz, step_mhz)
    if np.count_nonzero(inside) < 6:
        return empty
    selected = complex_values[inside]
    circle = _fit_circle(np.column_stack((selected.real, selected.imag)))
    if circle is None:
        return empty
    center_x, center_y, _radius, relative = circle
    offsets = (selected.real - center_x) + 1j * (selected.imag - center_y)
    offsets = offsets[np.abs(offsets) > 0]
    if offsets.size < 3:
        return empty
    increments = np.angle(offsets[1:] * np.conj(offsets[:-1]))
    total = float(np.sum(np.abs(increments)))
    if total <= 0:
        return empty
    net = float(abs(np.sum(increments)))
    return {
        "geometry_score": float(net / total),
        "arc_rad": net,
        "circle_residual": relative,
    }


def displacement_geometry(
    x: Any,
    complex_deviation: Any,
    center_mhz: float,
    hwhm_mhz: float,
    step_mhz: float,
) -> dict:
    """How consistently the I/Q deviation points one way across a feature.

    Driving a qubit moves the readout point from its ground-state position
    toward its excited-state one in proportion to the excited population, so
    every deviation near a real line is a *positive* multiple of the same
    complex vector and their directions are tightly concentrated. Noise
    deviations point every which way and cancel.

    Measured on the sample data, this separates traces containing a line
    (0.72 to 0.99) from noise-only traces (0.21 to 0.27), which magnitude
    alone does not: the two look alike in ``|I + iQ|`` until the directions
    are compared.
    """
    frequency = np.asarray(x, dtype=float).ravel()
    deviation = np.asarray(complex_deviation, dtype=complex).ravel()
    empty = {"geometry_score": 0.0, "directional_coherence": 0.0}
    inside = _window(frequency, center_mhz, hwhm_mhz, step_mhz)
    if np.count_nonzero(inside) < 6:
        return empty
    selected = deviation[inside]
    total = float(np.sum(np.abs(selected)))
    if total <= 0:
        return empty
    coherence = float(abs(np.sum(selected)) / total)
    return {"geometry_score": coherence, "directional_coherence": coherence}


def complex_deviation(
    values: Any,
    fine_sigma: float,
    wide_sigma: float,
) -> np.ndarray:
    """Rolling-baseline residual of an I/Q trace, kept complex."""
    complex_values = np.asarray(values, dtype=complex).ravel()

    def residual(component: np.ndarray) -> np.ndarray:
        smoothed = gaussian_filter1d(component, fine_sigma, mode="nearest")
        return smoothed - gaussian_filter1d(smoothed, wide_sigma, mode="nearest")

    return residual(complex_values.real) + 1j * residual(complex_values.imag)


@dataclass(frozen=True)
class FeatureCandidate:
    """One candidate resonance and the evidence supporting it."""

    center_mhz: float
    hwhm_mhz: float
    polarity: str
    prominence_snr: float
    width_samples: float
    detected_in: tuple
    channel_snr: Mapping[str, float]
    corroborating_channels: int
    control_snr: float
    geometry_score: float
    geometry: Mapping[str, float]
    evidence: float
    strong: bool
    basis: str

    @property
    def fwhm_mhz(self) -> float:
        return 2.0 * self.hwhm_mhz

    def as_dict(self) -> dict:
        return {
            "center_mhz": self.center_mhz,
            "hwhm_mhz": self.hwhm_mhz,
            "fwhm_mhz": self.fwhm_mhz,
            "polarity": self.polarity,
            "prominence_snr": self.prominence_snr,
            "width_samples": self.width_samples,
            "detected_in": list(self.detected_in),
            "channel_snr": dict(self.channel_snr),
            "corroborating_channels": self.corroborating_channels,
            "control_snr": self.control_snr,
            "geometry_score": self.geometry_score,
            "geometry": dict(self.geometry),
            "evidence": self.evidence,
            "strong": self.strong,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class SpectroscopyDetection:
    """Everything found in one trace, ranked by evidence."""

    kind: str
    x: np.ndarray
    iq: np.ndarray
    corrected: np.ndarray
    delay: DelayEstimate
    channels: Mapping[str, ChannelDecomposition]
    candidates: tuple
    marginal: tuple
    step_mhz: float
    span_mhz: float
    anisotropy: float

    @property
    def found(self) -> bool:
        return bool(self.candidates)

    @property
    def best(self) -> Optional[FeatureCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def best_marginal(self) -> Optional[FeatureCandidate]:
        """Strongest sub-threshold candidate, for diagnostics and plots."""
        return self.marginal[0] if self.marginal else None

    def require_best(self) -> FeatureCandidate:
        if self.candidates:
            return self.candidates[0]
        hint = (
            "Narrow the sweep around the expected frequency, average longer, "
            "or pass an explicit fit window to fit this trace anyway."
        )
        near = self.best_marginal
        if near is None:
            raise AnalysisError(
                "no spectroscopy feature was detected: no channel showed a "
                f"peak wider than {MINIMUM_WIDTH_SAMPLES:.0f} samples above "
                f"its noise. {hint}"
            )
        raise AnalysisError(
            "no credible spectroscopy feature was detected. The strongest "
            f"candidate sits at {near.center_mhz:.4f} MHz with a "
            f"prominence of {near.prominence_snr:.1f} sigma against a "
            f"{max(channel.prominence_cut for channel in self.channels.values()):.1f} "
            "sigma bar, and no second channel corroborated it "
            f"(per channel: {', '.join(f'{name} {value:.1f}' for name, value in near.channel_snr.items())}). "
            f"{hint}"
        )

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "delay": self.delay.as_dict(),
            "channels": {
                name: channel.as_dict() for name, channel in self.channels.items()
            },
            "candidate_count": len(self.candidates),
            "candidates": [candidate.as_dict() for candidate in self.candidates[:5]],
            "marginal_count": len(self.marginal),
            "marginal": [candidate.as_dict() for candidate in self.marginal[:3]],
            "step_mhz": self.step_mhz,
            "span_mhz": self.span_mhz,
            "anisotropy": self.anisotropy,
        }


def _channel_peaks(
    x: np.ndarray,
    channel: ChannelDecomposition,
    *,
    minimum_width_samples: float,
    maximum_width_fraction: float,
    limit: int,
) -> list:
    step = float(np.median(np.diff(x)))
    span = float(np.ptp(x))
    found = []
    for scale in channel.scales:
        for sign in (1.0, -1.0):
            peaks, properties = find_peaks(
                sign * scale.deviation,
                prominence=channel.deviation_cut * scale.noise,
                width=minimum_width_samples,
            )
            for position, index in enumerate(peaks):
                width_samples = float(properties["widths"][position])
                width_mhz = width_samples * step
                if span > 0 and width_mhz > maximum_width_fraction * span:
                    continue
                found.append(
                    {
                        "index": int(index),
                        "center_mhz": float(x[index]),
                        "prominence_snr": float(
                            properties["prominences"][position] / scale.noise
                        ),
                        "width_samples": width_samples,
                        "width_mhz": width_mhz,
                        "polarity": "peak" if sign > 0 else "dip",
                        "channel": channel.name,
                        "wide_sigma": scale.wide_sigma,
                    }
                )
    found.sort(key=lambda item: item["prominence_snr"], reverse=True)
    return found[:limit]


def _drop_baseline_lobes(
    candidates: list,
    channels: Mapping[str, ChannelDecomposition],
    step_mhz: float,
) -> list:
    """Remove the side lobes a rolling baseline leaves beside a strong line.

    Subtracting a wide Gaussian from a tall narrow feature lifts the baseline
    on either side of it, so the deviation dips below zero a baseline width
    out on each flank. Those dips are shaped like real features and are picked
    up as extra candidates, which inflates the count and can trip a
    multiple-feature warning on what is one clean line.
    """
    if len(candidates) < 2:
        return candidates
    reach = max(
        scale.wide_sigma
        for channel in channels.values()
        for scale in channel.scales
    ) * abs(step_mhz)
    kept: list = []
    for candidate in candidates:
        lobe = any(
            candidate.polarity != stronger.polarity
            and candidate.prominence_snr < 0.5 * stronger.prominence_snr
            and abs(candidate.center_mhz - stronger.center_mhz) <= reach
            for stronger in kept
        )
        if not lobe:
            kept.append(candidate)
    return kept


def detect_features(
    x: Any,
    iq: Any,
    *,
    kind: str = "resonator",
    minimum_width_samples: float = MINIMUM_WIDTH_SAMPLES,
    maximum_width_fraction: float = MAXIMUM_WIDTH_FRACTION,
    maximum_candidates: int = 8,
    window_mhz: Optional[Sequence[float]] = None,
) -> SpectroscopyDetection:
    """Find and rank the resonance candidates in one spectroscopy trace."""
    normalized_kind = normalize_kind(kind)
    definition = DETECTION_KINDS[normalized_kind]
    frequency = np.asarray(x, dtype=float).ravel()
    values = np.asarray(iq, dtype=complex).ravel()
    if frequency.size != values.size:
        raise AnalysisError("frequency and I/Q arrays must have equal lengths")
    if window_mhz is not None:
        if len(window_mhz) != 2:
            raise ConfigError("a detection window must have two values")
        lower, upper = sorted(float(value) for value in window_mhz)
        inside = (frequency >= lower) & (frequency <= upper)
        frequency, values = frequency[inside], values[inside]
    if frequency.size < 12:
        raise AnalysisError("feature detection requires at least 12 points")
    order = np.argsort(frequency)
    frequency, values = frequency[order], values[order]
    step = float(np.median(np.diff(frequency)))
    span = float(np.ptp(frequency))
    if step <= 0 or span <= 0:
        raise AnalysisError("the frequency axis must increase")

    if definition["remove_delay"]:
        corrected, delay = remove_electrical_delay(frequency, values)
    else:
        corrected, delay = values.copy(), estimate_electrical_delay(frequency, values)
        delay = DelayEstimate(
            slope_rad_per_mhz=delay.slope_rad_per_mhz,
            coherence=delay.coherence,
            applied=False,
            turns=delay.turns,
            aliased=delay.aliased,
        )

    along, across, anisotropy = displacement_axes(corrected)
    available = {
        "amplitude": np.abs(values),
        "group_delay": residual_group_delay(frequency, corrected),
        "projection": along,
        "quadrature": across,
    }
    control_name = definition["control"]
    channels = {
        name: decompose_channel(
            frequency,
            available[name],
            name,
            is_control=(name == control_name),
        )
        for name in definition["channels"]
    }
    reference_channel = channels[definition["primary"]]
    deviation_complex = complex_deviation(
        corrected,
        reference_channel.fine_sigma,
        reference_channel.primary.wide_sigma,
    )

    peaks = []
    for channel in channels.values():
        if channel.is_control:
            continue
        peaks.extend(
            _channel_peaks(
                frequency,
                channel,
                minimum_width_samples=minimum_width_samples,
                maximum_width_fraction=maximum_width_fraction,
                limit=maximum_candidates,
            )
        )

    clusters: list = []
    for peak in sorted(peaks, key=lambda item: item["prominence_snr"], reverse=True):
        for cluster in clusters:
            tolerance = max(
                2.0 * step,
                0.5 * max(cluster["width_mhz"], peak["width_mhz"]),
            )
            if abs(cluster["center_mhz"] - peak["center_mhz"]) <= tolerance:
                cluster["members"].append(peak)
                break
        else:
            clusters.append(
                {
                    "center_mhz": peak["center_mhz"],
                    "width_mhz": peak["width_mhz"],
                    "members": [peak],
                }
            )

    candidates = []
    for cluster in clusters:
        leader = max(cluster["members"], key=lambda item: item["prominence_snr"])
        hwhm = max(leader["width_mhz"] / 2.0, step)
        channel_snr = {
            name: channel.best_snr_near(frequency, cluster["center_mhz"])
            for name, channel in channels.items()
        }
        signal_channels = [
            name for name, channel in channels.items() if not channel.is_control
        ]
        corroborating = sum(
            1
            for name in signal_channels
            if channel_snr[name] >= channels[name].deviation_cut
        )
        # `strong` is judged on the leader's *prominence*, which is what the
        # prominence cut was calibrated against; ``channel_snr`` holds
        # single-sided deviation ratios, roughly half the size, and comparing
        # those against the trough-to-peak cut made the route unreachable.
        strong = leader["prominence_snr"] >= channels[
            leader["channel"]
        ].prominence_cut
        control_snr = (
            channel_snr[control_name] if control_name in channel_snr else 0.0
        )
        primary_name = definition["primary"]
        primary_detected = (
            channel_snr[primary_name] >= channels[primary_name].deviation_cut
        )
        if definition["geometry"] == "arc":
            geometry = arc_geometry(
                frequency, corrected, cluster["center_mhz"], hwhm, step
            )
        else:
            geometry = displacement_geometry(
                frequency, deviation_complex, cluster["center_mhz"], hwhm, step
            )
        # Three routes to acceptance. A lone channel clearing the
        # trough-to-peak extreme value stands on its own. Two agreeing
        # channels may each clear the single-sided value instead. Failing
        # both, a feature on the channel the physics puts it on is accepted
        # when the I/Q geometry independently agrees -- which matters for
        # qubit sweeps, where magnitude and displacement are two views of the
        # same two quadratures and so corroborate each other only weakly.
        if strong:
            basis = "single-channel"
        elif corroborating >= 2:
            basis = "cross-channel"
        elif primary_detected and geometry["geometry_score"] >= MINIMUM_GEOMETRY_SCORE:
            basis = "iq-geometry"
        else:
            # Kept, but not accepted. Reporting the best sub-threshold
            # candidate and the bar it missed is far more use than a bare
            # "nothing found" when deciding where to sweep next.
            basis = "below-threshold"
        # A control channel carries no signal, so any excursion it shows at the
        # same frequency is common-mode and argues against a real feature.
        control_penalty = 1.0
        if control_name is not None:
            control_penalty = float(
                np.clip(
                    channel_snr[control_name]
                    / max(channels[control_name].deviation_cut, 1e-9),
                    0.0,
                    1.0,
                )
            )
        candidates.append(
            FeatureCandidate(
                center_mhz=float(cluster["center_mhz"]),
                hwhm_mhz=float(hwhm),
                polarity=leader["polarity"],
                prominence_snr=float(leader["prominence_snr"]),
                width_samples=float(leader["width_samples"]),
                detected_in=tuple(
                    sorted({member["channel"] for member in cluster["members"]})
                ),
                channel_snr=channel_snr,
                corroborating_channels=int(corroborating),
                control_snr=float(control_snr),
                geometry_score=float(geometry["geometry_score"]),
                geometry=geometry,
                evidence=float(
                    leader["prominence_snr"]
                    * (1.0 + 0.5 * max(corroborating - 1, 0))
                    * (0.25 + geometry["geometry_score"])
                    * (1.0 - 0.5 * control_penalty)
                ),
                strong=bool(strong),
                basis=basis,
            )
        )

    candidates.sort(key=lambda candidate: candidate.evidence, reverse=True)
    candidates = _drop_baseline_lobes(candidates, channels, step)
    accepted = [
        candidate for candidate in candidates if candidate.basis != "below-threshold"
    ]
    marginal = [
        candidate for candidate in candidates if candidate.basis == "below-threshold"
    ]
    return SpectroscopyDetection(
        kind=normalized_kind,
        x=frequency,
        iq=values,
        corrected=corrected,
        delay=delay,
        channels=channels,
        candidates=tuple(accepted[:maximum_candidates]),
        marginal=tuple(marginal[:maximum_candidates]),
        step_mhz=step,
        span_mhz=span,
        anisotropy=float(anisotropy),
    )
