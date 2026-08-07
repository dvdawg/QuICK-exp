"""Flux-line characterization and predistortion from Hellings et al.

The public functions in this module deliberately separate acquisition from
filter construction.  Quick/Mercator can acquire the spectroscopy and
cryoscope maps used by the calibration, while the current local Quick API does
not expose a verified arbitrary-waveform upload path.  The resulting filters
can therefore be inspected, tested, and exported without silently claiming
that they have been applied to hardware.

Experiment times are expressed in microseconds.  DAC sample intervals are
accepted in nanoseconds at the public boundary and converted exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import convolution_matrix
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import sosfilt, zpk2sos
from scipy.interpolate import PchipInterpolator

from .errors import AnalysisError, ConfigError
from .fit_stats import bic, oriented_rotate_iq, r_squared
from .flux_lookup import frequency_from_record
from .native_fit import load_native_trace
from .native_map import load_native_map

# Importing this module registers the ``transmon_f01`` lookup evaluator.
from . import qubit_flux_fit as _qubit_flux_fit  # noqa: F401


PAPER_DOI = "10.1103/1qhb-r4fb"


def _vector(values: Any, label: str, *, minimum: int = 1) -> np.ndarray:
    result = np.asarray(values, dtype=float).ravel()
    if result.size < minimum or not np.all(np.isfinite(result)):
        raise AnalysisError(
            f"{label} must contain at least {minimum} finite value(s)"
        )
    return result


def _matched_vectors_with_order(
    x: Any,
    y: Any,
    x_label: str,
    y_label: str,
) -> tuple:
    x_values = _vector(x, x_label, minimum=2)
    y_values = _vector(y, y_label, minimum=2)
    if x_values.shape != y_values.shape:
        raise AnalysisError(f"{x_label} and {y_label} must have equal shapes")
    order = np.argsort(x_values, kind="stable")
    x_values, y_values = x_values[order], y_values[order]
    if np.any(np.diff(x_values) <= 0):
        raise AnalysisError(f"{x_label} must be strictly increasing")
    return x_values, y_values, order


def _matched_vectors(x: Any, y: Any, x_label: str, y_label: str) -> tuple:
    x_values, y_values, _order = _matched_vectors_with_order(
        x,
        y,
        x_label,
        y_label,
    )
    return x_values, y_values


@dataclass(frozen=True)
class SpectroscopyRow:
    time_us: float
    center_mhz: float
    uncertainty_mhz: float
    r_squared: float
    contrast_snr: float
    detection_sigma: float = 0.0
    width_mhz: float = 0.0
    line_shape: str = "unknown"


@dataclass(frozen=True)
class FluxStepTrace:
    source_csv: Path
    rows: tuple
    dropped_times_us: np.ndarray

    @property
    def time_us(self) -> np.ndarray:
        return np.asarray([row.time_us for row in self.rows], dtype=float)

    @property
    def center_mhz(self) -> np.ndarray:
        return np.asarray([row.center_mhz for row in self.rows], dtype=float)

    @property
    def uncertainty_mhz(self) -> np.ndarray:
        return np.asarray(
            [max(row.uncertainty_mhz, np.finfo(float).eps) for row in self.rows],
            dtype=float,
        )


@dataclass(frozen=True)
class StepResponseFit:
    time_us: np.ndarray
    measured: np.ndarray
    fitted: np.ndarray
    alphas: np.ndarray
    taus_us: np.ndarray
    dc_gain: float
    model_order: int
    bic: float
    candidate_bic: Mapping[int, float]
    statistics: Mapping[str, float]

    def response(self, time_us: Any) -> np.ndarray:
        time = np.asarray(time_us, dtype=float)
        terms = np.exp(-time[..., None] / self.taus_us)
        return self.dc_gain + terms @ self.alphas

    def as_dict(self) -> dict:
        return {
            "model": "normalized_sum_of_exponentials",
            "model_order": int(self.model_order),
            "dc_gain": float(self.dc_gain),
            "alphas": self.alphas.tolist(),
            "taus_us": self.taus_us.tolist(),
            "high_frequency_gain": float(self.dc_gain + np.sum(self.alphas)),
            "bic": float(self.bic),
            "candidate_bic": {
                str(order): float(value)
                for order, value in self.candidate_bic.items()
            },
            "statistics": dict(self.statistics),
        }


@dataclass(frozen=True)
class IIRInverseDesign:
    sample_interval_ns: float
    sos: np.ndarray
    continuous_zeros_per_us: np.ndarray
    continuous_poles_per_us: np.ndarray
    digital_zeros: np.ndarray
    digital_poles: np.ndarray
    leak_tau_us: Optional[float]
    maximum_pole_radius: float

    @property
    def stable(self) -> bool:
        # The faithful high-pass inverse has a pole exactly at z=1.  It is
        # marginal rather than asymptotically stable and is reported as such.
        return bool(self.maximum_pole_radius < 1.0 - 1e-12)

    @property
    def marginal(self) -> bool:
        return bool(abs(self.maximum_pole_radius - 1.0) <= 1e-10)

    def apply(self, waveform: Any) -> np.ndarray:
        values = _vector(waveform, "waveform")
        return sosfilt(self.sos, values)

    def as_dict(self) -> dict:
        def complex_pairs(values):
            return [
                [float(complex(value).real), float(complex(value).imag)]
                for value in values
            ]

        return {
            "method": "matched_z_normalized_inverse_sos",
            "sample_interval_ns": float(self.sample_interval_ns),
            "sos": np.asarray(self.sos, dtype=float).tolist(),
            "continuous_zeros_per_us": complex_pairs(
                self.continuous_zeros_per_us
            ),
            "continuous_poles_per_us": complex_pairs(
                self.continuous_poles_per_us
            ),
            "digital_zeros": complex_pairs(self.digital_zeros),
            "digital_poles": complex_pairs(self.digital_poles),
            "leak_tau_us": (
                None if self.leak_tau_us is None else float(self.leak_tau_us)
            ),
            "maximum_pole_radius": float(self.maximum_pole_radius),
            "stable": self.stable,
            "marginal": self.marginal,
        }


@dataclass(frozen=True)
class CryoscopeSchedule:
    center_time_us: np.ndarray
    delta_time_us: np.ndarray
    minus_time_us: np.ndarray
    plus_time_us: np.ndarray
    acquisition_time_us: np.ndarray
    sample_interval_ns: float


@dataclass(frozen=True)
class CryoscopePhaseTrace:
    source_csv: Path
    duration_us: np.ndarray
    accumulated_phase_rad: np.ndarray
    phase_uncertainty_rad: np.ndarray
    contrast: np.ndarray
    row_r_squared: np.ndarray


@dataclass(frozen=True)
class CryoscopeFrequencyTrace:
    time_us: np.ndarray
    delta_time_us: np.ndarray
    detuning_mhz: np.ndarray
    uncertainty_mhz: np.ndarray


@dataclass(frozen=True)
class ForwardFIRFit:
    coefficients: np.ndarray
    sample_interval_ns: float
    measured_time_us: np.ndarray
    measured_frequency_mhz: np.ndarray
    fitted_frequency_mhz: np.ndarray
    statistics: Mapping[str, float]
    regularization: Mapping[str, float]

    def as_dict(self) -> dict:
        return {
            "method": "nonlinear_forward_frequency_fir",
            "sample_interval_ns": float(self.sample_interval_ns),
            "coefficients": self.coefficients.tolist(),
            "statistics": dict(self.statistics),
            "regularization": dict(self.regularization),
        }


@dataclass(frozen=True)
class InverseFIRDesign:
    coefficients: np.ndarray
    target: np.ndarray
    realized: np.ndarray
    sample_interval_ns: float
    gaussian_sigma_ns: float
    latency_samples: int
    regularization: float
    statistics: Mapping[str, float]

    def apply(self, waveform: Any) -> np.ndarray:
        values = _vector(waveform, "waveform")
        return np.convolve(values, self.coefficients, mode="full")

    def as_dict(self) -> dict:
        return {
            "method": "regularized_sobolev_inverse_fir",
            "sample_interval_ns": float(self.sample_interval_ns),
            "gaussian_sigma_ns": float(self.gaussian_sigma_ns),
            "latency_samples": int(self.latency_samples),
            "regularization": float(self.regularization),
            "coefficients": self.coefficients.tolist(),
            "statistics": dict(self.statistics),
        }


@dataclass(frozen=True)
class WaveformCheck:
    finite: bool
    maximum_absolute: float
    maximum_fraction_of_full_scale: float
    maximum_step_per_ns: float
    amplitude_pass: bool
    slew_pass: Optional[bool]

    @property
    def passes(self) -> bool:
        return bool(
            self.finite
            and self.amplitude_pass
            and self.slew_pass is not False
        )


def _linear_coefficients(
    time_us: np.ndarray,
    measured: np.ndarray,
    taus_us: np.ndarray,
    weights: np.ndarray,
    dc_gain: Optional[float],
) -> tuple:
    exponentials = np.exp(-time_us[:, None] / taus_us)
    if dc_gain is None:
        matrix = np.column_stack((np.ones(time_us.size), exponentials))
        coefficients, *_ = np.linalg.lstsq(
            matrix * weights[:, None],
            measured * weights,
            rcond=None,
        )
        dc = float(coefficients[0])
        alpha = np.asarray(coefficients[1:], dtype=float)
    else:
        dc = float(dc_gain)
        alpha, *_ = np.linalg.lstsq(
            exponentials * weights[:, None],
            (measured - dc) * weights,
            rcond=None,
        )
    fitted = dc + exponentials @ alpha
    return dc, np.asarray(alpha, dtype=float), fitted


def fit_step_response(
    time_us: Any,
    normalized_response: Any,
    *,
    uncertainty: Optional[Any] = None,
    model_orders: Iterable[int] = range(1, 7),
    dc_gain: Optional[float] = 0.0,
    minimum_time_us: Optional[float] = None,
    maximum_time_us: Optional[float] = None,
    tau_bounds_us: Optional[Sequence[float]] = None,
    multistarts: int = 12,
    seed: int = 0,
) -> StepResponseFit:
    """Fit a normalized step response and select exponential order by BIC.

    Amplitudes are fitted freely, matching Appendix G. The inverse-filter
    construction subsequently drops their common high-frequency scale, as the
    paper does through ``H_inv=(1/H)/kappa``. Set ``dc_gain=0`` for the first
    high-pass pass, ``1`` after its dominant correction, or ``None`` to fit it.
    """
    time, measured, sort_order = _matched_vectors_with_order(
        time_us,
        normalized_response,
        "time_us",
        "normalized_response",
    )
    selected = np.ones(time.size, dtype=bool)
    if minimum_time_us is not None:
        selected &= time >= float(minimum_time_us)
    if maximum_time_us is not None:
        selected &= time <= float(maximum_time_us)
    time, measured = time[selected], measured[selected]
    if time.size < 8:
        raise AnalysisError("step-response fitting requires at least eight points")
    if np.any(time < 0):
        raise AnalysisError("step-response times cannot be negative")
    if uncertainty is None:
        weights = np.ones_like(time)
    else:
        raw_uncertainty = _vector(uncertainty, "uncertainty", minimum=2)
        if raw_uncertainty.size != selected.size:
            raise AnalysisError("uncertainty must match the unfiltered time array")
        sigma = raw_uncertainty[sort_order][selected]
        positive = sigma[np.isfinite(sigma) & (sigma > 0)]
        if not positive.size:
            raise AnalysisError("uncertainty must contain positive values")
        sigma = np.where(
            np.isfinite(sigma) & (sigma > 0),
            sigma,
            float(np.median(positive)),
        )
        weights = 1.0 / sigma
        weights /= float(np.median(weights))

    positive_times = time[time > 0]
    if not positive_times.size:
        raise AnalysisError("step-response fitting needs a positive time")
    typical_step = float(np.median(np.diff(time)))
    if tau_bounds_us is None:
        tau_min = max(
            min(float(np.min(positive_times)), typical_step) / 3.0,
            1e-9,
        )
        tau_max = max(float(np.max(time)) * 30.0, tau_min * 100.0)
    else:
        if len(tau_bounds_us) != 2:
            raise AnalysisError("tau_bounds_us must be [minimum, maximum]")
        tau_min, tau_max = map(float, tau_bounds_us)
        if not 0 < tau_min < tau_max:
            raise AnalysisError("tau bounds must be finite, positive, and ordered")
    order_values = sorted({int(order) for order in model_orders})
    if not order_values or order_values[0] < 1:
        raise AnalysisError("model_orders must contain positive integers")
    if max(order_values) * 3 >= time.size:
        raise AnalysisError("requested exponential order is too high for the data")
    starts = max(int(multistarts), 1)
    rng = np.random.default_rng(int(seed))
    log_lower, log_upper = math.log(tau_min), math.log(tau_max)
    candidates = {}
    candidate_bic = {}

    for order in order_values:
        best = None

        def evaluate(log_taus):
            taus = np.sort(np.exp(log_taus))
            dc, alpha, fitted = _linear_coefficients(
                time,
                measured,
                taus,
                weights,
                dc_gain,
            )
            return dc, alpha, fitted

        def residual(log_taus):
            return weights * (evaluate(log_taus)[2] - measured)

        base = np.linspace(log_lower, log_upper, order + 2)[1:-1]
        guesses = [base]
        for _ in range(starts - 1):
            jitter = rng.normal(0.0, 0.35, size=order)
            guesses.append(np.clip(base + jitter, log_lower, log_upper))
        for guess in guesses:
            try:
                result = least_squares(
                    residual,
                    guess,
                    bounds=(
                        np.full(order, log_lower),
                        np.full(order, log_upper),
                    ),
                    loss="soft_l1",
                    f_scale=1.0,
                    max_nfev=20_000,
                )
            except (ValueError, FloatingPointError):
                continue
            dc, alpha, fitted = evaluate(result.x)
            rss = float(np.sum((measured - fitted) ** 2))
            if result.success and (best is None or rss < best[0]):
                best = (rss, dc, alpha, np.sort(np.exp(result.x)), fitted)
        if best is None:
            continue
        rss, fitted_dc, alpha, taus, fitted = best
        linear_parameters = order + (1 if dc_gain is None else 0)
        n_parameters = order + linear_parameters
        score = bic(rss, time.size, n_parameters)
        candidate_bic[order] = score
        candidates[order] = (rss, fitted_dc, alpha, taus, fitted, score)

    if not candidates:
        raise AnalysisError("no sum-of-exponentials model converged")
    selected_order = min(candidate_bic, key=candidate_bic.get)
    rss, fitted_dc, alpha, taus, fitted, score = candidates[selected_order]
    residual_values = measured - fitted
    return StepResponseFit(
        time_us=time,
        measured=measured,
        fitted=fitted,
        alphas=np.asarray(alpha, dtype=float),
        taus_us=np.asarray(taus, dtype=float),
        dc_gain=float(fitted_dc),
        model_order=int(selected_order),
        bic=float(score),
        candidate_bic=dict(candidate_bic),
        statistics={
            "r_squared": r_squared(measured, fitted),
            "rmse": float(np.sqrt(np.mean(residual_values**2))),
            "maximum_absolute_residual": float(
                np.max(np.abs(residual_values))
            ),
            "rss": float(rss),
        },
    )


def _continuous_transfer_polynomials(fit: StepResponseFit) -> tuple:
    denominator = np.poly1d([1.0])
    factors = []
    for tau in fit.taus_us:
        factor = np.poly1d([float(tau), 1.0])
        factors.append(factor)
        denominator *= factor
    numerator = fit.dc_gain * denominator
    for index, (alpha, tau) in enumerate(zip(fit.alphas, fit.taus_us)):
        term = np.poly1d([float(alpha) * float(tau), 0.0])
        for other, factor in enumerate(factors):
            if other != index:
                term *= factor
        numerator += term
    numerator_coefficients = np.trim_zeros(
        np.asarray(numerator.c, dtype=float),
        trim="f",
    )
    denominator_coefficients = np.asarray(denominator.c, dtype=float)
    if numerator_coefficients.size < 2:
        raise AnalysisError("step-response transfer function is not invertible")
    return numerator_coefficients, denominator_coefficients


def design_iir_inverse(
    fit: StepResponseFit,
    *,
    sample_interval_ns: float,
    leak_tau_us: Optional[float] = None,
) -> IIRInverseDesign:
    """Construct the normalized matched-z inverse as second-order sections.

    With ``dc_gain=0`` the faithful inverse contains the paper's integrator
    pole at ``z=1``.  ``leak_tau_us`` moves that pole just inside the unit
    circle and is useful for long unattended sequences or non-net-zero pulses.
    """
    interval_ns = float(sample_interval_ns)
    if not np.isfinite(interval_ns) or interval_ns <= 0:
        raise AnalysisError("sample_interval_ns must be finite and positive")
    interval_us = interval_ns / 1000.0
    numerator, _denominator = _continuous_transfer_polynomials(fit)
    inverse_zeros = -1.0 / np.asarray(fit.taus_us, dtype=float)
    inverse_poles = np.roots(numerator)
    tolerance = max(1e-11, 1e-9 / max(float(np.max(fit.taus_us)), 1.0))
    if leak_tau_us is not None:
        leak = float(leak_tau_us)
        if not np.isfinite(leak) or leak <= 0:
            raise AnalysisError("leak_tau_us must be finite and positive")
        near_integrator = np.abs(inverse_poles) <= tolerance
        inverse_poles = np.where(near_integrator, -1.0 / leak, inverse_poles)
    else:
        leak = None
    if np.any(np.real(inverse_poles) > tolerance):
        raise AnalysisError(
            "fitted forward response is non-minimum-phase; its causal inverse "
            "would be unstable"
        )
    digital_zeros = np.exp(inverse_zeros * interval_us)
    digital_poles = np.exp(inverse_poles * interval_us)
    maximum_radius = float(np.max(np.abs(digital_poles)))
    if maximum_radius > 1.0 + 1e-10:
        raise AnalysisError("matched-z inverse has a pole outside the unit circle")
    sos = zpk2sos(
        digital_zeros,
        digital_poles,
        1.0,
        pairing="nearest",
    )
    sos = np.real_if_close(sos, tol=1000)
    if np.iscomplexobj(sos):
        raise AnalysisError("matched-z filter did not produce real coefficients")
    return IIRInverseDesign(
        sample_interval_ns=interval_ns,
        sos=np.asarray(sos, dtype=float),
        continuous_zeros_per_us=np.asarray(inverse_zeros, dtype=complex),
        continuous_poles_per_us=np.asarray(inverse_poles, dtype=complex),
        digital_zeros=np.asarray(digital_zeros, dtype=complex),
        digital_poles=np.asarray(digital_poles, dtype=complex),
        leak_tau_us=leak,
        maximum_pole_radius=maximum_radius,
    )


def _line_profile(
    frequency_mhz: np.ndarray,
    center: float,
    width: float,
    line_shape: str,
) -> np.ndarray:
    normalized_frequency = (frequency_mhz - center) / width
    if line_shape == "gaussian":
        return np.exp(-0.5 * normalized_frequency**2)
    if line_shape == "lorentzian":
        return 1.0 / (1.0 + 1j * normalized_frequency)
    raise AnalysisError(  # pragma: no cover - internal invariant
        f"unsupported line shape {line_shape!r}"
    )


def _fit_spectroscopy_row(frequency_mhz: np.ndarray, iq: np.ndarray) -> tuple:
    """Fit a complex Gaussian or Lorentzian resonance and keep the better fit.

    The paper uses a Gaussian spectroscopy pulse, so a Gaussian frequency
    profile is the faithful first candidate.  Native lab data can instead be
    power-broadened or resonator-like; the equal-parameter Lorentzian candidate
    avoids forcing that data into the wrong line shape.  Raw complex-IQ RSS is
    sufficient for selection because both candidates have eight parameters.

    Alongside the per-point contrast this returns ``detection_sigma``, the
    amplitude integrated over the profile as it was actually sampled.  A line
    that is resolved by many points is detected far more significantly than its
    per-point depth suggests, so the integrated figure is the one that survives
    a change of sweep span or step.
    """
    values = np.asarray(iq, dtype=complex).ravel()
    if values.size != frequency_mhz.size or not np.all(
        np.isfinite(values.real) & np.isfinite(values.imag)
    ):
        raise AnalysisError("spectroscopy row has non-finite IQ samples")
    span = float(np.ptp(frequency_mhz))
    step = float(np.median(np.diff(frequency_mhz)))
    reference = float(np.mean(frequency_mhz))
    edge_count = max(5, frequency_mhz.size // 10)
    edge_indices = np.r_[0:edge_count, frequency_mhz.size - edge_count:frequency_mhz.size]
    edge_design = np.column_stack(
        (
            np.ones(edge_indices.size),
            frequency_mhz[edge_indices] - reference,
        )
    )
    baseline_coefficients, *_ = np.linalg.lstsq(
        edge_design,
        values[edge_indices],
        rcond=None,
    )
    baseline = (
        baseline_coefficients[0]
        + baseline_coefficients[1] * (frequency_mhz - reference)
    )
    deviation = values - baseline
    smoothed_power = gaussian_filter1d(
        np.abs(deviation),
        1.5,
        mode="nearest",
    )
    center_index = int(np.argmax(smoothed_power))
    center_seed = float(frequency_mhz[center_index])
    amplitude_seed = complex(deviation[center_index])
    if abs(amplitude_seed) <= np.finfo(float).eps:
        amplitude_seed = complex(max(float(np.std(values)), 1e-6), 0.0)
    initial = np.asarray(
        [
            baseline_coefficients[0].real,
            baseline_coefficients[0].imag,
            baseline_coefficients[1].real,
            baseline_coefficients[1].imag,
            amplitude_seed.real,
            amplitude_seed.imag,
            center_seed,
            max(span / 40.0, step),
        ]
    )
    scale = max(float(np.ptp(np.abs(values))), float(np.std(values)), 1e-6)
    offset_real = float(baseline_coefficients[0].real)
    offset_imag = float(baseline_coefficients[0].imag)
    lower = np.asarray(
        [
            offset_real - 3 * scale,
            offset_imag - 3 * scale,
            -10 * scale / span,
            -10 * scale / span,
            -5 * scale,
            -5 * scale,
            float(frequency_mhz[0]),
            max(step / 3.0, 1e-9),
        ]
    )
    upper = np.asarray(
        [
            offset_real + 3 * scale,
            offset_imag + 3 * scale,
            10 * scale / span,
            10 * scale / span,
            5 * scale,
            5 * scale,
            float(frequency_mhz[-1]),
            max(span / 2.0, step),
        ]
    )

    def model(parameters, line_shape):
        offset = complex(parameters[0], parameters[1])
        slope = complex(parameters[2], parameters[3])
        amplitude = complex(parameters[4], parameters[5])
        profile = _line_profile(
            frequency_mhz,
            parameters[6],
            parameters[7],
            line_shape,
        )
        return (
            offset
            + slope * (frequency_mhz - reference)
            + amplitude * profile
        )

    candidates = []
    for line_shape in ("gaussian", "lorentzian"):
        def residual(parameters, *, _line_shape=line_shape):
            difference = model(parameters, _line_shape) - values
            return np.concatenate((difference.real, difference.imag))

        result = least_squares(
            residual,
            np.clip(initial, lower, upper),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=max(scale / 20.0, 1e-9),
            max_nfev=10_000,
        )
        fitted = model(result.x, line_shape)
        rss = float(np.sum(np.abs(values - fitted) ** 2))
        if result.success and np.isfinite(rss):
            candidates.append((rss, line_shape, result, fitted))
    if not candidates:
        raise AnalysisError("Gaussian and Lorentzian spectroscopy fits failed")
    _rss, line_shape, result, fitted = min(candidates, key=lambda item: item[0])
    residual_values = values - fitted
    dof = max(2 * values.size - result.x.size, 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac)
    covariance *= float(np.sum(np.abs(residual_values) ** 2)) / dof
    center_uncertainty = float(
        np.sqrt(max(float(covariance[6, 6]), 0.0))
    )
    noise = max(
        float(np.sqrt(np.mean(np.abs(residual_values) ** 2))),
        np.finfo(float).eps,
    )
    amplitude = abs(complex(result.x[4], result.x[5]))
    width = abs(float(result.x[7]))
    # Number of samples the profile is effectively worth, so a resolved line
    # keeps its significance when the sweep is resampled.
    effective_samples = float(
        np.sum(
            np.abs(
                _line_profile(frequency_mhz, result.x[6], result.x[7], line_shape)
            )
            ** 2
        )
    )
    return (
        float(result.x[6]),
        center_uncertainty,
        r_squared(values, fitted),
        amplitude / noise,
        amplitude * np.sqrt(max(effective_samples, 0.0)) / noise,
        width,
        effective_samples,
        line_shape,
    )


def _row_passes_qc(
    row_r2: float,
    contrast: float,
    detection: float,
    center: float,
    width: float,
    effective_samples: float,
    frequency_mhz: np.ndarray,
    *,
    minimum_row_r_squared: float,
    minimum_contrast_snr: float,
    minimum_detection_sigma: float,
    minimum_effective_samples: float,
    maximum_width_fraction: float,
) -> bool:
    """Accept a spectroscopy row on integrated significance and localisation.

    ``r_squared`` and the per-point ``contrast`` both measure the line against
    the whole sweep, so both shrink as the span widens or the step shrinks: a
    genuine 8 sigma line spread over 300 MHz of a 301-point scan lands near
    r^2 = 0.1 and contrast = 1.5, and no threshold on either can separate it
    from noise.  They are kept as optional floors, off by default.

    ``detection_sigma`` integrates the amplitude over the fitted profile and is
    invariant to that resampling.  It is only meaningful when the line really
    is a line, so two shape checks travel with it:

    * the profile must be worth at least a few samples of the grid it was
      measured on, or amplitude and width are not separably constrained and a
      single noisy bin fits as a very narrow, very significant "line";
    * the width must stay well inside the sweep, or a fit that absorbed the
      baseline into a span-wide "resonance" reports a large and entirely
      spurious significance.

    A centre that has railed against the edge of the sweep is rejected too: the
    line is then only partly inside the window, so its position is bounded
    rather than measured.
    """
    span = float(np.ptp(frequency_mhz))
    step = abs(float(np.median(np.diff(frequency_mhz))))
    low = min(float(frequency_mhz[0]), float(frequency_mhz[-1]))
    high = max(float(frequency_mhz[0]), float(frequency_mhz[-1]))
    if not np.isfinite(detection) or detection < minimum_detection_sigma:
        return False
    if not np.isfinite(center) or not low + step <= center <= high - step:
        return False
    if not np.isfinite(width) or width > maximum_width_fraction * span:
        return False
    if (
        not np.isfinite(effective_samples)
        or effective_samples < minimum_effective_samples
    ):
        return False
    if row_r2 < minimum_row_r_squared or contrast < minimum_contrast_snr:
        return False
    return True


def extract_step_spectroscopy(
    csv_path: Path,
    *,
    minimum_row_r_squared: float = 0.0,
    minimum_contrast_snr: float = 0.0,
    minimum_detection_sigma: float = 4.0,
    minimum_effective_samples: float = 3.0,
    maximum_width_fraction: float = 0.125,
) -> FluxStepTrace:
    native = load_native_map(csv_path)
    if "time" not in native.outer_label.lower():
        raise AnalysisError("flux-step map requires time as the outer axis")
    if "freq" not in native.inner_label.lower():
        raise AnalysisError("flux-step map requires frequency as the inner axis")
    rows = []
    dropped = list(map(float, native.incomplete_outer))
    for time_value, iq_row in zip(native.outer, native.complex_signal):
        try:
            (
                center,
                uncertainty,
                row_r2,
                contrast,
                detection,
                width,
                effective_samples,
                line_shape,
            ) = _fit_spectroscopy_row(native.inner, iq_row)
        except (AnalysisError, RuntimeError, ValueError, FloatingPointError):
            dropped.append(float(time_value))
            continue
        if not _row_passes_qc(
            row_r2,
            contrast,
            detection,
            center,
            width,
            effective_samples,
            native.inner,
            minimum_row_r_squared=minimum_row_r_squared,
            minimum_contrast_snr=minimum_contrast_snr,
            minimum_detection_sigma=minimum_detection_sigma,
            minimum_effective_samples=minimum_effective_samples,
            maximum_width_fraction=maximum_width_fraction,
        ):
            dropped.append(float(time_value))
            continue
        rows.append(
            SpectroscopyRow(
                time_us=float(time_value),
                center_mhz=center,
                uncertainty_mhz=max(uncertainty, np.finfo(float).eps),
                r_squared=float(row_r2),
                contrast_snr=float(contrast),
                detection_sigma=float(detection),
                width_mhz=float(width),
                line_shape=line_shape,
            )
        )
    rows.sort(key=lambda row: row.time_us)
    if len(rows) < 8:
        raise AnalysisError("fewer than eight flux-step spectra passed row QC")
    return FluxStepTrace(
        source_csv=native.source_csv,
        rows=tuple(rows),
        dropped_times_us=np.asarray(sorted(set(dropped)), dtype=float),
    )


def extract_step_spectroscopy_rows(
    csv_paths: Sequence[Path],
    *,
    minimum_row_r_squared: float = 0.0,
    minimum_contrast_snr: float = 0.0,
    minimum_detection_sigma: float = 4.0,
    minimum_effective_samples: float = 3.0,
    maximum_width_fraction: float = 0.125,
) -> FluxStepTrace:
    """Load adaptive one-spectrum-per-time acquisitions as one trace."""
    if not csv_paths:
        raise AnalysisError("at least one adaptive spectroscopy CSV is required")
    rows = []
    dropped = []
    sources = []
    unreadable = []
    for csv_path in csv_paths:
        # An aborted acquisition leaves a short CSV behind.  Losing that one
        # time point is not a reason to discard the rest of the campaign.
        try:
            native = load_native_trace(
                csv_path,
                quick_class="FluxStepSpectroscopy",
                axis_text="frequency",
                minimum_points=8,
            )
        except (AnalysisError, OSError, ValueError):
            unreadable.append(Path(csv_path))
            continue
        sources.append(native.source_csv)
        parameters = native.metadata.get("parameters", {})
        variables = parameters.get("var", parameters)
        try:
            time_value = float(variables["probe_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisError(
                f"{native.source_csv.name} has no scalar probe_time metadata"
            ) from error
        try:
            (
                center,
                uncertainty,
                row_r2,
                contrast,
                detection,
                width,
                effective_samples,
                line_shape,
            ) = _fit_spectroscopy_row(native.x, native.iq)
        except (AnalysisError, RuntimeError, ValueError, FloatingPointError):
            dropped.append(time_value)
            continue
        if not _row_passes_qc(
            row_r2,
            contrast,
            detection,
            center,
            width,
            effective_samples,
            native.x,
            minimum_row_r_squared=minimum_row_r_squared,
            minimum_contrast_snr=minimum_contrast_snr,
            minimum_detection_sigma=minimum_detection_sigma,
            minimum_effective_samples=minimum_effective_samples,
            maximum_width_fraction=maximum_width_fraction,
        ):
            dropped.append(time_value)
            continue
        rows.append(
            SpectroscopyRow(
                time_us=time_value,
                center_mhz=center,
                uncertainty_mhz=max(uncertainty, np.finfo(float).eps),
                r_squared=float(row_r2),
                contrast_snr=float(contrast),
                detection_sigma=float(detection),
                width_mhz=float(width),
                line_shape=line_shape,
            )
        )
    if unreadable and not sources:
        raise AnalysisError(
            "no adaptive spectroscopy CSV could be read; "
            f"{len(unreadable)} file(s) were empty or truncated"
        )
    # If a repeated time exists, prefer the newest path supplied by the caller.
    by_time = {row.time_us: row for row in rows}
    rows = [by_time[time] for time in sorted(by_time)]
    if len(rows) < 8:
        raise AnalysisError("fewer than eight adaptive spectra passed row QC")
    return FluxStepTrace(
        source_csv=sources[-1],
        rows=tuple(rows),
        dropped_times_us=np.asarray(sorted(set(dropped)), dtype=float),
    )


def frequency_to_flux(
    record: Mapping[str, Any],
    frequency_mhz: Any,
    *,
    branch_z: Sequence[float],
    grid_points: int = 20_001,
) -> np.ndarray:
    """Invert an accepted flux-frequency lookup on one monotonic branch."""
    if len(branch_z) != 2:
        raise AnalysisError("branch_z must be [minimum_z, maximum_z]")
    lower, upper = map(float, branch_z)
    if not np.isfinite(lower + upper) or lower >= upper:
        raise AnalysisError("branch_z must be finite and increasing")
    count = max(int(grid_points), 1001)
    z_grid = np.linspace(lower, upper, count)
    model_frequency = np.asarray(
        frequency_from_record(record, z_grid),
        dtype=float,
    )
    differences = np.diff(model_frequency)
    tolerance = max(float(np.ptp(model_frequency)) * 1e-10, 1e-10)
    significant = differences[np.abs(differences) > tolerance]
    if not significant.size or np.any(significant > 0) and np.any(significant < 0):
        raise AnalysisError(
            "selected branch is not monotonic; move its endpoints away from a sweet spot"
        )
    if model_frequency[-1] < model_frequency[0]:
        model_frequency = model_frequency[::-1]
        z_grid = z_grid[::-1]
    unique_frequency, unique_indices = np.unique(model_frequency, return_index=True)
    requested = np.asarray(frequency_mhz, dtype=float)
    if not np.all(np.isfinite(requested)):
        raise AnalysisError("frequency_mhz contains non-finite values")
    if np.any(requested < unique_frequency[0]) or np.any(
        requested > unique_frequency[-1]
    ):
        raise AnalysisError(
            "measured frequency lies outside the selected flux branch"
        )
    inverse = PchipInterpolator(
        unique_frequency,
        z_grid[unique_indices],
        extrapolate=False,
    )
    return np.asarray(inverse(requested), dtype=float)


def flux_command_from_phi0_fractions(
    record: Mapping[str, Any],
    *,
    baseline_phi0: float,
    step_phi0: float,
) -> tuple:
    """Convert offsets in flux quanta to this installation's Z-gain units.

    ``period_z`` is the fitted separation between equivalent upper sweet spots
    and therefore represents one flux quantum in the local Z coordinate.
    Both endpoints are domain-checked through the accepted lookup.
    """
    try:
        parameters = record["value"]["parameters"]
        period_z = abs(float(parameters["period_z"]))
        sweet_spot_z = float(parameters["sweet_spot_z"])
        baseline_fraction = float(baseline_phi0)
        step_fraction = float(step_phi0)
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(
            "qubit_vs_flux must contain transmon period_z and sweet_spot_z"
        ) from error
    if (
        record.get("model") != "transmon_f01"
        or not np.isfinite(period_z)
        or period_z <= 0
        or not np.isfinite(baseline_fraction + step_fraction)
    ):
        raise ConfigError(
            "Phi0-fraction conversion requires a finite accepted transmon_f01 fit"
        )
    baseline_z = sweet_spot_z + baseline_fraction * period_z
    step_z = step_fraction * period_z
    frequency_from_record(record, np.asarray([baseline_z, baseline_z + step_z]))
    return float(baseline_z), float(step_z)


def monotonic_branch_for_flux_step(
    record: Mapping[str, Any],
    *,
    baseline_z: float,
    commanded_step_z: float,
    sweet_spot_guard_fraction: float = 0.005,
) -> tuple:
    """Choose the accepted monotonic half-period containing a flux step."""
    try:
        parameters = record["value"]["parameters"]
        period_z = abs(float(parameters["period_z"]))
        sweet_spot_z = float(parameters["sweet_spot_z"])
        domain = record["valid_domain"]["z_gain"]
        domain_lower, domain_upper = sorted(map(float, domain))
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(
            "automatic branch selection needs period, sweet spot, and valid domain"
        ) from error
    baseline = float(baseline_z)
    target = baseline + float(commanded_step_z)
    offsets = np.asarray([baseline - sweet_spot_z, target - sweet_spot_z])
    if np.any(~np.isfinite(offsets)) or np.any(np.abs(offsets) >= 0.5 * period_z):
        raise ConfigError(
            "flux step is not contained between the selected upper and lower sweet spots"
        )
    signs = np.sign(offsets)
    if np.any(signs == 0) or signs[0] != signs[1]:
        raise ConfigError(
            "flux step touches or crosses the upper sweet spot; choose an explicit branch"
        )
    guard = float(sweet_spot_guard_fraction)
    if not 0 < guard < 0.25:
        raise ConfigError("sweet_spot_guard_fraction must lie between 0 and 0.25")
    side = float(signs[0])
    endpoints = np.asarray(
        [
            sweet_spot_z + side * guard * period_z,
            sweet_spot_z + side * (0.5 - guard) * period_z,
        ]
    )
    lower = max(float(np.min(endpoints)), domain_lower)
    upper = min(float(np.max(endpoints)), domain_upper)
    if lower >= upper or min(baseline, target) < lower or max(baseline, target) > upper:
        raise ConfigError(
            "accepted qubit_vs_flux domain does not cover the commanded monotonic branch"
        )
    return lower, upper


def spectroscopy_to_normalized_step(
    trace: FluxStepTrace,
    flux_frequency_record: Mapping[str, Any],
    *,
    branch_z: Sequence[float],
    baseline_z: float,
    commanded_step_z: float,
    spectroscopy_offset_mhz: float = 0.0,
    normalization: str = "measured_peak",
) -> tuple:
    step = float(commanded_step_z)
    if not np.isfinite(step) or step == 0:
        raise AnalysisError("commanded_step_z must be finite and non-zero")
    corrected_frequency = trace.center_mhz - float(spectroscopy_offset_mhz)
    measured_z = frequency_to_flux(
        flux_frequency_record,
        corrected_frequency,
        branch_z=branch_z,
    )
    flux_change = measured_z - float(baseline_z)
    if normalization == "measured_peak":
        scale = float(flux_change[np.argmax(np.abs(flux_change))])
        if not np.isfinite(scale) or abs(scale) <= np.finfo(float).eps:
            raise AnalysisError("measured flux excursion is zero or non-finite")
    elif normalization == "commanded_step":
        scale = step
    else:
        raise AnalysisError(
            "normalization must be 'measured_peak' or 'commanded_step'"
        )
    normalized = flux_change / scale
    # Propagate frequency uncertainty through a symmetric numerical inverse.
    plus = frequency_to_flux(
        flux_frequency_record,
        corrected_frequency + trace.uncertainty_mhz,
        branch_z=branch_z,
    )
    minus = frequency_to_flux(
        flux_frequency_record,
        corrected_frequency - trace.uncertainty_mhz,
        branch_z=branch_z,
    )
    response_uncertainty = np.abs(plus - minus) / (2.0 * abs(scale))
    return measured_z, normalized, response_uncertainty


def make_cryoscope_schedule(
    center_time_us: Any,
    *,
    pulse_duration_us: float,
    sample_interval_ns: float,
    edge_window_ns: float = 2.4,
    near_edge_window_ns: float = 30.0,
    near_edge_delta_samples: int = 3,
    interior_delta_samples: int = 8,
) -> CryoscopeSchedule:
    """Build the paper's Ts/3Ts/8Ts finite-difference schedule.

    Endpoints are snapped to integer DAC samples.  The returned center times
    are the actual pair midpoints and can differ by half a sample from the
    requested values.
    """
    requested = _vector(center_time_us, "center_time_us", minimum=1)
    interval_ns = float(sample_interval_ns)
    duration_us = float(pulse_duration_us)
    if not np.isfinite(interval_ns) or interval_ns <= 0:
        raise AnalysisError("sample_interval_ns must be finite and positive")
    if not np.isfinite(duration_us) or duration_us <= 0:
        raise AnalysisError("pulse_duration_us must be finite and positive")
    interval_us = interval_ns / 1000.0
    maximum_sample = int(round(duration_us / interval_us))
    if maximum_sample < 2:
        raise AnalysisError("pulse duration must span at least two samples")
    center_samples = np.rint(requested / interval_us).astype(int)
    if np.any(center_samples < 0) or np.any(center_samples > maximum_sample):
        raise AnalysisError("cryoscope center time lies outside the flux pulse")
    edge_distance_ns = np.minimum(
        center_samples,
        maximum_sample - center_samples,
    ) * interval_ns
    gaps = np.where(
        edge_distance_ns <= float(edge_window_ns),
        1,
        np.where(
            edge_distance_ns <= float(near_edge_window_ns),
            int(near_edge_delta_samples),
            int(interior_delta_samples),
        ),
    ).astype(int)
    if np.any(gaps < 1):
        raise AnalysisError("cryoscope difference gaps must be positive")
    minus = center_samples - gaps // 2
    plus = minus + gaps
    below = minus < 0
    plus[below] -= minus[below]
    minus[below] = 0
    above = plus > maximum_sample
    minus[above] -= plus[above] - maximum_sample
    plus[above] = maximum_sample
    acquisition = np.unique(np.concatenate((minus, plus)))
    return CryoscopeSchedule(
        center_time_us=(minus + plus) * interval_us / 2.0,
        delta_time_us=(plus - minus) * interval_us,
        minus_time_us=minus * interval_us,
        plus_time_us=plus * interval_us,
        acquisition_time_us=acquisition * interval_us,
        sample_interval_ns=interval_ns,
    )


def extract_cryoscope_phases(
    csv_path: Path,
    *,
    phase_sign: float = 1.0,
    phase_prior_detuning_mhz: Optional[float] = None,
) -> CryoscopePhaseTrace:
    native = load_native_map(csv_path)
    outer_label = native.outer_label.lower()
    if "time" not in outer_label and "duration" not in outer_label:
        raise AnalysisError("cryoscope map requires duration/time as outer axis")
    if "phase" not in native.inner_label.lower():
        raise AnalysisError("cryoscope map requires Ramsey phase as inner axis")
    phase_axis = np.asarray(native.inner, dtype=float)
    if "deg" in native.inner_unit.lower():
        phase_axis = np.deg2rad(phase_axis)
    elif np.ptp(phase_axis) > 2.5 * np.pi:
        raise AnalysisError(
            "Ramsey phase looks like degrees but its axis unit is not 'deg'"
        )
    iq = native.complex_signal
    _projected, orientation = oriented_rotate_iq(iq.ravel())
    angle = float(orientation["angle_rad"])
    projected = np.real(iq * np.exp(-1j * angle))
    design = np.column_stack(
        (np.ones(phase_axis.size), np.cos(phase_axis), np.sin(phase_axis))
    )
    phases = []
    uncertainties = []
    contrasts = []
    row_scores = []
    for row in projected:
        coefficients, *_ = np.linalg.lstsq(design, row, rcond=None)
        fitted = design @ coefficients
        residual = row - fitted
        covariance = np.linalg.pinv(design.T @ design)
        covariance *= float(np.sum(residual**2)) / max(row.size - 3, 1)
        a, b = float(coefficients[1]), float(coefficients[2])
        amplitude_squared = max(a * a + b * b, np.finfo(float).eps)
        phase = math.atan2(b, a)
        gradient = np.asarray([-b, a]) / amplitude_squared
        phase_covariance = covariance[1:3, 1:3]
        phase_variance = float(gradient @ phase_covariance @ gradient)
        phases.append(phase)
        uncertainties.append(math.sqrt(max(phase_variance, 0.0)))
        contrasts.append(math.sqrt(amplitude_squared))
        row_scores.append(r_squared(row, fitted))
    wrapped = float(phase_sign) * np.asarray(phases, dtype=float)
    if phase_prior_detuning_mhz is None:
        unwrapped = np.unwrap(wrapped)
    else:
        prior_detuning = float(phase_prior_detuning_mhz)
        if not np.isfinite(prior_detuning):
            raise AnalysisError("phase_prior_detuning_mhz must be finite")
        prior_phase = 2.0 * np.pi * prior_detuning * np.asarray(
            native.outer,
            dtype=float,
        )
        # Resolve the 2*pi cycle at each sampled duration using the expected
        # target detuning, then unwrap only the residual. This is required when
        # |detuning| exceeds the phase-sampling Nyquist limit, and makes the
        # result prior-guided rather than a fully independent measurement.
        unwrapped = wrapped + 2.0 * np.pi * np.rint(
            (prior_phase - wrapped) / (2.0 * np.pi)
        )
        residual = np.unwrap(unwrapped - prior_phase)
        unwrapped = prior_phase + residual
    return CryoscopePhaseTrace(
        source_csv=native.source_csv,
        duration_us=np.asarray(native.outer, dtype=float),
        accumulated_phase_rad=unwrapped,
        phase_uncertainty_rad=np.asarray(uncertainties, dtype=float),
        contrast=np.asarray(contrasts, dtype=float),
        row_r_squared=np.asarray(row_scores, dtype=float),
    )


def cryoscope_frequency(
    phase_trace: CryoscopePhaseTrace,
    schedule: CryoscopeSchedule,
) -> CryoscopeFrequencyTrace:
    order = np.argsort(phase_trace.duration_us)
    duration = phase_trace.duration_us[order]
    # ``extract_cryoscope_phases`` has already unwrapped, potentially using a
    # model prior when the detuning aliases at the duration spacing. Repeating
    # a blind unwrap here would destroy those resolved 2*pi cycle counts.
    phase = np.asarray(phase_trace.accumulated_phase_rad[order], dtype=float)
    uncertainty = phase_trace.phase_uncertainty_rad[order]
    if schedule.minus_time_us.min() < duration.min() - 1e-12 or (
        schedule.plus_time_us.max() > duration.max() + 1e-12
    ):
        raise AnalysisError(
            "cryoscope acquisition does not cover every finite-difference endpoint"
        )
    phase_interpolator = PchipInterpolator(duration, phase, extrapolate=False)
    plus_phase = phase_interpolator(schedule.plus_time_us)
    minus_phase = phase_interpolator(schedule.minus_time_us)
    plus_sigma = np.interp(schedule.plus_time_us, duration, uncertainty)
    minus_sigma = np.interp(schedule.minus_time_us, duration, uncertainty)
    delta = schedule.delta_time_us
    detuning = (plus_phase - minus_phase) / (2.0 * np.pi * delta)
    sigma = np.hypot(plus_sigma, minus_sigma) / (2.0 * np.pi * delta)
    return CryoscopeFrequencyTrace(
        time_us=schedule.center_time_us.copy(),
        delta_time_us=delta.copy(),
        detuning_mhz=np.asarray(detuning, dtype=float),
        uncertainty_mhz=np.asarray(sigma, dtype=float),
    )


def write_cryoscope_schedule(
    path: Path,
    schedule: CryoscopeSchedule,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically persist the exact finite-difference acquisition schedule."""
    destination = Path(path).expanduser().resolve()
    document = {
        "schema_version": 1,
        "metadata": dict(metadata or {}),
        "sample_interval_ns": float(schedule.sample_interval_ns),
        "center_time_us": schedule.center_time_us.tolist(),
        "delta_time_us": schedule.delta_time_us.tolist(),
        "minus_time_us": schedule.minus_time_us.tolist(),
        "plus_time_us": schedule.plus_time_us.tolist(),
        "acquisition_time_us": schedule.acquisition_time_us.tolist(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def write_step_campaign_manifest(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist the exact adaptive spectroscopy rows from one campaign."""
    normalized = []
    for row in rows:
        try:
            csv_path = str(Path(row["csv_path"]).expanduser().resolve())
            time_us = float(row["probe_time_us"])
            center_mhz = float(row["predicted_center_mhz"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisError(f"invalid step-campaign row: {row}") from error
        normalized.append(
            {
                "csv_path": csv_path,
                "probe_time_us": time_us,
                "predicted_center_mhz": center_mhz,
            }
        )
    if not normalized:
        raise AnalysisError("step campaign manifest requires at least one row")
    destination = Path(path).expanduser().resolve()
    document = {
        "schema_version": 1,
        "quick_class": "FluxStepSpectroscopy",
        "metadata": dict(metadata or {}),
        "rows": normalized,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _read_step_campaign_document(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise AnalysisError(f"cannot read step campaign {source}: {error}") from error
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != 1
        or document.get("quick_class") != "FluxStepSpectroscopy"
        or not isinstance(document.get("rows"), list)
    ):
        raise AnalysisError("unsupported step-campaign manifest")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise AnalysisError("step-campaign metadata must be a mapping")
    return document


def read_step_campaign_manifest(path: Path) -> tuple:
    document = _read_step_campaign_document(path)
    paths = tuple(
        Path(row["csv_path"]).expanduser().resolve()
        for row in document["rows"]
        if isinstance(row, Mapping) and row.get("csv_path")
    )
    if not paths or any(not path.is_file() for path in paths):
        raise AnalysisError("step-campaign manifest contains missing CSV paths")
    return paths


def read_step_campaign_metadata(path: Path) -> Mapping[str, Any]:
    """Return acquisition coordinates and settings recorded with a campaign."""
    document = _read_step_campaign_document(path)
    return dict(document.get("metadata", {}))


def _read_cryoscope_document(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise AnalysisError(f"cannot read cryoscope schedule {source}: {error}") from error
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise AnalysisError("unsupported cryoscope schedule schema")
    if not isinstance(document.get("metadata", {}), Mapping):
        raise AnalysisError("cryoscope schedule metadata must be a mapping")
    return document


def read_cryoscope_schedule(path: Path) -> CryoscopeSchedule:
    document = _read_cryoscope_document(path)
    fields = {
        name: _vector(document.get(name), name)
        for name in (
            "center_time_us",
            "delta_time_us",
            "minus_time_us",
            "plus_time_us",
            "acquisition_time_us",
        )
    }
    count = fields["center_time_us"].size
    for name in ("delta_time_us", "minus_time_us", "plus_time_us"):
        if fields[name].size != count:
            raise AnalysisError("cryoscope schedule arrays have inconsistent lengths")
    return CryoscopeSchedule(
        center_time_us=fields["center_time_us"],
        delta_time_us=fields["delta_time_us"],
        minus_time_us=fields["minus_time_us"],
        plus_time_us=fields["plus_time_us"],
        acquisition_time_us=fields["acquisition_time_us"],
        sample_interval_ns=float(document["sample_interval_ns"]),
    )


def read_cryoscope_schedule_metadata(path: Path) -> Mapping[str, Any]:
    return dict(_read_cryoscope_document(path).get("metadata", {}))


def settling_metrics(
    time_us: Any,
    measured_frequency_mhz: Any,
    *,
    target_frequency_mhz: float,
    full_excursion_mhz: float,
    settle_after_us: float = 0.015,
    relative_tolerance: float = 0.001,
) -> Mapping[str, Any]:
    """Evaluate the paper's post-edge 0.1% settling criterion."""
    time, frequency = _matched_vectors(
        time_us,
        measured_frequency_mhz,
        "time_us",
        "measured_frequency_mhz",
    )
    excursion = abs(float(full_excursion_mhz))
    if excursion <= 0 or not np.isfinite(excursion):
        raise AnalysisError("full_excursion_mhz must be finite and non-zero")
    mask = time >= float(settle_after_us)
    if not np.any(mask):
        raise AnalysisError("no verification samples occur after settle_after_us")
    error = frequency - float(target_frequency_mhz)
    relative = np.abs(error) / excursion
    threshold = float(relative_tolerance)
    return {
        "settle_after_us": float(settle_after_us),
        "relative_tolerance": threshold,
        "maximum_absolute_error_mhz": float(np.max(np.abs(error[mask]))),
        "rms_error_mhz": float(np.sqrt(np.mean(error[mask] ** 2))),
        "maximum_relative_error": float(np.max(relative[mask])),
        "fraction_within_tolerance": float(np.mean(relative[mask] <= threshold)),
        "passes": bool(np.all(relative[mask] <= threshold)),
    }


def recommended_shots_per_phase(
    delta_time_us: Any,
    *,
    target_frequency_sigma_mhz: float,
    phase_count: int = 4,
    normalized_contrast: float = 0.8,
    minimum_shots: int = 64,
    maximum_shots: int = 65_536,
) -> np.ndarray:
    """Allocate shots from the sinusoidal phase-fit Fisher scaling.

    Four orthogonal phases are information-complete for the sinusoidal model;
    the paper's 16 phases remain useful as a systematic-error cross-check.
    """
    delta = _vector(delta_time_us, "delta_time_us")
    sigma = float(target_frequency_sigma_mhz)
    contrast = float(normalized_contrast)
    count = int(phase_count)
    if sigma <= 0 or not np.isfinite(sigma):
        raise AnalysisError("target_frequency_sigma_mhz must be positive")
    if contrast <= 0 or not np.isfinite(contrast):
        raise AnalysisError("normalized_contrast must be positive")
    if count < 4:
        raise AnalysisError("phase_count must be at least four")
    estimate = 1.0 / (
        contrast**2 * count * np.pi**2 * delta**2 * sigma**2
    )
    shots = np.ceil(estimate).astype(int)
    return np.clip(shots, int(minimum_shots), int(maximum_shots))


def fit_forward_fir(
    command_delta_z: Any,
    *,
    sample_interval_ns: float,
    measured_time_us: Any,
    measured_frequency_mhz: Any,
    frequency_model: Callable[[np.ndarray], Any],
    baseline_z: float,
    coefficient_count: int = 120,
    integration_weight: Optional[Any] = None,
    energy_regularization: float = 1e-6,
    tail_regularization: float = 1e-4,
    tail_growth: float = 6.0,
    dc_regularization: float = 1e-2,
    maximum_evaluations: int = 3000,
) -> ForwardFIRFit:
    """Fit a causal FIR forward model through the nonlinear qubit model."""
    command = _vector(command_delta_z, "command_delta_z", minimum=3)
    measured_time, measured_frequency, sort_order = _matched_vectors_with_order(
        measured_time_us,
        measured_frequency_mhz,
        "measured_time_us",
        "measured_frequency_mhz",
    )
    interval_ns = float(sample_interval_ns)
    if not np.isfinite(interval_ns) or interval_ns <= 0:
        raise AnalysisError("sample_interval_ns must be finite and positive")
    interval_us = interval_ns / 1000.0
    command_time = np.arange(command.size, dtype=float) * interval_us
    if measured_time[0] < command_time[0] or measured_time[-1] > command_time[-1]:
        raise AnalysisError("measured times lie outside the command waveform")
    length = int(coefficient_count)
    if length < 2 or length > command.size:
        raise AnalysisError("coefficient_count must be between 2 and command length")
    if integration_weight is None:
        weights = np.ones_like(measured_time)
    else:
        weights = _vector(integration_weight, "integration_weight")
        if weights.shape != measured_time.shape or np.any(weights <= 0):
            raise AnalysisError(
                "integration_weight must be positive and match measured times"
            )
        weights = weights[sort_order]
        weights /= float(np.median(weights))
    regularizers = (
        float(energy_regularization),
        float(tail_regularization),
        float(dc_regularization),
    )
    if any(not np.isfinite(value) or value < 0 for value in regularizers):
        raise AnalysisError("FIR regularization values must be finite and non-negative")
    tail_scale = np.exp(
        float(tail_growth) * np.arange(length) / max(length - 1, 1)
    )

    def predict(coefficients):
        actual_delta = np.convolve(command, coefficients, mode="full")[: command.size]
        actual_z = float(baseline_z) + actual_delta
        frequency = np.asarray(frequency_model(actual_z), dtype=float)
        if frequency.shape != command.shape or not np.all(np.isfinite(frequency)):
            raise AnalysisError(
                "frequency_model must return one finite value per command sample"
            )
        return np.interp(measured_time, command_time, frequency)

    def objective(coefficients):
        # Appendix I defines ``w(t)`` as the coefficient of squared error, so
        # least-squares residuals carry sqrt(w), not w itself.
        components = [
            np.sqrt(weights) * (predict(coefficients) - measured_frequency)
        ]
        if energy_regularization:
            components.append(math.sqrt(energy_regularization) * coefficients)
        if tail_regularization:
            components.append(
                np.sqrt(tail_regularization * tail_scale) * coefficients
            )
        if dc_regularization:
            components.append(
                np.asarray(
                    [math.sqrt(dc_regularization) * (np.sum(coefficients) - 1.0)]
                )
            )
        return np.concatenate(components)

    initial = np.zeros(length, dtype=float)
    initial[0] = 1.0
    result = least_squares(
        objective,
        initial,
        loss="linear",
        max_nfev=max(int(maximum_evaluations), 1),
    )
    if not result.success:
        raise AnalysisError(f"forward FIR fit did not converge: {result.message}")
    fitted = predict(result.x)
    residual = measured_frequency - fitted
    return ForwardFIRFit(
        coefficients=np.asarray(result.x, dtype=float),
        sample_interval_ns=interval_ns,
        measured_time_us=measured_time,
        measured_frequency_mhz=measured_frequency,
        fitted_frequency_mhz=fitted,
        statistics={
            "r_squared": r_squared(measured_frequency, fitted),
            "rmse_mhz": float(np.sqrt(np.mean(residual**2))),
            "maximum_absolute_residual_mhz": float(np.max(np.abs(residual))),
            "coefficient_sum": float(np.sum(result.x)),
            "optimizer_evaluations": int(result.nfev),
        },
        regularization={
            "energy": float(energy_regularization),
            "tail": float(tail_regularization),
            "tail_growth": float(tail_growth),
            "dc": float(dc_regularization),
        },
    )


def design_inverse_fir(
    forward_coefficients: Any,
    *,
    sample_interval_ns: float,
    inverse_length: Optional[int] = None,
    gaussian_sigma_ns: float = 0.75,
    derivative_regularization: float = 1e-3,
    latency_samples: Optional[int] = None,
) -> InverseFIRDesign:
    """Solve the paper's regularized FIR inverse problem as linear algebra.

    When ``latency_samples`` is omitted, a causal latency is selected by
    minimizing the same data-plus-Sobolev objective.  This removes a manual
    tuning step while leaving the chosen latency explicit in the result.
    """
    forward = _vector(forward_coefficients, "forward_coefficients", minimum=2)
    interval_ns = float(sample_interval_ns)
    sigma_ns = float(gaussian_sigma_ns)
    regularization = float(derivative_regularization)
    if not np.isfinite(interval_ns) or interval_ns <= 0:
        raise AnalysisError("sample_interval_ns must be finite and positive")
    if not np.isfinite(sigma_ns) or sigma_ns <= 0:
        raise AnalysisError("gaussian_sigma_ns must be finite and positive")
    if not np.isfinite(regularization) or regularization < 0:
        raise AnalysisError("derivative_regularization cannot be negative")
    length = int(inverse_length if inverse_length is not None else forward.size)
    if length < 2:
        raise AnalysisError("inverse_length must be at least two")
    convolution = convolution_matrix(forward, length, mode="full")
    output_length = convolution.shape[0]
    derivative = np.diff(np.eye(length), axis=0)
    augmented_matrix = np.vstack(
        (
            convolution,
            math.sqrt(regularization) * derivative,
            math.sqrt(np.finfo(float).eps) * np.eye(length),
        )
    )
    if latency_samples is None:
        maximum_latency = min(output_length - 1, forward.size + length // 2)
        latencies = range(maximum_latency + 1)
    else:
        latency = int(latency_samples)
        if latency < 0 or latency >= output_length:
            raise AnalysisError("latency_samples lies outside the convolution support")
        latencies = (latency,)
    sample_index = np.arange(output_length, dtype=float)
    sigma_samples = sigma_ns / interval_ns
    best = None
    for latency in latencies:
        target = np.exp(-0.5 * ((sample_index - latency) / sigma_samples) ** 2)
        target /= float(np.sum(target))
        right = np.concatenate(
            (target, np.zeros(derivative.shape[0] + length, dtype=float))
        )
        coefficients, *_ = np.linalg.lstsq(augmented_matrix, right, rcond=None)
        realized = convolution @ coefficients
        objective = float(
            np.sum((target - realized) ** 2)
            + regularization * np.sum((derivative @ coefficients) ** 2)
        )
        peak = float(np.max(np.abs(coefficients)))
        tie_break = (objective, peak, latency)
        if best is None or tie_break < best[0]:
            best = (tie_break, latency, target, coefficients, realized)
    assert best is not None
    _score, latency, target, coefficients, realized = best
    residual = target - realized
    return InverseFIRDesign(
        coefficients=np.asarray(coefficients, dtype=float),
        target=np.asarray(target, dtype=float),
        realized=np.asarray(realized, dtype=float),
        sample_interval_ns=interval_ns,
        gaussian_sigma_ns=sigma_ns,
        latency_samples=int(latency),
        regularization=regularization,
        statistics={
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "maximum_absolute_residual": float(np.max(np.abs(residual))),
            "coefficient_sum": float(np.sum(coefficients)),
            "maximum_absolute_coefficient": float(
                np.max(np.abs(coefficients))
            ),
        },
    )


def apply_predistortion(
    waveform: Any,
    *,
    iir: Optional[IIRInverseDesign] = None,
    fir: Optional[InverseFIRDesign] = None,
    preserve_length: bool = True,
) -> np.ndarray:
    values = _vector(waveform, "waveform")
    result = iir.apply(values) if iir is not None else values.copy()
    if fir is not None:
        result = fir.apply(result)
    return result[: values.size] if preserve_length else result


def validate_waveform(
    waveform: Any,
    *,
    sample_interval_ns: float,
    full_scale: float = 2.5,
    maximum_fraction_of_full_scale: float = 0.24,
    maximum_step_per_ns: Optional[float] = None,
) -> WaveformCheck:
    values = np.asarray(waveform, dtype=float).ravel()
    finite = bool(values.size and np.all(np.isfinite(values)))
    maximum = float(np.max(np.abs(values))) if values.size else float("inf")
    full = abs(float(full_scale))
    interval = float(sample_interval_ns)
    if full <= 0 or interval <= 0:
        raise AnalysisError("full scale and sample interval must be positive")
    fraction = maximum / full
    slew = (
        float(np.max(np.abs(np.diff(values)))) / interval
        if values.size > 1 and finite
        else 0.0
    )
    slew_pass = (
        None
        if maximum_step_per_ns is None
        else bool(slew <= float(maximum_step_per_ns))
    )
    return WaveformCheck(
        finite=finite,
        maximum_absolute=maximum,
        maximum_fraction_of_full_scale=fraction,
        maximum_step_per_ns=slew,
        amplitude_pass=bool(finite and fraction <= maximum_fraction_of_full_scale),
        slew_pass=slew_pass,
    )


def write_filter_bundle(
    path: Path,
    *,
    step_fit: Optional[StepResponseFit] = None,
    iir: Optional[IIRInverseDesign] = None,
    forward_fir: Optional[ForwardFIRFit] = None,
    inverse_fir: Optional[InverseFIRDesign] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically write an inspection-ready JSON filter bundle."""
    destination = Path(path).expanduser().resolve()
    document = {
        "schema_version": 1,
        "paper_doi": PAPER_DOI,
        "status": "candidate_not_applied",
        "metadata": dict(metadata or {}),
    }
    if step_fit is not None:
        document["step_response"] = step_fit.as_dict()
    if iir is not None:
        document["iir_inverse"] = iir.as_dict()
    if forward_fir is not None:
        document["forward_fir"] = forward_fir.as_dict()
    if inverse_fir is not None:
        document["inverse_fir"] = inverse_fir.as_dict()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def read_filter_bundle(path: Path) -> Mapping[str, Any]:
    """Read and minimally validate a candidate filter JSON document."""
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise AnalysisError(f"cannot read filter bundle {source}: {error}") from error
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise AnalysisError("unsupported filter-bundle schema")
    if not any(name in document for name in ("iir_inverse", "inverse_fir")):
        raise AnalysisError("filter bundle contains no inverse filter")
    return dict(document)


def apply_filter_bundles(
    waveform: Any,
    bundles: Sequence[Any],
    *,
    preserve_length: bool = True,
) -> np.ndarray:
    """Cascade exported IIR/FIR candidates without implying hardware upload."""
    original = _vector(waveform, "waveform")
    result = original.copy()
    sample_interval_ns = None
    for item in bundles:
        document = (
            read_filter_bundle(item)
            if isinstance(item, (str, Path))
            else item
        )
        if not isinstance(document, Mapping):
            raise AnalysisError("each filter bundle must be a path or mapping")
        iir = document.get("iir_inverse")
        if isinstance(iir, Mapping):
            interval = float(iir.get("sample_interval_ns"))
            if sample_interval_ns is not None and not np.isclose(
                interval, sample_interval_ns, rtol=1e-9, atol=1e-12
            ):
                raise AnalysisError("filter bundles use different sample intervals")
            sample_interval_ns = interval
            sos = np.asarray(iir.get("sos"), dtype=float)
            if sos.ndim != 2 or sos.shape[1] != 6 or not np.all(np.isfinite(sos)):
                raise AnalysisError("filter bundle has invalid IIR SOS coefficients")
            result = sosfilt(sos, result)
        fir = document.get("inverse_fir")
        if isinstance(fir, Mapping):
            interval = float(fir.get("sample_interval_ns"))
            if sample_interval_ns is not None and not np.isclose(
                interval, sample_interval_ns, rtol=1e-9, atol=1e-12
            ):
                raise AnalysisError("filter bundles use different sample intervals")
            sample_interval_ns = interval
            coefficients = _vector(
                fir.get("coefficients"),
                "inverse_fir.coefficients",
                minimum=2,
            )
            result = np.convolve(result, coefficients, mode="full")
    return result[: original.size] if preserve_length else result


def upload_predistorted_waveform(
    uploader: Any,
    waveform: Any,
    *,
    channel: int,
    sample_interval_ns: float,
    name: str,
    dry_run: bool = True,
) -> Mapping[str, Any]:
    """Guarded integration point for a site-verified waveform uploader.

    The callback must explicitly advertise ``supports_arbitrary_waveforms``
    and implement ``upload_z_waveform``.  No local Quick class currently does
    so; this function makes that missing hardware gate executable and testable.
    """
    values = _vector(waveform, "waveform")
    capable = bool(getattr(uploader, "supports_arbitrary_waveforms", False))
    callback = getattr(uploader, "upload_z_waveform", None)
    if not capable or not callable(callback):
        raise ConfigError(
            "no verified arbitrary-waveform uploader is installed; export and "
            "inspect the candidate filter, then provide a site adapter that "
            "advertises supports_arbitrary_waveforms"
        )
    manifest = {
        "name": str(name),
        "channel": int(channel),
        "sample_interval_ns": float(sample_interval_ns),
        "samples": int(values.size),
        "maximum_absolute": float(np.max(np.abs(values))),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return manifest
    callback(
        name=str(name),
        channel=int(channel),
        samples=values,
        sample_interval_ns=float(sample_interval_ns),
    )
    return manifest
