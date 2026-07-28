"""Compact analysis primitives used by the explicit experiment modules."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .data import AnalysisResult
from .errors import AnalysisError


def _finite_xy(x: Any, y: Any, minimum: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y).ravel()
    if x_array.size != y_array.size:
        raise AnalysisError("fit axes and signals must have equal lengths")
    mask = np.isfinite(x_array)
    if np.iscomplexobj(y_array):
        mask &= np.isfinite(y_array.real) & np.isfinite(y_array.imag)
    else:
        y_array = y_array.astype(float)
        mask &= np.isfinite(y_array)
    x_array = x_array[mask]
    y_array = y_array[mask]
    if x_array.size < minimum:
        raise AnalysisError(f"fit requires at least {minimum} finite points")
    order = np.argsort(x_array)
    return x_array[order], y_array[order]


def rotate_iq(iq: Any) -> Tuple[np.ndarray, dict]:
    values = np.asarray(iq, dtype=complex).ravel()
    matrix = np.column_stack((values.real, values.imag))
    centered = matrix - np.mean(matrix, axis=0)
    if centered.shape[0] < 2:
        return centered[:, 0], {"angle_rad": 0.0}
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    direction = right[0]
    projected = centered @ direction
    angle = math.atan2(float(direction[1]), float(direction[0]))
    return projected, {"angle_rad": angle}


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(observed) - np.asarray(predicted)
    rss = float(np.sum(np.abs(residual) ** 2))
    centered = np.asarray(observed) - np.mean(observed)
    tss = float(np.sum(np.abs(centered) ** 2))
    return 1.0 - rss / tss if tss > 0 else float("nan")


def fit_spectral_feature(x: Any, iq: Any) -> AnalysisResult:
    x_array, iq_array = _finite_xy(x, np.asarray(iq, dtype=complex), minimum=7)
    signal, rotation = rotate_iq(iq_array)

    def model(values, offset, slope, amplitude, center, hwhm):
        return (
            offset
            + slope * (values - center)
            + amplitude / (1.0 + ((values - center) / hwhm) ** 2)
        )

    span = float(np.ptp(x_array))
    if span <= 0:
        raise AnalysisError("spectroscopy axis must span more than one value")
    baseline = float(np.median(signal))
    deviations = signal - baseline
    peak_index = int(np.argmax(np.abs(deviations)))
    amplitude = float(deviations[peak_index])
    step = float(np.median(np.diff(x_array)))
    p0 = [baseline, 0.0, amplitude, float(x_array[peak_index]), max(abs(step), span / 20)]
    bounds = (
        [-np.inf, -np.inf, -np.inf, float(x_array[0]), max(abs(step) / 10, 1e-12)],
        [np.inf, np.inf, np.inf, float(x_array[-1]), span * 2],
    )
    try:
        parameters, covariance = curve_fit(
            model,
            x_array,
            signal,
            p0=p0,
            bounds=bounds,
            maxfev=30000,
        )
    except Exception as error:
        raise AnalysisError(f"spectral fit failed: {error}") from error
    predicted = model(x_array, *parameters)
    uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0))
    center = float(parameters[3])
    hwhm = abs(float(parameters[4]))
    center_uncertainty = float(uncertainty[3])
    r2 = _r_squared(signal, predicted)
    edge_distance = min(center - x_array[0], x_array[-1] - center)
    valid = bool(
        np.isfinite(r2)
        and hwhm > 0
        and edge_distance >= 2 * abs(step)
    )
    return AnalysisResult(
        model="signed_lorentzian_with_slope",
        values={
            "center": center,
            "center_uncertainty": center_uncertainty,
            "hwhm": hwhm,
            "fwhm": 2 * hwhm,
            "amplitude": float(parameters[2]),
            "rotation": rotation,
        },
        quality={"r_squared": r2, "edge_distance": float(edge_distance)},
        valid=valid,
    )


def _dominant_frequency(x: np.ndarray, signal: np.ndarray) -> float:
    steps = np.diff(x)
    step = float(np.median(steps))
    if step <= 0 or not np.allclose(steps, step, rtol=0.05, atol=0):
        raise AnalysisError("oscillation fit requires a uniformly spaced axis")
    spectrum = np.fft.rfft(signal - np.mean(signal))
    frequencies = np.fft.rfftfreq(signal.size, d=step)
    if frequencies.size < 2:
        raise AnalysisError("oscillation axis is too short")
    index = int(np.argmax(np.abs(spectrum[1:])) + 1)
    return float(frequencies[index])


def fit_damped_oscillation(
    x: Any,
    iq: Any,
    *,
    frequency_hint: Optional[float] = None,
    model_name: str = "damped_cosine",
) -> AnalysisResult:
    x_array, iq_array = _finite_xy(x, np.asarray(iq, dtype=complex), minimum=10)
    signal, rotation = rotate_iq(iq_array)
    span = float(np.ptp(x_array))
    if span <= 0:
        raise AnalysisError("oscillation axis must span more than one value")
    frequency = (
        float(frequency_hint)
        if frequency_hint is not None
        else _dominant_frequency(x_array, signal)
    )

    def model(values, offset, amplitude, decay, fitted_frequency, phase, slope):
        shifted = values - values[0]
        return (
            offset
            + slope * shifted
            + amplitude
            * np.exp(-shifted / decay)
            * np.cos(2 * np.pi * fitted_frequency * shifted + phase)
        )

    amplitude = max(float(np.ptp(signal)) / 2, np.finfo(float).eps)
    p0 = [
        float(np.mean(signal)),
        amplitude,
        max(span, np.finfo(float).eps),
        max(frequency, 1 / (span * 10)),
        0.0,
        0.0,
    ]
    minimum_frequency = 1 / (span * 20)
    maximum_frequency = 0.5 / float(np.median(np.diff(x_array)))
    bounds = (
        [-np.inf, -np.inf, span / 1000, minimum_frequency, -4 * np.pi, -np.inf],
        [np.inf, np.inf, span * 1000, maximum_frequency, 4 * np.pi, np.inf],
    )
    try:
        parameters, covariance = curve_fit(
            model,
            x_array,
            signal,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
        )
    except Exception as error:
        raise AnalysisError(f"oscillation fit failed: {error}") from error
    predicted = model(x_array, *parameters)
    uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0))
    fitted_frequency = float(parameters[3])
    period = 1.0 / fitted_frequency
    return AnalysisResult(
        model=model_name,
        values={
            "frequency": fitted_frequency,
            "frequency_uncertainty": float(uncertainty[3]),
            "period": period,
            "half_period": period / 2,
            "decay": float(parameters[2]),
            "decay_uncertainty": float(uncertainty[2]),
            "phase": float(parameters[4]),
            "rotation": rotation,
        },
        quality={"r_squared": _r_squared(signal, predicted)},
        valid=bool(np.all(np.isfinite(parameters))),
    )


def fit_exponential_decay(x: Any, iq: Any) -> AnalysisResult:
    x_array, iq_array = _finite_xy(x, np.asarray(iq, dtype=complex), minimum=7)
    signal, rotation = rotate_iq(iq_array)
    shifted = x_array - x_array[0]
    span = float(np.ptp(x_array))

    def model(values, offset, amplitude, decay):
        return offset + amplitude * np.exp(-values / decay)

    p0 = [
        float(signal[-1]),
        float(signal[0] - signal[-1]),
        max(span / 3, np.finfo(float).eps),
    ]
    try:
        parameters, covariance = curve_fit(
            model,
            shifted,
            signal,
            p0=p0,
            bounds=([-np.inf, -np.inf, span / 1000], [np.inf, np.inf, span * 1000]),
            maxfev=30000,
        )
    except Exception as error:
        raise AnalysisError(f"T1 fit failed: {error}") from error
    predicted = model(shifted, *parameters)
    uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0))
    decay = float(parameters[2])
    return AnalysisResult(
        model="bounded_exponential",
        values={
            "decay": decay,
            "decay_uncertainty": float(uncertainty[2]),
            "rotation": rotation,
        },
        quality={"r_squared": _r_squared(signal, predicted), "span_over_decay": span / decay},
        valid=bool(decay > 0 and np.all(np.isfinite(parameters))),
    )


def fit_iq_classifier(
    i_ground: Any,
    q_ground: Any,
    i_excited: Any,
    q_excited: Any,
) -> AnalysisResult:
    ground = np.asarray(i_ground, dtype=float).ravel() + 1j * np.asarray(
        q_ground, dtype=float
    ).ravel()
    excited = np.asarray(i_excited, dtype=float).ravel() + 1j * np.asarray(
        q_excited, dtype=float
    ).ravel()
    if ground.size < 2 or excited.size < 2:
        raise AnalysisError("IQ classification requires at least two shots per state")
    center_ground = np.mean(ground)
    center_excited = np.mean(excited)
    direction = center_excited - center_ground
    if abs(direction) == 0:
        raise AnalysisError("ground and excited IQ centroids coincide")
    unit = direction / abs(direction)
    projected_ground = np.real((ground - center_ground) * np.conj(unit))
    projected_excited = np.real((excited - center_ground) * np.conj(unit))
    threshold = float(abs(direction) / 2)
    correct_ground = np.mean(projected_ground < threshold)
    correct_excited = np.mean(projected_excited >= threshold)
    fidelity = float((correct_ground + correct_excited) / 2)
    return AnalysisResult(
        model="centroid_axis_threshold",
        values={
            "threshold": threshold,
            "angle_rad": float(np.angle(unit)),
            "ground_center": [float(center_ground.real), float(center_ground.imag)],
            "excited_center": [float(center_excited.real), float(center_excited.imag)],
            "fidelity": fidelity,
        },
        quality={
            "ground_accuracy": float(correct_ground),
            "excited_accuracy": float(correct_excited),
            "shots_ground": int(ground.size),
            "shots_excited": int(excited.size),
        },
        valid=bool(np.isfinite(fidelity)),
        recommendation={"defaults.r_threshold": threshold},
    )


def feature_extremum(x: Sequence[float], y: Sequence[float], model: str) -> AnalysisResult:
    x_array, y_array = _finite_xy(x, y, minimum=3)
    centered = np.abs(y_array - np.median(y_array))
    index = int(np.argmax(centered))
    return AnalysisResult(
        model=model,
        values={"feature": float(x_array[index])},
        quality={"contrast": float(centered[index])},
        valid=bool(index not in (0, x_array.size - 1)),
    )

