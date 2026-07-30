"""Shared statistical diagnostics for native Quick fitters."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import AnalysisError


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return the ordinary coefficient of determination."""
    measured = np.asarray(observed)
    fitted = np.asarray(predicted)
    if measured.shape != fitted.shape:
        raise AnalysisError("observed and predicted arrays must have equal shapes")
    finite = np.isfinite(measured) & np.isfinite(fitted)
    if np.iscomplexobj(measured) or np.iscomplexobj(fitted):
        finite = (
            np.isfinite(measured.real)
            & np.isfinite(measured.imag)
            & np.isfinite(fitted.real)
            & np.isfinite(fitted.imag)
        )
    measured = measured[finite]
    fitted = fitted[finite]
    if measured.size == 0:
        return float("nan")
    residual_power = float(np.sum(np.abs(measured - fitted) ** 2))
    centered_power = float(np.sum(np.abs(measured - np.mean(measured)) ** 2))
    return (
        float(1.0 - residual_power / centered_power)
        if centered_power > 0
        else float("nan")
    )


def _named_values(values: Any) -> tuple:
    if isinstance(values, Mapping):
        names = tuple(values)
        array = np.asarray([values[name] for name in names], dtype=float)
        return names, array
    array = np.asarray(values, dtype=float).ravel()
    return tuple(range(array.size)), array


def _bound_values(bounds: Any, names: Sequence[Any]) -> np.ndarray:
    if isinstance(bounds, Mapping):
        return np.asarray([bounds[name] for name in names], dtype=float)
    array = np.asarray(bounds, dtype=float).ravel()
    if array.size != len(names):
        raise AnalysisError("parameter and bound arrays must have equal lengths")
    return array


def pinned_parameters(
    parameters: Any,
    lower: Any,
    upper: Any,
    *,
    rtol: float = 1e-4,
) -> list:
    """Return parameter names/indices numerically pinned to either bound."""
    names, values = _named_values(parameters)
    lower_values = _bound_values(lower, names)
    upper_values = _bound_values(upper, names)
    if rtol < 0:
        raise AnalysisError("rtol must be non-negative")
    scale = np.maximum(
        np.maximum(np.abs(lower_values), np.abs(upper_values)),
        1.0,
    )
    tolerance = float(rtol) * scale
    pinned = (
        np.isfinite(lower_values) & (np.abs(values - lower_values) <= tolerance)
    ) | (
        np.isfinite(upper_values) & (np.abs(values - upper_values) <= tolerance)
    )
    return [name for name, is_pinned in zip(names, pinned) if is_pinned]


def bic(rss: float, n_points: int, n_parameters: int) -> float:
    """Bayesian information criterion for a least-squares model."""
    n = int(n_points)
    k = int(n_parameters)
    if n < 1 or k < 0:
        raise AnalysisError("BIC requires n_points >= 1 and n_parameters >= 0")
    residual = float(rss)
    if not np.isfinite(residual) or residual < 0:
        raise AnalysisError("BIC residual sum of squares must be finite and non-negative")
    floor = np.finfo(float).tiny * n
    return float(n * math.log(max(residual, floor) / n) + k * math.log(n))


def _coerce_fit_result(result: Any, n_points: int) -> tuple:
    predicted = None
    parameters = result
    if isinstance(result, tuple) and len(result) == 2:
        possible_prediction = np.asarray(result[1])
        if possible_prediction.size == n_points:
            parameters = result[0]
            predicted = possible_prediction.reshape(n_points)
    if isinstance(parameters, Mapping):
        parameter_names = tuple(parameters)
        parameter_values = np.asarray(
            [parameters[name] for name in parameter_names],
            dtype=float,
        )
    else:
        parameter_names = None
        parameter_values = np.asarray(parameters, dtype=float).ravel()
    if parameter_values.size == 0 or not np.all(np.isfinite(parameter_values)):
        raise AnalysisError("bootstrap fit returned no finite parameters")
    return parameter_names, parameter_values, predicted


def bootstrap_1d(
    x: Any,
    y: Any,
    fit_function,
    n_resamples: int = 200,
    seed: int = 0,
) -> dict:
    """Bootstrap one-dimensional fit parameters.

    ``fit_function`` may return either parameters or ``(parameters,
    predicted_y)``. The latter enables residual resampling. When predictions
    are unavailable, the function falls back to paired resampling and records
    that method in the result.
    """
    x_values = np.asarray(x, dtype=float).ravel()
    y_values = np.asarray(y).ravel()
    if x_values.size != y_values.size or x_values.size < 3:
        raise AnalysisError("bootstrap requires equal x/y arrays with at least 3 points")
    finite = np.isfinite(x_values)
    if np.iscomplexobj(y_values):
        finite &= np.isfinite(y_values.real) & np.isfinite(y_values.imag)
    else:
        finite &= np.isfinite(y_values.astype(float))
    x_values = x_values[finite]
    y_values = y_values[finite]
    count = int(n_resamples)
    if count < 1:
        raise AnalysisError("n_resamples must be at least 1")

    names, initial, predicted = _coerce_fit_result(
        fit_function(x_values, y_values),
        x_values.size,
    )
    rng = np.random.default_rng(int(seed))
    samples = []
    method = "residual" if predicted is not None else "paired"
    residual = y_values - predicted if predicted is not None else None
    for _ in range(count):
        try:
            if residual is not None:
                sampled = predicted + rng.choice(
                    residual,
                    size=residual.size,
                    replace=True,
                )
                candidate = fit_function(x_values, sampled)
            else:
                indices = rng.integers(0, x_values.size, size=x_values.size)
                order = np.argsort(x_values[indices], kind="stable")
                candidate = fit_function(
                    x_values[indices][order],
                    y_values[indices][order],
                )
            candidate_names, values, _ = _coerce_fit_result(
                candidate,
                x_values.size,
            )
            if candidate_names != names or values.shape != initial.shape:
                continue
            samples.append(values)
        except (AnalysisError, RuntimeError, ValueError, FloatingPointError):
            continue
    if not samples:
        raise AnalysisError("all bootstrap refits failed")
    sample_array = np.asarray(samples, dtype=float)
    return {
        "samples": sample_array,
        "ci_low": np.percentile(sample_array, 2.5, axis=0),
        "ci_high": np.percentile(sample_array, 97.5, axis=0),
        "median": np.median(sample_array, axis=0),
        "parameter_names": names,
        "method": method,
        "successful_resamples": int(sample_array.shape[0]),
        "requested_resamples": count,
    }


def residual_ripple_fraction(x: Any, residual: Any) -> float:
    """Return the largest non-DC Fourier mode's share of residual power."""
    x_values = np.asarray(x, dtype=float).ravel()
    values = np.asarray(residual, dtype=float).ravel()
    if x_values.size != values.size or values.size < 4:
        return 0.0
    finite = np.isfinite(x_values) & np.isfinite(values)
    x_values, values = x_values[finite], values[finite]
    if values.size < 4:
        return 0.0
    order = np.argsort(x_values)
    x_values, values = x_values[order], values[order]
    design = np.column_stack((np.ones(values.size), x_values - np.mean(x_values)))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    detrended = values - design @ coefficients
    spectrum = np.fft.rfft(detrended)
    power = np.abs(spectrum[1:]) ** 2
    total = float(np.sum(power))
    return float(np.max(power) / total) if total > 0 else 0.0


def oriented_rotate_iq(iq: Any) -> tuple:
    """Project IQ onto its principal axis with a deterministic orientation."""
    values = np.asarray(iq, dtype=complex).ravel()
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.all(finite):
        values = values[finite]
    matrix = np.column_stack((values.real, values.imag))
    centered = matrix - np.mean(matrix, axis=0) if matrix.size else matrix
    if centered.shape[0] < 2:
        return centered[:, 0], {"angle_rad": 0.0, "flipped": False}
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    direction = right[0].copy()
    projected = centered @ direction
    deviation = projected - np.median(projected)
    extreme = int(np.argmax(np.abs(deviation)))
    flipped = bool(deviation[extreme] < 0)
    if flipped:
        direction *= -1
        projected *= -1
    angle = math.atan2(float(direction[1]), float(direction[0]))
    angle = (angle + math.pi / 2.0) % math.pi - math.pi / 2.0
    return projected, {"angle_rad": float(angle), "flipped": flipped}


def pi_consistency(fit: Any) -> dict:
    """Cross-check a Rabi π recommendation against the measured trace."""
    x = np.asarray(fit.x, dtype=float)
    measured = np.asarray(fit.projected, dtype=float)
    pi_value = float(fit.parameters["pi_value"])
    half_period = float(fit.parameters["half_period"])
    amplitude = abs(float(fit.parameters["amplitude"]))
    if x.size < 2 or amplitude <= 0 or not x[0] <= pi_value <= x[-1]:
        contrast = float("nan")
    else:
        at_pi = float(np.interp(pi_value, x, measured))
        at_zero = float(np.interp(0.0, x, measured))
        contrast = abs(at_pi - at_zero) / (2.0 * amplitude)
    ratio = pi_value / half_period if half_period > 0 else float("nan")
    nearest = int(round(ratio)) if np.isfinite(ratio) else 0
    return {
        "measured_contrast_at_pi": float(contrast),
        "pi_to_half_period_ratio": float(ratio),
        "nearest_odd_multiple": nearest,
        "odd_multiple_consistent": bool(
            nearest % 2 == 1 and abs(ratio - nearest) <= 0.15
        ) if np.isfinite(ratio) else False,
    }
