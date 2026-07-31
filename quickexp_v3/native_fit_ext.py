"""Extended native decay fitting, including the previously missing Echo path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from .errors import AnalysisError
from .fit_calibration import annotate_forced_write, write_calibration_records
from .fit_stats import bic, bootstrap_1d, pinned_parameters, r_squared
from .native_fit import _fit_signal, load_native_trace
from .trace_qc import qc_trace
from .util import utc_now


def stretched_exponential(
    time_us: Any,
    offset: float,
    amplitude: float,
    decay_us: float,
    exponent: float,
    *,
    time_zero_us: float = 0.0,
) -> np.ndarray:
    shifted = np.maximum(np.asarray(time_us, dtype=float) - time_zero_us, 0.0)
    return float(offset) + float(amplitude) * np.exp(
        -(shifted / float(decay_us)) ** float(exponent)
    )


@dataclass(frozen=True)
class DecayFit:
    source_csv: Path
    source_yml: Path
    quick_class: str
    signal: str
    signal_label: str
    time_us: np.ndarray
    iq: np.ndarray
    measured: np.ndarray
    fitted: np.ndarray
    parameters: Mapping[str, Any]
    statistics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def decay_us(self) -> float:
        return float(self.parameters["decay_us"])

    def passes(
        self,
        *,
        minimum_r_squared: float = 0.70,
        minimum_span_over_t: float = 0.75,
        maximum_relative_t_uncertainty: float = 0.25,
    ) -> bool:
        return bool(
            self.statistics["r_squared"] >= minimum_r_squared
            and self.statistics["span_over_decay"] >= minimum_span_over_t
            and self.statistics["relative_decay_uncertainty"]
            <= maximum_relative_t_uncertainty
            and not self.statistics.get("pinned_parameters")
            and not self.statistics.get("n_pinned", False)
        )


def _fit_decay_arrays(
    time_us: np.ndarray,
    measured: np.ndarray,
    *,
    free_exponent: bool,
) -> tuple:
    shifted = time_us - time_us[0]
    span = float(np.ptp(shifted))
    step = float(np.median(np.diff(shifted)))
    offset_guess = float(np.median(measured[-max(2, measured.size // 10) :]))
    amplitude_guess = float(measured[0] - offset_guess)
    if free_exponent:
        initial = np.asarray([offset_guess, amplitude_guess, max(span / 3.0, step), 1.2])
        lower = np.asarray([-np.inf, -np.inf, max(step / 10.0, span / 10000.0), 0.7])
        upper = np.asarray([np.inf, np.inf, span * 1000.0, 3.0])

        def prediction(parameters):
            return stretched_exponential(
                time_us,
                parameters[0],
                parameters[1],
                parameters[2],
                parameters[3],
                time_zero_us=float(time_us[0]),
            )
    else:
        initial = np.asarray([offset_guess, amplitude_guess, max(span / 3.0, step)])
        lower = np.asarray([-np.inf, -np.inf, max(step / 10.0, span / 10000.0)])
        upper = np.asarray([np.inf, np.inf, span * 1000.0])

        def prediction(parameters):
            return stretched_exponential(
                time_us,
                parameters[0],
                parameters[1],
                parameters[2],
                1.0,
                time_zero_us=float(time_us[0]),
            )
    result = least_squares(
        lambda parameters: prediction(parameters) - measured,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        max_nfev=30_000,
    )
    if not result.success:
        raise AnalysisError(f"decay fit failed: {result.message}")
    fitted = prediction(result.x)
    residual = measured - fitted
    dof = max(measured.size - result.x.size, 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * float(residual @ residual) / dof
    return result.x, covariance, fitted, lower, upper


def fit_stretched_decay(
    csv_path: Path,
    *,
    quick_class: str,
    signal: str = "IQ",
    bootstrap_resamples: int = 100,
) -> DecayFit:
    trace = load_native_trace(
        csv_path,
        quick_class=quick_class,
        axis_text="delay time",
        minimum_points=9,
    )
    measured, signal_label, rotation = _fit_signal(
        trace,
        signal,
        relaxation_axis=True,
    )
    time_us = trace.x
    fixed = _fit_decay_arrays(time_us, measured, free_exponent=False)
    stretched = _fit_decay_arrays(time_us, measured, free_exponent=True)
    rss_fixed = float(np.sum((measured - fixed[2]) ** 2))
    rss_stretched = float(np.sum((measured - stretched[2]) ** 2))
    delta_bic = bic(rss_fixed, measured.size, 3) - bic(
        rss_stretched,
        measured.size,
        4,
    )
    signal_power = float(
        np.sum((measured - np.mean(measured)) ** 2)
    )
    selected_stretched = bool(
        delta_bic > 10.0
        and abs(float(stretched[0][3]) - 1.0) > 0.05
        and rss_fixed > np.finfo(float).eps * max(signal_power, 1.0)
    )
    values, covariance, fitted, lower, upper = (
        stretched if selected_stretched else fixed
    )
    exponent = float(values[3]) if selected_stretched else 1.0
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    def refit(x_values, y_values):
        candidate = _fit_decay_arrays(
            np.asarray(x_values),
            np.asarray(y_values),
            free_exponent=selected_stretched,
        )
        parameters = candidate[0]
        if not selected_stretched:
            parameters = np.r_[parameters, 1.0]
        return parameters, candidate[2]

    bootstrap = bootstrap_1d(
        time_us,
        measured,
        refit,
        n_resamples=bootstrap_resamples,
        seed=0,
    )
    bootstrap_half_width = 0.5 * (
        bootstrap["ci_high"] - bootstrap["ci_low"]
    )
    decay_uncertainty = max(
        float(stderr[2]),
        float(bootstrap_half_width[2]),
    )
    exponent_uncertainty = (
        max(float(stderr[3]), float(bootstrap_half_width[3]))
        if selected_stretched
        else 0.0
    )
    parameter_names = (
        ("offset", "amplitude", "decay_us", "exponent")
        if selected_stretched
        else ("offset", "amplitude", "decay_us")
    )
    pinned = pinned_parameters(
        dict(zip(parameter_names, values)),
        dict(zip(parameter_names, lower)),
        dict(zip(parameter_names, upper)),
    )
    decay = float(values[2])
    cycle = (
        trace.metadata.get("parameters", {})
        .get("var", {})
        .get("cycle", 1)
    )
    parameters = {
        "decay_us": decay,
        "decay_uncertainty_us": decay_uncertainty,
        "exponent": exponent,
        "exponent_uncertainty": exponent_uncertainty,
        "offset": float(values[0]),
        "amplitude": float(values[1]),
        "cycle": int(cycle),
        "rotation_angle_rad": float(rotation.get("angle_rad", 0.0)),
    }
    residual = measured - fitted
    statistics = {
        "r_squared": r_squared(measured, fitted),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "span_over_decay": float(np.ptp(time_us) / decay),
        "relative_decay_uncertainty": float(decay_uncertainty / decay),
        "delta_bic_stretched_vs_exponential": float(delta_bic),
        "selected_stretched": bool(selected_stretched),
        "pinned_parameters": pinned,
        "n_pinned": bool(
            selected_stretched and "exponent" in pinned
        ),
        "bootstrap_ci_low": bootstrap["ci_low"].tolist(),
        "bootstrap_ci_high": bootstrap["ci_high"].tolist(),
        "qc": qc_trace(time_us, trace.iq).as_dict(),
    }
    return DecayFit(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        quick_class=quick_class,
        signal=signal,
        signal_label=signal_label,
        time_us=time_us,
        iq=trace.iq,
        measured=measured,
        fitted=fitted,
        parameters=parameters,
        statistics=statistics,
        metadata=trace.metadata,
    )


def fit_echo(
    csv_path: Path,
    *,
    signal: str = "IQ",
    bootstrap_resamples: int = 100,
) -> DecayFit:
    return fit_stretched_decay(
        csv_path,
        quick_class="T2Echo",
        signal=signal,
        bootstrap_resamples=bootstrap_resamples,
    )


def plot_decay_fit(fit: DecayFit):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(fit.time_us, fit.measured, "o", markersize=3, label="data")
    axes[0].plot(fit.time_us, fit.fitted, "-", label="fit")
    axes[0].set(
        xlabel="Delay time (us)",
        ylabel=fit.signal_label,
        title=(
            f"T={fit.decay_us:.4g} us, "
            f"n={fit.parameters['exponent']:.3g}"
        ),
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(fit.time_us, fit.measured - fit.fitted, ".-")
    axes[1].axhline(0.0, color="0.4")
    axes[1].set(xlabel="Delay time (us)", ylabel="Residual", title="Fit residual")
    axes[1].grid(alpha=0.3)
    return figure


def echo_calibration_record(fit: DecayFit) -> dict:
    return {
        "value": fit.decay_us,
        "unit": "us",
        "uncertainty": {
            "decay_us": float(fit.parameters["decay_uncertainty_us"]),
            "exponent": float(fit.parameters["exponent_uncertainty"]),
            "rmse": float(fit.statistics["rmse"]),
        },
        "provenance": {
            "source": str(fit.source_csv),
            "fitted_at": utc_now(),
            "analysis": "quickexp_v3.native_fit_ext.fit_echo",
        },
        "quality": dict(fit.statistics),
        "model": (
            "stretched_exponential"
            if fit.statistics["selected_stretched"]
            else "single_exponential"
        ),
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def accept_echo_fit(
    project_root: Path,
    fit: DecayFit,
    *,
    minimum_r_squared: float = 0.70,
    minimum_span_over_t: float = 0.75,
    maximum_relative_t_uncertainty: float = 0.25,
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_r_squared=minimum_r_squared,
        minimum_span_over_t=minimum_span_over_t,
        maximum_relative_t_uncertainty=maximum_relative_t_uncertainty,
    )
    if not gates_passed and not force_write:
        raise AnalysisError("echo fit did not pass acceptance gates")
    cycle = int(fit.parameters["cycle"])
    updates = annotate_forced_write(
        {f"derived.t2_echo_cycle_{cycle}": echo_calibration_record(fit)},
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)
