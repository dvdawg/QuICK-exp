"""Rabi and Ramsey chevron analysis for native two-dimensional maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from .analysis import fit_damped_oscillation
from .errors import AnalysisError
from .fit_stats import r_squared
from .native_map import NativeMap, load_native_map


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _frequency_rows(native: NativeMap) -> tuple:
    outer_frequency = "freq" in native.outer_label.lower()
    inner_frequency = "freq" in native.inner_label.lower()
    if outer_frequency == inner_frequency:
        raise AnalysisError("chevron map must have exactly one frequency axis")
    if outer_frequency:
        return (
            native.outer,
            native.inner,
            native.complex_signal,
            native.outer_label,
            native.inner_label,
        )
    return (
        native.inner,
        native.outer,
        native.complex_signal.T,
        native.inner_label,
        native.outer_label,
    )


def _oscillation_rows(
    frequencies: np.ndarray,
    sweep: np.ndarray,
    iq_rows: np.ndarray,
    *,
    minimum_r_squared: float,
) -> tuple:
    valid_frequency = []
    rates = []
    uncertainty = []
    row_r2 = []
    for frequency, iq in zip(frequencies, iq_rows):
        try:
            fit = fit_damped_oscillation(sweep, iq)
        except AnalysisError:
            continue
        rate = float(fit.values["frequency"])
        oscillations = rate * float(np.ptp(sweep))
        r2 = float(fit.quality["r_squared"])
        if (
            fit.valid
            and r2 >= minimum_r_squared
            and oscillations >= 1.0
            and np.isfinite(rate)
        ):
            valid_frequency.append(float(frequency))
            rates.append(rate)
            uncertainty.append(
                max(
                    float(fit.values["frequency_uncertainty"]),
                    np.finfo(float).eps,
                )
            )
            row_r2.append(r2)
    return (
        np.asarray(valid_frequency),
        np.asarray(rates),
        np.asarray(uncertainty),
        np.asarray(row_r2),
    )


@dataclass(frozen=True)
class RabiChevronFit:
    source_csv: Path
    mode: str
    frequencies_mhz: np.ndarray
    sweep: np.ndarray
    iq_map: np.ndarray
    valid_frequencies_mhz: np.ndarray
    rates_mhz: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def f0_mhz(self) -> float:
        return float(self.parameters["f0_mhz"])

    def passes(
        self,
        *,
        minimum_valid_columns: int = 5,
        f0_inside_scan: bool = True,
        minimum_r_squared_parabola: float = 0.9,
    ) -> bool:
        inside = (
            float(np.min(self.frequencies_mhz))
            <= self.f0_mhz
            <= float(np.max(self.frequencies_mhz))
        )
        return bool(
            self.statistics["valid_columns"] >= minimum_valid_columns
            and (inside or not f0_inside_scan)
            and self.statistics["r_squared_parabola"] >= minimum_r_squared_parabola
            and self.statistics.get("physics_slope_pass", True)
        )


def fit_rabi_chevron(
    csv_path: Path,
    *,
    minimum_row_r_squared: float = 0.6,
) -> RabiChevronFit:
    native = load_native_map(csv_path)
    frequencies, sweep, iq_rows, _frequency_label, sweep_label = _frequency_rows(native)
    lower_sweep = sweep_label.lower()
    if "gain" in lower_sweep or "power" in lower_sweep:
        projected_visibility = np.ptp(np.abs(iq_rows), axis=1)
        threshold = 0.5 * float(np.max(projected_visibility))
        weights = np.clip(projected_visibility - threshold, 0.0, None)
        if np.sum(weights) <= 0:
            raise AnalysisError("amplitude chevron has no visible resonance")
        center = float(np.sum(frequencies * weights) / np.sum(weights))
        parameters = {
            "f0_mhz": center,
            "omega0_mhz": float("nan"),
            "pi_time_us": float("nan"),
        }
        statistics = {
            "valid_columns": int(np.count_nonzero(weights > 0)),
            "r_squared_parabola": float("nan"),
            "physics_slope_pass": True,
            "diagnostic_only": True,
        }
        return RabiChevronFit(
            source_csv=native.source_csv,
            mode="amplitude_diagnostic",
            frequencies_mhz=np.asarray(frequencies),
            sweep=np.asarray(sweep),
            iq_map=iq_rows,
            valid_frequencies_mhz=np.asarray(frequencies[weights > 0]),
            rates_mhz=np.asarray(projected_visibility[weights > 0]),
            parameters=parameters,
            statistics=statistics,
            metadata=native.metadata,
        )
    valid_frequency, rates, uncertainty, row_r2 = _oscillation_rows(
        np.asarray(frequencies),
        np.asarray(sweep),
        iq_rows,
        minimum_r_squared=minimum_row_r_squared,
    )
    if valid_frequency.size < 3:
        raise AnalysisError("fewer than three Rabi-chevron columns passed row fits")
    weights = 1.0 / np.maximum(uncertainty, np.finfo(float).eps)
    coefficients, covariance = np.polyfit(
        valid_frequency,
        rates**2,
        2,
        w=weights,
        cov=True,
    )
    a, b, c = map(float, coefficients)
    if a <= 0:
        raise AnalysisError("Rabi chevron parabola has non-positive curvature")
    f0 = -b / (2.0 * a)
    omega_squared = c - b**2 / (4.0 * a)
    if omega_squared <= 0:
        raise AnalysisError("Rabi chevron produced a non-positive resonant rate")
    omega0 = float(np.sqrt(omega_squared))
    fitted_squared = np.polyval(coefficients, valid_frequency)
    parameters = {
        "f0_mhz": float(f0),
        "omega0_mhz": omega0,
        "pi_time_us": float(1.0 / (2.0 * omega0)),
        "physics_slope": a,
    }
    statistics = {
        "valid_columns": int(valid_frequency.size),
        "r_squared_parabola": r_squared(rates**2, fitted_squared),
        "physics_slope_pass": bool(abs(a - 1.0) <= 0.15),
        "row_r_squared_minimum": float(np.min(row_r2)),
        "coefficient_covariance": covariance.tolist(),
    }
    return RabiChevronFit(
        source_csv=native.source_csv,
        mode="duration",
        frequencies_mhz=np.asarray(frequencies),
        sweep=np.asarray(sweep),
        iq_map=iq_rows,
        valid_frequencies_mhz=valid_frequency,
        rates_mhz=rates,
        parameters=parameters,
        statistics=statistics,
        metadata=native.metadata,
    )


@dataclass(frozen=True)
class RamseyChevronFit:
    source_csv: Path
    frequencies_mhz: np.ndarray
    delays_us: np.ndarray
    iq_map: np.ndarray
    valid_frequencies_mhz: np.ndarray
    fringe_rates_mhz: np.ndarray
    parameters: Mapping[str, Any]
    statistics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def qubit_frequency_mhz(self) -> float:
        return float(self.parameters["qubit_frequency_mhz"])

    def passes(
        self,
        *,
        minimum_valid_columns: int = 5,
        vertex_inside_scan: bool = True,
        maximum_vertex_uncertainty_mhz: float = 0.5,
    ) -> bool:
        inside = (
            float(np.min(self.frequencies_mhz))
            <= self.qubit_frequency_mhz
            <= float(np.max(self.frequencies_mhz))
        )
        return bool(
            self.statistics["valid_columns"] >= minimum_valid_columns
            and (inside or not vertex_inside_scan)
            and self.parameters["qubit_frequency_uncertainty_mhz"]
            <= maximum_vertex_uncertainty_mhz
        )


def fit_ramsey_chevron(
    csv_path: Path,
    *,
    artificial_fringe_mhz: Optional[float] = None,
    expected_q_frequency_mhz: Optional[float] = None,
    minimum_row_r_squared: float = 0.6,
) -> RamseyChevronFit:
    native = load_native_map(csv_path)
    frequencies, delays, iq_rows, _frequency_label, _delay_label = _frequency_rows(native)
    valid_frequency, rates, uncertainty, row_r2 = _oscillation_rows(
        np.asarray(frequencies),
        np.asarray(delays),
        iq_rows,
        minimum_r_squared=minimum_row_r_squared,
    )
    if valid_frequency.size < 3:
        raise AnalysisError("fewer than three Ramsey-chevron columns passed row fits")
    programmed = _nested(native.metadata, "parameters", "var", "fringe_freq")
    fringe = float(
        artificial_fringe_mhz
        if artificial_fringe_mhz is not None
        else (programmed if programmed is not None else 0.0)
    )
    programmed_q = _nested(native.metadata, "parameters", "var", "q_freq")
    expected = float(
        expected_q_frequency_mhz
        if expected_q_frequency_mhz is not None
        else (
            programmed_q
            if programmed_q is not None
            else np.mean(frequencies)
        )
    )
    lower = float(np.min(frequencies) - abs(fringe))
    upper = float(np.max(frequencies) + abs(fringe))
    candidates = []
    for sign in (-1, 1):
        result = least_squares(
            lambda value: (
                np.abs(fringe + sign * (value[0] - valid_frequency)) - rates
            ) / uncertainty,
            np.asarray([expected]),
            bounds=([lower], [upper]),
            max_nfev=10_000,
        )
        prediction = np.abs(fringe + sign * (result.x[0] - valid_frequency))
        rss = float(np.sum((prediction - rates) ** 2))
        prior_penalty = 1e-9 * ((float(result.x[0]) - expected) / max(np.ptp(frequencies), 1.0)) ** 2
        candidates.append((rss + prior_penalty, sign, result, prediction))
    _score, sign, result, prediction = min(candidates, key=lambda candidate: candidate[0])
    residual = rates - prediction
    dof = max(rates.size - 1, 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * float(residual @ residual) / dof
    uncertainty_q = float(np.sqrt(max(covariance[0, 0], 0.0)))
    parameters = {
        "qubit_frequency_mhz": float(result.x[0]),
        "qubit_frequency_uncertainty_mhz": uncertainty_q,
        "detuning_sign_convention": int(sign),
        "artificial_fringe_mhz": fringe,
    }
    statistics = {
        "valid_columns": int(valid_frequency.size),
        "r_squared_v": r_squared(rates, prediction),
        "rmse_mhz": float(np.sqrt(np.mean(residual**2))),
        "row_r_squared_minimum": float(np.min(row_r2)),
    }
    return RamseyChevronFit(
        source_csv=native.source_csv,
        frequencies_mhz=np.asarray(frequencies),
        delays_us=np.asarray(delays),
        iq_map=iq_rows,
        valid_frequencies_mhz=valid_frequency,
        fringe_rates_mhz=rates,
        parameters=parameters,
        statistics=statistics,
        metadata=native.metadata,
    )


def plot_rabi_chevron(fit: RabiChevronFit):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].pcolormesh(
        fit.sweep,
        fit.frequencies_mhz,
        np.abs(fit.iq_map),
        shading="auto",
    )
    axes[0].set(xlabel="Pulse sweep", ylabel="Drive frequency (MHz)", title="Rabi chevron")
    axes[1].plot(fit.valid_frequencies_mhz, fit.rates_mhz**2, "o")
    if fit.mode == "duration":
        dense = np.linspace(np.min(fit.frequencies_mhz), np.max(fit.frequencies_mhz), 300)
        a = fit.parameters["physics_slope"]
        omega0 = fit.parameters["omega0_mhz"]
        f0 = fit.parameters["f0_mhz"]
        axes[1].plot(dense, omega0**2 + a * (dense - f0) ** 2, "-")
    axes[1].set(xlabel="Drive frequency (MHz)", ylabel="Extracted rate²", title="Resonance fit")
    axes[1].grid(alpha=0.3)
    return figure


def plot_ramsey_chevron(fit: RamseyChevronFit):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].pcolormesh(
        fit.delays_us,
        fit.frequencies_mhz,
        np.abs(fit.iq_map),
        shading="auto",
    )
    axes[0].set(xlabel="Delay (us)", ylabel="Drive frequency (MHz)", title="Ramsey chevron")
    dense = np.linspace(np.min(fit.frequencies_mhz), np.max(fit.frequencies_mhz), 300)
    sign = fit.parameters["detuning_sign_convention"]
    prediction = np.abs(
        fit.parameters["artificial_fringe_mhz"]
        + sign * (fit.qubit_frequency_mhz - dense)
    )
    axes[1].plot(fit.valid_frequencies_mhz, fit.fringe_rates_mhz, "o")
    axes[1].plot(dense, prediction, "-")
    axes[1].set(xlabel="Drive frequency (MHz)", ylabel="Fringe rate (MHz)", title="V fit")
    axes[1].grid(alpha=0.3)
    return figure
