"""Qubit-spectroscopy ridge extraction and constrained transmon flux fit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .errors import AnalysisError
from .fit_calibration import annotate_forced_write, write_calibration_records
from .fit_stats import oriented_rotate_iq, pinned_parameters, r_squared
from .flux_lookup import register_model
from .native_map import load_native_map
from .notch_fit import _fit_lorentz_components
from .util import utc_now


def transmon_frequency(
    z_gain: Any,
    *,
    f_max_mhz: float,
    period_z: float,
    sweet_spot_z: float,
    asymmetry: float,
    ec_mhz: float,
) -> np.ndarray:
    """Evaluate the SQUID-transmon f01 model in MHz."""
    z = np.asarray(z_gain, dtype=float)
    ec = float(ec_mhz)
    if ec <= 0 or period_z <= 0:
        raise AnalysisError("EC and the flux period must be positive")
    phase = np.pi * (z - float(sweet_spot_z)) / float(period_z)
    shape = np.sqrt(
        np.cos(phase) ** 2
        + float(asymmetry) ** 2 * np.sin(phase) ** 2
    )
    ej_sum = (float(f_max_mhz) + ec) ** 2 / (8.0 * ec)
    return np.sqrt(8.0 * ec * ej_sum * shape) - ec


def _evaluate_transmon_record(record: Mapping[str, Any], z_gain: Any):
    parameters = record["value"]["parameters"]
    result = transmon_frequency(z_gain, **parameters)
    return float(result) if np.asarray(result).ndim == 0 else result


register_model("transmon_f01", _evaluate_transmon_record)


@dataclass(frozen=True)
class RidgeRow:
    z_gain: float
    center_mhz: float
    uncertainty_mhz: float
    r_squared: float
    contrast_snr: float


@dataclass(frozen=True)
class QubitFluxFit:
    source_csv: Path
    z_gain: np.ndarray
    map_z_gain: np.ndarray
    frequencies_mhz: np.ndarray
    signal_map: np.ndarray
    ridge_rows: tuple
    fitted_frequencies_mhz: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, Any]
    identifiable: Mapping[str, bool]
    metadata: Mapping[str, Any]

    def frequency(self, z_gain: Any):
        return transmon_frequency(z_gain, **dict(self.parameters))

    def passes(
        self,
        *,
        minimum_ridge_rows: int = 6,
        minimum_r_squared: float = 0.95,
        maximum_rmse_mhz: float = 5.0,
    ) -> bool:
        return bool(
            len(self.ridge_rows) >= minimum_ridge_rows
            and self.statistics["r_squared"] >= minimum_r_squared
            and self.statistics["rmse_mhz"] <= maximum_rmse_mhz
            and self.identifiable.get("period", False)
            and not self.statistics.get("pinned_parameters")
        )


def _extract_row(
    frequencies: np.ndarray,
    iq: np.ndarray,
) -> Optional[tuple]:
    projected, _ = oriented_rotate_iq(iq)
    projected = np.asarray(projected, dtype=float)
    smoothed = gaussian_filter1d(projected, 2.0, mode="nearest")
    baseline = gaussian_filter1d(
        smoothed,
        max(10.0, min(100.0, frequencies.size / 12.0)),
        mode="nearest",
    )
    deviation = smoothed - baseline
    index = int(np.argmax(np.abs(deviation)))
    step = float(np.median(np.diff(frequencies)))
    half_width = max(20.0 * step, float(np.ptp(frequencies)) / 30.0)
    mask = np.abs(frequencies - frequencies[index]) <= half_width
    if np.count_nonzero(mask) < 12:
        return None
    x = frequencies[mask]
    y = projected[mask]
    try:
        values, covariance, fitted, _lower, _upper = _fit_lorentz_components(
            x,
            y,
            1,
        )
    except AnalysisError:
        return None
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    residual = y - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    center = float(values[3])
    uncertainty = float(errors[3])
    contrast = abs(float(values[2])) / max(rmse, np.finfo(float).eps)
    return center, uncertainty, r_squared(y, fitted), contrast, projected


def _fit_window_mask(
    values: np.ndarray,
    window: Optional[Sequence[float]],
    *,
    label: str,
    minimum_points: int,
) -> np.ndarray:
    """Return an inclusive, validated mask for one fit axis."""
    mask = np.ones(values.shape, dtype=bool)
    if window is None:
        return mask
    try:
        if len(window) != 2:
            raise AnalysisError(f"{label} fit window must have two values")
        lower, upper = sorted(float(value) for value in window)
    except (TypeError, ValueError) as error:
        raise AnalysisError(
            f"{label} fit window must contain two finite numeric values"
        ) from error
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise AnalysisError(
            f"{label} fit window must contain two finite numeric values"
        )
    mask = (values >= lower) & (values <= upper)
    if np.count_nonzero(mask) < minimum_points:
        raise AnalysisError(
            f"{label} fit window contains fewer than {minimum_points} points"
        )
    return mask


def fit_transmon_flux(
    z_gain: Any,
    frequencies_mhz: Any,
    *,
    uncertainty_mhz: Optional[Any] = None,
    ec_mhz: float = 180.0,
    period_hint: Optional[float] = None,
) -> tuple:
    z = np.asarray(z_gain, dtype=float)
    frequency = np.asarray(frequencies_mhz, dtype=float)
    if z.size < 4 or z.shape != frequency.shape:
        raise AnalysisError("transmon flux fitting requires at least four matched rows")
    order = np.argsort(z)
    z, frequency = z[order], frequency[order]
    sigma = (
        np.asarray(uncertainty_mhz, dtype=float)[order]
        if uncertainty_mhz is not None
        else np.ones_like(frequency)
    )
    positive_sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback_sigma = float(np.median(positive_sigma)) if positive_sigma.size else 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback_sigma)
    z_steps = np.diff(z)
    minimum_period = max(4.0 * float(np.median(z_steps)), 1e-4)
    z_span = float(np.ptp(z))
    if period_hint is not None:
        minimum_period = max(minimum_period, 0.5 * float(period_hint))
        maximum_period = max(minimum_period * 1.01, 2.0 * float(period_hint))
        period_seed = float(period_hint)
    else:
        maximum_period = max(2.0 * z_span, minimum_period * 1.5)
        period_seed = min(max(z_span, minimum_period * 1.1), maximum_period * 0.9)
    initial = np.asarray(
        [
            float(np.max(frequency)),
            period_seed,
            float(z[np.argmax(frequency)]),
            0.2,
        ]
    )
    frequency_span = max(float(np.ptp(frequency)), 1.0)
    lower = np.asarray(
        [
            float(np.max(frequency) - frequency_span),
            minimum_period,
            float(np.min(z) - maximum_period),
            0.001,
        ]
    )
    upper = np.asarray(
        [
            float(np.max(frequency) + 2.0 * frequency_span + 100.0),
            maximum_period,
            float(np.max(z) + maximum_period),
            0.999,
        ]
    )

    def prediction(parameters):
        return transmon_frequency(
            z,
            f_max_mhz=parameters[0],
            period_z=parameters[1],
            sweet_spot_z=parameters[2],
            asymmetry=parameters[3],
            ec_mhz=ec_mhz,
        )

    best = None
    for period in np.linspace(minimum_period, maximum_period, 8):
        for sweet_spot in (
            z[np.argmax(frequency)],
            float(np.mean(z)),
            z[np.argmax(frequency)] - period,
            z[np.argmax(frequency)] + period,
        ):
            guess = initial.copy()
            guess[1] = period
            guess[2] = np.clip(sweet_spot, lower[2], upper[2])
            try:
                result = least_squares(
                    lambda parameters: (prediction(parameters) - frequency) / sigma,
                    guess,
                    bounds=(lower, upper),
                    loss="soft_l1",
                    f_scale=1.0,
                    max_nfev=30_000,
                )
            except (ValueError, FloatingPointError):
                continue
            rss = float(np.sum((prediction(result.x) - frequency) ** 2))
            if result.success and (best is None or rss < best[0]):
                best = (rss, result)
    if best is None:
        raise AnalysisError("transmon flux fit did not converge")
    rss, result = best
    parameters = result.x.copy()
    parameters[2] += (
        round((float(np.mean(z)) - parameters[2]) / parameters[1])
        * parameters[1]
    )
    fitted = prediction(parameters)
    residual = frequency - fitted
    dof = max(z.size - parameters.size, 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * rss / dof
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    names = ("f_max_mhz", "period_z", "sweet_spot_z", "asymmetry")
    parameter_map = {
        "f_max_mhz": float(parameters[0]),
        "period_z": float(parameters[1]),
        "sweet_spot_z": float(parameters[2]),
        "asymmetry": float(parameters[3]),
        "ec_mhz": float(ec_mhz),
    }
    identifiable = {
        "ec": False,
        "period": bool(z_span >= 0.6 * parameters[1]),
        "sweet_spot": bool(
            np.min(z) <= parameters[2] <= np.max(z)
            or z_span >= 0.6 * parameters[1]
        ),
        "asymmetry": bool(z_span >= 0.6 * parameters[1]),
        "f_max": bool(
            np.min(np.abs(z - parameters[2])) <= 0.1 * parameters[1]
        ),
    }
    statistics = {
        "r_squared": r_squared(frequency, fitted),
        "rmse_mhz": float(np.sqrt(np.mean(residual**2))),
        "max_abs_residual_mhz": float(np.max(np.abs(residual))),
        "stderr": dict(zip(names, map(float, stderr))),
        "pinned_parameters": pinned_parameters(
            dict(zip(names, parameters)),
            dict(zip(names, lower)),
            dict(zip(names, upper)),
        ),
    }
    return parameter_map, fitted, statistics, identifiable


def fit_qubit_flux(
    csv_path: Path,
    *,
    ec_mhz: float = 180.0,
    period_hint: Optional[float] = None,
    frequency_window_mhz: Optional[Sequence[float]] = None,
    flux_window_z: Optional[Sequence[float]] = None,
    minimum_row_r_squared: float = 0.05,
    minimum_row_contrast_snr: float = 1.0,
) -> QubitFluxFit:
    native = load_native_map(csv_path)
    if "z" not in native.outer_label.lower() or "freq" not in native.inner_label.lower():
        raise AnalysisError("qubit flux fitting requires Z as outer and frequency as inner axis")
    frequency_mask = _fit_window_mask(
        native.inner,
        frequency_window_mhz,
        label="frequency",
        minimum_points=12,
    )
    flux_mask = _fit_window_mask(
        native.outer,
        flux_window_z,
        label="flux",
        minimum_points=4,
    )
    frequencies = native.inner[frequency_mask]
    map_z_gain = native.outer[flux_mask]
    iq_map = native.complex_signal[flux_mask][:, frequency_mask]
    signal_map = np.asarray(native.signals.get("amplitude"))[flux_mask][
        :, frequency_mask
    ]
    ridge_rows = []
    projected_rows = []
    for row_index, z_value in enumerate(map_z_gain):
        extracted = _extract_row(frequencies, iq_map[row_index])
        if extracted is None:
            continue
        center, uncertainty, row_r2, contrast, projected = extracted
        projected_rows.append(projected)
        if row_r2 >= minimum_row_r_squared and contrast >= minimum_row_contrast_snr:
            ridge_rows.append(
                RidgeRow(
                    z_gain=float(z_value),
                    center_mhz=center,
                    uncertainty_mhz=max(uncertainty, np.finfo(float).eps),
                    r_squared=row_r2,
                    contrast_snr=contrast,
                )
            )
    if len(ridge_rows) < 4:
        raise AnalysisError("fewer than four qubit-flux ridge rows could be fitted")
    z_values = np.asarray([row.z_gain for row in ridge_rows])
    centers = np.asarray([row.center_mhz for row in ridge_rows])
    uncertainties = np.asarray([row.uncertainty_mhz for row in ridge_rows])
    parameters, fitted, statistics, identifiable = fit_transmon_flux(
        z_values,
        centers,
        uncertainty_mhz=uncertainties,
        ec_mhz=ec_mhz,
        period_hint=period_hint,
    )
    statistics = {
        **statistics,
        "ridge_rows": len(ridge_rows),
        "ec_source": "hardware.q_delta" if ec_mhz == 180.0 else "explicit",
        "frequency_fit_window_mhz": [
            float(np.min(frequencies)),
            float(np.max(frequencies)),
        ],
        "flux_fit_window_z": [
            float(np.min(map_z_gain)),
            float(np.max(map_z_gain)),
        ],
    }
    return QubitFluxFit(
        source_csv=native.source_csv,
        z_gain=z_values,
        map_z_gain=map_z_gain,
        frequencies_mhz=frequencies,
        signal_map=signal_map,
        ridge_rows=tuple(ridge_rows),
        fitted_frequencies_mhz=fitted,
        parameters=parameters,
        statistics=statistics,
        identifiable=identifiable,
        metadata=native.metadata,
    )


def plot_qubit_flux_fit(fit: QubitFluxFit):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    mesh = axes[0].pcolormesh(
        fit.frequencies_mhz,
        fit.map_z_gain,
        fit.signal_map,
        shading="auto",
    )
    axes[0].plot(
        [row.center_mhz for row in fit.ridge_rows],
        [row.z_gain for row in fit.ridge_rows],
        "wo",
        markeredgecolor="black",
    )
    axes[0].set(xlabel="Qubit frequency (MHz)", ylabel="Z gain", title="Extracted ridge")
    figure.colorbar(mesh, ax=axes[0])
    z_dense = np.linspace(float(np.min(fit.z_gain)), float(np.max(fit.z_gain)), 500)
    axes[1].plot(fit.z_gain, [row.center_mhz for row in fit.ridge_rows], "o", label="ridge")
    axes[1].plot(z_dense, fit.frequency(z_dense), "-", label="transmon model")
    axes[1].set(xlabel="Z gain", ylabel="f01 (MHz)", title=f"R²={fit.statistics['r_squared']:.4f}")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    return figure


def qubit_flux_records(fit: QubitFluxFit) -> dict:
    common = {
        "provenance": {
            "source": str(fit.source_csv),
            "fitted_at": utc_now(),
            "analysis": "quickexp_v3.qubit_flux_fit.fit_qubit_flux",
        },
        "quality": {
            **dict(fit.statistics),
            "identifiable": dict(fit.identifiable),
        },
        "valid_domain": {
            "z_gain": [float(np.min(fit.z_gain)), float(np.max(fit.z_gain))]
        },
        "status": "accepted",
        "accepted_at": utc_now(),
    }
    return {
        "lookups.qubit_vs_flux": {
            **common,
            "value": {"parameters": dict(fit.parameters)},
            "unit": "MHz",
            "uncertainty": dict(fit.statistics.get("stderr", {})),
            "model": "transmon_f01",
        },
        "derived.flux_sweet_spot_z": {
            **common,
            "value": float(fit.parameters["sweet_spot_z"]),
            "unit": "a.u.",
            "uncertainty": {
                "sweet_spot_z": fit.statistics.get("stderr", {}).get("sweet_spot_z")
            },
            "model": "transmon_f01_sweet_spot",
        },
    }


def accept_qubit_flux_fit(
    project_root: Path,
    fit: QubitFluxFit,
    *,
    minimum_ridge_rows: int = 6,
    minimum_r_squared: float = 0.95,
    maximum_rmse_mhz: float = 5.0,
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_ridge_rows=minimum_ridge_rows,
        minimum_r_squared=minimum_r_squared,
        maximum_rmse_mhz=maximum_rmse_mhz,
    )
    if not gates_passed and not force_write:
        raise AnalysisError("qubit flux fit did not pass acceptance gates")
    updates = annotate_forced_write(
        qubit_flux_records(fit),
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)
