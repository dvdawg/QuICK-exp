"""Fit, validate, persist, and evaluate resonator frequency versus held Z."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from .errors import AnalysisError, ConfigError
from .fit_calibration import write_calibration_records
from .util import to_builtin, utc_now


DEFAULT_SCAN_NAME = "ResVsZ_held_bias"
MODEL_NAME = "cosine"


def _as_scalar_or_array(values: np.ndarray) -> Union[float, np.ndarray]:
    return float(values) if values.ndim == 0 else values


def cosine_frequency(
    z_gain: Any,
    *,
    center_frequency: float,
    amplitude: float,
    period: float,
    peak_bias: float,
) -> Union[float, np.ndarray]:
    """Evaluate the empirical periodic resonator calibration in MHz."""
    if not np.isfinite(period) or period <= 0:
        raise ConfigError("resonator flux-fit period must be positive and finite")
    z_values = np.asarray(z_gain, dtype=float)
    if not np.all(np.isfinite(z_values)):
        raise ConfigError("z_gain must contain only finite values")
    frequency = center_frequency + amplitude * np.cos(
        2.0 * np.pi * (z_values - peak_bias) / period
    )
    return _as_scalar_or_array(np.asarray(frequency))


def frequency_from_calibration_record(
    record: Mapping[str, Any],
    z_gain: Any,
) -> Union[float, np.ndarray]:
    """Evaluate one accepted ``resonator_vs_flux`` calibration record."""
    try:
        if record.get("status") != "accepted":
            raise KeyError("record is not accepted")
        if record.get("model") != MODEL_NAME:
            raise KeyError("record is not a cosine")
        parameters = record["value"]["parameters"]
        center_frequency = float(parameters["center_frequency"])
        amplitude = float(parameters["amplitude"])
        period = float(parameters["period"])
        peak_bias = float(parameters["peak_bias"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(
            "resonator_vs_flux must be an accepted cosine calibration"
        ) from error

    values = np.asarray(z_gain, dtype=float)
    domain = record.get("valid_domain", {}).get("z_gain")
    if domain is not None:
        try:
            minimum, maximum = map(float, domain)
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "resonator_vs_flux valid_domain.z_gain must be [min, max]"
            ) from error
        if np.any(values < minimum) or np.any(values > maximum):
            raise ConfigError(
                "requested z_gain is outside accepted resonator-fit domain "
                f"[{minimum}, {maximum}]"
            )

    return cosine_frequency(
        values,
        center_frequency=center_frequency,
        amplitude=amplitude,
        period=period,
        peak_bias=peak_bias,
    )


def find_latest_scan(data_directory: Path, scan_name: str) -> Path:
    """Return the newest non-empty Quick CSV matching ``scan_name``."""
    root = Path(data_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Quick data directory does not exist: {root}")
    candidates = [
        path
        for path in root.glob(f"*{scan_name}*.csv")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No '*{scan_name}*.csv' file found in {root}. "
            "Run 05c first or set INPUT_CSV explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_scan(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a Quick flux map, dropping only incomplete/non-finite Z rows."""
    source = Path(csv_path).expanduser().resolve()
    try:
        data = np.loadtxt(source, delimiter=",")
    except ValueError:
        data = np.loadtxt(source)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.ndim != 2 or data.shape[1] < 3:
        raise AnalysisError(
            f"expected Z, frequency, and amplitude columns; got {data.shape}"
        )

    finite_axes = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    data = data[finite_axes]
    if len(data) == 0:
        raise AnalysisError("resonator flux scan has no finite axis rows")
    z_all = np.unique(data[:, 0])
    frequencies = np.unique(data[:, 1])
    if len(frequencies) < 3:
        raise AnalysisError("at least three readout frequencies are required")

    amplitude = np.full((len(z_all), len(frequencies)), np.nan)
    counts = np.zeros_like(amplitude, dtype=int)
    z_index = np.searchsorted(z_all, data[:, 0])
    f_index = np.searchsorted(frequencies, data[:, 1])
    np.add.at(counts, (z_index, f_index), 1)
    if np.any(counts > 1):
        raise AnalysisError("resonator flux scan contains duplicate grid points")
    amplitude[z_index, f_index] = data[:, 2]

    valid_rows = np.all((counts == 1) & np.isfinite(amplitude), axis=1)
    dropped_z = z_all[~valid_rows]
    z_values = z_all[valid_rows]
    amplitude = amplitude[valid_rows]
    if len(z_values) < 6:
        raise AnalysisError(
            "at least six complete finite Z rows are required for a cosine fit"
        )

    frequency_steps = np.diff(frequencies)
    frequency_step = float(np.median(frequency_steps))
    if frequency_step <= 0 or not np.allclose(
        frequency_steps,
        frequency_step,
        rtol=1e-5,
        atol=max(abs(frequency_step) * 1e-8, 1e-12),
    ):
        raise AnalysisError("readout frequency axis must be uniformly spaced")
    return z_values, frequencies, amplitude, dropped_z


def extract_notch_centers(
    frequencies_mhz: np.ndarray,
    amplitude_db: np.ndarray,
    smooth_sigma_bins: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract each notch with smoothing and three-bin parabolic refinement."""
    if smooth_sigma_bins < 0:
        raise AnalysisError("smooth_sigma_bins must be non-negative")
    smoothed = (
        gaussian_filter1d(
            amplitude_db,
            smooth_sigma_bins,
            axis=1,
            mode="nearest",
        )
        if smooth_sigma_bins > 0
        else amplitude_db.copy()
    )
    minimum_indices = np.argmin(smoothed, axis=1)
    centers = np.empty(amplitude_db.shape[0], dtype=float)
    depths = np.empty_like(centers)

    for row_index, minimum_index in enumerate(minimum_indices):
        center = float(frequencies_mhz[minimum_index])
        if 0 < minimum_index < len(frequencies_mhz) - 1:
            local_frequency = frequencies_mhz[
                minimum_index - 1 : minimum_index + 2
            ]
            local_signal = smoothed[
                row_index,
                minimum_index - 1 : minimum_index + 2,
            ]
            offsets = local_frequency - center
            quadratic, linear, _ = np.polyfit(offsets, local_signal, deg=2)
            if quadratic > 0:
                correction = -linear / (2.0 * quadratic)
                limit = float(np.max(np.abs(offsets)))
                center += float(np.clip(correction, -limit, limit))
        centers[row_index] = center
        depths[row_index] = (
            float(np.median(amplitude_db[row_index]))
            - float(smoothed[row_index, minimum_index])
        )
    return centers, depths, smoothed


def _cosine_array(z_gain: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    return np.asarray(
        cosine_frequency(
            z_gain,
            center_frequency=float(parameters[0]),
            amplitude=float(parameters[1]),
            period=float(parameters[2]),
            peak_bias=float(parameters[3]),
        )
    )


def _initial_guess(
    z_gain: np.ndarray,
    centers_mhz: np.ndarray,
    period_bounds: tuple[float, float],
) -> np.ndarray:
    periods = np.linspace(period_bounds[0], period_bounds[1], 4000)
    best_rss = np.inf
    best_period = float(periods[0])
    best_coefficients = np.zeros(3)
    for period in periods:
        phase = 2.0 * np.pi * z_gain / period
        design = np.column_stack(
            (np.ones_like(z_gain), np.cos(phase), np.sin(phase))
        )
        coefficients, *_ = np.linalg.lstsq(
            design,
            centers_mhz,
            rcond=None,
        )
        residual = design @ coefficients - centers_mhz
        rss = float(residual @ residual)
        if rss < best_rss:
            best_rss = rss
            best_period = float(period)
            best_coefficients = coefficients

    center, cosine_coefficient, sine_coefficient = best_coefficients
    amplitude = float(np.hypot(cosine_coefficient, sine_coefficient))
    peak_bias = (
        np.arctan2(sine_coefficient, cosine_coefficient)
        * best_period
        / (2.0 * np.pi)
    )
    peak_bias += (
        round((float(np.mean(z_gain)) - peak_bias) / best_period)
        * best_period
    )
    return np.array([center, amplitude, best_period, peak_bias])


def fit_cosine(
    z_gain: np.ndarray,
    centers_mhz: np.ndarray,
    frequency_step_mhz: float,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Robustly fit the notebook's empirical periodic readout calibration."""
    z_steps = np.diff(np.sort(z_gain))
    positive_steps = z_steps[z_steps > 0]
    if len(z_gain) < 6 or len(positive_steps) == 0:
        raise AnalysisError("at least six distinct Z points are required")
    typical_step = float(np.median(positive_steps))
    z_span = float(np.ptp(z_gain))
    lower_period = (
        float(period_min)
        if period_min is not None
        else 4.0 * typical_step
    )
    upper_period = (
        float(period_max)
        if period_max is not None
        else 2.0 * z_span
    )
    if not (0 < lower_period < upper_period):
        raise AnalysisError("period bounds must satisfy 0 < minimum < maximum")

    initial = _initial_guess(
        z_gain,
        centers_mhz,
        (lower_period, upper_period),
    )
    center_span = max(float(np.ptp(centers_mhz)), frequency_step_mhz)
    lower_bounds = np.array(
        [
            centers_mhz.min() - center_span,
            0.0,
            lower_period,
            z_gain.min() - upper_period,
        ]
    )
    upper_bounds = np.array(
        [
            centers_mhz.max() + center_span,
            2.0 * center_span,
            upper_period,
            z_gain.max() + upper_period,
        ]
    )
    result = least_squares(
        lambda parameters: _cosine_array(z_gain, parameters) - centers_mhz,
        initial,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=max(frequency_step_mhz, 0.01),
        max_nfev=20_000,
    )
    if not result.success:
        raise AnalysisError(f"resonator cosine fit failed: {result.message}")

    parameters = result.x.copy()
    parameters[3] += (
        round(
            (float(np.mean(z_gain)) - parameters[3])
            / parameters[2]
        )
        * parameters[2]
    )
    fitted = _cosine_array(z_gain, parameters)
    residual = centers_mhz - fitted
    degrees_of_freedom = max(len(z_gain) - len(parameters), 1)
    residual_variance = float(residual @ residual) / degrees_of_freedom
    covariance = (
        np.linalg.pinv(result.jac.T @ result.jac) * residual_variance
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    total_variance = float(
        np.sum((centers_mhz - np.mean(centers_mhz)) ** 2)
    )
    r_squared = (
        float(1.0 - np.sum(residual**2) / total_variance)
        if total_variance > 0
        else float("nan")
    )
    statistics = {
        "center_frequency_stderr_mhz": float(standard_errors[0]),
        "amplitude_stderr_mhz": float(standard_errors[1]),
        "period_stderr": float(standard_errors[2]),
        "peak_bias_stderr": float(standard_errors[3]),
        "rmse_mhz": float(np.sqrt(np.mean(residual**2))),
        "max_abs_residual_mhz": float(np.max(np.abs(residual))),
        "r_squared": r_squared,
    }
    return parameters, fitted, statistics


@dataclass(frozen=True)
class ResonatorFluxFit:
    """One fitted resonator-vs-Z calibration and its diagnostic arrays."""

    source_csv: Path
    z_gain: np.ndarray
    frequencies_mhz: np.ndarray
    amplitude_db: np.ndarray
    smoothed_amplitude_db: np.ndarray
    extracted_frequencies_mhz: np.ndarray
    fitted_frequencies_mhz: np.ndarray
    notch_depth_db: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, float]
    smooth_sigma_bins: float
    dropped_z_gain: np.ndarray

    def frequency(self, z_gain: Any) -> Union[float, np.ndarray]:
        values = np.asarray(z_gain, dtype=float)
        minimum = float(np.min(self.z_gain))
        maximum = float(np.max(self.z_gain))
        if np.any(values < minimum) or np.any(values > maximum):
            raise ConfigError(
                "requested z_gain is outside fitted domain "
                f"[{minimum}, {maximum}]"
            )
        return cosine_frequency(values, **dict(self.parameters))

    def passes(
        self,
        *,
        minimum_r_squared: float,
        maximum_rmse_mhz: float,
    ) -> bool:
        r_squared = float(self.statistics["r_squared"])
        rmse = float(self.statistics["rmse_mhz"])
        return (
            np.isfinite(r_squared)
            and np.isfinite(rmse)
            and r_squared >= minimum_r_squared
            and rmse <= maximum_rmse_mhz
        )


def fit_resonator_flux(
    csv_path: Path,
    *,
    smooth_sigma_bins: float = 2.0,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
) -> ResonatorFluxFit:
    """Fit a Quick ``ResVsZ_held_bias`` CSV."""
    source = Path(csv_path).expanduser().resolve()
    z_gain, frequencies, amplitude, dropped_z = load_scan(source)
    centers, depths, smoothed = extract_notch_centers(
        frequencies,
        amplitude,
        smooth_sigma_bins,
    )
    frequency_step = float(np.median(np.diff(frequencies)))
    parameters_array, fitted, statistics = fit_cosine(
        z_gain,
        centers,
        frequency_step,
        period_min,
        period_max,
    )
    parameters = {
        "center_frequency": float(parameters_array[0]),
        "amplitude": float(parameters_array[1]),
        "period": float(parameters_array[2]),
        "peak_bias": float(parameters_array[3]),
    }
    return ResonatorFluxFit(
        source_csv=source,
        z_gain=z_gain,
        frequencies_mhz=frequencies,
        amplitude_db=amplitude,
        smoothed_amplitude_db=smoothed,
        extracted_frequencies_mhz=centers,
        fitted_frequencies_mhz=fitted,
        notch_depth_db=depths,
        parameters=parameters,
        statistics=statistics,
        smooth_sigma_bins=float(smooth_sigma_bins),
        dropped_z_gain=dropped_z,
    )


def plot_resonator_flux_fit(fit: ResonatorFluxFit):
    """Return the notebook-style map, fitted curve, and residual figure."""
    z_dense = np.linspace(
        float(np.min(fit.z_gain)),
        float(np.max(fit.z_gain)),
        1001,
    )
    frequency_dense = np.asarray(fit.frequency(z_dense))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.2),
        constrained_layout=True,
    )
    mesh = axes[0].pcolormesh(
        fit.frequencies_mhz,
        fit.z_gain,
        fit.amplitude_db,
        shading="auto",
        cmap="turbo",
    )
    fit_line = axes[0].plot(
        frequency_dense,
        z_dense,
        "k-",
        linewidth=2,
        label="cosine fit",
    )[0]
    fit_line.set_path_effects(
        [
            path_effects.Stroke(linewidth=4, foreground="white"),
            path_effects.Normal(),
        ]
    )
    axes[0].plot(
        fit.extracted_frequencies_mhz,
        fit.z_gain,
        "wo",
        markeredgecolor="black",
        markeredgewidth=0.7,
        markersize=4,
        label="extracted notch",
    )
    axes[0].set(
        xlabel="Readout frequency (MHz)",
        ylabel="Z gain",
        title="Fit over measured amplitude",
    )
    axes[0].legend(loc="best")
    figure.colorbar(mesh, ax=axes[0], label="Amplitude (dB)")

    axes[1].plot(z_dense, frequency_dense, "-", linewidth=2)
    axes[1].plot(
        fit.z_gain,
        fit.extracted_frequencies_mhz,
        "o",
        markersize=4,
    )
    axes[1].set(
        xlabel="Z gain",
        ylabel="Readout frequency (MHz)",
        title="Resonator flux dependence",
    )
    axes[1].grid(alpha=0.3)

    residual_khz = 1000.0 * (
        fit.extracted_frequencies_mhz - fit.fitted_frequencies_mhz
    )
    axes[2].axhline(0, color="0.4", linewidth=1)
    axes[2].plot(fit.z_gain, residual_khz, "o-", markersize=4)
    axes[2].set(
        xlabel="Z gain",
        ylabel="Extracted - fit (kHz)",
        title=(
            "Fit residuals "
            f"(RMSE {1000 * fit.statistics['rmse_mhz']:.0f} kHz)"
        ),
    )
    axes[2].grid(alpha=0.3)
    figure.suptitle("Fitted resonator readout calibration")
    return figure


def calibration_record(fit: ResonatorFluxFit) -> dict:
    """Build the accepted v3 lookup record for ``calibration.yml``."""
    return {
        "value": {"parameters": dict(fit.parameters)},
        "unit": "MHz",
        "uncertainty": {
            key: float(value)
            for key, value in fit.statistics.items()
            if key != "r_squared"
        },
        "provenance": {
            "source": str(fit.source_csv),
            "fitted_at": utc_now(),
            "analysis": "quickexp_v3.resonator_flux.fit_resonator_flux",
        },
        "quality": {
            "r_squared": float(fit.statistics["r_squared"]),
            "complete_z_rows": int(len(fit.z_gain)),
            "dropped_z_rows": to_builtin(fit.dropped_z_gain),
            "notch_depth_db": {
                "minimum": float(np.min(fit.notch_depth_db)),
                "median": float(np.median(fit.notch_depth_db)),
            },
        },
        "valid_domain": {
            "z_gain": [
                float(np.min(fit.z_gain)),
                float(np.max(fit.z_gain)),
            ]
        },
        "model": MODEL_NAME,
        "notes": (
            "Per-Z notch minima use Gaussian smoothing and three-bin "
            "parabolic center refinement."
        ),
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def accept_fit(
    project_root: Path,
    fit: ResonatorFluxFit,
    *,
    minimum_r_squared: float,
    maximum_rmse_mhz: float,
) -> Path:
    """Atomically install a quality-gated accepted fit in ``calibration.yml``."""
    if not fit.passes(
        minimum_r_squared=minimum_r_squared,
        maximum_rmse_mhz=maximum_rmse_mhz,
    ):
        raise AnalysisError(
            "fit was not accepted: "
            f"R^2={fit.statistics['r_squared']:.6f} "
            f"(minimum {minimum_r_squared}), "
            f"RMSE={fit.statistics['rmse_mhz']:.6f} MHz "
            f"(maximum {maximum_rmse_mhz} MHz)"
        )

    return write_calibration_records(
        project_root,
        {
            "lookups.resonator_vs_flux": calibration_record(fit),
        },
    )
