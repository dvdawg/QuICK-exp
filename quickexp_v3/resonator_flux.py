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
from .fit_calibration import annotate_forced_write, write_calibration_records
from .util import to_builtin, utc_now


DEFAULT_SCAN_NAME = "ResVsZ_held_bias"
MODEL_NAME = "cosine"
LOOKUP_MODEL_NAME = "linear_interpolation"
FIT_METHODS = ("fit", "min", "max")
FEATURE_POLARITIES = ("auto", "dip", "peak")
# The four-parameter cosine needs a well-conditioned Z axis; a sampled
# min/max lookup only needs two rows to interpolate between.
MINIMUM_COSINE_Z_ROWS = 6
MINIMUM_LOOKUP_Z_ROWS = 2


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


def sampled_frequency(
    z_gain: Any,
    *,
    sampled_z_gain: Any,
    sampled_frequencies_mhz: Any,
) -> Union[float, np.ndarray]:
    """Linearly interpolate a measured resonator-frequency lookup."""
    sample_z = np.asarray(sampled_z_gain, dtype=float)
    sample_frequency = np.asarray(sampled_frequencies_mhz, dtype=float)
    if (
        sample_z.ndim != 1
        or sample_frequency.ndim != 1
        or sample_z.size != sample_frequency.size
        or sample_z.size < MINIMUM_LOOKUP_Z_ROWS
        or not np.all(np.isfinite(sample_z))
        or not np.all(np.isfinite(sample_frequency))
        or not np.all(np.diff(sample_z) > 0)
    ):
        raise ConfigError(
            "resonator sampled lookup requires matching finite 1D arrays "
            "with at least two strictly increasing Z values"
        )

    values = np.asarray(z_gain, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ConfigError("z_gain must contain only finite values")
    minimum = float(sample_z[0])
    maximum = float(sample_z[-1])
    if np.any(values < minimum) or np.any(values > maximum):
        raise ConfigError(
            "requested z_gain is outside accepted resonator-fit domain "
            f"[{minimum}, {maximum}]"
        )
    frequency = np.interp(
        values.ravel(),
        sample_z,
        sample_frequency,
    ).reshape(values.shape)
    return _as_scalar_or_array(np.asarray(frequency))


def frequency_from_calibration_record(
    record: Mapping[str, Any],
    z_gain: Any,
) -> Union[float, np.ndarray]:
    """Evaluate one accepted ``resonator_vs_flux`` calibration record."""
    try:
        if record.get("status") != "accepted":
            raise KeyError("record is not accepted")
        model = str(record["model"])
        value = record["value"]
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(
            "resonator_vs_flux must be an accepted supported calibration"
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

    try:
        if model == MODEL_NAME:
            parameters = value["parameters"]
            return cosine_frequency(
                values,
                center_frequency=float(parameters["center_frequency"]),
                amplitude=float(parameters["amplitude"]),
                period=float(parameters["period"]),
                peak_bias=float(parameters["peak_bias"]),
            )
        if model == LOOKUP_MODEL_NAME:
            return sampled_frequency(
                values,
                sampled_z_gain=value["z_gain"],
                sampled_frequencies_mhz=value["r_freq_mhz"],
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(
            f"invalid resonator_vs_flux {model!r} calibration"
        ) from error
    raise ConfigError(f"unsupported resonator_vs_flux model {model!r}")


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
    *,
    minimum_z_rows: int = MINIMUM_COSINE_Z_ROWS,
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
    if np.unique(data[:, :2], axis=0).shape[0] != data.shape[0]:
        raise AnalysisError("resonator flux scan contains duplicate grid points")
    z_all = np.unique(data[:, 0])
    row_frequencies = []
    row_amplitudes = []
    z_values = []
    dropped = []
    expected_points = None
    expected_step = None
    for z_gain in z_all:
        row = data[data[:, 0] == z_gain]
        row = row[np.argsort(row[:, 1])]
        frequency = np.asarray(row[:, 1], dtype=float)
        amplitude_row = np.asarray(row[:, 2], dtype=float)
        valid = bool(
            frequency.size >= 3
            and np.all(np.isfinite(amplitude_row))
            and np.unique(frequency).size == frequency.size
        )
        if valid:
            steps = np.diff(frequency)
            step = float(np.median(steps))
            valid = bool(
                step > 0.0
                and np.allclose(
                    steps,
                    step,
                    rtol=1.0e-5,
                    atol=max(abs(step) * 1.0e-8, 1.0e-12),
                )
            )
        if expected_points is None and valid:
            expected_points = int(frequency.size)
            expected_step = step
        if valid:
            valid = bool(
                int(frequency.size) == expected_points
                and np.isclose(
                    step,
                    expected_step,
                    rtol=1.0e-5,
                    atol=max(abs(expected_step) * 1.0e-8, 1.0e-12),
                )
            )
        if not valid:
            dropped.append(float(z_gain))
            continue
        z_values.append(float(z_gain))
        row_frequencies.append(frequency)
        row_amplitudes.append(amplitude_row)
    dropped_z = np.asarray(dropped, dtype=float)
    z_values = np.asarray(z_values, dtype=float)
    if len(z_values) < minimum_z_rows:
        raise AnalysisError(
            f"at least {minimum_z_rows} complete finite Z rows are required "
            "for this calibration"
        )
    frequency_grid = np.vstack(row_frequencies)
    amplitude = np.vstack(row_amplitudes)
    frequencies = (
        frequency_grid[0]
        if all(
            np.allclose(row, frequency_grid[0], rtol=1.0e-10, atol=1.0e-12)
            for row in frequency_grid[1:]
        )
        else frequency_grid
    )
    return z_values, frequencies, amplitude, dropped_z


def _smooth_amplitude(
    amplitude_db: np.ndarray,
    smooth_sigma_bins: float,
) -> np.ndarray:
    if not np.isfinite(smooth_sigma_bins) or smooth_sigma_bins < 0:
        raise AnalysisError("smooth_sigma_bins must be finite and non-negative")
    return (
        gaussian_filter1d(
            amplitude_db,
            smooth_sigma_bins,
            axis=1,
            mode="nearest",
        )
        if smooth_sigma_bins > 0
        else amplitude_db.copy()
    )


def extract_feature_centers(
    frequencies_mhz: np.ndarray,
    amplitude_db: np.ndarray,
    smooth_sigma_bins: float,
    feature_polarity: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Extract consistently polarized peaks or dips from spectroscopy rows."""
    polarity = str(feature_polarity).strip().lower()
    if polarity not in FEATURE_POLARITIES:
        choices = ", ".join(repr(choice) for choice in FEATURE_POLARITIES)
        raise AnalysisError(f"feature_polarity must be one of {choices}")
    smoothed = _smooth_amplitude(amplitude_db, smooth_sigma_bins)
    frequency_rows = (
        np.broadcast_to(frequencies_mhz, amplitude_db.shape)
        if np.asarray(frequencies_mhz).ndim == 1
        else np.asarray(frequencies_mhz, dtype=float)
    )
    if smoothed.ndim != 2 or smoothed.shape != frequency_rows.shape:
        raise AnalysisError(
            "spectroscopy amplitude must have one row per outer-axis value "
            "and one column per frequency"
        )
    point_count = frequency_rows.shape[1]
    edge_count = max(2, min(20, point_count // 5))
    edge_indices = np.r_[:edge_count, point_count - edge_count : point_count]
    # Remove a per-row linear background so dip and peak contrast are
    # measured against the same reference instead of the row median.
    deviation = np.empty_like(smoothed)
    for row_index, row in enumerate(smoothed):
        row_frequency = frequency_rows[row_index]
        reference = float(np.mean(row_frequency))
        slope, offset = np.polyfit(
            row_frequency[edge_indices] - reference,
            row[edge_indices],
            1,
        )
        deviation[row_index] = row - (
            offset + slope * (row_frequency - reference)
        )
    peak_contrast = np.max(deviation, axis=1)
    dip_contrast = -np.min(deviation, axis=1)
    if polarity == "auto":
        polarity = (
            "peak"
            if float(np.median(peak_contrast))
            > float(np.median(dip_contrast))
            else "dip"
        )
    indices = (
        np.argmax(deviation, axis=1)
        if polarity == "peak"
        else np.argmin(deviation, axis=1)
    )
    centers = np.empty(amplitude_db.shape[0], dtype=float)
    contrasts = np.empty_like(centers)

    for row_index, feature_index in enumerate(indices):
        row_frequency = frequency_rows[row_index]
        center = float(row_frequency[feature_index])
        if 0 < feature_index < len(row_frequency) - 1:
            local_frequency = row_frequency[
                feature_index - 1 : feature_index + 2
            ]
            local_signal = deviation[
                row_index,
                feature_index - 1 : feature_index + 2,
            ]
            offsets = local_frequency - center
            quadratic, linear, _ = np.polyfit(offsets, local_signal, deg=2)
            curvature_matches = (
                quadratic < 0 if polarity == "peak" else quadratic > 0
            )
            if curvature_matches:
                correction = -linear / (2.0 * quadratic)
                limit = float(np.max(np.abs(offsets)))
                center += float(np.clip(correction, -limit, limit))
        centers[row_index] = center
        contrasts[row_index] = (
            float(deviation[row_index, feature_index])
            if polarity == "peak"
            else -float(deviation[row_index, feature_index])
        )
    return centers, contrasts, smoothed, polarity


def extract_notch_centers(
    frequencies_mhz: np.ndarray,
    amplitude_db: np.ndarray,
    smooth_sigma_bins: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible explicitly downward-notch extractor."""
    centers, contrasts, smoothed, _ = extract_feature_centers(
        frequencies_mhz,
        amplitude_db,
        smooth_sigma_bins,
        feature_polarity="dip",
    )
    return centers, contrasts, smoothed


def extract_extremum_frequencies(
    frequencies_mhz: np.ndarray,
    amplitude_db: np.ndarray,
    smooth_sigma_bins: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the exact min/max frequency bin from every smoothed Z row."""
    if method not in {"min", "max"}:
        raise AnalysisError("extremum method must be 'min' or 'max'")
    smoothed = _smooth_amplitude(amplitude_db, smooth_sigma_bins)
    indices = (
        np.argmin(smoothed, axis=1)
        if method == "min"
        else np.argmax(smoothed, axis=1)
    )
    row_indices = np.arange(amplitude_db.shape[0])
    selected = smoothed[row_indices, indices]
    baseline = np.median(amplitude_db, axis=1)
    contrast = (
        baseline - selected
        if method == "min"
        else selected - baseline
    )
    frequency_rows = (
        np.broadcast_to(frequencies_mhz, amplitude_db.shape)
        if np.asarray(frequencies_mhz).ndim == 1
        else np.asarray(frequencies_mhz, dtype=float)
    )
    selected_frequency = frequency_rows[row_indices, indices]
    return selected_frequency.astype(float), contrast, smoothed


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
    """One resonator-vs-Z calibration and its diagnostic arrays."""

    source_csv: Path
    fit_method: str
    model: str
    z_gain: np.ndarray
    frequencies_mhz: np.ndarray
    amplitude_db: np.ndarray
    smoothed_amplitude_db: np.ndarray
    extracted_frequencies_mhz: np.ndarray
    fitted_frequencies_mhz: np.ndarray
    feature_contrast_db: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, float]
    smooth_sigma_bins: float
    dropped_z_gain: np.ndarray
    feature_polarity: str = "unknown"

    @property
    def notch_depth_db(self) -> np.ndarray:
        """Backward-compatible name for the per-row feature contrast."""
        return self.feature_contrast_db

    def frequency(self, z_gain: Any) -> Union[float, np.ndarray]:
        values = np.asarray(z_gain, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ConfigError("z_gain must contain only finite values")
        minimum = float(np.min(self.z_gain))
        maximum = float(np.max(self.z_gain))
        if np.any(values < minimum) or np.any(values > maximum):
            raise ConfigError(
                "requested z_gain is outside fitted domain "
                f"[{minimum}, {maximum}]"
            )
        if self.model == MODEL_NAME:
            return cosine_frequency(values, **dict(self.parameters))
        if self.model == LOOKUP_MODEL_NAME:
            return sampled_frequency(
                values,
                sampled_z_gain=self.z_gain,
                sampled_frequencies_mhz=self.extracted_frequencies_mhz,
            )
        raise ConfigError(
            f"unsupported resonator calibration model {self.model!r}"
        )

    def passes(
        self,
        *,
        minimum_r_squared: float,
        maximum_rmse_mhz: float,
    ) -> bool:
        # A sampled lookup reproduces the measured rows exactly, so the
        # cosine R^2/RMSE gates do not apply. WRITE_ACCEPTED_FIT stays the
        # explicit acceptance latch; only structural validity is checked.
        if self.model == LOOKUP_MODEL_NAME:
            return (
                self.fit_method in {"min", "max"}
                and len(self.z_gain) >= MINIMUM_LOOKUP_Z_ROWS
                and np.all(np.isfinite(self.extracted_frequencies_mhz))
                and np.all(np.isfinite(self.feature_contrast_db))
            )
        if self.model != MODEL_NAME:
            return False
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
    fit_method: str = "fit",
    smooth_sigma_bins: float = 2.0,
    feature_polarity: str = "auto",
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
) -> ResonatorFluxFit:
    """Build a cosine fit or a sampled min/max lookup from a flux map."""
    method = str(fit_method).strip().lower()
    if method not in FIT_METHODS:
        choices = ", ".join(repr(choice) for choice in FIT_METHODS)
        raise AnalysisError(f"fit_method must be one of {choices}")
    source = Path(csv_path).expanduser().resolve()
    z_gain, frequencies, amplitude, dropped_z = load_scan(
        source,
        minimum_z_rows=(
            MINIMUM_COSINE_Z_ROWS
            if method == "fit"
            else MINIMUM_LOOKUP_Z_ROWS
        ),
    )
    frequency_step = float(np.median(np.diff(frequencies, axis=-1)))
    if method == "fit":
        centers, contrast, smoothed, selected_polarity = extract_feature_centers(
            frequencies,
            amplitude,
            smooth_sigma_bins,
            feature_polarity=feature_polarity,
        )
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
        model = MODEL_NAME
    else:
        centers, contrast, smoothed = extract_extremum_frequencies(
            frequencies,
            amplitude,
            smooth_sigma_bins,
            method,
        )
        fitted = centers.copy()
        parameters = {}
        statistics = {"frequency_bin_mhz": abs(frequency_step)}
        selected_polarity = "dip" if method == "min" else "peak"
        model = LOOKUP_MODEL_NAME
    return ResonatorFluxFit(
        source_csv=source,
        fit_method=method,
        feature_polarity=selected_polarity,
        model=model,
        z_gain=z_gain,
        frequencies_mhz=frequencies,
        amplitude_db=amplitude,
        smoothed_amplitude_db=smoothed,
        extracted_frequencies_mhz=centers,
        fitted_frequencies_mhz=fitted,
        feature_contrast_db=contrast,
        parameters=parameters,
        statistics=statistics,
        smooth_sigma_bins=float(smooth_sigma_bins),
        dropped_z_gain=dropped_z,
    )


def plot_resonator_flux_fit(fit: ResonatorFluxFit):
    """Return the measured map and method-appropriate diagnostics."""
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
    plot_y = (
        np.broadcast_to(fit.z_gain[:, None], fit.frequencies_mhz.shape)
        if np.asarray(fit.frequencies_mhz).ndim == 2
        else fit.z_gain
    )
    mesh = axes[0].pcolormesh(
        fit.frequencies_mhz,
        plot_y,
        fit.amplitude_db,
        shading="auto",
        cmap="turbo",
    )
    fit_line = axes[0].plot(
        frequency_dense,
        z_dense,
        "k-",
        linewidth=2,
        label=(
            "cosine fit"
            if fit.model == MODEL_NAME
            else f"{fit.fit_method} linear lookup"
        ),
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
        label=(
            f"refined {fit.feature_polarity}"
            if fit.fit_method == "fit"
            else f"selected {fit.fit_method} bin"
        ),
    )
    axes[0].set(
        xlabel="Readout frequency (MHz)",
        ylabel="Z gain",
        title="Calibration over measured amplitude",
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

    if fit.model == MODEL_NAME:
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
    else:
        axes[2].plot(
            fit.z_gain,
            fit.feature_contrast_db,
            "o-",
            markersize=4,
        )
        axes[2].set(
            xlabel="Z gain",
            ylabel="Selected feature contrast (dB)",
            title=f"Per-row {fit.fit_method} contrast",
        )
    axes[2].grid(alpha=0.3)
    figure.suptitle(f"Resonator readout calibration ({fit.fit_method})")
    return figure


def calibration_record(fit: ResonatorFluxFit) -> dict:
    """Build the accepted v3 lookup record for ``calibration.yml``."""
    quality = {
        "complete_z_rows": int(len(fit.z_gain)),
        "dropped_z_rows": to_builtin(fit.dropped_z_gain),
        "feature_polarity": fit.feature_polarity,
    }
    if fit.model == MODEL_NAME:
        value = {"parameters": dict(fit.parameters)}
        uncertainty = {
            key: float(item)
            for key, item in fit.statistics.items()
            if key != "r_squared"
        }
        quality.update(
            {
                "r_squared": float(fit.statistics["r_squared"]),
                "feature_contrast_db": {
                    "minimum": float(np.min(fit.feature_contrast_db)),
                    "median": float(np.median(fit.feature_contrast_db)),
                },
                # Backward-compatible alias retained for older reports.
                "notch_depth_db": {
                    "minimum": float(np.min(fit.feature_contrast_db)),
                    "median": float(np.median(fit.feature_contrast_db)),
                },
            }
        )
        notes = (
            f"Per-Z {fit.feature_polarity} features use Gaussian smoothing, "
            "linear-background removal, and three-bin parabolic center "
            "refinement before the cosine fit."
        )
    elif fit.model == LOOKUP_MODEL_NAME:
        value = {
            "z_gain": to_builtin(fit.z_gain),
            "r_freq_mhz": to_builtin(fit.extracted_frequencies_mhz),
            "selection_method": fit.fit_method,
        }
        uncertainty = {
            "frequency_bin_mhz": float(fit.statistics["frequency_bin_mhz"])
        }
        quality.update(
            {
                "selection_method": fit.fit_method,
                "feature_contrast_db": {
                    "minimum": float(np.min(fit.feature_contrast_db)),
                    "median": float(np.median(fit.feature_contrast_db)),
                },
            }
        )
        notes = (
            f"Each Z row uses the smoothed amplitude {fit.fit_method} "
            "frequency bin. Frequencies between measured Z rows use "
            "linear interpolation."
        )
    else:
        raise ConfigError(
            f"unsupported resonator calibration model {fit.model!r}"
        )
    return {
        "value": value,
        "unit": "MHz",
        "uncertainty": uncertainty,
        "provenance": {
            "source": str(fit.source_csv),
            "fitted_at": utc_now(),
            "analysis": "quickexp_v3.resonator_flux.fit_resonator_flux",
        },
        "quality": quality,
        "valid_domain": {
            "z_gain": [
                float(np.min(fit.z_gain)),
                float(np.max(fit.z_gain)),
            ]
        },
        "model": fit.model,
        "notes": notes,
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def accept_fit(
    project_root: Path,
    fit: ResonatorFluxFit,
    *,
    minimum_r_squared: float,
    maximum_rmse_mhz: float,
    force_write: bool = False,
) -> Path:
    """Atomically install an accepted fit or sampled lookup."""
    gates_passed = fit.passes(
        minimum_r_squared=minimum_r_squared,
        maximum_rmse_mhz=maximum_rmse_mhz,
    )
    if not gates_passed and not force_write:
        if fit.model == MODEL_NAME:
            detail = (
                f"R^2={fit.statistics['r_squared']:.6f} "
                f"(minimum {minimum_r_squared}), "
                f"RMSE={fit.statistics['rmse_mhz']:.6f} MHz "
                f"(maximum {maximum_rmse_mhz} MHz)"
            )
        else:
            detail = "sampled lookup contains invalid values"
        raise AnalysisError(f"calibration was not accepted: {detail}")

    updates = annotate_forced_write(
        {
            "lookups.resonator_vs_flux": calibration_record(fit),
        },
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)
