"""Qubit-spectroscopy ridge extraction and constrained transmon flux fit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .errors import AnalysisError
from .fit_calibration import annotate_forced_write, write_calibration_records
from .fit_stats import oriented_rotate_iq, pinned_parameters, r_squared
from .flux_lookup import register_model
from .native_map import load_native_map, load_native_row_map
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
    signal_map: Any
    ridge_rows: tuple
    fitted_frequencies_mhz: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, Any]
    identifiable: Mapping[str, bool]
    metadata: Mapping[str, Any]
    frequency_rows_mhz: tuple = ()
    signal_rows: tuple = ()

    def frequency(self, z_gain: Any):
        return transmon_frequency(z_gain, **dict(self.parameters))

    @property
    def is_ragged(self) -> bool:
        return bool(self.frequency_rows_mhz)

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


def _fit_window_bounds(
    window: Optional[Sequence[float]],
    *,
    label: str,
) -> Optional[tuple[float, float]]:
    if window is None:
        return None
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
    return lower, upper


def _contiguous_frequency_slices(frequencies: np.ndarray) -> tuple[slice, ...]:
    """Split one row at deliberate gaps before smoothing or fitting."""
    values = np.asarray(frequencies, dtype=float)
    if values.size < 2:
        return ()
    steps = np.diff(values)
    positive = steps[steps > 0]
    if positive.size == 0:
        return ()
    native_step = float(np.median(positive))
    tolerance = max(abs(native_step) * 1e-6, 1e-12)
    breaks = np.flatnonzero(steps > 1.5 * native_step + tolerance) + 1
    boundaries = np.concatenate(([0], breaks, [values.size]))
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
        if stop - start >= 12
    )


def _extract_segmented_row(
    frequencies: np.ndarray,
    iq: np.ndarray,
    *,
    minimum_row_r_squared: float,
    minimum_row_contrast_snr: float,
) -> Optional[tuple]:
    """Fit every contiguous band and return the strongest credible feature."""
    candidates = []
    for segment in _contiguous_frequency_slices(frequencies):
        extracted = _extract_row(frequencies[segment], iq[segment])
        if extracted is not None:
            candidates.append(extracted)
    if not candidates:
        return None
    passing = [
        candidate
        for candidate in candidates
        if candidate[2] >= minimum_row_r_squared
        and candidate[3] >= minimum_row_contrast_snr
    ]
    choices = passing or candidates
    return max(choices, key=lambda candidate: (candidate[3], candidate[2]))


def _extract_row_candidates(
    frequencies: np.ndarray,
    iq: np.ndarray,
    *,
    maximum_candidates: int = 10,
) -> tuple:
    """Fit several local Lorentz candidates across all contiguous bands."""
    candidates = []
    for segment in _contiguous_frequency_slices(frequencies):
        x = np.asarray(frequencies[segment], dtype=float)
        complex_row = np.asarray(iq[segment], dtype=complex)
        projected, _ = oriented_rotate_iq(complex_row)
        projected = np.asarray(projected, dtype=float)
        smoothed = gaussian_filter1d(projected, 2.0, mode="nearest")
        baseline = gaussian_filter1d(
            smoothed,
            max(10.0, min(100.0, x.size / 12.0)),
            mode="nearest",
        )
        deviation = np.abs(smoothed - baseline)
        prominence = max(
            float(np.std(deviation)) * 0.5,
            float(np.ptp(deviation)) * 0.03,
            np.finfo(float).eps,
        )
        peaks, properties = find_peaks(
            deviation,
            distance=8,
            prominence=prominence,
        )
        if peaks.size == 0:
            extracted = _extract_row(x, complex_row)
            if extracted is not None:
                candidates.append(extracted)
            continue
        order = np.argsort(deviation[peaks])[::-1][:maximum_candidates]
        step = float(np.median(np.diff(x)))
        half_width = max(12.0 * step, float(np.ptp(x)) / 60.0)
        for peak_index in peaks[order]:
            mask = np.abs(x - x[peak_index]) <= half_width
            if np.count_nonzero(mask) < 12:
                continue
            local_x = x[mask]
            local_y = projected[mask]
            try:
                values, covariance, fitted, _lower, _upper = (
                    _fit_lorentz_components(local_x, local_y, 1)
                )
            except AnalysisError:
                continue
            center = float(values[3])
            if center < float(local_x.min()) or center > float(local_x.max()):
                continue
            errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            residual = local_y - fitted
            rmse = float(np.sqrt(np.mean(residual**2)))
            contrast = abs(float(values[2])) / max(
                rmse,
                np.finfo(float).eps,
            )
            candidates.append(
                (
                    center,
                    float(errors[3]),
                    r_squared(local_y, fitted),
                    contrast,
                    local_y,
                )
            )
    if not candidates:
        return ()
    candidates.sort(key=lambda candidate: (candidate[3], candidate[2]), reverse=True)
    deduplicated = []
    typical_step = float(np.median(np.diff(frequencies)))
    separation = max(2.0 * typical_step, 0.5)
    for candidate in candidates:
        if all(abs(candidate[0] - kept[0]) > separation for kept in deduplicated):
            deduplicated.append(candidate)
    return tuple(deduplicated)


def _select_transmon_candidates(
    z_gain: np.ndarray,
    candidate_rows: Sequence[Sequence[tuple]],
    *,
    ec_mhz: float,
    period_hint: Optional[float],
    sweet_spot_hint: Optional[float],
) -> tuple[tuple[RidgeRow, ...], tuple, np.ndarray, Mapping, Mapping]:
    """Select one globally transmon-like candidate from each usable row."""
    usable = [index for index, row in enumerate(candidate_rows) if row]
    if len(usable) < 4:
        raise AnalysisError("fewer than four rows contain fitted ridge candidates")
    z = np.asarray([z_gain[index] for index in usable], dtype=float)
    candidates = [candidate_rows[index] for index in usable]
    z_step = float(np.median(np.diff(z))) if z.size > 1 else 0.01
    z_span = max(float(np.ptp(z)), 4.0 * z_step)
    if period_hint is not None:
        period_seed = float(period_hint)
        periods = np.linspace(0.5 * period_seed, 1.2 * period_seed, 18)
    else:
        periods = np.linspace(max(4.0 * z_step, z_span), 3.0 * z_span, 18)
    if sweet_spot_hint is not None:
        sweet_center = float(sweet_spot_hint)
        sweets = np.linspace(sweet_center - 0.025, sweet_center + 0.025, 11)
    else:
        sweets = np.linspace(float(z.min()), float(z.max()), 13)
    asymmetries = np.linspace(0.05, 0.95, 13)
    # Retain several distinct coarse paths.  A single best coarse template can
    # lock onto a nearby parallel feature; nonlinear refinement is a much
    # better discriminator once a complete candidate path is available.
    initial_paths = {}
    for sweet in sweets:
        nearest_rows = np.argsort(np.abs(z - sweet))[: min(3, z.size)]
        f_max_seeds = sorted(
            {
                float(candidate[0])
                for row_index in nearest_rows
                for candidate in candidates[int(row_index)]
            }
        )
        for period in periods:
            for f_max in f_max_seeds:
                for asymmetry in asymmetries:
                    prediction = transmon_frequency(
                        z,
                        f_max_mhz=f_max,
                        period_z=float(period),
                        sweet_spot_z=float(sweet),
                        asymmetry=float(asymmetry),
                        ec_mhz=ec_mhz,
                    )
                    indices = tuple(
                        int(
                            np.argmin(
                                [abs(item[0] - expected) for item in row]
                            )
                        )
                        for row, expected in zip(candidates, prediction)
                    )
                    centers = np.asarray(
                        [row[index][0] for row, index in zip(candidates, indices)],
                        dtype=float,
                    )
                    score = float(np.sqrt(np.mean((centers - prediction) ** 2)))
                    previous = initial_paths.get(indices)
                    if previous is None or score < previous:
                        initial_paths[indices] = score
    if not initial_paths:
        raise AnalysisError("could not initialize custom-path ridge tracking")

    best = None
    ranked_paths = sorted(initial_paths.items(), key=lambda item: item[1])[:12]
    for indices, coarse_score in ranked_paths:
        chosen = tuple(
            row[index] for row, index in zip(candidates, indices)
        )
        try:
            for _iteration in range(3):
                centers = np.asarray(
                    [candidate[0] for candidate in chosen], dtype=float
                )
                uncertainties = np.asarray(
                    [
                        max(candidate[1], np.finfo(float).eps)
                        for candidate in chosen
                    ],
                    dtype=float,
                )
                parameters, fitted, statistics, identifiable = fit_transmon_flux(
                    z,
                    centers,
                    uncertainty_mhz=uncertainties,
                    ec_mhz=ec_mhz,
                    period_hint=period_hint,
                )
                updated = tuple(
                    min(row, key=lambda item: abs(item[0] - expected))
                    for row, expected in zip(candidates, fitted)
                )
                if all(new is old for new, old in zip(updated, chosen)):
                    break
                chosen = updated

            # Refit after the final reassignment so the reported model and
            # statistics always correspond to the returned ridge points.
            centers = np.asarray(
                [candidate[0] for candidate in chosen], dtype=float
            )
            uncertainties = np.asarray(
                [max(candidate[1], np.finfo(float).eps) for candidate in chosen],
                dtype=float,
            )
            parameters, fitted, statistics, identifiable = fit_transmon_flux(
                z,
                centers,
                uncertainty_mhz=uncertainties,
                ec_mhz=ec_mhz,
                period_hint=period_hint,
            )
        except AnalysisError:
            continue
        final_score = float(statistics["rmse_mhz"])
        final_score += 1_000.0 * len(statistics["pinned_parameters"])
        result = (
            final_score,
            coarse_score,
            chosen,
            parameters,
            fitted,
            statistics,
            identifiable,
        )
        if best is None or result[:2] < best[:2]:
            best = result
    if best is None:
        raise AnalysisError("could not refine a custom-path transmon ridge")
    (
        _final_score,
        _coarse_score,
        chosen,
        parameters,
        fitted,
        statistics,
        identifiable,
    ) = best
    ridge_rows = tuple(
        RidgeRow(
            z_gain=float(z_value),
            center_mhz=float(candidate[0]),
            uncertainty_mhz=max(float(candidate[1]), np.finfo(float).eps),
            r_squared=float(candidate[2]),
            contrast_snr=float(candidate[3]),
        )
        for z_value, candidate in zip(z, chosen)
    )
    return ridge_rows, parameters, fitted, statistics, identifiable


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
    sweet_spot_hint: Optional[float] = None,
    frequency_window_mhz: Optional[Sequence[float]] = None,
    flux_window_z: Optional[Sequence[float]] = None,
    minimum_row_r_squared: float = 0.05,
    minimum_row_contrast_snr: float = 1.0,
) -> QubitFluxFit:
    try:
        native = load_native_map(csv_path)
        outer_values = np.asarray(native.outer, dtype=float)
        inner_rows = tuple(
            np.asarray(native.inner, dtype=float) for _ in native.outer
        )
        iq_rows = tuple(
            np.asarray(row, dtype=complex) for row in native.complex_signal
        )
        amplitude = native.signals.get("amplitude")
        amplitude_rows = (
            tuple(np.asarray(row, dtype=float) for row in amplitude)
            if amplitude is not None
            else tuple(np.abs(row) for row in iq_rows)
        )
        rectangular_source = True
    except AnalysisError:
        native = load_native_row_map(csv_path)
        outer_values = np.asarray(native.outer, dtype=float)
        inner_rows = tuple(
            np.asarray(row, dtype=float) for row in native.inner_rows
        )
        iq_rows = tuple(
            np.asarray(row, dtype=complex)
            for row in native.complex_signal_rows
        )
        amplitude = native.signals.get("amplitude")
        amplitude_rows = (
            tuple(np.asarray(row, dtype=float) for row in amplitude)
            if amplitude is not None
            else tuple(np.abs(row) for row in iq_rows)
        )
        rectangular_source = False
    if "z" not in native.outer_label.lower() or "freq" not in native.inner_label.lower():
        raise AnalysisError("qubit flux fitting requires Z as outer and frequency as inner axis")
    flux_mask = _fit_window_mask(
        outer_values,
        flux_window_z,
        label="flux",
        minimum_points=4,
    )
    frequency_bounds = _fit_window_bounds(
        frequency_window_mhz,
        label="frequency",
    )
    selected_indices = np.flatnonzero(flux_mask)
    map_rows = []
    map_frequency_rows = []
    map_signal_rows = []
    ridge_rows = []
    candidate_rows = []
    dropped_frequency_rows = []
    disjoint_rows = 0
    for row_index in selected_indices:
        z_value = float(outer_values[row_index])
        frequencies = np.asarray(inner_rows[row_index], dtype=float)
        iq = np.asarray(iq_rows[row_index], dtype=complex)
        signal = np.asarray(amplitude_rows[row_index], dtype=float)
        finite = (
            np.isfinite(frequencies)
            & np.isfinite(iq.real)
            & np.isfinite(iq.imag)
            & np.isfinite(signal)
        )
        if frequency_bounds is not None:
            finite &= (
                (frequencies >= frequency_bounds[0])
                & (frequencies <= frequency_bounds[1])
            )
        frequencies, iq, signal = (
            values[finite] for values in (frequencies, iq, signal)
        )
        if frequencies.size < 12:
            dropped_frequency_rows.append(z_value)
            continue
        order = np.argsort(frequencies)
        frequencies, iq, signal = (
            values[order] for values in (frequencies, iq, signal)
        )
        if np.any(np.diff(frequencies) <= 0):
            raise AnalysisError(
                f"qubit-flux row Z={z_value:+.6g} has duplicate frequencies"
            )
        segments = _contiguous_frequency_slices(frequencies)
        if len(segments) > 1:
            disjoint_rows += 1
        if not segments:
            dropped_frequency_rows.append(z_value)
            continue
        map_rows.append(z_value)
        map_frequency_rows.append(frequencies)
        map_signal_rows.append(signal)
        if rectangular_source:
            extracted = _extract_segmented_row(
                frequencies,
                iq,
                minimum_row_r_squared=minimum_row_r_squared,
                minimum_row_contrast_snr=minimum_row_contrast_snr,
            )
        else:
            candidates = tuple(
                candidate
                for candidate in _extract_row_candidates(frequencies, iq)
                if candidate[2] >= minimum_row_r_squared
                and candidate[3] >= minimum_row_contrast_snr
            )
            candidate_rows.append(candidates)
            continue
        if extracted is None:
            continue
        center, uncertainty, row_r2, contrast, projected = extracted
        if row_r2 >= minimum_row_r_squared and contrast >= minimum_row_contrast_snr:
            ridge_rows.append(
                RidgeRow(
                    z_gain=z_value,
                    center_mhz=center,
                    uncertainty_mhz=max(uncertainty, np.finfo(float).eps),
                    r_squared=row_r2,
                    contrast_snr=contrast,
                )
            )
    if len(map_rows) < 4:
        if frequency_window_mhz is not None:
            raise AnalysisError(
                "frequency fit window contains fewer than 12 acquired points "
                "in at least four rows"
            )
        raise AnalysisError(
            "fewer than four custom-path rows contain enough acquired points"
        )
    map_z_gain = np.asarray(map_rows, dtype=float)
    if rectangular_source:
        if len(ridge_rows) < 4:
            raise AnalysisError(
                "fewer than four qubit-flux ridge rows could be fitted"
            )
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
    else:
        (
            ridge_rows,
            parameters,
            fitted,
            statistics,
            identifiable,
        ) = _select_transmon_candidates(
            map_z_gain,
            candidate_rows,
            ec_mhz=ec_mhz,
            period_hint=period_hint,
            sweet_spot_hint=sweet_spot_hint,
        )
        z_values = np.asarray([row.z_gain for row in ridge_rows])
    statistics = {
        **statistics,
        "ridge_rows": len(ridge_rows),
        "ec_source": "hardware.q_delta" if ec_mhz == 180.0 else "explicit",
        "frequency_fit_window_mhz": [
            float(min(np.min(row) for row in map_frequency_rows)),
            float(max(np.max(row) for row in map_frequency_rows)),
        ],
        "flux_fit_window_z": [
            float(np.min(map_z_gain)),
            float(np.max(map_z_gain)),
        ],
        "ragged_path_map": not rectangular_source,
        "ridge_selection": (
            "global_transmon_candidates"
            if not rectangular_source
            else "strongest_feature_per_row"
        ),
        "period_hint": None if period_hint is None else float(period_hint),
        "sweet_spot_hint": (
            None if sweet_spot_hint is None else float(sweet_spot_hint)
        ),
        "row_point_counts": [int(row.size) for row in map_frequency_rows],
        "disjoint_frequency_rows": int(disjoint_rows),
        "dropped_frequency_window_rows": dropped_frequency_rows,
    }
    same_frequency_axis = all(
        row.shape == map_frequency_rows[0].shape
        and np.allclose(row, map_frequency_rows[0], rtol=1e-10, atol=1e-12)
        for row in map_frequency_rows[1:]
    )
    if same_frequency_axis:
        frequencies_mhz = map_frequency_rows[0]
        signal_map = np.vstack(map_signal_rows)
        frequency_rows_mhz = ()
        signal_rows = ()
    else:
        frequencies_mhz = np.unique(np.concatenate(map_frequency_rows))
        signal_map = tuple(map_signal_rows)
        frequency_rows_mhz = tuple(map_frequency_rows)
        signal_rows = tuple(map_signal_rows)
    return QubitFluxFit(
        source_csv=native.source_csv,
        z_gain=z_values,
        map_z_gain=map_z_gain,
        frequencies_mhz=frequencies_mhz,
        signal_map=signal_map,
        ridge_rows=tuple(ridge_rows),
        fitted_frequencies_mhz=fitted,
        parameters=parameters,
        statistics=statistics,
        identifiable=identifiable,
        metadata=native.metadata,
        frequency_rows_mhz=frequency_rows_mhz,
        signal_rows=signal_rows,
    )


def plot_qubit_flux_fit(fit: QubitFluxFit):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    if fit.is_ragged:
        plot_frequency = np.concatenate(fit.frequency_rows_mhz)
        plot_z = np.concatenate(
            [
                np.full(row.shape, fit.map_z_gain[index], dtype=float)
                for index, row in enumerate(fit.frequency_rows_mhz)
            ]
        )
        mesh = axes[0].scatter(
            plot_frequency,
            plot_z,
            c=np.concatenate(fit.signal_rows),
            marker="s",
            s=9,
            linewidths=0,
            cmap="turbo",
        )
    else:
        mesh = axes[0].pcolormesh(
            fit.frequencies_mhz,
            fit.map_z_gain,
            fit.signal_map,
            shading="auto",
            cmap="turbo",
        )
    axes[0].plot(
        [row.center_mhz for row in fit.ridge_rows],
        [row.z_gain for row in fit.ridge_rows],
        "wo",
        markeredgecolor="black",
    )
    axes[0].set(
        xlabel="Qubit frequency (MHz)",
        ylabel="Z gain",
        title=(
            "Extracted ridge (custom path)"
            if fit.is_ragged
            else "Extracted ridge"
        ),
    )
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
