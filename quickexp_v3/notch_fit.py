"""Complex resonator-notch and multi-feature spectroscopy fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .errors import AnalysisError, ConfigError
from .fit_calibration import write_calibration_records
from .fit_stats import (
    bic,
    oriented_rotate_iq,
    pinned_parameters,
    residual_ripple_fraction,
    r_squared,
)
from .native_fit import (
    NativeTrace,
    SPECTROSCOPY_KINDS,
    SpectroscopyFit,
    fit_spectroscopy,
    load_native_trace,
)
from .trace_qc import qc_trace
from .util import utc_now


NOTCH_PARAMETER_NAMES = (
    "center_mhz",
    "fwhm_mhz",
    "baseline_amplitude",
    "baseline_slope_per_mhz",
    "baseline_phase_rad",
    "cable_delay_rad_per_mhz",
    "coupling_real",
    "coupling_imag",
)


def complex_notch_model(
    frequency_mhz: Any,
    parameters: Sequence[float],
    *,
    reference_mhz: Optional[float] = None,
) -> np.ndarray:
    """Evaluate the eight-parameter complex notch model."""
    frequency = np.asarray(frequency_mhz, dtype=float)
    values = np.asarray(parameters, dtype=float)
    if values.size != 8:
        raise AnalysisError("complex notch model requires eight parameters")
    center, linewidth, a0, a1, phi0, tau, coupling_re, coupling_im = values
    shifted = frequency - center
    baseline = (a0 + a1 * shifted) * np.exp(
        1j * (phi0 + tau * shifted)
    )
    coupling = coupling_re + 1j * coupling_im
    return baseline * (1.0 - coupling / (1.0 + 2j * shifted / linewidth))


@dataclass(frozen=True)
class NotchFit:
    source_csv: Path
    source_yml: Path
    model: str
    x: np.ndarray
    iq: np.ndarray
    fitted_complex: np.ndarray
    parameters: Mapping[str, Any]
    statistics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def center_mhz(self) -> float:
        return float(self.parameters["center_mhz"])

    def passes(
        self,
        *,
        minimum_r_squared: float = 0.90,
        minimum_contrast_snr: float = 5.0,
        minimum_edge_distance_over_fwhm: float = 1.0,
    ) -> bool:
        return bool(
            np.isfinite(self.statistics.get("r_squared_complex", np.nan))
            and self.statistics["r_squared_complex"] >= minimum_r_squared
            and self.statistics["contrast_snr"] >= minimum_contrast_snr
            and self.statistics["edge_distance_over_fwhm"]
            >= minimum_edge_distance_over_fwhm
            and not self.statistics.get("pinned_parameters")
        )


def _notch_initial_guess(trace: NativeTrace) -> tuple:
    x = trace.x
    iq = trace.iq
    magnitude = np.abs(iq)
    smoothed = gaussian_filter1d(magnitude, 2.0, mode="nearest")
    minimum_index = int(np.argmin(smoothed))
    center = float(x[minimum_index])
    edge_count = max(3, min(30, x.size // 5))
    edge_indices = np.r_[:edge_count, x.size - edge_count : x.size]
    reference = float(np.mean(x))
    a1, a0 = np.polyfit(
        x[edge_indices] - reference,
        magnitude[edge_indices],
        1,
    )
    baseline_level = float(np.median(magnitude[edge_indices]))
    half_depth = float(smoothed[minimum_index] + (baseline_level - smoothed[minimum_index]) / 2)
    below = np.flatnonzero(smoothed <= half_depth)
    linewidth = (
        float(x[below[-1]] - x[below[0]])
        if below.size >= 2
        else float(np.ptp(x) / 20.0)
    )
    linewidth = max(linewidth, float(np.median(np.diff(x))))
    phase = np.unwrap(np.angle(iq))[edge_indices]
    tau, phi0 = np.polyfit(
        x[edge_indices] - reference,
        phase,
        1,
    )
    baseline_center = (a0 + a1 * (center - reference)) * np.exp(
        1j * (phi0 + tau * (center - reference))
    )
    coupling = (
        1.0 - iq[minimum_index] / baseline_center
        if abs(baseline_center) > np.finfo(float).eps
        else 0.5 + 0j
    )
    return np.asarray(
        [
            center,
            linewidth,
            max(float(a0), np.finfo(float).eps),
            float(a1),
            float(phi0),
            float(tau),
            float(coupling.real),
            float(coupling.imag),
        ]
    )


def fit_complex_notch(csv_path: Path) -> NotchFit:
    """Fit a native ResonatorSpectroscopy trace in the complex plane."""
    trace = load_native_trace(
        csv_path,
        quick_class="ResonatorSpectroscopy",
        axis_text="readout pulse frequency",
        minimum_points=12,
    )
    scalar = fit_spectroscopy(
        csv_path,
        kind="resonator",
        signal="amplitude",
    )
    center_seed = float(scalar.center_mhz)
    linewidth_seed = float(scalar.parameters["fwhm_mhz"])
    full_x = trace.x
    full_iq = trace.iq
    full_step = float(np.median(np.diff(full_x)))
    half_window = max(8.0 * linewidth_seed, 20.0 * full_step)
    half_window = min(
        max(half_window, 1.5),
        min(12.0, float(np.ptp(full_x)) / 2.0),
    )
    window = np.abs(full_x - center_seed) <= half_window
    x = full_x[window]
    iq = full_iq[window]
    if x.size < 16:
        raise AnalysisError("complex notch fit window contains fewer than 16 points")
    reference = center_seed
    span = float(np.ptp(x))
    step = float(np.median(np.diff(x)))
    window_trace = NativeTrace(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        quick_class=trace.quick_class,
        axis_label=trace.axis_label,
        axis_unit=trace.axis_unit,
        x=x,
        amplitude=np.abs(iq),
        phase=np.angle(iq),
        iq=iq,
        metadata=trace.metadata,
    )
    initial = _notch_initial_guess(window_trace)
    initial[0] = center_seed
    initial[1] = max(linewidth_seed, 2.0 * step)
    initial[6] = float(
        np.clip(
            abs(scalar.parameters["amplitude"])
            / max(initial[2], np.finfo(float).eps),
            0.005,
            1.5,
        )
    )
    initial[7] = 0.0
    amplitude_scale = max(float(np.max(np.abs(iq))), np.finfo(float).eps)
    lower = np.asarray(
        [
            max(float(x[0]), center_seed - min(2.0 * linewidth_seed, span / 4.0)),
            max(1.5 * step, span * 1e-9),
            0.0,
            -max(initial[2], 1e-6) / max(step, 1e-6),
            initial[4] - 2.0 * np.pi,
            initial[5] - 10.0,
            -2.0,
            -2.0,
        ]
    )
    upper = np.asarray(
        [
            min(float(x[-1]), center_seed + min(2.0 * linewidth_seed, span / 4.0)),
            min(span, 50.0),
            max(4.0 * initial[2], 4.0 * amplitude_scale, 1e-6),
            max(initial[2], 1e-6) / max(step, 1e-6),
            initial[4] + 2.0 * np.pi,
            initial[5] + 10.0,
            2.0,
            2.0,
        ]
    )
    candidate_indices = [int(np.argmin(np.abs(x - center_seed)))]
    noise = max(
        float(np.median(np.abs(np.diff(iq.real) - np.median(np.diff(iq.real))))) / 0.6745 / np.sqrt(2.0),
        float(np.median(np.abs(np.diff(iq.imag) - np.median(np.diff(iq.imag))))) / 0.6745 / np.sqrt(2.0),
        1e-9,
    )
    best = None
    for index in candidate_indices:
        for width_factor in (0.5, 1.0, 2.0, 4.0):
            guess = initial.copy()
            guess[0] = x[index]
            guess[1] = np.clip(initial[1] * width_factor, lower[1] * 1.01, upper[1] * 0.99)

            def residual(parameters):
                difference = complex_notch_model(
                    x,
                    parameters,
                    reference_mhz=reference,
                ) - iq
                return np.r_[difference.real, difference.imag] / noise

            try:
                result = least_squares(
                    residual,
                    guess,
                    bounds=(lower, upper),
                    loss="soft_l1",
                    f_scale=1.0,
                    max_nfev=10_000,
                )
            except (ValueError, FloatingPointError):
                continue
            rss = float(np.sum(residual(result.x) ** 2))
            if result.success and (best is None or rss < best[0]):
                best = (rss, result)
    if best is None:
        raise AnalysisError("complex notch fit did not converge")
    _scaled_rss, result = best
    fitted = complex_notch_model(x, result.x, reference_mhz=reference)
    residual_complex = iq - fitted
    rss = float(np.sum(np.abs(residual_complex) ** 2))
    dof = max(2 * x.size - result.x.size, 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * rss / dof
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    center, linewidth, a0, a1, phi0, tau, c_re, c_im = result.x
    coupling = abs(c_re + 1j * c_im)
    baseline_center = a0 * np.exp(1j * phi0)
    fitted_center = complex_notch_model(
        np.asarray([center]),
        result.x,
        reference_mhz=reference,
    )[0]
    depth_db = 20.0 * np.log10(
        max(abs(baseline_center), np.finfo(float).eps)
        / max(abs(fitted_center), np.finfo(float).eps)
    )
    rmse = float(np.sqrt(np.mean(np.abs(residual_complex) ** 2)))
    magnitude_depth = max(
        float(np.median(np.abs(iq)[np.r_[: max(2, x.size // 5), -max(2, x.size // 5) :]]))
        - float(np.min(np.abs(iq))),
        0.0,
    )
    names = list(NOTCH_PARAMETER_NAMES)
    pinned = pinned_parameters(
        dict(zip(names, result.x)),
        dict(zip(names, lower)),
        dict(zip(names, upper)),
    )
    complex_r2 = r_squared(iq, fitted)
    amplitude_r2 = r_squared(np.abs(iq), np.abs(fitted))
    reported_center = (
        float(center)
        if amplitude_r2 >= 0.98
        else center_seed
    )
    reported_uncertainty = max(
        float(scalar.parameters["center_uncertainty_mhz"]),
        float(standard_errors[0]),
    )
    parameters = {
        "center_mhz": reported_center,
        "center_uncertainty_mhz": reported_uncertainty,
        "complex_model_center_mhz": float(center),
        "fwhm_mhz": float(abs(linewidth)),
        "fwhm_uncertainty_mhz": float(standard_errors[1]),
        "loaded_q": float(abs(center / linewidth)),
        "coupling_q": float(abs(center / linewidth) / coupling) if coupling > 0 else float("inf"),
        "depth_db": float(depth_db),
        "baseline_amplitude": float(a0),
        "baseline_slope_per_mhz": float(a1),
        "baseline_phase_rad": float(phi0),
        "cable_delay_rad_per_mhz": float(tau),
        "coupling_real": float(c_re),
        "coupling_imag": float(c_im),
        "reference_mhz": float(center),
    }
    edge_distance = float(
        min(reported_center - x[0], x[-1] - reported_center)
    )
    statistics = {
        "r_squared_complex": complex_r2,
        "r_squared_amplitude": amplitude_r2,
        "rmse": rmse,
        "contrast_snr": float(magnitude_depth / max(rmse, np.finfo(float).eps)),
        "edge_distance_over_fwhm": float(edge_distance / abs(linewidth)),
        "edge_distance_mhz": edge_distance,
        "pinned_parameters": pinned,
        "qc": qc_trace(x, iq).as_dict(),
        "points": int(x.size),
    }
    return NotchFit(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        model="complex_notch",
        x=x,
        iq=iq,
        fitted_complex=fitted,
        parameters=parameters,
        statistics=statistics,
        metadata=trace.metadata,
    )


def fit_notch_amplitude(csv_path: Path) -> NotchFit:
    """Adapt the established Lorentzian fitter to the :class:`NotchFit` API."""
    fit = fit_spectroscopy(csv_path, kind="resonator", signal="amplitude")
    fitted_complex = fit.fitted * np.exp(1j * np.angle(fit.iq))
    parameters = dict(fit.parameters)
    statistics = {
        "r_squared_complex": float(fit.statistics["r_squared"]),
        "r_squared_amplitude": float(fit.statistics["r_squared"]),
        "rmse": float(fit.statistics["rmse"]),
        "contrast_snr": float(fit.statistics["contrast_snr"]),
        "edge_distance_over_fwhm": float(
            fit.statistics["edge_distance_mhz"] / fit.parameters["fwhm_mhz"]
        ),
        "edge_distance_mhz": float(fit.statistics["edge_distance_mhz"]),
        "pinned_parameters": [],
        "qc": qc_trace(fit.x, fit.iq).as_dict(),
        "points": int(len(fit.x)),
    }
    return NotchFit(
        source_csv=fit.source_csv,
        source_yml=fit.source_yml,
        model="amplitude_lorentzian_fallback",
        x=fit.x,
        iq=fit.iq,
        fitted_complex=fitted_complex,
        parameters=parameters,
        statistics=statistics,
        metadata=fit.metadata,
    )


def fit_notch(csv_path: Path, *, allow_fallback: bool = True) -> NotchFit:
    try:
        fit = fit_complex_notch(csv_path)
        if fit.statistics["r_squared_complex"] < 0.75:
            raise AnalysisError("complex notch fit has implausibly low R-squared")
        return fit
    except (AnalysisError, ValueError, FloatingPointError):
        if not allow_fallback:
            raise
        return fit_notch_amplitude(csv_path)


def plot_notch_fit(fit: NotchFit):
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
    axes[0].plot(fit.x, np.abs(fit.iq), "o", markersize=3, label="data")
    axes[0].plot(fit.x, np.abs(fit.fitted_complex), "-", linewidth=2, label=fit.model)
    axes[0].axvline(fit.center_mhz, color="tab:red", linestyle="--")
    axes[0].set(xlabel="Readout frequency (MHz)", ylabel="|S21|", title="Resonator notch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(fit.x, np.abs(fit.iq) - np.abs(fit.fitted_complex), ".-")
    axes[1].axhline(0.0, color="0.4")
    axes[1].set(xlabel="Readout frequency (MHz)", ylabel="Magnitude residual", title="Residual")
    axes[1].grid(alpha=0.3)
    axes[2].plot(fit.iq.real, fit.iq.imag, "o", markersize=3, label="data")
    axes[2].plot(fit.fitted_complex.real, fit.fitted_complex.imag, "-", label="fit")
    axes[2].set(xlabel="I", ylabel="Q", title="Complex trajectory")
    axes[2].axis("equal")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    figure.suptitle(f"Fit from {fit.source_csv.name}")
    return figure


def notch_calibration_record(fit: NotchFit) -> dict:
    return {
        "value": fit.center_mhz,
        "unit": "MHz",
        "uncertainty": {
            "center_mhz": float(fit.parameters["center_uncertainty_mhz"]),
            "fwhm_mhz": float(fit.parameters["fwhm_mhz"]),
            "rmse": float(fit.statistics["rmse"]),
        },
        "provenance": {
            "source": str(fit.source_csv),
            "fitted_at": utc_now(),
            "analysis": "quickexp_v3.notch_fit.fit_notch",
        },
        "quality": dict(fit.statistics),
        "valid_domain": {
            "z_gain": [
                float(
                    fit.metadata.get("parameters", {})
                    .get("var", {})
                    .get("z_gain", 0.0)
                )
            ]
            * 2
        },
        "model": fit.model,
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def accept_notch_fit(
    project_root: Path,
    fit: NotchFit,
    *,
    minimum_r_squared: float = 0.90,
    minimum_contrast_snr: float = 5.0,
    minimum_edge_distance_over_fwhm: float = 1.0,
) -> Path:
    if not fit.passes(
        minimum_r_squared=minimum_r_squared,
        minimum_contrast_snr=minimum_contrast_snr,
        minimum_edge_distance_over_fwhm=minimum_edge_distance_over_fwhm,
    ):
        raise AnalysisError("complex notch fit did not pass acceptance gates")
    return write_calibration_records(
        project_root,
        {"defaults.r_freq": notch_calibration_record(fit)},
    )


def _lorentz_model(x, parameters, components, reference):
    values = parameters[0] + parameters[1] * (x - reference)
    cursor = 2
    for _ in range(components):
        amplitude, center, hwhm = parameters[cursor : cursor + 3]
        values = values + amplitude / (1.0 + ((x - center) / hwhm) ** 2)
        cursor += 3
    return values


def _fit_lorentz_components(x, measured, components):
    reference = float(np.mean(x))
    span = float(np.ptp(x))
    step = float(np.median(np.diff(x)))
    edge = max(2, min(20, x.size // 5))
    indices = np.r_[:edge, x.size - edge : x.size]
    slope, offset = np.polyfit(x[indices] - reference, measured[indices], 1)
    deviation = measured - (offset + slope * (x - reference))
    seeds = []
    residual = deviation.copy()
    width = max(2 * step, span / 20)
    for _ in range(components):
        index = int(np.argmax(np.abs(residual)))
        seeds.extend([float(residual[index]), float(x[index]), width])
        residual = residual - seeds[-3] / (1.0 + ((x - seeds[-2]) / width) ** 2)
    initial = np.asarray([offset, slope] + seeds)
    lower = [-np.inf, -np.inf]
    upper = [np.inf, np.inf]
    for _ in range(components):
        lower.extend([-np.inf, float(x[0]), max(step / 10.0, span * 1e-9)])
        upper.extend([np.inf, float(x[-1]), 2.0 * span])
    result = least_squares(
        lambda parameters: _lorentz_model(x, parameters, components, reference) - measured,
        initial,
        bounds=(np.asarray(lower), np.asarray(upper)),
        loss="soft_l1",
        max_nfev=50_000,
    )
    if not result.success:
        raise AnalysisError(f"{components}-feature spectroscopy fit failed")
    fitted = _lorentz_model(x, result.x, components, reference)
    residual = measured - fitted
    dof = max(x.size - result.x.size, 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * float(residual @ residual) / dof
    return result.x, covariance, fitted, np.asarray(lower), np.asarray(upper)


@dataclass(frozen=True)
class FeatureSpectroscopyFit:
    source_csv: Path
    source_yml: Path
    kind: str
    signal: str
    signal_label: str
    x: np.ndarray
    iq: np.ndarray
    measured: np.ndarray
    fitted: np.ndarray
    parameters: Mapping[str, Any]
    statistics: Mapping[str, Any]
    metadata: Mapping[str, Any]
    explicit_window: bool

    @property
    def center_mhz(self) -> float:
        return float(self.parameters["center_mhz"])

    def passes(
        self,
        *,
        minimum_r_squared: float,
        minimum_contrast_snr: float,
        maximum_center_uncertainty_fraction_of_fwhm: float,
    ) -> bool:
        return bool(
            self.statistics["r_squared"] >= minimum_r_squared
            and self.statistics["contrast_snr"] >= minimum_contrast_snr
            and self.statistics["center_uncertainty_fraction_of_fwhm"]
            <= maximum_center_uncertainty_fraction_of_fwhm
            and not self.statistics.get("pinned_parameters")
            and (
                not self.statistics.get("multi_feature")
                or self.explicit_window
            )
        )


def fit_spectroscopy_features(
    csv_path: Path,
    *,
    kind: str = "qubit",
    signal: str = "IQ",
    window_mhz: Optional[Sequence[float]] = None,
) -> FeatureSpectroscopyFit:
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in SPECTROSCOPY_KINDS:
        raise ConfigError("spectroscopy kind must be resonator or qubit")
    definition = SPECTROSCOPY_KINDS[normalized_kind]
    trace = load_native_trace(
        csv_path,
        quick_class=definition["quick_class"],
        axis_text=definition["axis_text"],
        minimum_points=12,
    )
    if str(signal).upper() != "IQ":
        legacy = fit_spectroscopy(
            csv_path,
            kind=normalized_kind,
            signal=signal,
            window_mhz=window_mhz,
        )
        return _feature_from_legacy(legacy, window_mhz is not None)
    x = trace.x.copy()
    iq = trace.iq.copy()
    if window_mhz is not None:
        lower_window, upper_window = sorted(map(float, window_mhz))
        mask = (x >= lower_window) & (x <= upper_window)
        x, iq = x[mask], iq[mask]
        if x.size < 12:
            raise AnalysisError("spectroscopy fit window contains fewer than 12 points")
    measured, rotation = oriented_rotate_iq(iq)
    measured = np.asarray(measured, dtype=float)
    one = _fit_lorentz_components(x, measured, 1)
    two = _fit_lorentz_components(x, measured, 2)
    rss_one = float(np.sum((measured - one[2]) ** 2))
    rss_two = float(np.sum((measured - two[2]) ** 2))
    delta_bic = bic(rss_one, x.size, 5) - bic(rss_two, x.size, 8)
    selected = two if delta_bic > 10.0 else one
    components = 2 if delta_bic > 10.0 else 1
    values, covariance, fitted, lower, upper = selected
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    features = []
    for component in range(components):
        cursor = 2 + 3 * component
        features.append(
            {
                "amplitude": float(values[cursor]),
                "center_mhz": float(values[cursor + 1]),
                "center_uncertainty_mhz": float(errors[cursor + 1]),
                "hwhm_mhz": abs(float(values[cursor + 2])),
            }
        )
    features.sort(key=lambda feature: abs(feature["amplitude"]), reverse=True)
    primary = features[0]
    fwhm = 2.0 * primary["hwhm_mhz"]
    rmse = float(np.sqrt(np.mean((measured - fitted) ** 2)))
    names = [f"p{index}" for index in range(values.size)]
    pinned = pinned_parameters(
        dict(zip(names, values)),
        dict(zip(names, lower)),
        dict(zip(names, upper)),
    )
    parameters = {
        "center_mhz": primary["center_mhz"],
        "center_uncertainty_mhz": primary["center_uncertainty_mhz"],
        "hwhm_mhz": primary["hwhm_mhz"],
        "fwhm_mhz": fwhm,
        "amplitude": primary["amplitude"],
        "offset": float(values[0]),
        "slope_per_mhz": float(values[1]),
        "rotation_angle_rad": float(rotation["angle_rad"]),
        "rotation_flipped": bool(rotation["flipped"]),
        "features": features,
    }
    statistics = {
        "r_squared": r_squared(measured, fitted),
        "rmse": rmse,
        "contrast_snr": abs(primary["amplitude"]) / max(rmse, np.finfo(float).eps),
        "center_uncertainty_fraction_of_fwhm": primary["center_uncertainty_mhz"] / fwhm,
        "edge_distance_mhz": min(primary["center_mhz"] - x[0], x[-1] - primary["center_mhz"]),
        "multi_feature": components == 2,
        "delta_bic_two_vs_one": float(delta_bic),
        "residual_ripple_fraction": residual_ripple_fraction(x, measured - fitted),
        "ripple_suspected": residual_ripple_fraction(x, measured - fitted) > 0.3,
        "pinned_parameters": pinned,
        "qc": qc_trace(x, iq).as_dict(),
        "points": int(x.size),
    }
    return FeatureSpectroscopyFit(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        kind=normalized_kind,
        signal="IQ",
        signal_label="Oriented I/Q principal-axis projection (a.u.)",
        x=x,
        iq=iq,
        measured=measured,
        fitted=fitted,
        parameters=parameters,
        statistics=statistics,
        metadata=trace.metadata,
        explicit_window=window_mhz is not None,
    )


def _feature_from_legacy(
    fit: SpectroscopyFit,
    explicit_window: bool,
) -> FeatureSpectroscopyFit:
    statistics = dict(fit.statistics)
    statistics.update(
        {
            "multi_feature": False,
            "delta_bic_two_vs_one": float("nan"),
            "residual_ripple_fraction": residual_ripple_fraction(
                fit.x,
                fit.measured - fit.fitted,
            ),
            "ripple_suspected": False,
            "pinned_parameters": [],
            "qc": qc_trace(fit.x, fit.iq).as_dict(),
        }
    )
    return FeatureSpectroscopyFit(
        source_csv=fit.source_csv,
        source_yml=fit.source_yml,
        kind=fit.kind,
        signal=fit.signal,
        signal_label=fit.signal_label,
        x=fit.x,
        iq=fit.iq,
        measured=fit.measured,
        fitted=fit.fitted,
        parameters=fit.parameters,
        statistics=statistics,
        metadata=fit.metadata,
        explicit_window=explicit_window,
    )


fit_spectroscopy_multifeature = fit_spectroscopy_features
