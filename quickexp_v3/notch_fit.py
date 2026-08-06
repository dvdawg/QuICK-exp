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
from .fit_calibration import annotate_forced_write, write_calibration_records
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
from .spectroscopy_detect import detect_features
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
    """Evaluate the eight-parameter complex notch model.

    ``reference_mhz`` fixes the frequency the baseline is expanded about. It
    matters during fitting: expanding about the free ``center`` instead makes
    the baseline slope, phase, and cable delay reparametrise every time the
    centre moves, and that degeneracy is what drives the optimiser onto its
    bounds and lets the centre slide to the edge of the sweep. It defaults to
    the centre so that existing callers evaluating the model directly are
    unaffected.
    """
    frequency = np.asarray(frequency_mhz, dtype=float)
    values = np.asarray(parameters, dtype=float)
    if values.size != 8:
        raise AnalysisError("complex notch model requires eight parameters")
    center, linewidth, a0, a1, phi0, tau, coupling_re, coupling_im = values
    reference = center if reference_mhz is None else float(reference_mhz)
    shifted = frequency - center
    from_reference = frequency - reference
    baseline = (a0 + a1 * from_reference) * np.exp(
        1j * (phi0 + tau * from_reference)
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

    def acceptance_gates(
        self,
        *,
        minimum_r_squared: float = 0.90,
        minimum_contrast_snr: float = 5.0,
        minimum_edge_distance_over_fwhm: float = 1.0,
    ) -> Mapping[str, tuple]:
        """Each gate as ``(passed, measured, threshold)``."""
        statistics = self.statistics
        complex_r2 = float(statistics.get("r_squared_complex", float("nan")))
        amplitude_r2 = float(
            statistics.get("r_squared_amplitude", float("nan"))
        )
        return {
            "r_squared_complex": (
                bool(np.isfinite(complex_r2) and complex_r2 >= minimum_r_squared),
                complex_r2,
                float(minimum_r_squared),
            ),
            # Checked separately because the complex residual is dominated by
            # the baseline and cable delay and stays near one even when the
            # modelled magnitude is nonsense.
            "r_squared_amplitude": (
                bool(np.isfinite(amplitude_r2) and amplitude_r2 >= 0.0),
                amplitude_r2,
                0.0,
            ),
            "contrast_snr": (
                bool(statistics["contrast_snr"] >= minimum_contrast_snr),
                float(statistics["contrast_snr"]),
                float(minimum_contrast_snr),
            ),
            "edge_distance_over_fwhm": (
                bool(
                    statistics["edge_distance_over_fwhm"]
                    >= minimum_edge_distance_over_fwhm
                ),
                float(statistics["edge_distance_over_fwhm"]),
                float(minimum_edge_distance_over_fwhm),
            ),
            "no_pinned_parameters": (
                not statistics.get("pinned_parameters"),
                len(statistics.get("pinned_parameters") or []),
                0.0,
            ),
        }

    def passes(
        self,
        *,
        minimum_r_squared: float = 0.90,
        minimum_contrast_snr: float = 5.0,
        minimum_edge_distance_over_fwhm: float = 1.0,
    ) -> bool:
        return all(
            gate[0]
            for gate in self.acceptance_gates(
                minimum_r_squared=minimum_r_squared,
                minimum_contrast_snr=minimum_contrast_snr,
                minimum_edge_distance_over_fwhm=(
                    minimum_edge_distance_over_fwhm
                ),
            ).values()
        )


def _notch_initial_guess(
    trace: NativeTrace,
    *,
    center_mhz: Optional[float] = None,
) -> tuple:
    x = trace.x
    iq = trace.iq
    magnitude = np.abs(iq)
    smoothed = gaussian_filter1d(magnitude, 2.0, mode="nearest")
    edge_count = max(3, min(30, x.size // 5))
    edge_indices = np.r_[:edge_count, x.size - edge_count : x.size]
    reference = float(np.mean(x))
    a1, a0 = np.polyfit(
        x[edge_indices] - reference,
        magnitude[edge_indices],
        1,
    )
    baseline = a0 + a1 * (x - reference)
    deviation = smoothed - baseline
    feature_index = (
        int(np.argmax(np.abs(deviation)))
        if center_mhz is None
        else int(np.argmin(np.abs(x - float(center_mhz))))
    )
    center = float(x[feature_index])
    half_contrast = 0.5 * float(deviation[feature_index])
    if half_contrast >= 0:
        inside = deviation >= half_contrast
    else:
        inside = deviation <= half_contrast
    left = feature_index
    right = feature_index
    while left > 0 and inside[left - 1]:
        left -= 1
    while right < x.size - 1 and inside[right + 1]:
        right += 1
    linewidth = (
        float(x[right] - x[left])
        if right > left
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
        1.0 - iq[feature_index] / baseline_center
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
    initial = _notch_initial_guess(window_trace, center_mhz=center_seed)
    initial[0] = center_seed
    initial[1] = max(linewidth_seed, 2.0 * step)
    initial[6:8] = np.clip(initial[6:8], -1.5, 1.5)
    if np.hypot(initial[6], initial[7]) < 0.005:
        initial[6] = float(
            np.clip(
                -float(scalar.parameters["amplitude"])
                / max(initial[2], np.finfo(float).eps),
                -1.5,
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
    magnitude_contrast = abs(float(scalar.parameters["amplitude"]))
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
        "feature_polarity": scalar.parameters["feature_polarity"],
    }
    edge_distance = float(
        min(reported_center - x[0], x[-1] - reported_center)
    )
    statistics = {
        "r_squared_complex": complex_r2,
        "r_squared_amplitude": amplitude_r2,
        "rmse": rmse,
        "contrast_snr": float(
            magnitude_contrast / max(rmse, np.finfo(float).eps)
        ),
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
        # The complex residual is dominated by the baseline and cable delay,
        # so it stays near one even when the modelled magnitude is nonsense --
        # observed at -232 while the complex R-squared read 0.98. Checking the
        # magnitude separately is what catches that.
        if fit.statistics["r_squared_amplitude"] < 0.0:
            raise AnalysisError(
                "complex notch fit reproduces the phase but not the magnitude"
            )
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
    axes[0].set(
        xlabel="Readout frequency (MHz)",
        ylabel="|S21|",
        title=f"Resonator {fit.parameters.get('feature_polarity', 'feature')}",
    )
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
        "quality": {
            **dict(fit.statistics),
            "feature_polarity": fit.parameters.get("feature_polarity"),
        },
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
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_r_squared=minimum_r_squared,
        minimum_contrast_snr=minimum_contrast_snr,
        minimum_edge_distance_over_fwhm=minimum_edge_distance_over_fwhm,
    )
    if not gates_passed and not force_write:
        raise AnalysisError("complex notch fit did not pass acceptance gates")
    updates = annotate_forced_write(
        {"defaults.r_freq": notch_calibration_record(fit)},
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)


def _lorentz_model(x, parameters, components, reference):
    values = parameters[0] + parameters[1] * (x - reference)
    cursor = 2
    for _ in range(components):
        amplitude, center, hwhm = parameters[cursor : cursor + 3]
        values = values + amplitude / (1.0 + ((x - center) / hwhm) ** 2)
        cursor += 3
    return values


def _seed_lorentz_hwhm(residual, index, step, span):
    """Estimate a component width from its local half-height crossings.

    A span-derived seed is much wider than the narrow lines in a coarse
    spectroscopy trace.  With two components that caused the first seed to
    subtract most of the second feature and let both optimized centres
    collapse onto the dominant line.  The local half-height scale gives the
    optimizer distinct, physically plausible starting basins without adding
    a feature-selection threshold.
    """
    values = np.asarray(residual, dtype=float)
    peak = abs(float(values[index]))
    minimum = max(2.0 * abs(float(step)), np.finfo(float).eps)
    fallback = max(minimum, abs(float(span)) / 50.0)
    if not np.isfinite(peak) or peak <= np.finfo(float).eps:
        return fallback

    polarity = np.sign(values[index])
    half_height = 0.5 * peak
    left = int(index)
    while (
        left > 0
        and np.sign(values[left - 1]) == polarity
        and abs(float(values[left - 1])) >= half_height
    ):
        left -= 1
    right = int(index)
    while (
        right + 1 < values.size
        and np.sign(values[right + 1]) == polarity
        and abs(float(values[right + 1])) >= half_height
    ):
        right += 1

    distances = []
    if left < index:
        distances.append((index - left) * abs(float(step)))
    if right > index:
        distances.append((right - index) * abs(float(step)))
    width = float(np.mean(distances)) if distances else fallback
    return float(np.clip(width, minimum, max(abs(float(span)) / 4.0, minimum)))


def _fit_lorentz_components(x, measured, components, seed_centers=None):
    reference = float(np.mean(x))
    span = float(np.ptp(x))
    step = float(np.median(np.diff(x)))
    edge = max(2, min(20, x.size // 5))
    indices = np.r_[:edge, x.size - edge : x.size]
    slope, offset = np.polyfit(x[indices] - reference, measured[indices], 1)
    deviation = measured - (offset + slope * (x - reference))
    seeds = []
    residual = deviation.copy()
    provided = list(seed_centers or ())
    for component in range(components):
        if component < len(provided):
            index = int(np.argmin(np.abs(x - float(provided[component]))))
        else:
            # Falls back to the largest residual only when the detector had
            # nothing to offer; on a noisy trace that is a single sample.
            index = int(np.argmax(np.abs(residual)))
        width = _seed_lorentz_hwhm(residual, index, step, span)
        seeds.extend([float(residual[index]), float(x[index]), width])
        residual = residual - seeds[-3] / (1.0 + ((x - seeds[-2]) / width) ** 2)
    initial = np.asarray([offset, slope] + seeds)
    lower = [-np.inf, -np.inf]
    upper = [np.inf, np.inf]
    for _ in range(components):
        # A line narrower than the sampling is a spike on one sample, and one
        # wider than the sweep is background wearing a Lorentzian's clothes.
        lower.extend([-np.inf, float(x[0]), 0.75 * step])
        upper.extend([np.inf, float(x[-1]), 0.25 * span])
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
    #: Evidence from :mod:`quickexp_v3.spectroscopy_detect`; shared with
    #: :class:`~quickexp_v3.native_fit.SpectroscopyFit` so one plotting path
    #: serves both.
    detection: Any = None
    fit_mask: Optional[np.ndarray] = None

    @property
    def center_mhz(self) -> float:
        return float(self.parameters["center_mhz"])

    def acceptance_gates(
        self,
        *,
        minimum_r_squared: float,
        minimum_contrast_snr: float,
        maximum_center_uncertainty_fraction_of_fwhm: float,
    ) -> Mapping[str, tuple]:
        """Each gate as ``(passed, measured, threshold)``."""
        statistics = self.statistics
        return {
            "r_squared": (
                bool(statistics["r_squared"] >= minimum_r_squared),
                float(statistics["r_squared"]),
                float(minimum_r_squared),
            ),
            "contrast_snr": (
                bool(statistics["contrast_snr"] >= minimum_contrast_snr),
                float(statistics["contrast_snr"]),
                float(minimum_contrast_snr),
            ),
            "center_uncertainty_fraction_of_fwhm": (
                bool(
                    statistics["center_uncertainty_fraction_of_fwhm"]
                    <= maximum_center_uncertainty_fraction_of_fwhm
                ),
                float(statistics["center_uncertainty_fraction_of_fwhm"]),
                float(maximum_center_uncertainty_fraction_of_fwhm),
            ),
            "no_pinned_parameters": (
                not statistics.get("pinned_parameters"),
                len(statistics.get("pinned_parameters") or []),
                0.0,
            ),
            "single_feature_or_windowed": (
                bool(
                    not statistics.get("multi_feature")
                    or self.explicit_window
                ),
                float(statistics.get("detected_candidates", 1)),
                1.0,
            ),
        }

    def passes(
        self,
        *,
        minimum_r_squared: float,
        minimum_contrast_snr: float,
        maximum_center_uncertainty_fraction_of_fwhm: float,
    ) -> bool:
        return all(
            gate[0]
            for gate in self.acceptance_gates(
                minimum_r_squared=minimum_r_squared,
                minimum_contrast_snr=minimum_contrast_snr,
                maximum_center_uncertainty_fraction_of_fwhm=(
                    maximum_center_uncertainty_fraction_of_fwhm
                ),
            ).values()
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
    normalized_signal = str(signal).strip().lower()
    if normalized_signal not in {"iq", "amplitude"}:
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
    if normalized_signal == "iq":
        measured, rotation = oriented_rotate_iq(iq)
        measured = np.asarray(measured, dtype=float)
        signal_name = "IQ"
        signal_label = "Oriented I/Q principal-axis projection (a.u.)"
    else:
        measured = np.asarray(np.abs(iq), dtype=float)
        rotation = {"angle_rad": float("nan"), "flipped": False}
        signal_name = "amplitude"
        signal_label = "Amplitude (a.u.)"
    detection = detect_features(x, iq, kind=normalized_kind)
    seeds = list(detection.candidates)
    if not seeds:
        # An explicit window is the operator asserting where to look, so it
        # licenses fitting the best sub-threshold candidate; without one, a
        # confident centre drawn from noise is worse than no answer at all.
        if window_mhz is not None and detection.marginal:
            seeds = list(detection.marginal[:3])
        else:
            detection.require_best()
    leader = seeds[0]

    # Fit near the strongest feature rather than across the whole sweep. The
    # background is only straight locally, and over a full sweep a Lorentzian
    # will happily widen until it becomes the background -- which is how this
    # fitter used to report linewidths as wide as the trace.
    step = float(np.median(np.diff(x)))
    reach = max(10.0 * float(leader.hwhm_mhz), 20.0 * step)
    inside = np.abs(x - float(leader.center_mhz)) <= reach
    if np.count_nonzero(inside) >= 12:
        x, iq, measured = x[inside], iq[inside], measured[inside]

    seed_centers = [
        candidate.center_mhz
        for candidate in seeds
        if float(x[0]) <= candidate.center_mhz <= float(x[-1])
    ]
    one = _fit_lorentz_components(x, measured, 1, seed_centers[:1])
    two = _fit_lorentz_components(x, measured, 2, seed_centers[:2])
    rss_one = float(np.sum((measured - one[2]) ** 2))
    rss_two = float(np.sum((measured - two[2]) ** 2))
    delta_bic = bic(rss_one, x.size, 5) - bic(rss_two, x.size, 8)
    two_values = two[0]
    two_center_separation = abs(
        float(two_values[3]) - float(two_values[6])
    )
    two_resolution_floor = max(
        3.0 * float(np.median(np.diff(x))),
        0.5
        * (
            abs(float(two_values[4]))
            + abs(float(two_values[7]))
        ),
    )
    two_feature_resolved = two_center_separation >= two_resolution_floor
    # A second component only counts when the detector independently found a
    # second feature. A narrow line sitting on a broad pedestal improves the
    # residual sum enormously by splitting into two displaced components, so
    # the information criterion alone reports two features where there is one.
    detected_candidates = len(detection.candidates)
    use_two = bool(
        delta_bic > 10.0 and two_feature_resolved and detected_candidates >= 2
    )
    selected = two if use_two else one
    components = 2 if use_two else 1
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
                "feature_polarity": (
                    "peak" if float(values[cursor]) >= 0 else "dip"
                ),
            }
        )
    features.sort(key=lambda feature: abs(feature["amplitude"]), reverse=True)
    primary = features[0]

    # Features outside the local fit window are never seen by the component
    # fit, but they remain real candidates the caller has to choose between.
    # Carry the detector's view of them through, at its own resolution and
    # ranked after everything actually fitted, so downstream adjudication
    # still sees every option.
    reference_channel = detection.channels[
        "projection" if normalized_signal == "iq" else "amplitude"
    ]
    channel_noise = reference_channel.primary.noise
    known = [feature["center_mhz"] for feature in features]
    for candidate in detection.candidates:
        if any(
            abs(candidate.center_mhz - center)
            <= max(candidate.hwhm_mhz, 2.0 * float(np.median(np.diff(x))))
            for center in known
        ):
            continue
        sign = 1.0 if candidate.polarity == "peak" else -1.0
        features.append(
            {
                "amplitude": float(sign * candidate.prominence_snr * channel_noise),
                "center_mhz": float(candidate.center_mhz),
                "center_uncertainty_mhz": float(candidate.hwhm_mhz),
                "hwhm_mhz": float(candidate.hwhm_mhz),
                "feature_polarity": candidate.polarity,
                "detected_only": True,
            }
        )
        known.append(candidate.center_mhz)
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
        "feature_polarity": primary["feature_polarity"],
    }
    statistics = {
        "r_squared": r_squared(measured, fitted),
        "rmse": rmse,
        "contrast_snr": abs(primary["amplitude"]) / max(rmse, np.finfo(float).eps),
        "center_uncertainty_fraction_of_fwhm": primary["center_uncertainty_mhz"] / fwhm,
        "edge_distance_mhz": min(primary["center_mhz"] - x[0], x[-1] - primary["center_mhz"]),
        # Warn whenever more than one credible feature exists anywhere in the
        # sweep, not only when the local two-component model wins: a false
        # warning costs a fit window, calibrating the wrong line costs more.
        "multi_feature": bool(components == 2 or detected_candidates >= 2),
        "detected_candidates": int(detected_candidates),
        "detection_prominence_snr": float(leader.prominence_snr),
        "detection_basis": leader.basis,
        "detection_channel_snr": dict(leader.channel_snr),
        "detection_geometry_score": float(leader.geometry_score),
        "delta_bic_two_vs_one": float(delta_bic),
        "two_feature_center_separation_mhz": float(
            two_center_separation
        ),
        "two_feature_resolution_floor_mhz": float(
            two_resolution_floor
        ),
        "two_feature_resolved": bool(two_feature_resolved),
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
        signal=signal_name,
        signal_label=signal_label,
        x=x,
        iq=iq,
        measured=measured,
        fitted=fitted,
        parameters=parameters,
        statistics=statistics,
        metadata=trace.metadata,
        explicit_window=window_mhz is not None,
        detection=detection,
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
