"""Fit one-dimensional native Quick spectroscopy and coherence files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import yaml

from .analysis import rotate_iq
from .errors import AnalysisError, ConfigError
from .fit_calibration import annotate_forced_write, write_calibration_records
from .spectroscopy_detect import detect_features, resonance_projection
from .util import utc_now


SIGNAL_CHOICES = ("amplitude", "phase", "I", "Q", "IQ")
SPECTROSCOPY_KINDS = {
    "resonator": {
        "quick_class": "ResonatorSpectroscopy",
        "axis_text": "readout pulse frequency",
        "axis_label": "Readout frequency",
        "record": "r_freq",
    },
    "qubit": {
        "quick_class": "QubitSpectroscopy",
        "axis_text": "qubit pulse frequency",
        "axis_label": "Qubit frequency",
        "record": "q_freq",
    },
}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _normalize_signal(signal: str) -> str:
    key = str(signal).strip().lower().replace(" ", "")
    aliases = {
        "amp": "amplitude",
        "magnitude": "amplitude",
        "mag": "amplitude",
        "angle": "phase",
        "i": "I",
        "q": "Q",
        "iq": "IQ",
        "i/q": "IQ",
        "projection": "IQ",
        "iqprojection": "IQ",
    }
    normalized = aliases.get(key, key)
    if normalized not in SIGNAL_CHOICES:
        raise ConfigError(
            f"signal must be one of {', '.join(SIGNAL_CHOICES)}"
        )
    return normalized


@dataclass(frozen=True)
class NativeTrace:
    source_csv: Path
    source_yml: Path
    quick_class: str
    axis_label: str
    axis_unit: str
    x: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    iq: np.ndarray
    metadata: Mapping[str, Any]


def inspect_native_file(
    csv_path: Path,
    *,
    quick_class: str,
    axis_text: str,
) -> tuple[Path, dict, str, str]:
    """Validate a one-dimensional native Quick CSV/YML pair."""
    source = Path(csv_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(
            f"native Quick CSV does not exist or is empty: {source}"
        )
    yml_path = source.with_suffix(".yml")
    if not yml_path.is_file():
        raise FileNotFoundError(f"paired Quick YML does not exist: {yml_path}")
    metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise AnalysisError(f"{yml_path} must contain a YAML mapping")

    detected_class = _nested(metadata, "parameters", "quick_experiment")
    if detected_class != quick_class:
        raise AnalysisError(
            f"{source.name} is {detected_class!r}, not {quick_class!r}"
        )
    independent = metadata.get("independent")
    if not isinstance(independent, list) or len(independent) != 1:
        raise AnalysisError(
            "fitting requires exactly one native Quick independent axis"
        )
    entry = independent[0]
    if not isinstance(entry, (list, tuple)) or not entry:
        raise AnalysisError("native Quick YML has an invalid independent axis")
    label = str(entry[0])
    unit = str(entry[1]) if len(entry) > 1 else ""
    if axis_text.lower() not in label.strip().lower():
        raise AnalysisError(
            f"{source.name} has independent axis {label!r}; "
            f"expected one containing {axis_text!r}"
        )
    return yml_path, dict(metadata), label, unit


def find_latest_native(
    data_directory: Path,
    *,
    quick_class: str,
    axis_text: str,
) -> Path:
    """Return the newest complete matching one-dimensional Quick pair."""
    root = Path(data_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Quick data directory does not exist: {root}")
    candidates = []
    for csv_path in root.glob("*.csv"):
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            continue
        try:
            inspect_native_file(
                csv_path,
                quick_class=quick_class,
                axis_text=axis_text,
            )
        except (
            AnalysisError,
            FileNotFoundError,
            OSError,
            yaml.YAMLError,
        ):
            continue
        candidates.append(csv_path)
    if not candidates:
        raise FileNotFoundError(
            f"No complete one-dimensional {quick_class} CSV/YML pair "
            f"with axis {axis_text!r} was found in {root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _dependent_column(
    metadata: Mapping[str, Any],
    name: str,
) -> Optional[int]:
    dependent = metadata.get("dependent")
    if not isinstance(dependent, list):
        return None
    for index, entry in enumerate(dependent):
        if (
            isinstance(entry, (list, tuple))
            and entry
            and str(entry[0]).strip().lower() == name.lower()
        ):
            return index + 1
    return None


def load_native_trace(
    csv_path: Path,
    *,
    quick_class: str,
    axis_text: str,
    minimum_points: int,
) -> NativeTrace:
    """Load axis, amplitude, phase, I, and Q from a native Quick pair."""
    source = Path(csv_path).expanduser().resolve()
    yml_path, metadata, axis_label, axis_unit = inspect_native_file(
        source,
        quick_class=quick_class,
        axis_text=axis_text,
    )
    try:
        matrix = np.loadtxt(source, delimiter=",")
    except ValueError:
        matrix = np.loadtxt(source)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    if matrix.ndim != 2 or matrix.shape[1] < 5:
        raise AnalysisError(
            "native Quick CSV must contain axis, amplitude, phase, I, and Q"
        )

    columns = {
        name: _dependent_column(metadata, name)
        for name in ("Amplitude", "Phase", "I", "Q")
    }
    if columns["I"] is None or columns["Q"] is None:
        columns["I"], columns["Q"] = matrix.shape[1] - 2, matrix.shape[1] - 1
    if (
        columns["Amplitude"] is None
        or columns["Amplitude"] >= matrix.shape[1]
    ):
        columns["Amplitude"] = matrix.shape[1] - 4
    if columns["Phase"] is None or columns["Phase"] >= matrix.shape[1]:
        columns["Phase"] = matrix.shape[1] - 3
    if max(int(value) for value in columns.values()) >= matrix.shape[1]:
        raise AnalysisError(
            "native Quick dependent columns do not match the CSV"
        )

    x = matrix[:, 0]
    amplitude = matrix[:, int(columns["Amplitude"])]
    phase = matrix[:, int(columns["Phase"])]
    iq = (
        matrix[:, int(columns["I"])]
        + 1j * matrix[:, int(columns["Q"])]
    )
    finite = (
        np.isfinite(x)
        & np.isfinite(amplitude)
        & np.isfinite(phase)
        & np.isfinite(iq.real)
        & np.isfinite(iq.imag)
    )
    x, amplitude, phase, iq = (
        values[finite] for values in (x, amplitude, phase, iq)
    )
    if x.size < minimum_points:
        raise AnalysisError(
            f"{quick_class} fitting requires at least "
            f"{minimum_points} finite points"
        )
    order = np.argsort(x)
    x, amplitude, phase, iq = (
        values[order] for values in (x, amplitude, phase, iq)
    )
    if np.any(np.diff(x) <= 0):
        raise AnalysisError("native Quick sweep axis must be unique")
    return NativeTrace(
        source_csv=source,
        source_yml=yml_path,
        quick_class=quick_class,
        axis_label=axis_label,
        axis_unit=axis_unit,
        x=x,
        amplitude=amplitude,
        phase=phase,
        iq=iq,
        metadata=metadata,
    )


def _fit_signal(
    trace: NativeTrace,
    signal: str,
    *,
    relaxation_axis: bool = False,
) -> tuple[np.ndarray, str, Mapping[str, float]]:
    normalized = _normalize_signal(signal)
    if normalized == "amplitude":
        return np.asarray(trace.amplitude, dtype=float), "Amplitude", {}
    if normalized == "phase":
        return np.unwrap(trace.phase), "Unwrapped phase (rad)", {}
    if normalized == "I":
        return trace.iq.real.copy(), "I", {}
    if normalized == "Q":
        return trace.iq.imag.copy(), "Q", {}

    if relaxation_axis:
        count = max(1, min(10, len(trace.iq) // 5))
        late = np.mean(trace.iq[-count:])
        direction = np.mean(trace.iq[:count]) - late
        if not np.isclose(abs(direction), 0.0):
            projected = np.real(
                (trace.iq - late) * np.conj(direction) / abs(direction)
            )
            return (
                np.asarray(projected, dtype=float),
                "I/Q relaxation-axis projection (a.u.)",
                {"angle_rad": float(np.angle(direction))},
            )

    projected, rotation = rotate_iq(trace.iq)
    return (
        np.asarray(projected, dtype=float),
        "I/Q principal-axis projection (a.u.)",
        rotation,
    )


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(observed) - np.asarray(predicted)
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return (
        float(1.0 - np.sum(residual**2) / total)
        if total > 0
        else float("nan")
    )


def _spectral_model(
    x: np.ndarray,
    offset: float,
    slope: float,
    amplitude: float,
    center: float,
    hwhm: float,
    x_reference: float,
) -> np.ndarray:
    return (
        offset
        + slope * (x - x_reference)
        + amplitude / (1.0 + ((x - center) / hwhm) ** 2)
    )


#: Half-width of the local fit window, in detected half-widths. Wide enough to
#: pin the background and the Lorentzian wings, narrow enough that a straight
#: background still describes it.
FIT_WINDOW_IN_HWHM = 10.0

#: Fitted amplitude over local residual scatter, above which a candidate is
#: taken as confirmed and the remaining candidates are not tried.
MINIMUM_FITTED_CONTRAST = 3.0


def _fit_local_lorentzian(
    x: np.ndarray,
    measured: np.ndarray,
    seed,
    *,
    step: float,
    x_reference: float,
):
    """Fit one detected feature on a local window.

    Fitting the whole trace forces a single straight line to describe a
    background that curves and ripples, and the Lorentzian ends up absorbing
    the mismatch -- which is how a fit reports a linewidth wider than the
    sweep. Restricting to a neighbourhood of the detected feature keeps the
    straight background honest.
    """
    reach = max(
        FIT_WINDOW_IN_HWHM * float(seed.hwhm_mhz),
        20.0 * abs(step),
    )
    mask = np.abs(x - float(seed.center_mhz)) <= reach
    if np.count_nonzero(mask) < 9:
        # Too few points nearby: widen to the smallest usable neighbourhood.
        index = int(np.argmin(np.abs(x - float(seed.center_mhz))))
        low = max(0, index - 6)
        mask = np.zeros(x.size, dtype=bool)
        mask[low : low + 13] = True
    if np.count_nonzero(mask) < 9:
        return None

    window_x = x[mask]
    window_y = measured[mask]
    window_span = float(np.ptp(window_x))
    if window_span <= 0:
        return None

    edge = max(2, min(20, window_x.size // 5))
    edges = np.r_[:edge, window_x.size - edge : window_x.size]
    slope_guess, offset_guess = np.polyfit(
        window_x[edges] - x_reference,
        window_y[edges],
        1,
    )
    baseline = offset_guess + slope_guess * (window_x - x_reference)
    amplitude_guess = float(
        window_y[int(np.argmin(np.abs(window_x - float(seed.center_mhz))))]
        - baseline[int(np.argmin(np.abs(window_x - float(seed.center_mhz))))]
    )
    if amplitude_guess == 0.0:
        amplitude_guess = float(np.ptp(window_y)) or 1.0

    # A resonance cannot be narrower than the sampling nor wider than the
    # window it was found in. Without the lower bound the optimiser happily
    # returns a delta function sitting on one noisy sample; without the upper
    # bound it returns a "line" as wide as the sweep, which is background.
    minimum_hwhm = 0.75 * abs(step)
    maximum_hwhm = max(window_span / 2.0, minimum_hwhm * 2.0)

    def model(values, offset, slope, amplitude, center, hwhm):
        return _spectral_model(
            values, offset, slope, amplitude, center, hwhm, x_reference
        )

    best = None
    for width in np.unique(
        np.clip(
            np.asarray(
                [
                    float(seed.hwhm_mhz),
                    0.5 * float(seed.hwhm_mhz),
                    2.0 * float(seed.hwhm_mhz),
                    2.0 * abs(step),
                ]
            ),
            minimum_hwhm * 1.01,
            maximum_hwhm * 0.99,
        )
    ):
        try:
            parameters, covariance = curve_fit(
                model,
                window_x,
                window_y,
                p0=[
                    float(offset_guess),
                    float(slope_guess),
                    amplitude_guess,
                    float(seed.center_mhz),
                    float(width),
                ],
                bounds=(
                    [
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        float(window_x[0]),
                        minimum_hwhm,
                    ],
                    [
                        np.inf,
                        np.inf,
                        np.inf,
                        float(window_x[-1]),
                        maximum_hwhm,
                    ],
                ),
                maxfev=100000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        predicted = model(window_x, *parameters)
        squared_error = float(np.sum((window_y - predicted) ** 2))
        if best is None or squared_error < best[0]:
            best = (squared_error, parameters, covariance)
    if best is None:
        return None
    _error, parameters, covariance = best
    return mask, parameters, covariance, model(x, *parameters)


@dataclass(frozen=True)
class SpectroscopyFit:
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
    #: Points that entered the least squares. The feature is fitted on a local
    #: window because a straight background is only a good description near
    #: the line; the rest of the trace is retained for plotting.
    fit_mask: Optional[np.ndarray] = None
    #: Evidence from :mod:`quickexp_v3.spectroscopy_detect`, or None when an
    #: explicit window bypassed detection.
    detection: Any = None

    @property
    def center_mhz(self) -> float:
        return float(self.parameters["center_mhz"])

    @property
    def fit_x(self) -> np.ndarray:
        return self.x if self.fit_mask is None else self.x[self.fit_mask]

    def acceptance_gates(
        self,
        *,
        minimum_r_squared: float,
        minimum_contrast_snr: float,
        maximum_center_uncertainty_fraction_of_fwhm: float,
    ) -> Mapping[str, tuple]:
        """Each gate as ``(passed, measured, threshold)``.

        Reporting the measured value beside its threshold turns a bare
        failure into something actionable: it says whether to average longer,
        narrow the sweep, or re-centre it.
        """
        statistics = self.statistics
        return {
            "r_squared": (
                bool(
                    np.isfinite(statistics["r_squared"])
                    and statistics["r_squared"] >= minimum_r_squared
                ),
                float(statistics["r_squared"]),
                float(minimum_r_squared),
            ),
            "contrast_snr": (
                bool(
                    np.isfinite(statistics["contrast_snr"])
                    and statistics["contrast_snr"] >= minimum_contrast_snr
                ),
                float(statistics["contrast_snr"]),
                float(minimum_contrast_snr),
            ),
            "center_uncertainty_fraction_of_fwhm": (
                bool(
                    np.isfinite(
                        statistics["center_uncertainty_fraction_of_fwhm"]
                    )
                    and statistics["center_uncertainty_fraction_of_fwhm"]
                    <= maximum_center_uncertainty_fraction_of_fwhm
                ),
                float(statistics["center_uncertainty_fraction_of_fwhm"]),
                float(maximum_center_uncertainty_fraction_of_fwhm),
            ),
            "center_inside_sweep": (
                bool(
                    float(np.min(self.x))
                    < self.center_mhz
                    < float(np.max(self.x))
                ),
                self.center_mhz,
                float("nan"),
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


def fit_spectroscopy(
    csv_path: Path,
    *,
    kind: str,
    signal: str = "IQ",
    window_mhz: Optional[Sequence[float]] = None,
) -> SpectroscopyFit:
    """Fit a signed Lorentzian feature with a linear background."""
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in SPECTROSCOPY_KINDS:
        raise ConfigError("spectroscopy kind must be resonator or qubit")
    definition = SPECTROSCOPY_KINDS[normalized_kind]
    trace = load_native_trace(
        csv_path,
        quick_class=definition["quick_class"],
        axis_text=definition["axis_text"],
        minimum_points=9,
    )
    measured, signal_label, rotation = _fit_signal(trace, signal)
    x = trace.x.copy()
    iq = trace.iq.copy()
    if window_mhz is not None:
        if len(window_mhz) != 2:
            raise ConfigError("spectroscopy fit window must have two values")
        lower, upper = sorted(float(value) for value in window_mhz)
        mask = (x >= lower) & (x <= upper)
        x, iq, measured = x[mask], iq[mask], measured[mask]
        if len(x) < 9:
            raise AnalysisError(
                "spectroscopy fit window contains fewer than 9 points"
            )

    span = float(np.ptp(x))
    step = float(np.median(np.diff(x)))
    if span <= 0 or step <= 0:
        raise AnalysisError("spectroscopy axis must span a positive interval")
    x_reference = float(np.mean(x))

    detection = detect_features(x, iq, kind=normalized_kind)
    if _normalize_signal(signal) == "IQ" and detection.best is not None:
        # Project the de-delayed trace onto the direction the resonance moves
        # it. On a wide resonator sweep the cable delay turns the I/Q point
        # through tens of full revolutions, so the principal axis of the raw
        # cloud is the delay itself and the projection is a sinusoid with the
        # resonance buried in it.
        measured, angle = resonance_projection(
            x, detection.corrected, detection.best.center_mhz
        )
        rotation = {"angle_rad": angle}
        signal_label = "I/Q resonance-axis projection (a.u.)"
    seeds = list(detection.candidates)
    if not seeds:
        # An explicit window is the operator asserting where to look, so it
        # licenses fitting the best sub-threshold candidate; without one, a
        # confident centre drawn from noise is worse than no answer at all.
        if window_mhz is not None and detection.marginal:
            seeds = list(detection.marginal[:3])
        else:
            detection.require_best()

    # Seeds arrive ranked by detection evidence, which is the quantity that
    # was calibrated against noise. The fit's job is to refine and to reject,
    # not to re-rank: scoring candidates by residual size instead would prefer
    # a flat quiet stretch of trace over a strong, noisy line.
    best = None
    for seed in seeds:
        outcome = _fit_local_lorentzian(
            x,
            measured,
            seed,
            step=step,
            x_reference=x_reference,
        )
        if outcome is None:
            continue
        fit_mask, parameters, covariance, fitted = outcome
        residual = measured[fit_mask] - fitted[fit_mask]
        local_rmse = float(np.sqrt(np.mean(residual**2)))
        contrast = abs(float(parameters[2])) / max(local_rmse, np.finfo(float).eps)
        candidate = (contrast, parameters, covariance, fitted, fit_mask, seed)
        if best is None or contrast > best[0]:
            best = candidate
        if contrast >= MINIMUM_FITTED_CONTRAST:
            best = candidate
            break
    if best is None:
        raise AnalysisError(
            "spectroscopy fit did not converge from any detected feature"
        )

    _contrast, parameters, covariance, fitted, fit_mask, seed = best
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    residual = measured[fit_mask] - fitted[fit_mask]
    rmse = float(np.sqrt(np.mean(residual**2)))
    hwhm = abs(float(parameters[4]))
    fwhm = 2.0 * hwhm
    center = float(parameters[3])
    center_uncertainty = float(errors[3])
    contrast_snr = abs(float(parameters[2])) / max(
        rmse,
        np.finfo(float).eps,
    )
    parameter_map = {
        "center_mhz": center,
        "center_uncertainty_mhz": center_uncertainty,
        "hwhm_mhz": hwhm,
        "fwhm_mhz": fwhm,
        "amplitude": float(parameters[2]),
        "offset": float(parameters[0]),
        "slope_per_mhz": float(parameters[1]),
        "rotation_angle_rad": float(rotation.get("angle_rad", 0.0)),
        "feature_polarity": (
            "peak" if float(parameters[2]) >= 0 else "dip"
        ),
    }
    statistics = {
        "r_squared": _r_squared(measured[fit_mask], fitted[fit_mask]),
        "rmse": rmse,
        "contrast_snr": float(contrast_snr),
        "center_uncertainty_fraction_of_fwhm": float(
            center_uncertainty / fwhm
        ),
        "edge_distance_mhz": float(
            min(center - x[0], x[-1] - center)
        ),
        "points": int(np.count_nonzero(fit_mask)),
        "trace_points": int(len(x)),
        "detection_prominence_snr": float(seed.prominence_snr),
        "detection_basis": seed.basis,
        "detection_channel_snr": dict(seed.channel_snr),
        "detection_corroborating_channels": int(seed.corroborating_channels),
        "detection_geometry_score": float(seed.geometry_score),
        "detection_candidates": int(len(detection.candidates)),
        "delay_ns": float(detection.delay.delay_ns),
        "delay_applied": bool(detection.delay.applied),
    }
    if normalized_kind == "resonator" and fwhm > 0:
        parameter_map["loaded_q"] = abs(center / fwhm)
    return SpectroscopyFit(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        kind=normalized_kind,
        signal=_normalize_signal(signal),
        signal_label=signal_label,
        x=x,
        iq=iq,
        measured=measured,
        fitted=fitted,
        parameters=parameter_map,
        statistics=statistics,
        metadata=trace.metadata,
        fit_mask=fit_mask,
        detection=detection,
    )


def plot_spectroscopy_fit(fit: SpectroscopyFit):
    """Six panels: where the feature is, why, and how well it fitted.

    The top row locates the feature in the whole sweep, then zooms on the
    fitted window. The bottom row is the evidence: every detection channel in
    units of its own noise against the thresholds, and the I/Q plane with the
    cable delay removed, where a real resonance is a clean arc.
    """
    definition = SPECTROSCOPY_KINDS[fit.kind]
    axis_label = f"{definition['axis_label']} (MHz)"
    p = fit.parameters
    # Also serves FeatureSpectroscopyFit, which carries the same two fields.
    detection = getattr(fit, "detection", None)
    stored_mask = getattr(fit, "fit_mask", None)
    mask = (
        np.ones(fit.x.size, dtype=bool)
        if stored_mask is None
        else np.asarray(stored_mask, dtype=bool)
    )
    window_x = fit.x[mask]
    dense_x = np.linspace(float(window_x[0]), float(window_x[-1]), 2000)
    dense_fit = _spectral_model(
        dense_x,
        p["offset"],
        p["slope_per_mhz"],
        p["amplitude"],
        p["center_mhz"],
        p["hwhm_mhz"],
        float(np.mean(fit.x)),
    )

    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.4), constrained_layout=True)

    # --- whole sweep, with every candidate the detector ranked -------------
    overview = axes[0, 0]
    overview.plot(fit.x, fit.measured, "-", linewidth=0.7, color="0.6", label="data")
    overview.axvspan(
        float(window_x[0]),
        float(window_x[-1]),
        color="tab:blue",
        alpha=0.12,
        label="fitted window",
    )
    overview.axvline(fit.center_mhz, color="tab:red", linestyle="--", linewidth=1.5)
    if detection is not None:
        for rank, candidate in enumerate(detection.candidates[1:5], start=2):
            overview.axvline(
                candidate.center_mhz,
                color="tab:orange",
                linestyle=":",
                linewidth=1.2,
            )
            overview.annotate(
                f"#{rank}",
                (candidate.center_mhz, float(np.max(fit.measured))),
                fontsize=7,
                color="tab:orange",
                ha="center",
                va="top",
            )
        for candidate in detection.marginal[:2]:
            overview.axvline(
                candidate.center_mhz,
                color="0.5",
                linestyle=":",
                linewidth=1.0,
            )
    extra = (
        len(detection.candidates) if detection is not None else 1
    )
    overview.set(
        xlabel=axis_label,
        ylabel=fit.signal_label,
        title=f"Full sweep - {extra} candidate(s) detected",
    )
    overview.grid(alpha=0.3)
    overview.legend(loc="best", fontsize=8)

    # --- the fitted window --------------------------------------------------
    zoom = axes[0, 1]
    zoom.plot(window_x, fit.measured[mask], "o", markersize=3, label="data")
    zoom.plot(dense_x, dense_fit, "-", linewidth=2, label="fit")
    zoom.axvline(
        fit.center_mhz,
        color="tab:red",
        linestyle="--",
        label=f"{fit.center_mhz:.6f} MHz",
    )
    half = 0.5 * float(p["fwhm_mhz"])
    zoom.axvspan(
        fit.center_mhz - half,
        fit.center_mhz + half,
        color="tab:red",
        alpha=0.10,
        label=f"FWHM {p['fwhm_mhz']:.4g} MHz",
    )
    zoom.set(
        xlabel=axis_label,
        ylabel=fit.signal_label,
        title=(
            f"{fit.kind.capitalize()} {p['feature_polarity']} "
            f"($R^2$={fit.statistics['r_squared']:.4f}, "
            f"SNR={fit.statistics['contrast_snr']:.1f})"
        ),
    )
    zoom.grid(alpha=0.3)
    zoom.legend(loc="best", fontsize=8)

    # --- residuals over the fitted window ----------------------------------
    residual_axis = axes[0, 2]
    residual = fit.measured - fit.fitted
    residual_axis.axhline(0.0, color="0.4", linewidth=1)
    residual_axis.plot(window_x, residual[mask], "o-", markersize=3, linewidth=0.8)
    residual_axis.set(
        xlabel=axis_label,
        ylabel="Measured - fit",
        title=f"Residuals (RMSE={fit.statistics['rmse']:.3g})",
    )
    residual_axis.grid(alpha=0.3)

    # --- detection channels, each in units of its own noise ----------------
    evidence = axes[1, 0]
    if detection is not None:
        for name, channel in detection.channels.items():
            scale = channel.primary
            evidence.plot(
                detection.x,
                scale.deviation / max(scale.noise, np.finfo(float).eps),
                linewidth=0.9,
                label=f"{name}{' (control)' if channel.is_control else ''}",
            )
        reference = next(iter(detection.channels.values()))
        evidence.axhline(
            reference.prominence_cut,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label=f"single-channel cut {reference.prominence_cut:.1f}",
        )
        evidence.axhline(-reference.prominence_cut, color="tab:red", linestyle="--", linewidth=1)
        evidence.axhline(reference.deviation_cut, color="tab:orange", linestyle=":", linewidth=1)
        evidence.axhline(-reference.deviation_cut, color="tab:orange", linestyle=":", linewidth=1)
        evidence.axvline(fit.center_mhz, color="tab:red", linestyle="--", linewidth=1.2)
    evidence.set(
        xlabel=axis_label,
        ylabel="Deviation / noise",
        title="Detection channels vs thresholds",
    )
    evidence.grid(alpha=0.3)
    evidence.legend(loc="best", fontsize=7)

    # --- the I/Q plane, cable delay removed --------------------------------
    plane = axes[1, 1]
    # The detection ran on the whole trace; the fit may have been restricted
    # to a window, so the two carry different lengths. Pair each set of
    # complex values with its own frequency axis.
    if detection is not None and detection.corrected.size == detection.x.size:
        values, plane_x = detection.corrected, detection.x
    else:
        values, plane_x = fit.iq, fit.x
    scatter = plane.scatter(
        np.real(values), np.imag(values), c=plane_x, s=16, cmap="viridis"
    )
    near = np.abs(plane_x - fit.center_mhz) <= max(
        float(p["fwhm_mhz"]), 3.0 * float(np.median(np.diff(plane_x)))
    )
    if np.any(near):
        plane.plot(
            np.real(values)[near],
            np.imag(values)[near],
            "o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="tab:red",
            label="within FWHM",
        )
        plane.legend(loc="best", fontsize=8)
    delay_note = (
        f"delay {detection.delay.delay_ns:.1f} ns removed"
        if detection is not None and detection.delay.applied
        else "no delay removed"
    )
    plane.set(xlabel="I", ylabel="Q", title=f"I/Q plane ({delay_note})")
    plane.axis("equal")
    plane.grid(alpha=0.3)
    figure.colorbar(scatter, ax=plane, label="Frequency (MHz)")

    # --- the numbers --------------------------------------------------------
    summary = axes[1, 2]
    summary.axis("off")
    lines = [
        f"centre      {fit.center_mhz:.6f} MHz",
        f"  +/-       {p['center_uncertainty_mhz']:.3g} MHz",
        f"FWHM        {p['fwhm_mhz']:.6g} MHz",
    ]
    if "loaded_q" in p:
        lines.append(f"loaded Q    {p['loaded_q']:.6g}")
    statistics = fit.statistics
    lines += [
        "",
        f"R^2         {statistics['r_squared']:.4f}",
        f"contrast    {statistics['contrast_snr']:.2f}",
        f"points      {statistics['points']} of {statistics.get('trace_points', statistics['points'])}",
    ]
    if "detection_basis" in statistics:
        lines += [
            "",
            "detection",
            f"  basis     {statistics['detection_basis']}",
            f"  prominence {statistics['detection_prominence_snr']:.1f} sigma",
            f"  geometry  {statistics['detection_geometry_score']:.2f}",
        ]
        for name, value in statistics.get("detection_channel_snr", {}).items():
            lines.append(f"  {name:<12s}{value:.1f} sigma")
    summary.text(
        0.0,
        1.0,
        "\n".join(lines),
        family="monospace",
        fontsize=9,
        va="top",
        ha="left",
        transform=summary.transAxes,
    )

    figure.suptitle(f"Fit from {fit.source_csv.name}")
    return figure


@dataclass(frozen=True)
class T1Fit:
    source_csv: Path
    source_yml: Path
    signal: str
    signal_label: str
    time_us: np.ndarray
    iq: np.ndarray
    measured: np.ndarray
    fitted: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, float]
    metadata: Mapping[str, Any]

    @property
    def t1_us(self) -> float:
        return float(self.parameters["t1_us"])

    def passes(
        self,
        *,
        minimum_r_squared: float,
        minimum_span_over_t1: float,
        maximum_relative_t1_uncertainty: float,
    ) -> bool:
        return bool(
            np.isfinite(self.statistics["r_squared"])
            and np.isfinite(self.statistics["relative_t1_uncertainty"])
            and self.statistics["r_squared"] >= minimum_r_squared
            and self.statistics["span_over_t1"] >= minimum_span_over_t1
            and self.statistics["relative_t1_uncertainty"]
            <= maximum_relative_t1_uncertainty
            and self.t1_us > 0
        )


def _decay_model(
    time_us: np.ndarray,
    amplitude: float,
    decay_us: float,
    offset: float,
    origin_us: float,
) -> np.ndarray:
    return amplitude * np.exp(-(time_us - origin_us) / decay_us) + offset


def fit_t1(
    csv_path: Path,
    *,
    signal: str = "IQ",
) -> T1Fit:
    trace = load_native_trace(
        csv_path,
        quick_class="T1",
        axis_text="delay time",
        minimum_points=7,
    )
    measured, signal_label, rotation = _fit_signal(
        trace,
        signal,
        relaxation_axis=True,
    )
    time_us = trace.x
    span = float(np.ptp(time_us))
    if span <= 0:
        raise AnalysisError("T1 delay axis must span a positive interval")
    count = max(1, min(10, len(measured) // 5))
    offset_guess = float(np.mean(measured[-count:]))
    amplitude_guess = float(np.mean(measured[:count]) - offset_guess)
    minimum_t1 = max(np.finfo(float).eps, span * 1e-9)
    maximum_t1 = span * 100.0
    origin = float(time_us[0])

    def model(values, amplitude, decay_us, offset):
        return _decay_model(values, amplitude, decay_us, offset, origin)

    try:
        parameters, covariance = curve_fit(
            model,
            time_us,
            measured,
            p0=[amplitude_guess, span / 3.0, offset_guess],
            bounds=(
                [-np.inf, minimum_t1, -np.inf],
                [np.inf, maximum_t1, np.inf],
            ),
            maxfev=100000,
        )
    except Exception as error:
        raise AnalysisError(f"T1 fit failed: {error}") from error
    fitted = model(time_us, *parameters)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t1_us = float(parameters[1])
    t1_uncertainty = float(errors[1])
    residual = measured - fitted
    statistics = {
        "r_squared": _r_squared(measured, fitted),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "span_over_t1": float(span / t1_us),
        "relative_t1_uncertainty": float(t1_uncertainty / t1_us),
        "points": int(len(time_us)),
    }
    return T1Fit(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        signal=_normalize_signal(signal),
        signal_label=signal_label,
        time_us=time_us,
        iq=trace.iq,
        measured=measured,
        fitted=fitted,
        parameters={
            "t1_us": t1_us,
            "t1_uncertainty_us": t1_uncertainty,
            "amplitude": float(parameters[0]),
            "offset": float(parameters[2]),
            "rotation_angle_rad": float(rotation.get("angle_rad", 0.0)),
        },
        statistics=statistics,
        metadata=trace.metadata,
    )


def plot_t1_fit(fit: T1Fit):
    dense_time = np.linspace(
        float(fit.time_us[0]),
        float(fit.time_us[-1]),
        1500,
    )
    dense_fit = _decay_model(
        dense_time,
        fit.parameters["amplitude"],
        fit.parameters["t1_us"],
        fit.parameters["offset"],
        float(fit.time_us[0]),
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.2),
        constrained_layout=True,
    )
    axes[0].plot(fit.time_us, fit.measured, "o", markersize=3, label="data")
    axes[0].plot(dense_time, dense_fit, "-", linewidth=2, label="fit")
    axes[0].set(
        xlabel="Delay time (us)",
        ylabel=fit.signal_label,
        title=(
            f"T1 = {fit.t1_us:.6g} +/- "
            f"{fit.parameters['t1_uncertainty_us']:.2g} us"
        ),
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    residual = fit.measured - fit.fitted
    axes[1].axhline(0.0, color="0.4", linewidth=1)
    axes[1].plot(fit.time_us, residual, "o-", markersize=3)
    axes[1].set(
        xlabel="Delay time (us)",
        ylabel="Measured - fit",
        title=f"Residuals ($R^2$={fit.statistics['r_squared']:.4f})",
    )
    axes[1].grid(alpha=0.3)

    scatter = axes[2].scatter(
        fit.iq.real,
        fit.iq.imag,
        c=fit.time_us,
        s=18,
        cmap="viridis",
    )
    axes[2].set(xlabel="I", ylabel="Q", title="IQ relaxation trajectory")
    axes[2].axis("equal")
    axes[2].grid(alpha=0.3)
    figure.colorbar(scatter, ax=axes[2], label="Delay (us)")
    figure.suptitle(f"T1 fit from {fit.source_csv.name}")
    return figure


@dataclass(frozen=True)
class RamseyFit:
    source_csv: Path
    source_yml: Path
    signal: str
    signal_label: str
    time_us: np.ndarray
    iq: np.ndarray
    measured: np.ndarray
    fitted: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, float]
    metadata: Mapping[str, Any]

    @property
    def t2_star_us(self) -> float:
        return float(self.parameters["t2_star_us"])

    def passes(
        self,
        *,
        minimum_r_squared: float,
        minimum_oscillations: float,
        maximum_relative_t2_uncertainty: float,
    ) -> bool:
        return bool(
            np.isfinite(self.statistics["r_squared"])
            and np.isfinite(self.statistics["relative_t2_uncertainty"])
            and self.statistics["r_squared"] >= minimum_r_squared
            and self.statistics["oscillations"] >= minimum_oscillations
            and self.statistics["relative_t2_uncertainty"]
            <= maximum_relative_t2_uncertainty
            and self.t2_star_us > 0
        )


def _ramsey_model(
    time_us: np.ndarray,
    amplitude: float,
    t2_star_us: float,
    frequency_mhz: float,
    phase_rad: float,
    offset: float,
    origin_us: float,
) -> np.ndarray:
    elapsed = time_us - origin_us
    return (
        amplitude
        * np.exp(-elapsed / t2_star_us)
        * np.cos(2.0 * np.pi * frequency_mhz * elapsed + phase_rad)
        + offset
    )


def fit_ramsey(
    csv_path: Path,
    *,
    signal: str = "IQ",
) -> RamseyFit:
    trace = load_native_trace(
        csv_path,
        quick_class="T2Ramsey",
        axis_text="delay time",
        minimum_points=10,
    )
    measured, signal_label, rotation = _fit_signal(trace, signal)
    time_us = trace.x
    steps = np.diff(time_us)
    step = float(np.median(steps))
    span = float(np.ptp(time_us))
    if (
        span <= 0
        or step <= 0
        or not np.allclose(
            steps,
            step,
            rtol=0.05,
            atol=max(step * 1e-8, 1e-12),
        )
    ):
        raise AnalysisError(
            "Ramsey fitting requires a uniformly spaced positive delay axis"
        )
    nyquist_mhz = 0.5 / step
    programmed = _nested(trace.metadata, "parameters", "var", "fringe_freq")
    drive = _nested(trace.metadata, "parameters", "var", "q_freq")
    programmed_fringe_mhz = (
        float(programmed) if programmed is not None else float("nan")
    )
    drive_frequency_mhz = (
        float(drive) if drive is not None else float("nan")
    )

    centered = measured - np.mean(measured)
    frequencies = np.fft.rfftfreq(len(time_us), d=step)
    spectrum = np.abs(np.fft.rfft(centered))
    ranked = np.argsort(spectrum[1:])[::-1] + 1
    candidates = []
    if (
        np.isfinite(programmed_fringe_mhz)
        and 0 < programmed_fringe_mhz < nyquist_mhz
    ):
        candidates.append(programmed_fringe_mhz)
    candidates.extend(
        float(frequencies[index])
        for index in ranked[: min(8, len(ranked))]
        if 0 < frequencies[index] < nyquist_mhz
    )
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise AnalysisError("Ramsey fringe frequency could not be estimated")

    origin = float(time_us[0])
    amplitude_guess = max(
        float(np.ptp(measured)) / 2.0,
        np.finfo(float).eps,
    )
    offset_guess = float(np.mean(measured))
    minimum_t2 = max(np.finfo(float).eps, step * 0.01)
    maximum_t2 = span * 100.0

    def model(
        values,
        amplitude,
        t2_star_us,
        frequency_mhz,
        phase_rad,
        offset,
    ):
        return _ramsey_model(
            values,
            amplitude,
            t2_star_us,
            frequency_mhz,
            phase_rad,
            offset,
            origin,
        )

    best = None
    for frequency_guess in candidates:
        for phase_guess in (0.0, np.pi / 2.0, -np.pi / 2.0, np.pi):
            try:
                parameters, covariance = curve_fit(
                    model,
                    time_us,
                    measured,
                    p0=[
                        amplitude_guess,
                        max(span / 2.0, step),
                        frequency_guess,
                        phase_guess,
                        offset_guess,
                    ],
                    bounds=(
                        [
                            -np.inf,
                            minimum_t2,
                            0.0,
                            -4.0 * np.pi,
                            -np.inf,
                        ],
                        [
                            np.inf,
                            maximum_t2,
                            nyquist_mhz,
                            4.0 * np.pi,
                            np.inf,
                        ],
                    ),
                    maxfev=100000,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            fitted = model(time_us, *parameters)
            squared_error = float(np.sum((measured - fitted) ** 2))
            if best is None or squared_error < best[0]:
                best = (
                    squared_error,
                    parameters,
                    covariance,
                    fitted,
                )
    if best is None:
        raise AnalysisError(
            "Ramsey fit did not converge from any initial frequency"
        )

    _, parameters, covariance, fitted = best
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    fitted_fringe_mhz = float(parameters[2])
    fitted_fringe_uncertainty = float(errors[2])
    t2_star_us = float(parameters[1])
    t2_uncertainty = float(errors[1])
    detuning_mhz = (
        fitted_fringe_mhz - programmed_fringe_mhz
        if np.isfinite(programmed_fringe_mhz)
        else float("nan")
    )
    predicted_drive_mhz = (
        drive_frequency_mhz + detuning_mhz
        if np.isfinite(drive_frequency_mhz) and np.isfinite(detuning_mhz)
        else float("nan")
    )
    alternate_drive_mhz = (
        drive_frequency_mhz - detuning_mhz
        if np.isfinite(drive_frequency_mhz) and np.isfinite(detuning_mhz)
        else float("nan")
    )
    residual = measured - fitted
    return RamseyFit(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        signal=_normalize_signal(signal),
        signal_label=signal_label,
        time_us=time_us,
        iq=trace.iq,
        measured=measured,
        fitted=fitted,
        parameters={
            "t2_star_us": t2_star_us,
            "t2_star_uncertainty_us": t2_uncertainty,
            "fitted_fringe_mhz": fitted_fringe_mhz,
            "fitted_fringe_uncertainty_mhz": fitted_fringe_uncertainty,
            "programmed_fringe_mhz": programmed_fringe_mhz,
            "drive_frequency_mhz": drive_frequency_mhz,
            "detuning_mhz": detuning_mhz,
            "predicted_drive_mhz": predicted_drive_mhz,
            "alternate_drive_mhz": alternate_drive_mhz,
            "amplitude": float(parameters[0]),
            "phase_rad": float(parameters[3]),
            "offset": float(parameters[4]),
            "rotation_angle_rad": float(rotation.get("angle_rad", 0.0)),
        },
        statistics={
            "r_squared": _r_squared(measured, fitted),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "oscillations": float(span * fitted_fringe_mhz),
            "relative_t2_uncertainty": float(
                t2_uncertainty / t2_star_us
            ),
            "points": int(len(time_us)),
        },
        metadata=trace.metadata,
    )


def plot_ramsey_fit(fit: RamseyFit):
    dense_time = np.linspace(
        float(fit.time_us[0]),
        float(fit.time_us[-1]),
        2000,
    )
    p = fit.parameters
    dense_fit = _ramsey_model(
        dense_time,
        p["amplitude"],
        p["t2_star_us"],
        p["fitted_fringe_mhz"],
        p["phase_rad"],
        p["offset"],
        float(fit.time_us[0]),
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.2),
        constrained_layout=True,
    )
    axes[0].plot(fit.time_us, fit.measured, "o", markersize=3, label="data")
    axes[0].plot(dense_time, dense_fit, "-", linewidth=2, label="fit")
    axes[0].set(
        xlabel="Delay time (us)",
        ylabel=fit.signal_label,
        title=(
            f"T2* = {fit.t2_star_us:.6g} us; "
            f"fringe = {p['fitted_fringe_mhz']:.6g} MHz"
        ),
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    residual = fit.measured - fit.fitted
    axes[1].axhline(0.0, color="0.4", linewidth=1)
    axes[1].plot(fit.time_us, residual, "o-", markersize=3)
    axes[1].set(
        xlabel="Delay time (us)",
        ylabel="Measured - fit",
        title=f"Residuals ($R^2$={fit.statistics['r_squared']:.4f})",
    )
    axes[1].grid(alpha=0.3)

    scatter = axes[2].scatter(
        fit.iq.real,
        fit.iq.imag,
        c=fit.time_us,
        s=18,
        cmap="viridis",
    )
    axes[2].set(xlabel="I", ylabel="Q", title="IQ Ramsey trajectory")
    axes[2].axis("equal")
    axes[2].grid(alpha=0.3)
    figure.colorbar(scatter, ax=axes[2], label="Delay (us)")
    figure.suptitle(f"Ramsey fit from {fit.source_csv.name}")
    return figure


@dataclass(frozen=True)
class LoopbackFit:
    source_csv: Path
    source_yml: Path
    time_us: np.ndarray
    iq: np.ndarray
    magnitude: np.ndarray
    projection: np.ndarray
    smoothed_projection: np.ndarray
    local_time_us: np.ndarray
    local_fitted: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, float]
    metadata: Mapping[str, Any]

    @property
    def recommended_r_offset_us(self) -> float:
        return float(self.parameters["recommended_r_offset_us"])

    def passes(
        self,
        *,
        minimum_edge_snr: float,
        minimum_r_squared: float,
        maximum_edge_uncertainty_us: float,
    ) -> bool:
        return all(
            self.acceptance_gates(
                minimum_edge_snr=minimum_edge_snr,
                minimum_r_squared=minimum_r_squared,
                maximum_edge_uncertainty_us=maximum_edge_uncertainty_us,
            ).values()
        )

    def acceptance_gates(
        self,
        *,
        minimum_edge_snr: float,
        minimum_r_squared: float,
        maximum_edge_uncertainty_us: float,
    ) -> Mapping[str, bool]:
        step = float(np.median(np.diff(self.time_us)))
        edge = float(self.parameters["edge_in_trace_us"])
        edge_snr = float(self.statistics["edge_snr"])
        r_squared = float(self.statistics["r_squared"])
        edge_uncertainty = float(self.parameters["edge_uncertainty_us"])
        return {
            "edge_snr": bool(
                np.isfinite(edge_snr) and edge_snr >= minimum_edge_snr
            ),
            "r_squared": bool(
                np.isfinite(r_squared) and r_squared >= minimum_r_squared
            ),
            "edge_uncertainty_us": bool(
                np.isfinite(edge_uncertainty)
                and edge_uncertainty <= maximum_edge_uncertainty_us
            ),
            "edge_inside_usable_record": bool(
                np.isfinite(edge)
                and self.time_us[0] + 2.0 * step
                < edge
                < self.time_us[-1] - 2.0 * step
            ),
        }


def _two_complex_centers(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitudes = np.abs(values)
    centers = np.asarray(
        [
            values[int(np.argmin(magnitudes))],
            values[int(np.argmax(magnitudes))],
        ],
        dtype=complex,
    )
    labels = np.zeros(len(values), dtype=int)
    for _ in range(50):
        distances = np.column_stack(
            [np.abs(values - center) for center in centers]
        )
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for index in (0, 1):
            members = values[new_labels == index]
            if len(members):
                new_centers[index] = np.mean(members)
        if np.array_equal(labels, new_labels) and np.allclose(
            centers,
            new_centers,
        ):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return centers, labels


def _edge_model(
    time_us: np.ndarray,
    baseline: float,
    contrast: float,
    edge_us: float,
    rise_us: float,
) -> np.ndarray:
    return baseline + 0.5 * contrast * (
        1.0 + np.tanh((time_us - edge_us) / (2.0 * rise_us))
    )


def fit_loopback(
    csv_path: Path,
    *,
    smooth_sigma_bins: float = 5.0,
) -> LoopbackFit:
    trace = load_native_trace(
        csv_path,
        quick_class="LoopBack",
        axis_text="time",
        minimum_points=50,
    )
    if smooth_sigma_bins < 0:
        raise ConfigError("loopback smoothing sigma cannot be negative")
    centers, labels = _two_complex_centers(trace.iq)
    off_index = int(np.argmin(np.abs(centers)))
    on_index = 1 - off_index
    direction = centers[on_index] - centers[off_index]
    if np.isclose(abs(direction), 0.0):
        raise AnalysisError("loopback IQ states are indistinguishable")
    unit = direction / abs(direction)
    projection = np.real(
        (trace.iq - centers[off_index]) * np.conj(unit)
    )
    smoothed = (
        gaussian_filter1d(projection, smooth_sigma_bins)
        if smooth_sigma_bins > 0
        else projection.copy()
    )
    low_values = smoothed[labels == off_index]
    high_values = smoothed[labels == on_index]
    if len(low_values) < 3 or len(high_values) < 3:
        raise AnalysisError("loopback trace does not contain two stable states")
    baseline_guess = float(np.median(low_values))
    high_guess = float(np.median(high_values))
    contrast_guess = high_guess - baseline_guess
    if contrast_guess <= 0:
        raise AnalysisError("loopback pulse contrast is not positive")
    normalized = (smoothed - baseline_guess) / contrast_guess
    crossings = np.flatnonzero(
        (normalized[:-1] < 0.5) & (normalized[1:] >= 0.5)
    )
    if not len(crossings):
        raise AnalysisError(
            "no rising loopback edge was found; rerun 02 with "
            "READOUT_OFFSET_US = 0 so the edge lies inside the ADC trace"
        )
    slopes = normalized[crossings + 1] - normalized[crossings]
    crossing = int(crossings[int(np.argmax(slopes))])
    fraction = (
        (0.5 - normalized[crossing])
        / (normalized[crossing + 1] - normalized[crossing])
    )
    edge_guess = float(
        trace.x[crossing]
        + fraction * (trace.x[crossing + 1] - trace.x[crossing])
    )

    step = float(np.median(np.diff(trace.x)))
    span = float(np.ptp(trace.x))
    half_window = max(30.0 * step, 0.05 * span)
    local_mask = (
        (trace.x >= edge_guess - half_window)
        & (trace.x <= edge_guess + half_window)
    )
    local_time = trace.x[local_mask]
    local_signal = smoothed[local_mask]
    if len(local_time) < 12:
        raise AnalysisError("loopback edge has too few local points to fit")
    minimum_rise = max(step / 20.0, np.finfo(float).eps)
    maximum_rise = max(half_window, minimum_rise * 2.0)
    try:
        parameters, covariance = curve_fit(
            _edge_model,
            local_time,
            local_signal,
            p0=[
                baseline_guess,
                contrast_guess,
                edge_guess,
                max(step, minimum_rise),
            ],
            bounds=(
                [
                    -np.inf,
                    0.0,
                    float(local_time[0]),
                    minimum_rise,
                ],
                [
                    np.inf,
                    np.inf,
                    float(local_time[-1]),
                    maximum_rise,
                ],
            ),
            maxfev=100000,
        )
    except Exception as error:
        raise AnalysisError(f"loopback edge fit failed: {error}") from error
    local_fitted = _edge_model(local_time, *parameters)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    edge_us = float(parameters[2])
    edge_uncertainty = float(errors[2])
    current_offset = _nested(
        trace.metadata,
        "parameters",
        "var",
        "r_offset",
    )
    current_offset_us = (
        float(current_offset) if current_offset is not None else 0.0
    )
    recommended_offset = current_offset_us + edge_us
    low_residual = projection[labels == off_index] - np.median(
        projection[labels == off_index]
    )
    noise = 1.4826 * float(np.median(np.abs(low_residual)))
    edge_snr = float(parameters[1]) / max(noise, np.finfo(float).eps)
    return LoopbackFit(
        source_csv=trace.source_csv,
        source_yml=trace.source_yml,
        time_us=trace.x,
        iq=trace.iq,
        magnitude=np.abs(trace.iq),
        projection=projection,
        smoothed_projection=smoothed,
        local_time_us=local_time,
        local_fitted=local_fitted,
        parameters={
            "edge_in_trace_us": edge_us,
            "edge_uncertainty_us": edge_uncertainty,
            "current_r_offset_us": current_offset_us,
            "recommended_r_offset_us": recommended_offset,
            "rise_10_to_90_us": float(
                4.394449154672439 * parameters[3]
            ),
            "baseline": float(parameters[0]),
            "contrast": float(parameters[1]),
            "rotation_angle_rad": float(np.angle(unit)),
        },
        statistics={
            "r_squared": _r_squared(local_signal, local_fitted),
            "edge_snr": edge_snr,
            "points": int(len(trace.x)),
        },
        metadata=trace.metadata,
    )


def plot_loopback_fit(fit: LoopbackFit):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.2),
        constrained_layout=True,
    )
    axes[0].plot(fit.time_us, fit.magnitude, linewidth=1)
    axes[0].axvline(
        fit.parameters["edge_in_trace_us"],
        color="tab:red",
        linestyle="--",
        label="fitted edge",
    )
    axes[0].set(
        xlabel="ADC time (us)",
        ylabel="|I + iQ|",
        title="Raw loopback magnitude",
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        fit.time_us,
        fit.projection,
        color="0.7",
        linewidth=0.8,
        label="projected IQ",
    )
    axes[1].plot(
        fit.time_us,
        fit.smoothed_projection,
        linewidth=1.5,
        label="smoothed",
    )
    axes[1].plot(
        fit.local_time_us,
        fit.local_fitted,
        linewidth=2,
        label="edge fit",
    )
    axes[1].set(
        xlabel="ADC time (us)",
        ylabel="Pulse-axis projection",
        title=(
            f"r_offset recommendation = "
            f"{fit.recommended_r_offset_us:.6f} us"
        ),
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    scatter = axes[2].scatter(
        fit.iq.real,
        fit.iq.imag,
        c=fit.time_us,
        s=8,
        cmap="viridis",
    )
    axes[2].set(xlabel="I", ylabel="Q", title="Loopback IQ trajectory")
    axes[2].axis("equal")
    axes[2].grid(alpha=0.3)
    figure.colorbar(scatter, ax=axes[2], label="ADC time (us)")
    figure.suptitle(f"Loopback edge fit from {fit.source_csv.name}")
    return figure


def _source_var(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _nested(metadata, "parameters", "var")
    return value if isinstance(value, Mapping) else {}


def _record(
    *,
    value: float,
    unit: str,
    uncertainty: Mapping[str, float],
    source_csv: Path,
    source_yml: Path,
    analysis: str,
    quality: Mapping[str, Any],
    valid_domain: Mapping[str, Any],
    model: str,
    notes: str,
) -> dict:
    return {
        "value": float(value),
        "unit": unit,
        "uncertainty": dict(uncertainty),
        "provenance": {
            "source": str(source_csv),
            "source_yml": str(source_yml),
            "fitted_at": utc_now(),
            "analysis": analysis,
        },
        "quality": dict(quality),
        "valid_domain": dict(valid_domain),
        "model": model,
        "notes": notes,
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def _fixed_domain(metadata: Mapping[str, Any]) -> dict:
    source_var = _source_var(metadata)
    domain = {}
    for name in ("z_gain", "r_power", "q_gain", "q_length", "q_freq"):
        if name not in source_var:
            continue
        try:
            value = float(source_var[name])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            domain[name] = [value, value]
    return domain


def accept_spectroscopy_fit(
    project_root: Path,
    fit: SpectroscopyFit,
    *,
    minimum_r_squared: float,
    minimum_contrast_snr: float,
    maximum_center_uncertainty_fraction_of_fwhm: float,
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_r_squared=minimum_r_squared,
        minimum_contrast_snr=minimum_contrast_snr,
        maximum_center_uncertainty_fraction_of_fwhm=(
            maximum_center_uncertainty_fraction_of_fwhm
        ),
    )
    if not gates_passed and not force_write:
        raise AnalysisError("spectroscopy fit did not pass its acceptance gates")
    name = SPECTROSCOPY_KINDS[fit.kind]["record"]
    record = _record(
        value=fit.center_mhz,
        unit="MHz",
        uncertainty={
            "center_mhz": fit.parameters["center_uncertainty_mhz"],
            "fwhm_mhz": fit.parameters["fwhm_mhz"],
        },
        source_csv=fit.source_csv,
        source_yml=fit.source_yml,
        analysis="quickexp_v3.native_fit.fit_spectroscopy",
        quality=fit.statistics,
        valid_domain=_fixed_domain(fit.metadata),
        model="signed_lorentzian_with_linear_background",
        notes=(
            f"Center fitted from the {fit.signal_label} channel of a "
            f"fixed-flux {fit.kind} spectroscopy sweep."
        ),
    )
    updates = annotate_forced_write(
        {f"defaults.{name}": record},
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)


def accept_t1_fit(
    project_root: Path,
    fit: T1Fit,
    *,
    minimum_r_squared: float,
    minimum_span_over_t1: float,
    maximum_relative_t1_uncertainty: float,
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_r_squared=minimum_r_squared,
        minimum_span_over_t1=minimum_span_over_t1,
        maximum_relative_t1_uncertainty=maximum_relative_t1_uncertainty,
    )
    if not gates_passed and not force_write:
        raise AnalysisError("T1 fit did not pass its acceptance gates")
    record = _record(
        value=fit.t1_us,
        unit="us",
        uncertainty={"t1_us": fit.parameters["t1_uncertainty_us"]},
        source_csv=fit.source_csv,
        source_yml=fit.source_yml,
        analysis="quickexp_v3.native_fit.fit_t1",
        quality=fit.statistics,
        valid_domain=_fixed_domain(fit.metadata),
        model="bounded_exponential",
        notes=f"T1 fitted from the {fit.signal_label} channel.",
    )
    updates = annotate_forced_write(
        {"derived.t1": record},
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)


def accept_ramsey_fit(
    project_root: Path,
    fit: RamseyFit,
    *,
    minimum_r_squared: float,
    minimum_oscillations: float,
    maximum_relative_t2_uncertainty: float,
    update_q_frequency: bool,
    correction_sign: int,
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_r_squared=minimum_r_squared,
        minimum_oscillations=minimum_oscillations,
        maximum_relative_t2_uncertainty=maximum_relative_t2_uncertainty,
    )
    if not gates_passed and not force_write:
        raise AnalysisError("Ramsey fit did not pass its acceptance gates")
    if correction_sign not in (-1, 1):
        raise ConfigError("Ramsey correction_sign must be +1 or -1")
    updates = {
        "derived.t2_ramsey": _record(
            value=fit.t2_star_us,
            unit="us",
            uncertainty={
                "t2_star_us": fit.parameters["t2_star_uncertainty_us"],
                "fringe_mhz": fit.parameters[
                    "fitted_fringe_uncertainty_mhz"
                ],
            },
            source_csv=fit.source_csv,
            source_yml=fit.source_yml,
            analysis="quickexp_v3.native_fit.fit_ramsey",
            quality=fit.statistics,
            valid_domain=_fixed_domain(fit.metadata),
            model="decaying_cosine",
            notes=f"T2* fitted from the {fit.signal_label} channel.",
        )
    }
    if update_q_frequency:
        drive = fit.parameters["drive_frequency_mhz"]
        detuning = fit.parameters["detuning_mhz"]
        if not np.isfinite(drive) or not np.isfinite(detuning):
            raise AnalysisError(
                "Ramsey metadata lacks q_freq or fringe_freq; "
                "q_freq cannot be updated"
            )
        corrected = float(drive + correction_sign * detuning)
        updates["defaults.q_freq"] = _record(
            value=corrected,
            unit="MHz",
            uncertainty={
                "fringe_mhz": fit.parameters[
                    "fitted_fringe_uncertainty_mhz"
                ]
            },
            source_csv=fit.source_csv,
            source_yml=fit.source_yml,
            analysis="quickexp_v3.native_fit.fit_ramsey",
            quality=fit.statistics,
            valid_domain=_fixed_domain(fit.metadata),
            model="ramsey_fringe_frequency_correction",
            notes=(
                f"q_freq correction used sign {correction_sign:+d}. A scalar "
                "Ramsey trace cannot determine the detuning sign independently; "
                "confirm by repeating Ramsey at this frequency."
            ),
        )
    updates = annotate_forced_write(
        updates,
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)


def accept_loopback_fit(
    project_root: Path,
    fit: LoopbackFit,
    *,
    minimum_edge_snr: float,
    minimum_r_squared: float,
    maximum_edge_uncertainty_us: float,
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_edge_snr=minimum_edge_snr,
        minimum_r_squared=minimum_r_squared,
        maximum_edge_uncertainty_us=maximum_edge_uncertainty_us,
    )
    if not gates_passed and not force_write:
        raise AnalysisError("loopback fit did not pass its acceptance gates")
    record = _record(
        value=fit.recommended_r_offset_us,
        unit="us",
        uncertainty={
            "edge_us": fit.parameters["edge_uncertainty_us"],
            "rise_10_to_90_us": fit.parameters["rise_10_to_90_us"],
        },
        source_csv=fit.source_csv,
        source_yml=fit.source_yml,
        analysis="quickexp_v3.native_fit.fit_loopback",
        quality=fit.statistics,
        valid_domain={},
        model="smoothed_iq_logistic_edge",
        notes=(
            "Recommended offset equals the acquisition's existing r_offset "
            "plus the fitted pulse edge position within the ADC trace."
        ),
    )
    updates = annotate_forced_write(
        {"defaults.r_offset": record},
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)
