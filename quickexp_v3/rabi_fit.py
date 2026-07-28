"""Fit native Quick Rabi files and accept pi-pulse calibration values."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import yaml

from .analysis import rotate_iq
from .config import ConfigRepository, SCHEMA_VERSION
from .errors import AnalysisError, ConfigError
from .util import to_builtin, utc_now


RABI_VARIABLES = {
    "q_length": {
        "label": "Qubit Pulse Length",
        "unit": "us",
    },
    "q_gain": {
        "label": "Qubit Pulse Gain",
        "unit": "a.u.",
    },
}
MODEL_NAME = "damped_cosine_phase_corrected"


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _axis_variable(metadata: Mapping[str, Any]) -> tuple[str, str]:
    independent = metadata.get("independent")
    if not isinstance(independent, list) or len(independent) != 1:
        raise AnalysisError(
            "Rabi fitting requires exactly one native Quick independent axis"
        )
    entry = independent[0]
    if not isinstance(entry, (list, tuple)) or not entry:
        raise AnalysisError("native Quick YML has an invalid independent axis")
    label = str(entry[0])
    unit = str(entry[1]) if len(entry) > 1 else ""
    normalized = label.strip().lower()
    if "pulse length" in normalized:
        return "q_length", unit or "us"
    if "pulse gain" in normalized:
        return "q_gain", unit or "a.u."
    raise AnalysisError(f"unsupported Rabi independent axis {label!r}")


def inspect_rabi_file(
    csv_path: Path,
    *,
    variable: str = "auto",
) -> tuple[Path, dict, str, str]:
    """Inspect one native CSV/YML pair and identify its Rabi sweep variable."""
    source = Path(csv_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Rabi CSV does not exist or is empty: {source}")
    yml_path = source.with_suffix(".yml")
    if not yml_path.is_file():
        raise FileNotFoundError(f"paired Quick YML does not exist: {yml_path}")
    metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise AnalysisError(f"{yml_path} must contain a YAML mapping")
    quick_class = _nested(metadata, "parameters", "quick_experiment")
    if quick_class not in (None, "Rabi"):
        raise AnalysisError(
            f"{source.name} is {quick_class!r}, not a Quick Rabi acquisition"
        )
    detected, unit = _axis_variable(metadata)
    requested = str(variable).strip().lower()
    if requested not in {"auto", *RABI_VARIABLES}:
        raise ConfigError("Rabi variable must be auto, q_length, or q_gain")
    if requested != "auto" and requested != detected:
        raise AnalysisError(
            f"{source.name} sweeps {detected}, not requested {requested}"
        )
    return yml_path, dict(metadata), detected, unit


def find_latest_rabi(
    data_directory: Path,
    *,
    variable: str = "auto",
) -> Path:
    """Find the newest complete one-dimensional native Quick Rabi file."""
    root = Path(data_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Quick data directory does not exist: {root}")
    candidates = []
    for csv_path in root.glob("*.csv"):
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            continue
        try:
            inspect_rabi_file(csv_path, variable=variable)
        except (AnalysisError, FileNotFoundError, ConfigError, yaml.YAMLError):
            continue
        candidates.append(csv_path)
    if not candidates:
        suffix = "" if variable == "auto" else f" sweeping {variable}"
        raise FileNotFoundError(
            f"No complete one-dimensional Quick Rabi CSV/YML pair{suffix} "
            f"was found in {root}."
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


def _load_native_data(
    csv_path: Path,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        matrix = np.loadtxt(csv_path, delimiter=",")
    except ValueError:
        matrix = np.loadtxt(csv_path)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    if matrix.ndim != 2 or matrix.shape[1] < 5:
        raise AnalysisError(
            "native Rabi CSV must contain axis, amplitude, phase, I, and Q"
        )

    i_column = _dependent_column(metadata, "I")
    q_column = _dependent_column(metadata, "Q")
    amplitude_column = _dependent_column(metadata, "Amplitude")
    phase_column = _dependent_column(metadata, "Phase")
    if i_column is None or q_column is None:
        i_column, q_column = matrix.shape[1] - 2, matrix.shape[1] - 1
    if max(i_column, q_column) >= matrix.shape[1]:
        raise AnalysisError("native Rabi dependent columns do not match its CSV")

    x = matrix[:, 0]
    iq = matrix[:, i_column] + 1j * matrix[:, q_column]
    amplitude = (
        matrix[:, amplitude_column]
        if amplitude_column is not None and amplitude_column < matrix.shape[1]
        else np.abs(iq)
    )
    phase = (
        matrix[:, phase_column]
        if phase_column is not None and phase_column < matrix.shape[1]
        else np.angle(iq)
    )
    finite = (
        np.isfinite(x)
        & np.isfinite(iq.real)
        & np.isfinite(iq.imag)
        & np.isfinite(amplitude)
        & np.isfinite(phase)
    )
    x, iq, amplitude, phase = (
        values[finite] for values in (x, iq, amplitude, phase)
    )
    if x.size < 10:
        raise AnalysisError("Rabi fitting requires at least 10 finite points")
    order = np.argsort(x)
    x, iq, amplitude, phase = (
        values[order] for values in (x, iq, amplitude, phase)
    )
    if np.any(np.diff(x) <= 0):
        raise AnalysisError("Rabi sweep axis must be unique and increasing")
    steps = np.diff(x)
    step = float(np.median(steps))
    if not np.allclose(steps, step, rtol=0.05, atol=max(step * 1e-8, 1e-12)):
        raise AnalysisError("Rabi fitting requires a uniformly spaced sweep")
    return x, iq, amplitude, phase


def _model(x: np.ndarray, parameters: np.ndarray, x_start: float) -> np.ndarray:
    offset, amplitude, decay, frequency, phase, slope = parameters
    shifted = x - x_start
    return (
        offset
        + slope * shifted
        + amplitude
        * np.exp(-shifted / decay)
        * np.cos(2.0 * np.pi * frequency * shifted + phase)
    )


def _frequency_candidates(x: np.ndarray, signal: np.ndarray) -> np.ndarray:
    step = float(np.median(np.diff(x)))
    detrended = signal - np.polyval(np.polyfit(x, signal, 1), x)
    spectrum = np.abs(np.fft.rfft(detrended))
    frequencies = np.fft.rfftfreq(len(x), d=step)
    if len(frequencies) < 2:
        raise AnalysisError("Rabi sweep is too short for frequency estimation")
    order = np.argsort(spectrum[1:])[::-1] + 1
    selected = frequencies[order[: min(5, len(order))]]
    return np.asarray(selected[selected > 0], dtype=float)


def _pi_recommendation(
    parameters: np.ndarray,
    covariance: np.ndarray,
    x_start: float,
) -> tuple[float, float, float]:
    frequency = float(parameters[3])
    phase = float(parameters[4])
    phase_at_zero = phase - 2.0 * np.pi * frequency * x_start
    target = np.pi if np.cos(phase_at_zero) >= 0 else 0.0
    raw_cycles = (target - phase_at_zero) / (2.0 * np.pi)
    branch = int(np.ceil(-raw_cycles))
    recommendation = (raw_cycles + branch) / frequency
    while recommendation <= 0:
        branch += 1
        recommendation = (raw_cycles + branch) / frequency

    derivative_frequency = -(recommendation - x_start) / frequency
    derivative_phase = -1.0 / (2.0 * np.pi * frequency)
    local_covariance = covariance[np.ix_([3, 4], [3, 4])]
    gradient = np.array([derivative_frequency, derivative_phase])
    variance = float(gradient @ local_covariance @ gradient)
    uncertainty = float(np.sqrt(max(variance, 0.0)))
    return float(recommendation), uncertainty, float(phase_at_zero)


@dataclass(frozen=True)
class RabiFit:
    source_csv: Path
    source_yml: Path
    variable: str
    unit: str
    x: np.ndarray
    iq: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    projected: np.ndarray
    fitted: np.ndarray
    parameters: Mapping[str, float]
    statistics: Mapping[str, float]
    metadata: Mapping[str, Any]

    @property
    def pi_value(self) -> float:
        return float(self.parameters["pi_value"])

    def passes(
        self,
        *,
        minimum_r_squared: float,
        minimum_oscillations: float,
        maximum_relative_pi_uncertainty: float,
    ) -> bool:
        r_squared = float(self.statistics["r_squared"])
        oscillations = float(self.statistics["oscillations"])
        relative_uncertainty = float(
            self.statistics["relative_pi_uncertainty"]
        )
        return bool(
            np.isfinite(r_squared)
            and np.isfinite(relative_uncertainty)
            and r_squared >= minimum_r_squared
            and oscillations >= minimum_oscillations
            and relative_uncertainty <= maximum_relative_pi_uncertainty
            and float(np.min(self.x)) <= self.pi_value <= float(np.max(self.x))
        )


def fit_rabi(
    csv_path: Path,
    *,
    variable: str = "auto",
) -> RabiFit:
    """Fit one native one-dimensional Quick Rabi CSV/YML pair."""
    source = Path(csv_path).expanduser().resolve()
    yml_path, metadata, detected, unit = inspect_rabi_file(
        source,
        variable=variable,
    )
    x, iq, amplitude, phase = _load_native_data(source, metadata)
    projected, rotation = rotate_iq(iq)
    projected = np.asarray(projected, dtype=float)
    span = float(np.ptp(x))
    step = float(np.median(np.diff(x)))
    if span <= 0 or step <= 0:
        raise AnalysisError("Rabi sweep must span more than one value")

    candidates = _frequency_candidates(x, projected)
    if candidates.size == 0:
        raise AnalysisError("Rabi oscillation frequency could not be estimated")
    minimum_frequency = 1.0 / (span * 20.0)
    maximum_frequency = 0.5 / step
    signal_span = max(float(np.ptp(projected)), np.finfo(float).eps)
    offset = float(np.mean(projected))
    amplitude_guess = signal_span / 2.0
    slope_guess = float((projected[-1] - projected[0]) / span)
    lower = np.array(
        [-np.inf, -np.inf, span / 1000.0, minimum_frequency, -4 * np.pi, -np.inf]
    )
    upper = np.array(
        [np.inf, np.inf, span * 1000.0, maximum_frequency, 4 * np.pi, np.inf]
    )

    best = None
    best_rss = np.inf
    for frequency in candidates:
        frequency = float(np.clip(frequency, minimum_frequency, maximum_frequency))
        for phase_guess in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
            initial = np.array(
                [
                    offset,
                    amplitude_guess,
                    max(span, np.finfo(float).eps),
                    frequency,
                    phase_guess,
                    slope_guess,
                ]
            )
            try:
                result = least_squares(
                    lambda values: _model(x, values, float(x[0])) - projected,
                    initial,
                    bounds=(lower, upper),
                    loss="soft_l1",
                    f_scale=max(0.05 * signal_span, np.finfo(float).eps),
                    x_scale="jac",
                    max_nfev=50_000,
                )
            except (ValueError, FloatingPointError):
                continue
            prediction = _model(x, result.x, float(x[0]))
            rss = float(np.sum((projected - prediction) ** 2))
            if result.success and rss < best_rss:
                best, best_rss = result, rss
    if best is None:
        raise AnalysisError("Rabi damped-cosine fit did not converge")

    parameters = np.asarray(best.x, dtype=float)
    degrees_of_freedom = max(len(x) - len(parameters), 1)
    residual_variance = best_rss / degrees_of_freedom
    covariance = np.linalg.pinv(best.jac.T @ best.jac) * residual_variance
    if parameters[1] < 0:
        parameters[1] *= -1
        parameters[4] += np.pi
    parameters[4] = (parameters[4] + np.pi) % (2.0 * np.pi) - np.pi
    fitted = _model(x, parameters, float(x[0]))
    residual = projected - fitted
    total_variance = float(np.sum((projected - np.mean(projected)) ** 2))
    r_squared = (
        float(1.0 - np.sum(residual**2) / total_variance)
        if total_variance > 0
        else float("nan")
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    pi_value, pi_uncertainty, phase_at_zero = _pi_recommendation(
        parameters,
        covariance,
        float(x[0]),
    )
    frequency = float(parameters[3])
    half_period = 0.5 / frequency
    half_period_uncertainty = float(
        0.5 * standard_errors[3] / (frequency**2)
    )
    relative_pi_uncertainty = (
        pi_uncertainty / abs(pi_value) if pi_value != 0 else float("inf")
    )
    parameter_map = {
        "pi_value": pi_value,
        "pi_uncertainty": pi_uncertainty,
        "half_period": half_period,
        "half_period_uncertainty": half_period_uncertainty,
        "frequency": frequency,
        "frequency_uncertainty": float(standard_errors[3]),
        "decay": float(parameters[2]),
        "decay_uncertainty": float(standard_errors[2]),
        "phase_at_first_point": float(parameters[4]),
        "phase_at_zero": phase_at_zero,
        "offset": float(parameters[0]),
        "amplitude": float(parameters[1]),
        "slope": float(parameters[5]),
        "rotation_angle_rad": float(rotation["angle_rad"]),
    }
    statistics = {
        "r_squared": r_squared,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "normalized_rmse": float(
            np.sqrt(np.mean(residual**2)) / signal_span
        ),
        "oscillations": float(span * frequency),
        "relative_pi_uncertainty": float(relative_pi_uncertainty),
        "points": int(len(x)),
    }
    return RabiFit(
        source_csv=source,
        source_yml=yml_path,
        variable=detected,
        unit=unit,
        x=x,
        iq=iq,
        amplitude=amplitude,
        phase=phase,
        projected=projected,
        fitted=fitted,
        parameters=parameter_map,
        statistics=statistics,
        metadata=metadata,
    )


def plot_rabi_fit(fit: RabiFit):
    """Plot projected Rabi data, residuals, and the measured IQ trajectory."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.2),
        constrained_layout=True,
    )
    dense_x = np.linspace(float(fit.x[0]), float(fit.x[-1]), 2000)
    p = fit.parameters
    dense_parameters = np.array(
        [
            p["offset"],
            p["amplitude"],
            p["decay"],
            p["frequency"],
            p["phase_at_first_point"],
            p["slope"],
        ]
    )
    dense_fit = _model(dense_x, dense_parameters, float(fit.x[0]))
    axes[0].plot(fit.x, fit.projected, "o", markersize=3, label="projected IQ")
    axes[0].plot(dense_x, dense_fit, "-", linewidth=2, label="damped-cosine fit")
    axes[0].axvline(
        fit.pi_value,
        color="tab:red",
        linestyle="--",
        label=f"pi = {fit.pi_value:.6g} {fit.unit}",
    )
    axes[0].set(
        xlabel=f"{RABI_VARIABLES[fit.variable]['label']} ({fit.unit})",
        ylabel="Rotated IQ projection",
        title=f"Rabi fit ($R^2$={fit.statistics['r_squared']:.4f})",
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    residual = fit.projected - fit.fitted
    axes[1].axhline(0.0, color="0.4", linewidth=1)
    axes[1].plot(fit.x, residual, "o-", markersize=3)
    axes[1].set(
        xlabel=f"{fit.variable} ({fit.unit})",
        ylabel="Measured - fit",
        title=f"Residuals (RMSE={fit.statistics['rmse']:.3g})",
    )
    axes[1].grid(alpha=0.3)

    scatter = axes[2].scatter(
        fit.iq.real,
        fit.iq.imag,
        c=fit.x,
        s=18,
        cmap="viridis",
    )
    axes[2].set(xlabel="I", ylabel="Q", title="IQ trajectory")
    axes[2].axis("equal")
    axes[2].grid(alpha=0.3)
    figure.colorbar(scatter, ax=axes[2], label=f"{fit.variable} ({fit.unit})")
    figure.suptitle(f"Rabi calibration from {fit.source_csv.name}")
    return figure


def calibration_record(fit: RabiFit) -> dict:
    """Build one accepted scalar q_length or q_gain calibration record."""
    source_var = _nested(fit.metadata, "parameters", "var")
    source_var = source_var if isinstance(source_var, Mapping) else {}
    valid_domain = {
        fit.variable: [float(np.min(fit.x)), float(np.max(fit.x))]
    }
    for name in ("z_gain", "q_freq", "q_gain", "q_length"):
        if name == fit.variable or name not in source_var:
            continue
        try:
            value = float(source_var[name])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            valid_domain[name] = [value, value]
    return {
        "value": fit.pi_value,
        "unit": fit.unit,
        "uncertainty": {
            "pi_value": float(fit.parameters["pi_uncertainty"]),
            "frequency": float(fit.parameters["frequency_uncertainty"]),
        },
        "provenance": {
            "source": str(fit.source_csv),
            "source_yml": str(fit.source_yml),
            "fitted_at": utc_now(),
            "analysis": "quickexp_v3.rabi_fit.fit_rabi",
        },
        "quality": dict(fit.statistics),
        "valid_domain": valid_domain,
        "model": MODEL_NAME,
        "notes": (
            "Pi recommendation is the first fitted oscillation extremum "
            "opposite the extrapolated zero-pulse state."
        ),
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def _atomic_yaml(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            yaml.safe_dump(
                to_builtin(document),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def accept_rabi_fit(
    project_root: Path,
    fit: RabiFit,
    *,
    minimum_r_squared: float,
    minimum_oscillations: float,
    maximum_relative_pi_uncertainty: float,
) -> Path:
    """Atomically accept a quality-gated q_length or q_gain pi calibration."""
    if not fit.passes(
        minimum_r_squared=minimum_r_squared,
        minimum_oscillations=minimum_oscillations,
        maximum_relative_pi_uncertainty=maximum_relative_pi_uncertainty,
    ):
        raise AnalysisError(
            "Rabi fit was not accepted: "
            f"R^2={fit.statistics['r_squared']:.6f}, "
            f"oscillations={fit.statistics['oscillations']:.3f}, "
            "relative pi uncertainty="
            f"{fit.statistics['relative_pi_uncertainty']:.3%}"
        )

    root = Path(project_root).resolve()
    target = root / "calibration.yml"
    source = target if target.exists() else root / "calibration.example.yml"
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{source} must contain a YAML mapping")
    document = deepcopy(dict(loaded))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"calibration schema_version must be {SCHEMA_VERSION}"
        )

    defaults = document.setdefault("records", {}).setdefault("defaults", {})
    previous = deepcopy(defaults.get(fit.variable))
    defaults[fit.variable] = calibration_record(fit)
    if previous is not None:
        document.setdefault("history", []).append(
            {
                "record": f"defaults.{fit.variable}",
                "superseded_at": utc_now(),
                "previous": previous,
            }
        )
    document["revision"] = int(document.get("revision", 0)) + 1
    document["updated_at"] = date.today().isoformat()

    hardware_path = (
        root / "hardware.yml"
        if (root / "hardware.yml").exists()
        else root / "hardware.example.yml"
    )
    presets_path = (
        root / "presets.yml"
        if (root / "presets.yml").exists()
        else root / "presets.example.yml"
    )
    repository = ConfigRepository.from_files(
        hardware_path,
        None,
        presets_path,
    )
    ConfigRepository(
        repository.hardware,
        document,
        {"schema_version": SCHEMA_VERSION, "presets": repository.presets},
    )
    _atomic_yaml(target, document)
    return target
