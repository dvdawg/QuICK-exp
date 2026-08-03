"""Registered perturbation probes and pure response extraction.

The registry only names experiments that already exist in Quick-exp.  This
module plans overrides and analyses native CSV/YML pairs; acquisition remains
owned by :mod:`quickexp_v3.ide` and its safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
import yaml

from ...notch_fit import fit_spectroscopy_features
from ...iq_gmm import fit_readout_optimization
from ...rabi_fit import fit_rabi
from .candidates import Candidate


@dataclass(frozen=True)
class Probe:
    probe_id: str
    experiment: str
    preset: str
    build_overrides: Callable[[Candidate, Mapping[str, Any]], Mapping[str, Any]]
    extract_response: Callable[[Sequence[Path]], Mapping[str, float]]
    estimated_seconds: Callable[[Mapping[str, Any]], float]


def _positive_levels(base: float, cap: float, factors: Sequence[float]) -> Tuple[float, ...]:
    floor = np.finfo(float).eps
    values = {
        min(max(abs(float(base)) * float(factor), floor), abs(float(cap)))
        for factor in factors
    }
    return tuple(sorted(values))


def _spectroscopy_axis(candidate: Candidate, points: int = 121) -> np.ndarray:
    half_width = max(3.0 * abs(float(candidate.fwhm_mhz)), 1.0)
    return np.linspace(
        float(candidate.center_mhz) - half_width,
        float(candidate.center_mhz) + half_width,
        int(points),
    )


def _drive_power_overrides(
    candidate: Candidate,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    base = abs(float(context.get("q_gain", 0.1)))
    cap = abs(float(context.get("q_gain_max", max(base * 2.0, 0.2))))
    levels = _positive_levels(base, cap, (0.4, 0.7, 1.0, 1.6))
    return {
        "q_freq": _spectroscopy_axis(candidate),
        "q_gain": levels,
    }


def _flux_nudge_overrides(
    candidate: Candidate,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    center = float(context.get("z_gain", 0.0))
    period = abs(float(context.get("resonator_flux_period_z", 0.3)))
    # A nudge is deliberately much smaller than a flux map.  A percent-scale
    # fraction of a period can move a transmon hundreds of MHz and leave the
    # narrow tracking window entirely.
    span = abs(float(context.get("flux_nudge_span_z", period / 1500.0)))
    span = max(span, np.finfo(float).eps)
    return {
        "q_freq": _spectroscopy_axis(candidate),
        "q_gain": float(context.get("q_gain", 0.1)),
        "z_gain": tuple(center + span * offset for offset in (-1.0, -0.5, 0.0, 0.5, 1.0)),
    }


def _rabi_ping_overrides(
    candidate: Candidate,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    base = abs(float(context.get("q_gain", 0.1)))
    cap = abs(float(context.get("q_gain_max", max(base * 2.0, 0.2))))
    levels = _positive_levels(base, cap, (0.75, 1.5))
    if len(levels) < 2:
        levels = _positive_levels(base, cap, (0.5, 1.0))
    duration_max = max(float(context.get("rabi_probe_length_us", 1.6)), 0.1)
    return {
        "q_freq": float(candidate.center_mhz),
        "q_gain": levels,
        "q_length": np.linspace(0.02, duration_max, 96),
    }


def _dispersive_overrides(
    candidate: Candidate,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    readout_center = float(context.get("r_freq", 6884.0))
    linewidth = abs(float(context.get("resonator_linewidth_mhz", 0.5)))
    half_span = max(8.0 * linewidth, 4.0)
    gain = min(
        abs(float(context.get("dispersive_q_gain", context.get("q_gain", 0.3)))),
        abs(float(context.get("q_gain_max", 0.8))),
    )
    return {
        "r_freq": np.linspace(readout_center - half_span, readout_center + half_span, 241),
        "q_freq": float(candidate.center_mhz),
        "q_gain": gain,
    }


def expand_probe_runs(
    probe: Probe,
    candidate: Candidate,
    context: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    """Expand a multi-level probe plan into ordinary scalar experiment runs."""
    overrides = dict(probe.build_overrides(candidate, context))
    series_key = {
        "drive_power_ladder": "q_gain",
        "flux_nudge": "z_gain",
        "rabi_ping": "q_gain",
        "dispersive_response": "state",
    }.get(probe.probe_id)
    if series_key is None or series_key not in overrides:
        return (overrides,)
    values = tuple(overrides.pop(series_key))
    runs = []
    for value in values:
        run = dict(overrides)
        run[series_key] = value
        runs.append(run)
    return tuple(runs)


def _variables(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    metadata = yaml.safe_load(source.with_suffix(".yml").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("native probe metadata must be a mapping")
    parameters = metadata.get("parameters", {})
    variables = parameters.get("var", {}) if isinstance(parameters, Mapping) else {}
    if not isinstance(variables, Mapping):
        raise ValueError("native probe variables must be a mapping")
    return variables


def _log_slope(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = np.log(x[mask])
    y = np.log(y[mask])
    if x.size < 2 or np.ptp(x) <= np.finfo(float).eps:
        raise ValueError("probe response requires at least two distinct positive levels")
    coefficients = np.polyfit(x, y, 1)
    prediction = np.polyval(coefficients, x)
    if x.size > 2:
        residual_scale = float(np.sqrt(np.sum((y - prediction) ** 2) / (x.size - 2)))
        uncertainty = residual_scale / max(
            float(np.sqrt(np.sum((x - np.mean(x)) ** 2))),
            np.finfo(float).eps,
        )
    else:
        uncertainty = 0.0
    return float(coefficients[0]), float(uncertainty)


def _linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size < 2 or np.ptp(x) <= np.finfo(float).eps:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _extract_drive_power(paths: Sequence[Path]) -> Mapping[str, float]:
    gains = []
    centers = []
    widths = []
    contrasts = []
    for path in paths:
        fit = fit_spectroscopy_features(Path(path), kind="qubit", signal="amplitude")
        variables = _variables(Path(path))
        gains.append(abs(float(variables["q_gain"])))
        centers.append(float(fit.parameters["center_mhz"]))
        widths.append(abs(float(fit.parameters["fwhm_mhz"])))
        contrasts.append(abs(float(fit.parameters["amplitude"])))
    exponent, uncertainty = _log_slope(gains, contrasts)
    return {
        "contrast_power_exponent": exponent,
        "power_exponent_uncertainty": uncertainty,
        "linewidth_vs_power_slope": _linear_slope(gains, widths),
        "center_shift_vs_power_slope": _linear_slope(gains, centers),
        "gain_levels": float(len(gains)),
    }


def _extract_flux_nudge(paths: Sequence[Path]) -> Mapping[str, float]:
    z_values = []
    centers = []
    uncertainties = []
    for path in paths:
        fit = fit_spectroscopy_features(Path(path), kind="qubit", signal="amplitude")
        variables = _variables(Path(path))
        z_values.append(float(variables["z_gain"]))
        centers.append(float(fit.parameters["center_mhz"]))
        uncertainties.append(abs(float(fit.parameters["center_uncertainty_mhz"])))
    z = np.asarray(z_values, dtype=float)
    center = np.asarray(centers, dtype=float)
    degree = 2 if z.size >= 3 else 1
    coefficients = np.polyfit(z, center, degree)
    if degree == 2:
        slope = float(2.0 * coefficients[0] * np.mean(z) + coefficients[1])
        curvature = float(2.0 * coefficients[0])
    else:
        slope = float(coefficients[0])
        curvature = 0.0
    prediction = np.polyval(coefficients, z)
    residual = center - prediction
    return {
        "flux_slope_mhz_per_z": slope,
        "flux_curvature_mhz_per_z2": curvature,
        "flux_fit_rmse_mhz": float(np.sqrt(np.mean(residual ** 2))),
        "flux_center_uncertainty_mhz": float(np.mean(uncertainties)),
        "flux_levels": float(z.size),
    }


def _extract_rabi_ping(paths: Sequence[Path]) -> Mapping[str, float]:
    gains = []
    rates = []
    contrasts = []
    rate_uncertainties = []
    for path in paths:
        fit = fit_rabi(Path(path), variable="q_length")
        variables = _variables(Path(path))
        gains.append(abs(float(variables["q_gain"])))
        rates.append(abs(float(fit.parameters["frequency"])))
        rate_uncertainties.append(abs(float(fit.parameters["frequency_uncertainty"])))
        contrasts.append(2.0 * abs(float(fit.parameters["amplitude"])))
    exponent, uncertainty = _log_slope(gains, rates)
    return {
        "rabi_gain_exponent": exponent,
        "rabi_exponent_uncertainty": uncertainty,
        "rabi_contrast": float(np.mean(contrasts)),
        "rabi_rate_uncertainty_mhz": float(np.mean(rate_uncertainties)),
        "gain_levels": float(len(gains)),
    }


def _parabolic_minimum(x: np.ndarray, y: np.ndarray) -> float:
    index = int(np.argmin(y))
    if index == 0 or index == x.size - 1:
        return float(x[index])
    local_x = x[index - 1 : index + 2]
    local_y = y[index - 1 : index + 2]
    quadratic, linear, _constant = np.polyfit(local_x - x[index], local_y, 2)
    if quadratic <= np.finfo(float).eps:
        return float(x[index])
    correction = float(
        np.clip(-linear / (2.0 * quadratic), local_x[0] - x[index], local_x[-1] - x[index])
    )
    return float(x[index] + correction)


def _extract_dispersive(paths: Sequence[Path]) -> Mapping[str, float]:
    if len(paths) != 1:
        raise ValueError("dispersive response requires exactly one native trace")
    source = Path(paths[0]).expanduser().resolve()
    optimization = fit_readout_optimization(source)
    metadata = yaml.safe_load(source.with_suffix(".yml").read_text(encoding="utf-8"))
    matrix = np.atleast_2d(np.loadtxt(source, delimiter=","))
    columns = {}
    for index, entry in enumerate(metadata.get("dependent", ())):
        if isinstance(entry, (list, tuple)) and entry:
            columns[str(entry[0]).strip().lower().replace(" ", "_")] = index + 1
    def state_column(primary: str, alternate: str) -> int:
        return columns[primary] if primary in columns else columns[alternate]

    ground = (
        matrix[:, state_column("i_0", "i_ground")]
        + 1j * matrix[:, state_column("q_0", "q_ground")]
    )
    excited = (
        matrix[:, state_column("i_1", "i_excited")]
        + 1j * matrix[:, state_column("q_1", "q_excited")]
    )
    frequency = np.asarray(matrix[:, 0], dtype=float)
    ground_center = _parabolic_minimum(frequency, np.abs(ground))
    excited_center = _parabolic_minimum(frequency, np.abs(excited))
    return {
        "dispersive_shift_mhz": float(excited_center - ground_center),
        "ground_resonator_mhz": ground_center,
        "excited_resonator_mhz": excited_center,
        "dispersive_snr": float(optimization.snr_at_optimum),
    }


def _estimate_runs(count: int) -> Callable[[Mapping[str, Any]], float]:
    def estimate(context: Mapping[str, Any]) -> float:
        return float(count) * float(context.get("estimated_probe_run_seconds", 20.0))

    return estimate


_PROBES = (
    Probe(
        "drive_power_ladder",
        "qubit_spectroscopy",
        "qubit_fine",
        _drive_power_overrides,
        _extract_drive_power,
        _estimate_runs(4),
    ),
    Probe(
        "flux_nudge",
        "qubit_spectroscopy",
        "qubit_fine",
        _flux_nudge_overrides,
        _extract_flux_nudge,
        _estimate_runs(5),
    ),
    Probe(
        "dispersive_response",
        "dispersive_spectroscopy",
        "dispersive",
        _dispersive_overrides,
        _extract_dispersive,
        _estimate_runs(1),
    ),
    Probe(
        "rabi_ping",
        "rabi",
        "rabi_length",
        _rabi_ping_overrides,
        _extract_rabi_ping,
        _estimate_runs(2),
    ),
)


_PROBE_BY_ID: Dict[str, Probe] = {probe.probe_id: probe for probe in _PROBES}


def probe_ids() -> Tuple[str, ...]:
    return tuple(probe.probe_id for probe in _PROBES)


def get_probe(probe_id: str) -> Probe:
    try:
        return _PROBE_BY_ID[str(probe_id)]
    except KeyError as exc:
        raise ValueError("unknown hypothesis probe: " + str(probe_id)) from exc
