"""Resonator-sensed long-timescale flux-step transients.

The readout resonator is used as a slow flux sensor. A static ``f_r(z)``
calibration fixes the transduction; the flux line is then stepped and the
complex resonator response is probed at a set of observation times while the
line is still held at the stepped level. Inverting the static calibration turns
that response into an effective flux trajectory ``z_hat(t)``, which feeds the
same normalized exponential fit and matched-z IIR inverse that the pulsed
spectroscopy campaign in :mod:`quickexp_v3.flux_step_campaign` uses.

Why this is much cheaper than 17a: the pulsed method spends a full qubit
spectrum (typically 201 frequencies) per observation time. Here each
observation time costs one short mini-spectrum of the resonator, so the same
70-point time grid runs in minutes rather than hours.

What it cannot do: the resonator has finite bandwidth. The measured transient is
the flux-line response filtered by the cavity, with field time constant
``tau_r = 2/kappa = 1/(pi*FWHM)``. Observation times shorter than a few
``tau_r`` are not flux-line information and are masked out of the fit rather
than being fitted and quietly believed. This method targets bias-tee droop and
other multi-microsecond tails; short-time distortion belongs to the cryoscope.

Three acquisitions make up one campaign, all through the same authored program
so that no gain or phase convention is transferred between Quick classes:

``reference``
    Wide resonator sweep with the line held at the baseline. Fitted with the
    eight-parameter complex notch model to fix the line shape, the linewidth,
    and the complex baseline/coupling terms.
``settled``
    Wide sweep at the longest observation time with the step applied, giving
    the settled ``f_r`` and hence the measured excursion.
``transient``
    Two-dimensional ``probe_time`` by ``r_freq`` map: a narrow mini-spectrum at
    every observation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import AnalysisError, ConfigError
from .flux_compensation import (
    IIRInverseDesign,
    StepResponseFit,
    design_iir_inverse,
    fit_step_response,
)
from .ide import load_repository, run_experiment
from .naming import number_tag
from .notch_fit import NOTCH_PARAMETER_NAMES, fit_complex_notch


EXPERIMENT_NAME = "resonator_flux_transient"
PRESET_NAME = "resonator_flux_transient"


def _default_probe_times_us() -> np.ndarray:
    # The 17a grid, so a resonator-sensed campaign is directly comparable to
    # the pulsed one row for row. Points below the cavity/readout limit are
    # acquired but masked from the fit; see ``analysis_mask``.
    return np.geomspace(0.025, 100.0, 70)


# ----------------------------------------------------------------------------
# Static flux calibration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResonatorFluxCalibration:
    """Empirical cosine calibration ``f_r(z)`` with its residual scale.

    ``rmse_mhz`` is the scatter of the *static extraction*, not the noise floor
    of the transient. It bounds how well the flux axis is scaled, and therefore
    the fitted amplitudes; because the step response is normalized before
    fitting, it does not directly limit the recovered time constants.
    """

    center_frequency_mhz: float
    amplitude_mhz: float
    period_z: float
    peak_bias_z: float
    rmse_mhz: float = 0.0
    domain_z: Tuple[float, float] = (-0.4, 0.4)

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude_mhz) or self.amplitude_mhz <= 0:
            raise ConfigError("resonator flux amplitude must be positive")
        if not np.isfinite(self.period_z) or self.period_z <= 0:
            raise ConfigError("resonator flux period must be positive")
        if not np.isfinite(self.center_frequency_mhz):
            raise ConfigError("resonator centre frequency must be finite")
        if not np.isfinite(self.peak_bias_z):
            raise ConfigError("resonator peak bias must be finite")
        lower, upper = (float(value) for value in self.domain_z)
        if not lower < upper:
            raise ConfigError("resonator calibration domain must be increasing")

    def frequency(self, z_gain: Any) -> Any:
        values = np.asarray(z_gain, dtype=float)
        angle = 2.0 * np.pi * (values - self.peak_bias_z) / self.period_z
        result = self.center_frequency_mhz + self.amplitude_mhz * np.cos(angle)
        return float(result) if result.ndim == 0 else result

    def slope_mhz_per_z(self, z_gain: Any) -> Any:
        values = np.asarray(z_gain, dtype=float)
        angle = 2.0 * np.pi * (values - self.peak_bias_z) / self.period_z
        scale = self.amplitude_mhz * 2.0 * np.pi / self.period_z
        result = -scale * np.sin(angle)
        return float(result) if result.ndim == 0 else result

    @property
    def maximum_slope_mhz_per_z(self) -> float:
        return float(self.amplitude_mhz * 2.0 * np.pi / self.period_z)

    def maximum_slope_biases(self) -> np.ndarray:
        """Every ``z`` of steepest transduction inside the accepted domain."""
        lower, upper = self.domain_z
        quarter = self.peak_bias_z + 0.25 * self.period_z
        half = 0.5 * self.period_z
        first = int(np.floor((lower - quarter) / half))
        last = int(np.ceil((upper - quarter) / half))
        candidates = quarter + half * np.arange(first, last + 1)
        return candidates[(candidates >= lower) & (candidates <= upper)]

    def branch(self, baseline_z: float, target_z: float) -> Tuple[float, float]:
        """Monotonic half-period containing both step endpoints."""
        offsets = np.asarray(
            [float(baseline_z) - self.peak_bias_z, float(target_z) - self.peak_bias_z],
            dtype=float,
        )
        if np.any(~np.isfinite(offsets)):
            raise ConfigError("flux step endpoints must be finite")
        if np.any(np.abs(offsets) > 0.5 * self.period_z):
            raise ConfigError(
                "flux step leaves the half-period containing the baseline; "
                "f_r(z) is not invertible across it"
            )
        signs = np.sign(offsets)
        if np.any(signs == 0) or signs[0] != signs[1]:
            raise ConfigError(
                "flux step touches or crosses the resonator flux extremum at "
                f"z={self.peak_bias_z:+.6g}; f_r(z) is not monotonic there. "
                "Move the operating point onto one side of the extremum."
            )
        side = float(signs[0])
        endpoints = (
            self.peak_bias_z,
            self.peak_bias_z + side * 0.5 * self.period_z,
        )
        lower = max(min(endpoints), self.domain_z[0])
        upper = min(max(endpoints), self.domain_z[1])
        if lower >= upper:
            raise ConfigError(
                "accepted calibration domain does not cover a monotonic branch"
            )
        if min(baseline_z, target_z) < lower or max(baseline_z, target_z) > upper:
            raise ConfigError(
                f"flux step [{min(baseline_z, target_z):+.6g}, "
                f"{max(baseline_z, target_z):+.6g}] leaves the accepted branch "
                f"[{lower:+.6g}, {upper:+.6g}]"
            )
        return lower, upper

    def flux_from_frequency(self, frequency_mhz: Any, *, side: float) -> np.ndarray:
        """Invert ``f_r`` onto one signed half-period around the extremum."""
        values = np.asarray(frequency_mhz, dtype=float)
        cosine = (values - self.center_frequency_mhz) / self.amplitude_mhz
        clipped = np.clip(cosine, -1.0, 1.0)
        angle = np.arccos(clipped)
        return self.peak_bias_z + float(np.sign(side)) * angle * self.period_z / (
            2.0 * np.pi
        )

    def as_dict(self) -> dict:
        return {
            "model": "cosine",
            "center_frequency_mhz": float(self.center_frequency_mhz),
            "amplitude_mhz": float(self.amplitude_mhz),
            "period_z": float(self.period_z),
            "peak_bias_z": float(self.peak_bias_z),
            "rmse_mhz": float(self.rmse_mhz),
            "domain_z": [float(value) for value in self.domain_z],
        }


def calibration_from_accepted_record(
    record: Mapping[str, Any],
) -> ResonatorFluxCalibration:
    """Build a calibration from an accepted ``lookups.resonator_vs_flux`` record."""
    try:
        parameters = record["value"]["parameters"]
        domain = record["valid_domain"]["z_gain"]
        return ResonatorFluxCalibration(
            center_frequency_mhz=float(parameters["center_frequency"]),
            amplitude_mhz=float(parameters["amplitude"]),
            period_z=float(parameters["period"]),
            peak_bias_z=float(parameters["peak_bias"]),
            rmse_mhz=float(
                (record.get("uncertainty") or {}).get("rmse_mhz", 0.0) or 0.0
            ),
            domain_z=(float(domain[0]), float(domain[1])),
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ConfigError(
            "accepted resonator_vs_flux record is missing cosine parameters, "
            "uncertainty or valid_domain"
        ) from error


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResonatorTransientParameters:
    """Everything one resonator-sensed flux-step campaign needs."""

    live_hardware: bool = False

    # Flux command. Both are absolute levels on a DC-coupled fast line, matching
    # the baseline_on_fast_line mode of the pulsed campaign.
    baseline_z: float = -0.080
    commanded_step_z: float = 0.015

    # Static calibration. use_accepted_resonator_flux_fit reads
    # lookups.resonator_vs_flux; otherwise the explicit cosine below is used.
    use_accepted_resonator_flux_fit: bool = False
    calibration: Optional[ResonatorFluxCalibration] = None

    # Observation times and the mini-spectrum acquired at each of them.
    probe_times_us: Any = field(default_factory=_default_probe_times_us)
    transient_probe_points: int = 5
    transient_span_linewidths: float = 2.0

    # Wide sweeps used for the reference and settled notch fits.
    reference_probe_points: int = 41
    reference_span_mhz: float = 5.0
    reference_probe_time_us: float = 1.0
    reference_hard_avg: int = 3000

    # Readout and repetition. r_relax doubles as the baseline settle interval:
    # the line rests at baseline_z for its whole duration, so it must exceed
    # several times the longest flux time constant being measured.
    readout_power_db: float = -35.0
    readout_length_us: float = 2.0
    readout_relax_us: float = 300.0
    hard_avg: int = 2048
    soft_avg: int = 1
    z_set_length_us: float = 0.008

    # Analysis. fit_dc_gain=None fits the settled level, which is what a
    # DC-coupled line needs; use 0.0 only when a bias tee makes the response
    # genuinely high-pass, as in the reference paper.
    cavity_lifetimes_to_mask: float = 5.0
    fit_dc_gain: Optional[float] = None
    model_orders: Iterable[int] = field(default_factory=lambda: range(1, 7))
    # None bounds the time constants to the observation window actually
    # measured. Outside it they are not identifiable: a tau shorter than the
    # first kept sample has already decayed, and one longer than the last is
    # indistinguishable from a constant. fit_step_response otherwise allows up
    # to 30x the longest time, which would put unconstrained poles into the
    # designed inverse.
    tau_bounds_us: Optional[Sequence[float]] = None
    filter_sample_interval_ns: float = 1.669
    leak_tau_us: Optional[float] = None
    frequency_search_points: int = 4001

    # Bookkeeping.
    show_plot: bool = True
    seed: int = 23
    title_prefix: str = "ResFluxTransient"
    schedule_path: Any = Path(
        "analysis_cache/flux_compensation/resonator_transient.json"
    )

    @property
    def step_level_z(self) -> float:
        return float(self.baseline_z) + float(self.commanded_step_z)


def resolve_calibration(
    project_root: Path,
    parameters: ResonatorTransientParameters,
) -> ResonatorFluxCalibration:
    if parameters.use_accepted_resonator_flux_fit:
        repository = load_repository(Path(project_root))
        record = repository.calibration.get("lookups", {}).get("resonator_vs_flux")
        if not isinstance(record, Mapping):
            raise ConfigError(
                "no accepted lookups.resonator_vs_flux record is available; "
                "supply an explicit calibration instead"
            )
        return calibration_from_accepted_record(record)
    if parameters.calibration is None:
        raise ConfigError(
            "set calibration=ResonatorFluxCalibration(...) or "
            "use_accepted_resonator_flux_fit=True"
        )
    return parameters.calibration


# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TransientPreflight:
    baseline_z: float
    step_level_z: float
    branch_z: Tuple[float, float]
    baseline_frequency_mhz: float
    step_frequency_mhz: float
    excursion_mhz: float
    slope_mhz_per_z: float
    maximum_slope_mhz_per_z: float
    curvature_percent: float
    calibration_rmse_mhz: float
    probe_centre_mhz: float

    @property
    def slope_fraction_of_maximum(self) -> float:
        if self.maximum_slope_mhz_per_z == 0:
            return 0.0
        return abs(self.slope_mhz_per_z) / self.maximum_slope_mhz_per_z

    def report(self) -> str:
        lines = [
            f"  baseline      z={self.baseline_z:+.6g}  "
            f"f_r={self.baseline_frequency_mhz:.4f} MHz",
            f"  stepped       z={self.step_level_z:+.6g}  "
            f"f_r={self.step_frequency_mhz:.4f} MHz",
            f"  excursion     {self.excursion_mhz * 1e3:+.1f} kHz",
            f"  local slope   {self.slope_mhz_per_z:+.3f} MHz/z "
            f"({self.slope_fraction_of_maximum * 100:.0f}% of the "
            f"{self.maximum_slope_mhz_per_z:.3f} MHz/z maximum)",
            f"  monotonic on  z in [{self.branch_z[0]:+.6g}, {self.branch_z[1]:+.6g}]",
            f"  curvature     {self.curvature_percent:.1f}% deviation from linear "
            "across the step",
            f"  probe centre  {self.probe_centre_mhz:.4f} MHz "
            "(resonance at the step midpoint)",
        ]
        if self.calibration_rmse_mhz > 0:
            ratio = abs(self.excursion_mhz) / self.calibration_rmse_mhz
            lines.append(
                f"  calibration   RMSE {self.calibration_rmse_mhz * 1e3:.0f} kHz "
                f"= {1.0 / ratio if ratio else float('inf'):.2f}x the excursion; "
                "this scales the fitted amplitudes, not the time constants"
            )
        return "\n".join(lines)


def preflight(
    parameters: ResonatorTransientParameters,
    calibration: ResonatorFluxCalibration,
) -> TransientPreflight:
    """Check invertibility and report transduction before any acquisition."""
    baseline = float(parameters.baseline_z)
    step_level = parameters.step_level_z
    if float(parameters.commanded_step_z) == 0:
        raise ConfigError("commanded_step_z must be non-zero")
    for label, value in (("baseline_z", baseline), ("step level", step_level)):
        if not calibration.domain_z[0] <= value <= calibration.domain_z[1]:
            raise ConfigError(
                f"{label} z={value:+.6g} is outside the calibration domain "
                f"[{calibration.domain_z[0]:+.6g}, {calibration.domain_z[1]:+.6g}]"
            )
    branch = calibration.branch(baseline, step_level)

    baseline_frequency = float(calibration.frequency(baseline))
    step_frequency = float(calibration.frequency(step_level))
    excursion = step_frequency - baseline_frequency

    # Local curvature across the step: how far the midpoint departs from the
    # chord. The normalized fit assumes z_hat is a faithful reading of the
    # trajectory shape, which needs f_r(z) to be near-linear over the interval.
    midpoint = 0.5 * (baseline + step_level)
    chord = 0.5 * (baseline_frequency + step_frequency)
    curvature = float(calibration.frequency(midpoint)) - chord
    curvature_percent = (
        abs(curvature) / abs(excursion) * 100.0 if excursion else float("inf")
    )

    return TransientPreflight(
        baseline_z=baseline,
        step_level_z=step_level,
        branch_z=branch,
        baseline_frequency_mhz=baseline_frequency,
        step_frequency_mhz=step_frequency,
        excursion_mhz=excursion,
        slope_mhz_per_z=float(calibration.slope_mhz_per_z(midpoint)),
        maximum_slope_mhz_per_z=calibration.maximum_slope_mhz_per_z,
        curvature_percent=curvature_percent,
        calibration_rmse_mhz=float(calibration.rmse_mhz),
        probe_centre_mhz=float(calibration.frequency(midpoint)),
    )


def cavity_time_constant_us(linewidth_mhz: float) -> float:
    """``tau_r = 2/kappa = 1/(pi * FWHM)`` in microseconds for FWHM in MHz."""
    width = float(linewidth_mhz)
    if not np.isfinite(width) or width <= 0:
        raise AnalysisError("resonator linewidth must be positive and finite")
    return 1.0 / (np.pi * width)


def analysis_mask(
    probe_times_us: Any,
    *,
    linewidth_mhz: float,
    readout_length_us: float,
    cavity_lifetimes: float,
) -> Tuple[np.ndarray, float]:
    """Times the cavity and the readout window make interpretable."""
    times = np.asarray(probe_times_us, dtype=float)
    tau_r = cavity_time_constant_us(linewidth_mhz)
    threshold = max(
        float(cavity_lifetimes) * tau_r,
        float(readout_length_us),
    )
    return times >= threshold, threshold


# ----------------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------------


def _common_overrides(parameters: ResonatorTransientParameters) -> dict:
    return {
        "z_baseline_gain": float(parameters.baseline_z),
        "z_set_length": float(parameters.z_set_length_us),
        "r_power": float(parameters.readout_power_db),
        "r_length": float(parameters.readout_length_us),
        "r_relax": float(parameters.readout_relax_us),
        "hard_avg": int(parameters.hard_avg),
        "soft_avg": int(parameters.soft_avg),
    }


def _native_csv(result) -> Optional[Path]:
    for path in result.data.metadata.get("native_files", ()):
        candidate = Path(path)
        if candidate.suffix.lower() == ".csv":
            return candidate.resolve()
    return None


@dataclass(frozen=True)
class TransientCampaign:
    """One acquired campaign and the paths it produced."""

    calibration: ResonatorFluxCalibration
    preflight: TransientPreflight
    reference_csv: Optional[Path]
    settled_csv: Optional[Path]
    transient_csv: Optional[Path]
    reference_result: Any
    settled_result: Any
    transient_result: Any
    reference_fit: Any = None
    settled_fit: Any = None


def run_resonator_flux_transient(
    project_root: Path,
    parameters: ResonatorTransientParameters,
) -> TransientCampaign:
    """Acquire the reference, settled and transient runs of one campaign."""
    project_root = Path(project_root).resolve()
    calibration = resolve_calibration(project_root, parameters)
    checks = preflight(parameters, calibration)

    probe_times = np.asarray(parameters.probe_times_us, dtype=float)
    if probe_times.ndim != 1 or probe_times.size < 2:
        raise ConfigError("probe_times_us must be a 1-D array of at least two times")
    if np.any(probe_times <= 0) or np.any(~np.isfinite(probe_times)):
        raise ConfigError("probe_times_us must be positive and finite")
    if int(parameters.transient_probe_points) < 2:
        raise ConfigError(
            "transient_probe_points must be at least 2; the r_freq axis needs "
            "more than one point"
        )

    print("Resonator-sensed flux-step transient")
    print(checks.report())
    reference_frequencies = checks.probe_centre_mhz + np.linspace(
        -0.5 * float(parameters.reference_span_mhz),
        +0.5 * float(parameters.reference_span_mhz),
        int(parameters.reference_probe_points),
    )
    total_points = (
        probe_times.size * int(parameters.transient_probe_points)
        + 2 * int(parameters.reference_probe_points)
    )
    averages = int(parameters.hard_avg) * int(parameters.soft_avg)
    ideal_minutes = (
        total_points * averages * float(parameters.readout_relax_us) / 6.0e7
    )
    print(
        f"  campaign      {probe_times.size} times x "
        f"{int(parameters.transient_probe_points)} probes + 2 x "
        f"{int(parameters.reference_probe_points)} reference points, "
        f"{averages} averages, ideal lower bound {ideal_minutes:.1f} min"
    )

    common = _common_overrides(parameters)
    campaign_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def held_sweep(title: str, z_gain: float):
        """One static wide sweep with the line held at ``z_gain``.

        These are ordinary resonator spectroscopy measurements, so they use the
        static held-Z experiment rather than the stepping program. Any gain or
        phase difference between that path and the transient map is absorbed by
        the single complex gain fitted in :func:`invert_transient`.
        """
        return run_experiment(
            project_root,
            experiment="resonator_spectroscopy",
            preset="resonator_fine",
            title=title,
            live_hardware=parameters.live_hardware,
            fixed_z_gain=float(z_gain),
            overrides={
                "r_freq": reference_frequencies,
                "r_power": float(parameters.readout_power_db),
                "r_length": float(parameters.readout_length_us),
                "hard_avg": int(parameters.reference_hard_avg),
                "soft_avg": int(parameters.soft_avg),
            },
            analyze=False,
            show_plot=False,
            seed=int(parameters.seed),
        )

    # 1. Reference sweep at the baseline fixes the line shape and the linewidth.
    print("Reference sweep at the baseline level ...")
    reference_result = held_sweep(
        f"{parameters.title_prefix}_{campaign_id}_reference",
        float(parameters.baseline_z),
    )
    reference_csv = _native_csv(reference_result)
    reference_fit = None
    linewidth_mhz = None
    if reference_csv is not None:
        reference_fit = fit_complex_notch(reference_csv)
        linewidth_mhz = float(reference_fit.parameters["fwhm_mhz"])
        tau_r = cavity_time_constant_us(linewidth_mhz)
        print(
            f"  linewidth {linewidth_mhz * 1e3:.0f} kHz "
            f"(Q_L {reference_fit.parameters['loaded_q']:.0f}) -> "
            f"tau_r = {tau_r * 1e3:.0f} ns; "
            f"excursion is {abs(checks.excursion_mhz) / linewidth_mhz:.2f} linewidths"
        )
        if not reference_fit.passes():
            print(
                "  WARNING: the reference notch fit did not pass its acceptance "
                "gates. The inversion will refuse this campaign."
            )

    # 2. Settled sweep at the stepped level gives the measured excursion.
    print("Settled sweep at the stepped level ...")
    settled_result = held_sweep(
        f"{parameters.title_prefix}_{campaign_id}_settled",
        checks.step_level_z,
    )
    settled_csv = _native_csv(settled_result)
    settled_fit = fit_complex_notch(settled_csv) if settled_csv is not None else None
    if settled_fit is not None and reference_fit is not None:
        measured = settled_fit.center_mhz - reference_fit.center_mhz
        print(
            f"  measured excursion {measured * 1e3:+.1f} kHz against "
            f"{checks.excursion_mhz * 1e3:+.1f} kHz predicted "
            f"({measured / checks.excursion_mhz:.2f}x)"
        )

    # 3. Transient map, with the probe window sized by the measured linewidth.
    print(f"Transient map over {probe_times.size} observation times ...")
    transient_result = run_experiment(
        project_root,
        experiment=EXPERIMENT_NAME,
        preset=PRESET_NAME,
        title=(
            f"{parameters.title_prefix}_{campaign_id}"
            f"_z{number_tag(checks.step_level_z)}"
        ),
        live_hardware=parameters.live_hardware,
        overrides={
            **common,
            "z_step_gain": checks.step_level_z,
            "probe_time": probe_times,
            "r_freq": _transient_frequencies(parameters, checks, linewidth_mhz),
        },
        analyze=False,
        show_plot=False,
        seed=int(parameters.seed),
    )

    return TransientCampaign(
        calibration=calibration,
        preflight=checks,
        reference_csv=reference_csv,
        settled_csv=settled_csv,
        transient_csv=_native_csv(transient_result),
        reference_result=reference_result,
        settled_result=settled_result,
        transient_result=transient_result,
        reference_fit=reference_fit,
        settled_fit=settled_fit,
    )


def _transient_frequencies(
    parameters: ResonatorTransientParameters,
    checks: TransientPreflight,
    linewidth_mhz: Optional[float],
) -> np.ndarray:
    """Mini-spectrum around the step-midpoint resonance.

    Complex S21 is steepest exactly on resonance, so the grid is centred there
    rather than detuned. Before the reference fit is available the span falls
    back to the excursion itself, which keeps both endpoints inside the window.
    """
    points = int(parameters.transient_probe_points)
    if linewidth_mhz is not None and np.isfinite(linewidth_mhz) and linewidth_mhz > 0:
        span = float(parameters.transient_span_linewidths) * float(linewidth_mhz)
    else:
        span = max(4.0 * abs(checks.excursion_mhz), 0.5)
    return checks.probe_centre_mhz + np.linspace(-0.5 * span, +0.5 * span, points)


# ----------------------------------------------------------------------------
# Inversion
# ----------------------------------------------------------------------------


def _notch_parameter_vector(fit) -> np.ndarray:
    """Pack a complex notch fit into ``complex_notch_model`` order.

    The baseline slope, phase and cable delay are all expressed against the
    model's own centre, so element 0 must be ``complex_model_center_mhz`` rather
    than the separately refined ``center_mhz``.
    """
    parameters = fit.parameters
    try:
        values = [float(parameters[name]) for name in NOTCH_PARAMETER_NAMES]
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError(
            "reference notch fit did not report the eight-parameter complex "
            "model; refit with fit_complex_notch"
        ) from error
    model_centre = parameters.get("complex_model_center_mhz")
    if model_centre is not None and np.isfinite(float(model_centre)):
        values[0] = float(model_centre)
    return np.asarray(values, dtype=float)


def _model_s21(
    probe_mhz: np.ndarray,
    resonance_mhz: Any,
    notch: np.ndarray,
) -> np.ndarray:
    """Complex S21 with the resonance moved but the baseline held fixed.

    The amplitude slope and cable delay are properties of the *probe* frequency
    and stay anchored to the reference centre; only the Lorentzian follows
    ``f_r``. Writing it the other way round would let cable delay masquerade as
    flux.
    """
    reference_centre, linewidth, a0, a1, phi0, tau, coupling_re, coupling_im = notch
    probe = np.asarray(probe_mhz, dtype=float)
    resonance = np.asarray(resonance_mhz, dtype=float)
    baseline_delta = probe - reference_centre
    baseline = (a0 + a1 * baseline_delta) * np.exp(
        1j * (phi0 + tau * baseline_delta)
    )
    resonance_delta = probe - resonance[..., None]
    coupling = coupling_re + 1j * coupling_im
    return baseline * (
        1.0 - coupling / (1.0 + 2j * resonance_delta / linewidth)
    )


@dataclass(frozen=True)
class ResonatorTransientTrace:
    """Recovered flux trajectory and everything needed to audit it."""

    probe_times_us: np.ndarray
    effective_times_us: np.ndarray
    resonance_mhz: np.ndarray
    resonance_uncertainty_mhz: np.ndarray
    flux_z: np.ndarray
    normalized: np.ndarray
    normalized_uncertainty: np.ndarray
    mask: np.ndarray
    mask_threshold_us: float
    linewidth_mhz: float
    cavity_tau_us: float
    complex_gain: complex
    clipped: np.ndarray
    residual_rms: np.ndarray

    @property
    def fitted_times_us(self) -> np.ndarray:
        return self.effective_times_us[self.mask]

    @property
    def fitted_normalized(self) -> np.ndarray:
        return self.normalized[self.mask]

    @property
    def fitted_uncertainty(self) -> np.ndarray:
        return self.normalized_uncertainty[self.mask]


def _reshape_map(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the long-format transient table into ``(times, probes, iq)``."""
    if "probe_time" not in data.axes or "r_freq" not in data.axes:
        raise AnalysisError(
            "transient data needs both probe_time and r_freq axes"
        )
    times = np.asarray(data.axes["probe_time"], dtype=float)
    probes = np.asarray(data.axes["r_freq"], dtype=float)
    iq = np.asarray(data.iq)
    unique_times = np.unique(times)
    unique_probes = np.unique(probes)
    if unique_times.size * unique_probes.size != iq.size:
        raise AnalysisError(
            "transient map is not a complete probe_time by r_freq grid"
        )
    grid = np.full((unique_times.size, unique_probes.size), np.nan + 0j)
    time_index = np.searchsorted(unique_times, times)
    probe_index = np.searchsorted(unique_probes, probes)
    grid[time_index, probe_index] = iq
    if np.any(~np.isfinite(grid)):
        raise AnalysisError("transient map has missing grid points")
    return unique_times, unique_probes, grid


def invert_transient(
    campaign: TransientCampaign,
    parameters: ResonatorTransientParameters,
    *,
    reference_fit=None,
    settled_fit=None,
) -> ResonatorTransientTrace:
    """Turn the measured mini-spectra into a flux trajectory ``z_hat(t)``."""
    if reference_fit is None:
        reference_fit = campaign.reference_fit
    if reference_fit is None:
        if campaign.reference_csv is None:
            raise AnalysisError(
                "campaign has no reference CSV to fit the line shape"
            )
        reference_fit = fit_complex_notch(campaign.reference_csv)
    if not reference_fit.passes():
        raise AnalysisError(
            "reference notch fit failed its acceptance gates; the line shape "
            "cannot be trusted for inversion"
        )
    notch = _notch_parameter_vector(reference_fit)
    linewidth = float(notch[1])

    if settled_fit is None:
        settled_fit = campaign.settled_fit
    if settled_fit is None and campaign.settled_csv is not None:
        settled_fit = fit_complex_notch(campaign.settled_csv)

    times, probes, measured = _reshape_map(campaign.transient_result.data)
    calibration = campaign.calibration
    checks = campaign.preflight
    side = np.sign(checks.baseline_z - calibration.peak_bias_z)

    # Search f_r across the full monotonic branch so the minimum is global and
    # an excursion that leaves the branch is detectable rather than silently
    # clamped to an endpoint.
    branch_frequencies = calibration.frequency(np.asarray(checks.branch_z))
    lower = float(np.min(branch_frequencies))
    upper = float(np.max(branch_frequencies))
    grid = np.linspace(lower, upper, int(parameters.frequency_search_points))
    model_grid = _model_s21(probes, grid, notch)

    # A single complex gain absorbs drift between the reference sweep and the
    # transient map. Estimated from the settled rows, where f_r is known.
    settled_frequency = (
        float(settled_fit.center_mhz)
        if settled_fit is not None
        else float(checks.step_frequency_mhz)
    )
    settled_rows = times >= 0.5 * float(np.max(times))
    if not np.any(settled_rows):
        settled_rows = np.zeros_like(times, dtype=bool)
        settled_rows[-1] = True
    settled_model = _model_s21(probes, np.asarray([settled_frequency]), notch)[0]
    numerator = np.sum(np.conj(settled_model) * measured[settled_rows], axis=None)
    denominator = np.sum(np.abs(settled_model) ** 2) * int(
        np.count_nonzero(settled_rows)
    )
    complex_gain = complex(numerator / denominator) if denominator else 1.0 + 0j

    scaled_model = complex_gain * model_grid
    # |measured - model|^2 summed over probe frequencies, for every grid point.
    residuals = np.sum(
        np.abs(measured[:, None, :] - scaled_model[None, :, :]) ** 2,
        axis=2,
    )
    best = np.argmin(residuals, axis=1)
    resonance = grid[best]

    # Parabolic refinement and a curvature-based standard error.
    step = grid[1] - grid[0]
    interior = (best > 0) & (best < grid.size - 1)
    left = residuals[np.arange(times.size), np.clip(best - 1, 0, None)]
    centre = residuals[np.arange(times.size), best]
    right = residuals[np.arange(times.size), np.clip(best + 1, None, grid.size - 1)]
    curvature = left - 2.0 * centre + right
    shift = np.where(
        interior & (curvature > 0),
        0.5 * (left - right) / np.where(curvature > 0, curvature, 1.0),
        0.0,
    )
    resonance = resonance + shift * step

    degrees_of_freedom = max(2 * probes.size - 1, 1)
    residual_variance = centre / degrees_of_freedom
    residual_rms = np.sqrt(np.maximum(residual_variance, 0.0))
    # chi^2 curvature: sigma^2 = 2 * var / (d2chi2/dfr2)
    second_derivative = np.where(curvature > 0, curvature / step**2, np.nan)
    resonance_uncertainty = np.sqrt(
        np.abs(2.0 * residual_variance / second_derivative)
    )
    resonance_uncertainty = np.where(
        np.isfinite(resonance_uncertainty), resonance_uncertainty, step
    )

    clipped = (best == 0) | (best == grid.size - 1)

    flux = calibration.flux_from_frequency(resonance, side=side)
    flux_change = flux - float(checks.baseline_z)
    peak = flux_change[np.argmax(np.abs(flux_change))]
    if not np.isfinite(peak) or abs(peak) <= np.finfo(float).eps:
        raise AnalysisError("recovered flux excursion is zero or non-finite")
    normalized = flux_change / peak

    # Propagate the frequency error through the same inverse.
    upper_flux = calibration.flux_from_frequency(
        resonance + resonance_uncertainty, side=side
    )
    lower_flux = calibration.flux_from_frequency(
        resonance - resonance_uncertainty, side=side
    )
    normalized_uncertainty = np.abs(upper_flux - lower_flux) / (2.0 * abs(peak))

    mask, threshold = analysis_mask(
        times,
        linewidth_mhz=linewidth,
        readout_length_us=float(parameters.readout_length_us),
        cavity_lifetimes=float(parameters.cavity_lifetimes_to_mask),
    )
    if not np.any(mask):
        raise AnalysisError(
            f"every observation time falls below the {threshold:.3g} us "
            "cavity/readout limit; lengthen probe_times_us or shorten the readout"
        )

    return ResonatorTransientTrace(
        probe_times_us=times,
        # Centroid of the boxcar the readout integrates over.
        effective_times_us=times + 0.5 * float(parameters.readout_length_us),
        resonance_mhz=resonance,
        resonance_uncertainty_mhz=resonance_uncertainty,
        flux_z=flux,
        normalized=normalized,
        normalized_uncertainty=normalized_uncertainty,
        mask=mask,
        mask_threshold_us=threshold,
        linewidth_mhz=linewidth,
        cavity_tau_us=cavity_time_constant_us(linewidth),
        complex_gain=complex_gain,
        clipped=clipped,
        residual_rms=residual_rms,
    )


# ----------------------------------------------------------------------------
# Tail fit and inverse filter
# ----------------------------------------------------------------------------


def fit_transient_tail(
    trace: ResonatorTransientTrace,
    parameters: ResonatorTransientParameters,
) -> Tuple[StepResponseFit, IIRInverseDesign]:
    """Fit the masked tail and design the matched-z inverse."""
    if int(np.count_nonzero(trace.mask)) < 4:
        raise AnalysisError(
            "fewer than four observation times survive the cavity/readout mask; "
            "extend probe_times_us to longer delays"
        )
    times = trace.fitted_times_us
    bounds = parameters.tau_bounds_us
    if bounds is None:
        # Only time constants inside the measured window are identifiable.
        bounds = (float(np.min(times)), float(np.max(times)))
    fit = fit_step_response(
        times,
        trace.fitted_normalized,
        uncertainty=trace.fitted_uncertainty,
        model_orders=parameters.model_orders,
        dc_gain=parameters.fit_dc_gain,
        tau_bounds_us=bounds,
    )
    inverse = design_iir_inverse(
        fit,
        sample_interval_ns=float(parameters.filter_sample_interval_ns),
        leak_tau_us=parameters.leak_tau_us,
    )
    return fit, inverse
