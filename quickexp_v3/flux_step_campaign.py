"""Parameter-driven long-timescale flux-step spectroscopy campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from .backend import SyntheticBackend
from .errors import ConfigError
from .flux_compensation import (
    PAPER_DOI,
    flux_command_from_phi0_fractions,
    write_step_campaign_manifest,
)
from .flux_lookup import frequency_from_record
from .ide import (
    load_repository,
    plot_data,
    resolve_readout_frequency,
)
from .lab import connect_quick
from .naming import number_tag
from .runtime import ExperimentRunner


def _paper_probe_times() -> np.ndarray:
    return np.geomspace(0.010, 100.0, 70)


def _paper_frequency_offsets() -> np.ndarray:
    return np.linspace(-100.0, 100.0, 201)


def _default_prediction_alphas() -> np.ndarray:
    return np.asarray([0.10, 0.22, 0.68])


def _default_prediction_taus() -> np.ndarray:
    return np.asarray([0.055, 1.3, 20.0])


@dataclass(frozen=True)
class LongTimescaleFluxParameters:
    """Everything needed to run one adaptive long-timescale campaign.

    ``baseline_z`` and ``commanded_step_z`` are optional final commands in the
    installation's local Z units. Set both to bypass Phi0 conversion. When
    they are omitted, an accepted ``qubit_vs_flux`` fit converts the requested
    Phi0 fractions. ``step_scale`` applies only to the Phi0 form.

    ``q_frequency_centers_mhz`` may be one scalar center or one center per
    probe time. When omitted, centers are predicted from the accepted flux
    lookup and the exponential seed model. The prediction only places each
    spectroscopy window; it is not used by the subsequent fit.
    """

    live_hardware: bool = False
    external_baseline_confirmed: bool = False

    baseline_phi0: float = -0.127
    step_phi0: float = -0.217
    step_scale: float = 0.20
    baseline_z: Optional[float] = None
    commanded_step_z: Optional[float] = None

    probe_times_us: Any = field(default_factory=_paper_probe_times)
    q_frequency_offsets_mhz: Any = field(default_factory=_paper_frequency_offsets)
    q_frequency_centers_mhz: Optional[Any] = None
    fallback_q_frequency_mhz: float = 5200.0
    q_nyquist_zone: Optional[int] = None

    prediction_alphas: Any = field(default_factory=_default_prediction_alphas)
    prediction_taus_us: Any = field(default_factory=_default_prediction_taus)

    # Appendix F readout-return model for the first, uncompensated pass.
    first_uncompensated_pass: bool = True
    bias_tee_tau_us: float = 20.0
    post_probe_to_return_us: float = 0.070
    return_to_readout_us: float = 0.110
    z_idle_gain: float = 0.0
    z_set_length_us: float = 0.004

    use_accepted_resonator_flux_fit: bool = True
    readout_frequency_mhz: float = 6884.0
    q_gain: float = 0.15
    q_length_us: float = 0.040
    readout_power_db: float = -35.0
    readout_length_us: float = 2.0
    readout_relax_us: float = 300.0
    hard_avg: int = 2048
    soft_avg: int = 1

    show_plot: bool = True
    seed: int = 17
    title_prefix: str = "FluxStepAdaptive"
    campaign_manifest: Any = Path(
        "analysis_cache/flux_compensation/flux_step_campaign.json"
    )


@dataclass(frozen=True)
class _ResolvedFluxCommand:
    baseline_z: float
    commanded_step_z: float
    source: str


@dataclass(frozen=True)
class _ResolvedCampaign:
    probe_times_us: np.ndarray
    q_frequency_offsets_mhz: np.ndarray
    q_frequency_centers_mhz: np.ndarray
    q_nyquist_zone: int
    flux: _ResolvedFluxCommand


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{label} must be a finite number") from error
    if not np.isfinite(result):
        raise ConfigError(f"{label} must be a finite number")
    return result


def _positive_integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    result = int(number)
    if result < 1 or result != number:
        raise ConfigError(f"{label} must be a positive integer")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    result = int(number)
    if result < 0 or result != number:
        raise ConfigError(f"{label} must be a nonnegative integer")
    return result


def _vector(values: Any, label: str, *, minimum: int) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{label} must contain finite numbers") from error
    if result.ndim != 1 or result.size < minimum or not np.all(np.isfinite(result)):
        raise ConfigError(f"{label} must contain at least {minimum} finite values")
    return result.copy()


def _increasing_vector(
    values: Any,
    label: str,
    *,
    minimum: int,
    positive: bool = False,
) -> np.ndarray:
    result = _vector(values, label, minimum=minimum)
    if np.any(np.diff(result) <= 0):
        raise ConfigError(f"{label} must be strictly increasing")
    if positive and np.any(result <= 0):
        raise ConfigError(f"{label} must contain only positive values")
    return result


def _qubit_flux_record(repository) -> Optional[Mapping[str, Any]]:
    try:
        record = repository.calibration["records"]["lookups"]["qubit_vs_flux"]
    except (KeyError, TypeError):
        return None
    return record if record.get("status", "accepted") == "accepted" else None


def _resolve_flux_command(
    parameters: LongTimescaleFluxParameters,
    record: Optional[Mapping[str, Any]],
) -> _ResolvedFluxCommand:
    explicit = (
        parameters.baseline_z is not None or parameters.commanded_step_z is not None
    )
    if explicit:
        if parameters.baseline_z is None or parameters.commanded_step_z is None:
            raise ConfigError("set both baseline_z and commanded_step_z, or neither")
        baseline_z = _finite(parameters.baseline_z, "baseline_z")
        commanded_step_z = _finite(
            parameters.commanded_step_z,
            "commanded_step_z",
        )
        source = "explicit local Z parameters"
    else:
        baseline_phi0 = _finite(parameters.baseline_phi0, "baseline_phi0")
        step_phi0 = _finite(parameters.step_phi0, "step_phi0")
        step_scale = _finite(parameters.step_scale, "step_scale")
        if not 0.0 < step_scale <= 1.0:
            raise ConfigError("step_scale must lie in the interval (0, 1]")
        scaled_step_phi0 = step_phi0 * step_scale
        if record is not None:
            baseline_z, commanded_step_z = flux_command_from_phi0_fractions(
                record,
                baseline_phi0=baseline_phi0,
                step_phi0=scaled_step_phi0,
            )
            source = "accepted qubit_vs_flux Phi0 conversion"
        elif parameters.live_hardware:
            raise ConfigError(
                "live Phi0 parameters require an accepted qubit_vs_flux "
                "calibration; run 06b and 06e, or set both local Z parameters"
            )
        else:
            # Synthetic runs have no physical Z units. Keeping the requested
            # fractions makes the offline path deterministic and, importantly,
            # applies the same commissioning scale as the live conversion.
            baseline_z = baseline_phi0
            commanded_step_z = scaled_step_phi0
            source = "offline Phi0-as-Z fallback"
    if commanded_step_z == 0.0:
        raise ConfigError("commanded_step_z must be nonzero")
    return _ResolvedFluxCommand(baseline_z, commanded_step_z, source)


def _resolve_centers(
    parameters: LongTimescaleFluxParameters,
    probe_times_us: np.ndarray,
    flux: _ResolvedFluxCommand,
    record: Optional[Mapping[str, Any]],
) -> np.ndarray:
    requested = parameters.q_frequency_centers_mhz
    if requested is not None:
        try:
            centers = np.asarray(requested, dtype=float)
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "q_frequency_centers_mhz must be one finite center or one per time"
            ) from error
        if centers.ndim == 0:
            centers = np.full(probe_times_us.size, float(centers))
        elif centers.ndim != 1 or centers.size != probe_times_us.size:
            raise ConfigError(
                "q_frequency_centers_mhz must be one finite center or one per "
                "probe time"
            )
    elif record is not None:
        alphas = _vector(
            parameters.prediction_alphas,
            "prediction_alphas",
            minimum=1,
        )
        taus_us = _vector(
            parameters.prediction_taus_us,
            "prediction_taus_us",
            minimum=1,
        )
        if alphas.shape != taus_us.shape:
            raise ConfigError(
                "prediction_alphas and prediction_taus_us must have equal lengths"
            )
        if np.any(taus_us <= 0):
            raise ConfigError("prediction_taus_us must contain only positive values")
        response = np.sum(
            alphas[None, :] * np.exp(-probe_times_us[:, None] / taus_us[None, :]),
            axis=1,
        )
        predicted_z = flux.baseline_z + flux.commanded_step_z * response
        centers = np.asarray(frequency_from_record(record, predicted_z), dtype=float)
    elif parameters.live_hardware:
        raise ConfigError(
            "live adaptive spectroscopy requires either explicit "
            "q_frequency_centers_mhz or an accepted qubit_vs_flux calibration"
        )
    else:
        center = _finite(
            parameters.fallback_q_frequency_mhz,
            "fallback_q_frequency_mhz",
        )
        centers = np.full(probe_times_us.size, center)
    if centers.shape != probe_times_us.shape or not np.all(np.isfinite(centers)):
        raise ConfigError("resolved q-frequency centers must be finite")
    return centers


def _resolve_campaign(
    parameters: LongTimescaleFluxParameters,
    repository,
) -> _ResolvedCampaign:
    probe_times_us = _increasing_vector(
        parameters.probe_times_us,
        "probe_times_us",
        minimum=1,
        positive=True,
    )
    offsets_mhz = _increasing_vector(
        parameters.q_frequency_offsets_mhz,
        "q_frequency_offsets_mhz",
        minimum=8,
    )
    record = _qubit_flux_record(repository)
    flux = _resolve_flux_command(parameters, record)
    centers_mhz = _resolve_centers(
        parameters,
        probe_times_us,
        flux,
        record,
    )

    configured_zone = parameters.q_nyquist_zone
    if configured_zone is None:
        configured_zone = repository.hardware["channels"]["q"].get(
            "nyquist_zone",
            1,
        )
    zone = _positive_integer(configured_zone, "q_nyquist_zone")
    if zone not in (1, 2):
        raise ConfigError("q_nyquist_zone must be 1 or 2")

    q_limits = repository.hardware.get("limits", {}).get("q_freq")
    if q_limits is not None:
        lower, upper = map(float, q_limits)
        requested_min = float(np.min(centers_mhz) + offsets_mhz[0])
        requested_max = float(np.max(centers_mhz) + offsets_mhz[-1])
        if requested_min < lower or requested_max > upper:
            raise ConfigError(
                "adaptive q-frequency windows exceed hardware limits "
                f"[{lower:g}, {upper:g}] MHz"
            )

    _validate_scalar_parameters(parameters)
    campaign = _ResolvedCampaign(
        probe_times_us=probe_times_us,
        q_frequency_offsets_mhz=offsets_mhz,
        q_frequency_centers_mhz=centers_mhz,
        q_nyquist_zone=zone,
        flux=flux,
    )
    _validate_hardware_ranges(parameters, campaign, repository)
    return campaign


def _validate_scalar_parameters(parameters: LongTimescaleFluxParameters) -> None:
    positive = (
        ("q_length_us", parameters.q_length_us),
        ("readout_length_us", parameters.readout_length_us),
        ("readout_relax_us", parameters.readout_relax_us),
        ("z_set_length_us", parameters.z_set_length_us),
    )
    for label, value in positive:
        if _finite(value, label) <= 0:
            raise ConfigError(f"{label} must be positive")
    nonnegative = (
        ("post_probe_to_return_us", parameters.post_probe_to_return_us),
        ("return_to_readout_us", parameters.return_to_readout_us),
    )
    for label, value in nonnegative:
        if _finite(value, label) < 0:
            raise ConfigError(f"{label} must be nonnegative")
    for label, value in (
        ("readout_frequency_mhz", parameters.readout_frequency_mhz),
        ("q_gain", parameters.q_gain),
        ("readout_power_db", parameters.readout_power_db),
        ("z_idle_gain", parameters.z_idle_gain),
    ):
        _finite(value, label)
    _positive_integer(parameters.hard_avg, "hard_avg")
    _positive_integer(parameters.soft_avg, "soft_avg")
    _nonnegative_integer(parameters.seed, "seed")
    if not str(parameters.title_prefix).strip():
        raise ConfigError("title_prefix must not be empty")
    if parameters.first_uncompensated_pass:
        tau_us = _finite(parameters.bias_tee_tau_us, "bias_tee_tau_us")
        if tau_us <= 0:
            raise ConfigError("bias_tee_tau_us must be positive")
        if float(parameters.readout_relax_us) < 15.0 * tau_us:
            raise ConfigError(
                "the first uncompensated pass requires readout_relax_us >= "
                "15 * bias_tee_tau_us"
            )


def _validate_hardware_ranges(
    parameters: LongTimescaleFluxParameters,
    campaign: _ResolvedCampaign,
    repository,
) -> None:
    fast_line = repository.hardware.get("flux_lines", {}).get("z_rf", {})
    if {"minimum", "maximum"}.issubset(fast_line):
        lower = _finite(fast_line["minimum"], "flux_lines.z_rf.minimum")
        upper = _finite(fast_line["maximum"], "flux_lines.z_rf.maximum")
    else:
        z_limits = repository.hardware.get("limits", {}).get("z_gain")
        if z_limits is None:
            return
        lower, upper = map(float, z_limits)
    if lower > upper:
        raise ConfigError("the configured fast-line Z limits are reversed")
    levels = {
        "commanded_step_z": campaign.flux.commanded_step_z,
        "z_idle_gain": float(parameters.z_idle_gain),
        "readout-return zero": 0.0,
    }
    for label, value in levels.items():
        if not lower <= value <= upper:
            raise ConfigError(
                f"{label}={value:g} is outside the fast-line Z limits "
                f"[{lower:g}, {upper:g}]"
            )

    length_limits = repository.hardware.get("limits", {}).get("z_length")
    if length_limits is not None:
        minimum, maximum = map(float, length_limits)
        length = float(parameters.z_set_length_us)
        if not minimum <= length <= maximum:
            raise ConfigError(
                f"z_set_length_us={length:g} is outside hardware Z-length "
                f"limits [{minimum:g}, {maximum:g}]"
            )


def _readout_return_gain(
    parameters: LongTimescaleFluxParameters,
    probe_time_us: float,
    commanded_step_z: float,
) -> float:
    if not parameters.first_uncompensated_pass:
        return 0.0
    # The return happens after the probe pulse and the configured post-probe
    # delay, so changing either parameter cannot silently desynchronize Eq. F1.
    truncation_time_us = (
        float(probe_time_us)
        + float(parameters.q_length_us)
        + float(parameters.post_probe_to_return_us)
    )
    return float(commanded_step_z) * (
        1.0 - np.exp(-truncation_time_us / float(parameters.bias_tee_tau_us))
    )


def _common_overrides(
    parameters: LongTimescaleFluxParameters,
    campaign: _ResolvedCampaign,
    readout_frequency_mhz: float,
) -> dict:
    return {
        "r_freq": float(readout_frequency_mhz),
        "r_power": float(parameters.readout_power_db),
        "r_length": float(parameters.readout_length_us),
        "r_relax": float(parameters.readout_relax_us),
        "q_gain": float(parameters.q_gain),
        "q_length": float(parameters.q_length_us),
        "p1_nqz": campaign.q_nyquist_zone,
        "z_step_gain": campaign.flux.commanded_step_z,
        "z_return_gain": 0.0,
        "z_idle_gain": float(parameters.z_idle_gain),
        "z_set_length": float(parameters.z_set_length_us),
        "post_probe_to_return": float(parameters.post_probe_to_return_us),
        "return_guard": float(parameters.return_to_readout_us),
        "hard_avg": _positive_integer(parameters.hard_avg, "hard_avg"),
        "soft_avg": _positive_integer(parameters.soft_avg, "soft_avg"),
        "rep": 0,
    }


def _manifest_path(project_root: Path, configured: Any) -> Path:
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _native_csv(result) -> Optional[Path]:
    for path in result.data.metadata.get("native_files", ()):
        candidate = Path(path)
        if candidate.suffix.lower() == ".csv":
            return candidate.resolve()
    return None


def _manifest_metadata(
    parameters: LongTimescaleFluxParameters,
    campaign: _ResolvedCampaign,
    *,
    campaign_id: str,
    completed_rows: int,
    campaign_complete: bool,
) -> dict:
    metadata = {
        "paper_doi": PAPER_DOI,
        "campaign_id": campaign_id,
        "campaign_complete": bool(campaign_complete),
        "completed_rows": int(completed_rows),
        "requested_rows": int(campaign.probe_times_us.size),
        "flux_coordinate_source": campaign.flux.source,
        "baseline_z": campaign.flux.baseline_z,
        "commanded_step_z": campaign.flux.commanded_step_z,
        "first_uncompensated_pass": bool(parameters.first_uncompensated_pass),
        "q_nyquist_zone": campaign.q_nyquist_zone,
        "q_length_us": float(parameters.q_length_us),
        "post_probe_to_return_us": float(parameters.post_probe_to_return_us),
        "return_to_readout_us": float(parameters.return_to_readout_us),
        "hard_avg": int(parameters.hard_avg),
        "soft_avg": int(parameters.soft_avg),
    }
    if parameters.baseline_z is None:
        metadata.update(
            {
                "baseline_phi0": float(parameters.baseline_phi0),
                "step_phi0": float(parameters.step_phi0),
                "step_scale": float(parameters.step_scale),
                "commanded_step_phi0": (
                    float(parameters.step_phi0) * float(parameters.step_scale)
                ),
            }
        )
    return metadata


def run_long_timescale_flux_step(
    project_root: Path,
    parameters: LongTimescaleFluxParameters,
):
    """Run one moving-window spectroscopy row for every configured probe time."""
    project_root = Path(project_root).resolve()
    repository = load_repository(project_root)
    campaign = _resolve_campaign(parameters, repository)

    if parameters.live_hardware and not parameters.external_baseline_confirmed:
        raise ConfigError(
            "set and read back the external DC baseline at "
            f"Z={campaign.flux.baseline_z:+.8g}, then set "
            "external_baseline_confirmed=True for this supervised run"
        )

    readout_frequency_mhz = resolve_readout_frequency(
        project_root,
        campaign.flux.baseline_z,
        use_accepted_fit=parameters.use_accepted_resonator_flux_fit,
        fixed_frequency_mhz=float(parameters.readout_frequency_mhz),
    )
    row_count = campaign.probe_times_us.size
    frequencies_per_row = campaign.q_frequency_offsets_mhz.size
    averages = int(parameters.hard_avg) * int(parameters.soft_avg)
    ideal_hours = (
        row_count
        * frequencies_per_row
        * averages
        * float(parameters.readout_relax_us)
        / 3.6e9
    )
    print(f"Protocol source: DOI {PAPER_DOI}")
    print(
        f"Flux command ({campaign.flux.source}): external baseline "
        f"Z={campaign.flux.baseline_z:+.8g}, fast-step delta "
        f"Z={campaign.flux.commanded_step_z:+.8g}."
    )
    print(
        f"Adaptive long-time campaign: {row_count} rows x "
        f"{frequencies_per_row} frequencies, {averages} averages/point, "
        f"ideal lower bound {ideal_hours:.2f} h."
    )

    common = _common_overrides(
        parameters,
        campaign,
        readout_frequency_mhz,
    )
    campaign_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results = []
    manifest_rows = []
    manifest_path = _manifest_path(project_root, parameters.campaign_manifest)

    connection = None
    backend = None
    try:
        if parameters.live_hardware:
            connection = connect_quick(repository)
            backend = connection.backend
        else:
            backend = SyntheticBackend(seed=int(parameters.seed))
        runner = ExperimentRunner(repository, backend)
        for index, (time_us, center_mhz) in enumerate(
            zip(
                campaign.probe_times_us,
                campaign.q_frequency_centers_mhz,
            ),
            start=1,
        ):
            print(
                f"Row {index}/{row_count}: t={time_us:.8g} us, "
                f"center={center_mhz:.8g} MHz"
            )
            result = runner.run(
                "flux_step_spectroscopy",
                "flux_step_spectroscopy",
                title=(
                    f"{parameters.title_prefix}_{campaign_id}"
                    f"_t{number_tag(time_us)}"
                    f"_q{number_tag(center_mhz)}"
                ),
                overrides={
                    **common,
                    "probe_time": float(time_us),
                    "q_freq": (float(center_mhz) + campaign.q_frequency_offsets_mhz),
                    "z_return_gain": _readout_return_gain(
                        parameters,
                        float(time_us),
                        campaign.flux.commanded_step_z,
                    ),
                },
                run_options={"population": False},
                analyze=False,
                close_backend=False,
                park_flux_on_exit=False,
            )
            results.append(result)
            native_csv = _native_csv(result)
            if native_csv is not None:
                manifest_rows.append(
                    {
                        "csv_path": native_csv,
                        "probe_time_us": float(time_us),
                        "predicted_center_mhz": float(center_mhz),
                    }
                )
                write_step_campaign_manifest(
                    manifest_path,
                    manifest_rows,
                    metadata=_manifest_metadata(
                        parameters,
                        campaign,
                        campaign_id=campaign_id,
                        completed_rows=len(manifest_rows),
                        campaign_complete=False,
                    ),
                )
    finally:
        if connection is not None:
            connection.close()
        elif backend is not None:
            backend.close()

    if manifest_rows:
        manifest_complete = len(manifest_rows) == row_count
        write_step_campaign_manifest(
            manifest_path,
            manifest_rows,
            metadata=_manifest_metadata(
                parameters,
                campaign,
                campaign_id=campaign_id,
                completed_rows=len(manifest_rows),
                campaign_complete=manifest_complete,
            ),
        )
        print(f"Campaign manifest written to {manifest_path}")
        if not manifest_complete:
            print(
                "WARNING: the campaign completed acquisition, but one or more "
                "rows did not report a native CSV path; the manifest remains "
                "marked incomplete."
            )
    if parameters.show_plot and results:
        plot_data(results[-1].data, "Last long-timescale flux-step spectrum")
        plt.show()
    return results
