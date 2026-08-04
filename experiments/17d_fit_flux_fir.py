"""Fit the cryoscope forward response and export an inverse FIR candidate."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.errors import ConfigError
from quickexp_v3.flux_compensation import (
    cryoscope_frequency,
    design_inverse_fir,
    extract_cryoscope_phases,
    fit_forward_fir,
    flux_command_from_phi0_fractions,
    make_cryoscope_schedule,
    read_cryoscope_schedule,
    read_cryoscope_schedule_metadata,
    write_filter_bundle,
)
from quickexp_v3.flux_lookup import frequency_from_record
from quickexp_v3.ide import load_repository
from quickexp_v3.native_index import NativeIndex


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False  # Analysis only; never connects to QICK.
INPUT_CSV = None
SCHEDULE_JSON = PROJECT_ROOT / "analysis_cache/flux_compensation/cryoscope_schedule.json"
PHASE_SIGN = 1.0
MINIMUM_PHASE_ROW_R_SQUARED = 0.80
MAXIMUM_PHASE_UNCERTAINTY_RAD = 0.20

PAPER_BASELINE_PHI0 = 0.0
PAPER_STEP_PHI0 = -0.217
BASELINE_Z_OVERRIDE = None
COMMANDED_STEP_Z_OVERRIDE = None
OFFLINE_DEMO_BASELINE_Z = 0.0
OFFLINE_DEMO_STEP_Z = -0.217
PULSE_DURATION_US = 0.100
# Measured data must be fitted at the acquisition schedule's actual DAC rate.
# None selects that rate. A numeric override must match it. Synthetic demo data
# use the paper's 0.4167 ns interval.
FILTER_SAMPLE_INTERVAL_NS = None
PAPER_SAMPLE_INTERVAL_NS = 0.4167
FIR_SUPPORT_NS = 50.0
FIR_COEFFICIENT_COUNT = None  # Derived from FIR_SUPPORT_NS when None (120 paper taps).
INVERSE_FIR_LENGTH = None     # Matches the forward length when None.
GAUSSIAN_TARGET_SIGMA_NS = 0.75
ENERGY_REGULARIZATION = 1e-6
TAIL_REGULARIZATION = 1e-4
TAIL_GROWTH = 6.0
DC_REGULARIZATION = 1e-2
DERIVATIVE_REGULARIZATION = 1e-3
MAXIMUM_EVALUATIONS = 3000
MAXIMUM_EDGE_GAP_NS = 2.4
ALLOW_PARTIAL_EDGE_FIT = False

OUTPUT_JSON = PROJECT_ROOT / "analysis_cache/flux_compensation/fir_candidate.json"
WRITE_CANDIDATE = True
USE_SYNTHETIC_DEMO = False
SHOW_PLOT = True
# ============================================================================


def _input_path(repository):
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    root = Path(repository.hardware["storage"]["quick_native_root"])
    return NativeIndex(root).refresh().latest(
        quick_class="Cryoscope",
        n_axes=2,
    ).csv_path


def _qubit_flux_record(repository):
    try:
        record = repository.calibration["records"]["lookups"]["qubit_vs_flux"]
    except (KeyError, TypeError):
        raise ConfigError(
            "an accepted records.lookups.qubit_vs_flux calibration is required; "
            "run 06b and 06e first"
        )
    if record.get("status", "accepted") != "accepted":
        raise ConfigError("qubit_vs_flux calibration is not accepted")
    return record


def _resolved_flux_command(record, schedule_metadata=None):
    if (BASELINE_Z_OVERRIDE is None) != (COMMANDED_STEP_Z_OVERRIDE is None):
        raise ConfigError(
            "set both BASELINE_Z_OVERRIDE and COMMANDED_STEP_Z_OVERRIDE, or neither"
        )
    metadata = dict(schedule_metadata or {})
    if BASELINE_Z_OVERRIDE is not None:
        return float(BASELINE_Z_OVERRIDE), float(COMMANDED_STEP_Z_OVERRIDE)
    if {"baseline_z", "commanded_step_z"}.issubset(metadata):
        baseline_z = float(metadata["baseline_z"])
        commanded_step_z = float(metadata["commanded_step_z"])
        # Domain-check the exact coordinates used for acquisition.
        frequency_from_record(
            record,
            np.asarray([baseline_z, baseline_z + commanded_step_z]),
        )
        return baseline_z, commanded_step_z
    return flux_command_from_phi0_fractions(
        record,
        baseline_phi0=PAPER_BASELINE_PHI0,
        step_phi0=PAPER_STEP_PHI0,
    )


def _demo_data():
    interval_ns = (
        PAPER_SAMPLE_INTERVAL_NS
        if FILTER_SAMPLE_INTERVAL_NS is None
        else float(FILTER_SAMPLE_INTERVAL_NS)
    )
    interval_us = interval_ns / 1000.0
    samples = int(np.ceil(PULSE_DURATION_US / interval_us)) + 1
    command = np.full(samples, OFFLINE_DEMO_STEP_Z)
    coefficient_count = (
        int(FIR_COEFFICIENT_COUNT)
        if FIR_COEFFICIENT_COUNT is not None
        else max(int(round(FIR_SUPPORT_NS / interval_ns)), 2)
    )
    true_forward = np.zeros(max(coefficient_count, 3))
    true_forward[:3] = [0.78, 0.16, 0.06]
    actual_z = np.convolve(command, true_forward, mode="full")[:samples]

    model = lambda z: 5600.0 - 4200.0 * np.asarray(z, dtype=float) ** 2

    all_time = np.arange(samples) * interval_us
    stride = max(samples // 45, 1)
    measured_time = all_time[1::stride]
    measured_frequency = model(actual_z)[1::stride]
    measured_frequency += np.random.default_rng(23).normal(
        0.0, 0.04, measured_frequency.size
    )
    weights = np.ones_like(measured_time)
    return (
        command,
        measured_time,
        measured_frequency,
        weights,
        model,
        "synthetic",
        interval_ns,
        OFFLINE_DEMO_BASELINE_Z,
    )


def _measured_data(repository):
    record = _qubit_flux_record(repository)
    schedule_metadata = (
        read_cryoscope_schedule_metadata(SCHEDULE_JSON)
        if Path(SCHEDULE_JSON).is_file()
        else {}
    )
    if "pulse_duration_us" in schedule_metadata and not np.isclose(
        float(schedule_metadata["pulse_duration_us"]),
        PULSE_DURATION_US,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ConfigError(
            "PULSE_DURATION_US does not match the persisted cryoscope schedule"
        )
    baseline_z, commanded_step_z = _resolved_flux_command(
        record,
        schedule_metadata,
    )

    model = lambda z: np.asarray(frequency_from_record(record, z), dtype=float)
    idle_frequency = float(model(np.asarray([baseline_z]))[0])
    target_frequency = float(
        model(np.asarray([baseline_z + commanded_step_z]))[0]
    )
    phase_trace = extract_cryoscope_phases(
        _input_path(repository),
        phase_sign=PHASE_SIGN,
        phase_prior_detuning_mhz=target_frequency - idle_frequency,
    )
    poor_rows = (
        (phase_trace.row_r_squared < MINIMUM_PHASE_ROW_R_SQUARED)
        | (phase_trace.phase_uncertainty_rad > MAXIMUM_PHASE_UNCERTAINTY_RAD)
    )
    if np.any(poor_rows):
        raise ConfigError(
            f"{int(np.sum(poor_rows))} cryoscope phase rows failed QC; "
            "increase averaging or inspect readout contrast before fitting FIR"
        )
    if Path(SCHEDULE_JSON).is_file():
        schedule = read_cryoscope_schedule(SCHEDULE_JSON)
    else:
        # Explicit fallback for imported data. Prefer the persisted schedule,
        # because this reconstruction assumes the configured spacing.
        if FILTER_SAMPLE_INTERVAL_NS is None:
            raise ConfigError(
                "the persisted cryoscope schedule is required unless "
                "FILTER_SAMPLE_INTERVAL_NS is set explicitly for imported data"
            )
        centers = np.linspace(
            float(phase_trace.duration_us.min()),
            float(phase_trace.duration_us.max()),
            max(phase_trace.duration_us.size // 2, 8),
        )
        schedule = make_cryoscope_schedule(
            centers,
            pulse_duration_us=PULSE_DURATION_US,
            sample_interval_ns=float(FILTER_SAMPLE_INTERVAL_NS),
        )
    interval_ns = float(schedule.sample_interval_ns)
    if FILTER_SAMPLE_INTERVAL_NS is not None and not np.isclose(
        interval_ns,
        float(FILTER_SAMPLE_INTERVAL_NS),
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ConfigError(
            "FILTER_SAMPLE_INTERVAL_NS does not match the persisted cryoscope "
            "schedule; do not fit data on a different sample grid"
        )
    frequency_trace = cryoscope_frequency(phase_trace, schedule)
    measured_frequency = idle_frequency + frequency_trace.detuning_mhz
    samples = int(np.ceil(PULSE_DURATION_US * 1000.0 / interval_ns)) + 1
    command = np.full(samples, commanded_step_z)
    weights = frequency_trace.delta_time_us
    return (
        command,
        frequency_trace.time_us,
        measured_frequency,
        weights,
        model,
        str(phase_trace.source_csv),
        interval_ns,
        baseline_z,
    )


def main():
    repository = load_repository(PROJECT_ROOT)
    data = _demo_data() if USE_SYNTHETIC_DEMO else _measured_data(repository)
    (
        command,
        measured_time,
        measured_frequency,
        weights,
        model,
        source,
        sample_interval_ns,
        baseline_z,
    ) = data
    if not USE_SYNTHETIC_DEMO:
        edge_gaps_ns = np.asarray(
            [
                1000.0 * float(np.min(measured_time)),
                1000.0 * float(PULSE_DURATION_US - np.max(measured_time)),
            ]
        )
        if np.any(edge_gaps_ns > MAXIMUM_EDGE_GAP_NS):
            message = (
                "cryoscope frequency samples do not cover both pulse edges "
                f"within {MAXIMUM_EDGE_GAP_NS:g} ns (gaps "
                f"{edge_gaps_ns[0]:.3g}, {edge_gaps_ns[1]:.3g} ns)"
            )
            if not ALLOW_PARTIAL_EDGE_FIT:
                raise ConfigError(
                    message
                    + "; do not export a short-time inverse from interior-only "
                    "data unless ALLOW_PARTIAL_EDGE_FIT is explicitly enabled"
                )
            print(f"WARNING: {message}; candidate is diagnostic only.")
    coefficient_count = (
        int(FIR_COEFFICIENT_COUNT)
        if FIR_COEFFICIENT_COUNT is not None
        else max(int(round(FIR_SUPPORT_NS / sample_interval_ns)), 2)
    )
    coefficient_count = min(coefficient_count, command.size)
    inverse_length = (
        int(INVERSE_FIR_LENGTH)
        if INVERSE_FIR_LENGTH is not None
        else coefficient_count
    )
    forward = fit_forward_fir(
        command,
        sample_interval_ns=sample_interval_ns,
        measured_time_us=measured_time,
        measured_frequency_mhz=measured_frequency,
        frequency_model=model,
        baseline_z=baseline_z,
        coefficient_count=coefficient_count,
        integration_weight=weights,
        energy_regularization=ENERGY_REGULARIZATION,
        tail_regularization=TAIL_REGULARIZATION,
        tail_growth=TAIL_GROWTH,
        dc_regularization=DC_REGULARIZATION,
        maximum_evaluations=MAXIMUM_EVALUATIONS,
    )
    inverse = design_inverse_fir(
        forward.coefficients,
        sample_interval_ns=sample_interval_ns,
        inverse_length=inverse_length,
        gaussian_sigma_ns=GAUSSIAN_TARGET_SIGMA_NS,
        derivative_regularization=DERIVATIVE_REGULARIZATION,
    )
    print(f"Fit source: {source}")
    print(
        f"Forward FIR ({forward.coefficients.size} taps): "
        f"R^2={forward.statistics['r_squared']:.7f}, "
        f"RMSE={forward.statistics['rmse_mhz']:.5g} MHz."
    )
    print(
        f"Inverse FIR ({inverse.coefficients.size} taps): latency "
        f"{inverse.latency_samples} samples, convolution RMSE "
        f"{inverse.statistics['rmse']:.5g}."
    )
    if WRITE_CANDIDATE:
        path = write_filter_bundle(
            OUTPUT_JSON,
            forward_fir=forward,
            inverse_fir=inverse,
            metadata={
                "source": source,
                "iteration": "short_time_candidate",
                "hardware_applied": False,
                "acquisition_sample_interval_ns": sample_interval_ns,
                "baseline_z": baseline_z,
                "commanded_step_z": float(command[0]),
                "partial_edge_fit": bool(ALLOW_PARTIAL_EDGE_FIT),
            },
        )
        print(f"Candidate (not applied to hardware) written to {path}")
    if SHOW_PLOT:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        axes[0].plot(measured_time, measured_frequency, "o", label="measured")
        axes[0].plot(measured_time, forward.fitted_frequency_mhz, "-", label="fit")
        axes[0].set(xlabel="Time (us)", ylabel="Qubit frequency (MHz)")
        axes[0].legend()
        time_ns = np.arange(forward.coefficients.size) * sample_interval_ns
        axes[1].plot(time_ns, forward.coefficients, ".-")
        axes[1].set(xlabel="Tap time (ns)", ylabel="Forward FIR coefficient")
        axes[2].plot(inverse.target, label="Gaussian target")
        axes[2].plot(inverse.realized, label="forward * inverse")
        axes[2].set(xlabel="Sample", ylabel="Impulse response")
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.3)
        plt.show()
    return forward


if __name__ == "__main__":
    FIT = main()
