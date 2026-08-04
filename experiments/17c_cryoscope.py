"""Short-timescale Ramsey cryoscope acquisition for flux predistortion."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.flux_compensation import (
    flux_command_from_phi0_fractions,
    make_cryoscope_schedule,
    recommended_shots_per_phase,
    write_cryoscope_schedule,
)
from quickexp_v3.errors import ConfigError
from quickexp_v3.flux_lookup import frequency_from_record
from quickexp_v3.ide import load_repository, resolve_readout_frequency, run_experiment


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False

# The paper used 0.4167 ns samples and sigma=0.5 ns. This repository's verified
# Mercator path is fabric-clock limited, so the safe defaults below are 1.6693
# ns timing and sigma=2 ns. Do not select the paper values until a faster
# arbitrary-waveform upload path is verified on the scope.
SAMPLE_INTERVAL_NS = 1000.0 / 599.04
Z_SIGMA_NS = 2.0
PULSE_DURATION_US = 0.100
CENTER_TIMES_US = np.linspace(0.020, 0.095, 40)
ALLOW_GUIDED_PHASE_UNWRAP = False

PAPER_BASELINE_PHI0 = 0.0
PAPER_STEP_PHI0 = -0.217
BASELINE_Z_OVERRIDE = None
Z_GAIN_OVERRIDE = None
OFFLINE_FALLBACK_BASELINE_Z = 0.0
OFFLINE_FALLBACK_Z_GAIN = -0.217

# 16 phases and 65536 repetitions reproduce the paper. Four orthogonal phases
# are the efficient information-complete setting for iterative passes.
RAMSEY_PHASE_DEG = np.linspace(0.0, 360.0, 16, endpoint=False)
HARD_AVG = 65_536
TARGET_FREQUENCY_SIGMA_MHZ = 0.75
NORMALIZED_READOUT_CONTRAST = 0.8
PAPER_EDGE_COVERAGE_NS = 2.4

USE_ACCEPTED_RESONATOR_FLUX_FIT = True
READOUT_FREQUENCY_MHZ = 6884.0
Q_FREQUENCY_MHZ = 5600.0
Q_PI_OVER_TWO_GAIN = 0.2
Q_LENGTH_US = 0.040
R_POWER_DB = -35.0
R_LENGTH_US = 2.0
R_RELAX_US = 60.0
SOFT_AVG = 1
SHOW_PLOT = True
SCHEDULE_JSON = PROJECT_ROOT / "analysis_cache/flux_compensation/cryoscope_schedule.json"
# ============================================================================


def _qubit_flux_record(repository):
    try:
        record = repository.calibration["records"]["lookups"]["qubit_vs_flux"]
    except (KeyError, TypeError):
        return None
    return record if record.get("status", "accepted") == "accepted" else None


def _resolved_flux_command(repository):
    if (BASELINE_Z_OVERRIDE is None) != (Z_GAIN_OVERRIDE is None):
        raise ConfigError("set both BASELINE_Z_OVERRIDE and Z_GAIN_OVERRIDE, or neither")
    if BASELINE_Z_OVERRIDE is not None:
        return float(BASELINE_Z_OVERRIDE), float(Z_GAIN_OVERRIDE)
    record = _qubit_flux_record(repository)
    if record is not None:
        return flux_command_from_phi0_fractions(
            record,
            baseline_phi0=PAPER_BASELINE_PHI0,
            step_phi0=PAPER_STEP_PHI0,
        )
    if LIVE_HARDWARE:
        raise ConfigError("live cryoscope requires an accepted qubit_vs_flux lookup")
    return OFFLINE_FALLBACK_BASELINE_Z, OFFLINE_FALLBACK_Z_GAIN


def main():
    repository = load_repository(PROJECT_ROOT)
    baseline_z, z_gain = _resolved_flux_command(repository)
    record = _qubit_flux_record(repository)
    if record is not None:
        expected_detuning_mhz = float(
            frequency_from_record(record, baseline_z + z_gain)
            - frequency_from_record(record, baseline_z)
        )
    else:
        expected_detuning_mhz = None
    print(
        f"Resolved local coordinates: baseline Z={baseline_z:+.8g}, "
        f"flux-pulse delta Z={z_gain:+.8g}."
    )
    phase_nyquist_mhz = 500.0 / SAMPLE_INTERVAL_NS
    if expected_detuning_mhz is not None:
        print(
            f"Expected target detuning {expected_detuning_mhz:+.6g} MHz; "
            f"phase-sampling Nyquist limit {phase_nyquist_mhz:.6g} MHz."
        )
        if (
            LIVE_HARDWARE
            and abs(expected_detuning_mhz) >= phase_nyquist_mhz
            and not ALLOW_GUIDED_PHASE_UNWRAP
        ):
            raise ConfigError(
                "expected cryoscope detuning aliases at this timing resolution; "
                "reduce Z_GAIN_OVERRIDE (with its paired baseline override) or "
                "explicitly enable prior-guided phase unwrapping"
            )
    schedule = make_cryoscope_schedule(
        CENTER_TIMES_US,
        pulse_duration_us=PULSE_DURATION_US,
        sample_interval_ns=SAMPLE_INTERVAL_NS,
    )
    minimum_shots = recommended_shots_per_phase(
        schedule.delta_time_us,
        target_frequency_sigma_mhz=TARGET_FREQUENCY_SIGMA_MHZ,
        phase_count=RAMSEY_PHASE_DEG.size,
        normalized_contrast=NORMALIZED_READOUT_CONTRAST,
    )
    print(
        f"Cryoscope schedule: {schedule.center_time_us.size} frequency points, "
        f"{schedule.acquisition_time_us.size} unique pulse durations, "
        f"recommended max {int(np.max(minimum_shots))} shots/phase."
    )
    edge_gaps_ns = np.asarray(
        [
            1000.0 * float(np.min(schedule.center_time_us)),
            1000.0
            * float(PULSE_DURATION_US - np.max(schedule.center_time_us)),
        ]
    )
    if np.any(edge_gaps_ns > PAPER_EDGE_COVERAGE_NS):
        print(
            "WARNING: this fabric-safe schedule does not sample both pulse "
            "edges within the paper's 2.4 ns window; it is an interior "
            "diagnostic, not a complete short-time FIR calibration."
        )
    if HARD_AVG < int(np.max(minimum_shots)):
        print(
            "WARNING: HARD_AVG is below the Fisher-scaling shot estimate for "
            "the requested frequency precision."
        )
    readout_frequency_mhz = resolve_readout_frequency(
        PROJECT_ROOT,
        baseline_z,
        use_accepted_fit=USE_ACCEPTED_RESONATOR_FLUX_FIT,
        fixed_frequency_mhz=READOUT_FREQUENCY_MHZ,
    )
    result = run_experiment(
        PROJECT_ROOT,
        experiment="cryoscope",
        preset="cryoscope",
        title="CryoscopeFlux100ns",
        live_hardware=LIVE_HARDWARE,
        overrides={
            "r_freq": readout_frequency_mhz,
            "r_power": R_POWER_DB,
            "r_length": R_LENGTH_US,
            "r_relax": R_RELAX_US,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_PI_OVER_TWO_GAIN,
            "q_length": Q_LENGTH_US,
            "z_gain": z_gain,
            "z_sigma": Z_SIGMA_NS / 1000.0,
            "flux_time": schedule.acquisition_time_us,
            "ramsey_phase": RAMSEY_PHASE_DEG,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
            "rep": 1,
        },
        run_options={"population": False},
        analyze=False,
        show_plot=SHOW_PLOT,
        seed=23,
    )
    path = write_cryoscope_schedule(
        SCHEDULE_JSON,
        schedule,
        metadata={
            "baseline_z": baseline_z,
            "commanded_step_z": z_gain,
            "paper_baseline_phi0": PAPER_BASELINE_PHI0,
            "paper_step_phi0": PAPER_STEP_PHI0,
            "pulse_duration_us": PULSE_DURATION_US,
            "z_sigma_ns": Z_SIGMA_NS,
            "expected_detuning_mhz": expected_detuning_mhz,
            "guided_phase_unwrap_authorized": ALLOW_GUIDED_PHASE_UNWRAP,
        },
    )
    print(f"Exact finite-difference schedule written to {path}")
    return result


if __name__ == "__main__":
    RESULT = main()
