"""Long-timescale flux-step spectroscopy from Hellings et al. (2025)."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.errors import ConfigError
from quickexp_v3.flux_compensation import (
    PAPER_DOI,
    flux_command_from_phi0_fractions,
    write_step_campaign_manifest,
)
from quickexp_v3.flux_lookup import frequency_from_record
from quickexp_v3.ide import (
    load_repository,
    plot_data,
    resolve_readout_frequency,
    run_experiment,
)
from quickexp_v3.lab import connect_quick
from quickexp_v3.naming import number_tag
from quickexp_v3.runtime import ExperimentRunner


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False

# Adaptive rows reproduce the paper's +/-100 MHz window around a predicted
# center without measuring a prohibitively wide rectangular frequency map.
ADAPTIVE_ROW_MODE = True
PROBE_TIMES_US = np.geomspace(0.010, 100.0, 70)
Q_FREQUENCY_OFFSETS_MHZ = np.linspace(-100.0, 100.0, 201)
FALLBACK_Q_FREQUENCY_MHZ = 5200.0  # Offline only; live mode requires a lookup.

# Initial prediction for window placement only. The measured data, not these
# values, are used by 17b. They approximate the paper's bias-tee hierarchy.
PREDICTION_ALPHAS = np.asarray([0.10, 0.22, 0.68])
PREDICTION_TAUS_US = np.asarray([0.055, 1.3, 20.0])
# Paper coordinates relative to an upper sweet spot. They are converted to
# local Z-gain units using the accepted fit's period_z. Set both overrides only
# when the external DC and fast paths have their own cross-calibrated units.
PAPER_BASELINE_PHI0 = -0.127
PAPER_STEP_PHI0 = -0.217
BASELINE_Z_OVERRIDE = None
COMMANDED_STEP_Z_OVERRIDE = None
OFFLINE_FALLBACK_BASELINE_Z = -0.127
OFFLINE_FALLBACK_STEP_Z = -0.217
# The paper supplies the baseline through a separate DC path and sends only the
# transient step through the fast line. This launcher cannot own the injected
# external instrument yet; live execution is latched until the operator has
# set/read back that bias through the lab's normal instrument control.
EXTERNAL_BASELINE_CONFIRMED = False
MODEL_READOUT_RETURN = True
BIAS_TEE_MODEL_TAU_US = 20.0
STEP_TO_RETURN_OFFSET_US = 0.110
POST_PROBE_TO_RETURN_US = 0.070
RETURN_TO_READOUT_US = 0.110
Z_IDLE_GAIN = 0.0
Z_SET_LENGTH_US = 0.004

USE_ACCEPTED_RESONATOR_FLUX_FIT = True
READOUT_FREQUENCY_MHZ = 6884.0
Q_GAIN = 0.15
Q_LENGTH_US = 0.040
R_POWER_DB = -35.0
R_LENGTH_US = 2.0
R_RELAX_US = 300.0  # >=15*19.2 us paper bias-tee time constant.
HARD_AVG = 2048
SOFT_AVG = 1
SHOW_PLOT = True
CAMPAIGN_MANIFEST_JSON = (
    PROJECT_ROOT / "analysis_cache/flux_compensation/flux_step_campaign.json"
)
# ============================================================================


def _qubit_flux_record(repository):
    try:
        record = repository.calibration["records"]["lookups"]["qubit_vs_flux"]
    except (KeyError, TypeError):
        return None
    return record if record.get("status", "accepted") == "accepted" else None


def _resolved_flux_command(repository):
    if (BASELINE_Z_OVERRIDE is None) != (COMMANDED_STEP_Z_OVERRIDE is None):
        raise ConfigError(
            "set both BASELINE_Z_OVERRIDE and COMMANDED_STEP_Z_OVERRIDE, or neither"
        )
    if BASELINE_Z_OVERRIDE is not None:
        return float(BASELINE_Z_OVERRIDE), float(COMMANDED_STEP_Z_OVERRIDE)
    record = _qubit_flux_record(repository)
    if record is not None:
        return flux_command_from_phi0_fractions(
            record,
            baseline_phi0=PAPER_BASELINE_PHI0,
            step_phi0=PAPER_STEP_PHI0,
        )
    if LIVE_HARDWARE:
        raise ConfigError(
            "live flux commands require an accepted qubit_vs_flux calibration"
        )
    return OFFLINE_FALLBACK_BASELINE_Z, OFFLINE_FALLBACK_STEP_Z


def _predicted_centers(repository, baseline_z, commanded_step_z):
    record = _qubit_flux_record(repository)
    if record is None:
        if LIVE_HARDWARE:
            raise ConfigError(
                "adaptive live spectroscopy requires an accepted "
                "records.lookups.qubit_vs_flux calibration; run 06b and 06e first"
            )
        return np.full(PROBE_TIMES_US.size, FALLBACK_Q_FREQUENCY_MHZ)
    response = np.sum(
        PREDICTION_ALPHAS[None, :]
        * np.exp(-PROBE_TIMES_US[:, None] / PREDICTION_TAUS_US[None, :]),
        axis=1,
    )
    predicted_z = baseline_z + commanded_step_z * response
    return np.asarray(frequency_from_record(record, predicted_z), dtype=float)


def _readout_return_gain(time_us, commanded_step_z):
    if not MODEL_READOUT_RETURN:
        return 0.0
    return commanded_step_z * (
        1.0
        - np.exp(
            -(float(time_us) + STEP_TO_RETURN_OFFSET_US)
            / BIAS_TEE_MODEL_TAU_US
        )
    )


def _common_overrides(readout_frequency_mhz, commanded_step_z):
    return {
        "r_freq": readout_frequency_mhz,
        "r_power": R_POWER_DB,
        "r_length": R_LENGTH_US,
        "r_relax": R_RELAX_US,
        "q_gain": Q_GAIN,
        "q_length": Q_LENGTH_US,
        "z_step_gain": commanded_step_z,
        "z_return_gain": 0.0,
        "z_idle_gain": Z_IDLE_GAIN,
        "z_set_length": Z_SET_LENGTH_US,
        "post_probe_to_return": POST_PROBE_TO_RETURN_US,
        "return_guard": RETURN_TO_READOUT_US,
        "hard_avg": HARD_AVG,
        "soft_avg": SOFT_AVG,
        "rep": 0,
    }


def _adaptive_rows(
    repository,
    readout_frequency_mhz,
    baseline_z,
    commanded_step_z,
):
    backend = None
    connection = None
    if LIVE_HARDWARE:
        connection = connect_quick(repository)
        backend = connection.backend
    else:
        backend = SyntheticBackend(seed=17)
    runner = ExperimentRunner(repository, backend)
    results = []
    manifest_rows = []
    centers = _predicted_centers(repository, baseline_z, commanded_step_z)
    try:
        for time_us, center_mhz in zip(PROBE_TIMES_US, centers):
            result = runner.run(
                "flux_step_spectroscopy",
                "flux_step_spectroscopy",
                title=(
                    "FluxStepAdaptive"
                    f"_t{number_tag(time_us)}"
                    f"_q{number_tag(center_mhz)}"
                ),
                overrides={
                    **_common_overrides(readout_frequency_mhz, commanded_step_z),
                    "probe_time": float(time_us),
                    "q_freq": center_mhz + Q_FREQUENCY_OFFSETS_MHZ,
                    "z_return_gain": _readout_return_gain(
                        time_us,
                        commanded_step_z,
                    ),
                },
                run_options={"population": False},
                analyze=False,
                close_backend=False,
                park_flux_on_exit=False,
            )
            results.append(result)
            native_csv = next(
                (
                    path
                    for path in result.data.metadata.get("native_files", ())
                    if str(path).lower().endswith(".csv")
                ),
                None,
            )
            if native_csv is not None:
                manifest_rows.append(
                    {
                        "csv_path": native_csv,
                        "probe_time_us": float(time_us),
                        "predicted_center_mhz": float(center_mhz),
                    }
                )
    finally:
        if connection is not None:
            connection.close()
        elif backend is not None:
            backend.close()
    if manifest_rows:
        manifest = write_step_campaign_manifest(
            CAMPAIGN_MANIFEST_JSON,
            manifest_rows,
            metadata={
                "paper_doi": PAPER_DOI,
                "baseline_z": baseline_z,
                "commanded_step_z": commanded_step_z,
                "paper_baseline_phi0": PAPER_BASELINE_PHI0,
                "paper_step_phi0": PAPER_STEP_PHI0,
                "model_readout_return": MODEL_READOUT_RETURN,
            },
        )
        print(f"Adaptive campaign manifest written to {manifest}")
    if SHOW_PLOT and results:
        plot_data(results[-1].data, "Last adaptive flux-step spectrum")
        plt.show()
    return results


def main():
    if LIVE_HARDWARE and not EXTERNAL_BASELINE_CONFIRMED:
        raise ConfigError(
            "set and read back the external DC baseline, then set "
            "EXTERNAL_BASELINE_CONFIRMED=True for this supervised run"
        )
    repository = load_repository(PROJECT_ROOT)
    baseline_z, commanded_step_z = _resolved_flux_command(repository)
    if not ADAPTIVE_ROW_MODE and MODEL_READOUT_RETURN:
        raise ConfigError(
            "the uncompensated readout-return level depends on probe_time; "
            "use adaptive rows or set MODEL_READOUT_RETURN=False only after "
            "the dominant IIR correction is active"
        )
    readout_frequency_mhz = resolve_readout_frequency(
        PROJECT_ROOT,
        baseline_z,
        use_accepted_fit=USE_ACCEPTED_RESONATOR_FLUX_FIT,
        fixed_frequency_mhz=READOUT_FREQUENCY_MHZ,
    )
    print(f"Protocol source: DOI {PAPER_DOI}")
    print(
        f"Resolved local coordinates: baseline Z={baseline_z:+.8g}, "
        f"fast-step delta Z={commanded_step_z:+.8g}."
    )
    if ADAPTIVE_ROW_MODE:
        print(
            f"Adaptive spectroscopy: {PROBE_TIMES_US.size} time rows, "
            f"{Q_FREQUENCY_OFFSETS_MHZ.size} frequencies per row."
        )
        return _adaptive_rows(
            repository,
            readout_frequency_mhz,
            baseline_z,
            commanded_step_z,
        )
    predicted_centers = _predicted_centers(
        repository,
        baseline_z,
        commanded_step_z,
    )
    center = float(np.mean(predicted_centers))
    span = 2.0 * (
        np.max(np.abs(predicted_centers - center))
        + np.max(np.abs(Q_FREQUENCY_OFFSETS_MHZ))
    )
    points = max(int(np.ceil(span / 1.0)) + 1, 201)
    q_frequency = np.linspace(center - span / 2.0, center + span / 2.0, points)
    print(
        "Rectangular-map mode can be much slower than adaptive rows: "
        f"{PROBE_TIMES_US.size} x {q_frequency.size} points."
    )
    return run_experiment(
        PROJECT_ROOT,
        experiment="flux_step_spectroscopy",
        preset="flux_step_spectroscopy",
        title="FluxStepRectangular",
        live_hardware=LIVE_HARDWARE,
        overrides={
            **_common_overrides(readout_frequency_mhz, commanded_step_z),
            "probe_time": PROBE_TIMES_US,
            "q_freq": q_frequency,
        },
        run_options={"population": False},
        analyze=False,
        show_plot=SHOW_PLOT,
        seed=17,
    )


if __name__ == "__main__":
    RESULT = main()
