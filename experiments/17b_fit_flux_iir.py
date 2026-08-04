"""Fit the long-time flux response and export a candidate inverse IIR."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.errors import ConfigError
from quickexp_v3.flux_compensation import (
    design_iir_inverse,
    extract_step_spectroscopy,
    extract_step_spectroscopy_rows,
    fit_step_response,
    flux_command_from_phi0_fractions,
    monotonic_branch_for_flux_step,
    read_step_campaign_manifest,
    read_step_campaign_metadata,
    spectroscopy_to_normalized_step,
    write_filter_bundle,
)
from quickexp_v3.ide import load_repository
from quickexp_v3.native_index import NativeIndex


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False  # Analysis only; never connects to QICK.
INPUT_CSV = None       # One rectangular 2D map, if used.
INPUT_CSVS = ()        # Adaptive 1D rows; explicit paths take precedence.
ADAPTIVE_ROW_COUNT = 70
CAMPAIGN_MANIFEST_JSON = (
    PROJECT_ROOT / "analysis_cache/flux_compensation/flux_step_campaign.json"
)
USE_CAMPAIGN_FLUX_COORDINATES = True

PAPER_BASELINE_PHI0 = -0.127
PAPER_STEP_PHI0 = -0.217
BASELINE_Z_OVERRIDE = None
COMMANDED_STEP_Z_OVERRIDE = None
MONOTONIC_BRANCH_Z_OVERRIDE = None
SPECTROSCOPY_OFFSET_MHZ = 1.1
NORMALIZATION = "measured_peak"  # Appendix G1; or "commanded_step" for QA.
FIT_MINIMUM_TIME_US = 0.025
FIT_MAXIMUM_TIME_US = 50.0
MODEL_ORDERS = tuple(range(1, 7))
TAU_BOUNDS_US = (0.005, 300.0)
MULTISTARTS = 12
# Appendix G fixes alpha_0=0 on the uncompensated pass and alpha_0=1 after
# the dominant high-pass response is corrected. Change this for IIR iteration 2.
FIT_DC_GAIN = 0.0

# 0.4167 ns reproduces the 2.4-GSa/s AWG in the paper. It is an export design
# rate until this repository has a verified arbitrary-waveform upload adapter.
FILTER_SAMPLE_INTERVAL_NS = 0.4167
LEAK_TAU_US = None  # Faithful integrator at z=1; e.g. 1000 gives a safe leak.
OUTPUT_JSON = PROJECT_ROOT / "analysis_cache/flux_compensation/iir_candidate.json"
WRITE_CANDIDATE = True
USE_SYNTHETIC_DEMO = False
SHOW_PLOT = True
# ============================================================================


def _candidate_inputs(repository):
    explicit_rows = [Path(path).expanduser().resolve() for path in INPUT_CSVS]
    if explicit_rows:
        return "rows", explicit_rows, {}
    if INPUT_CSV is not None:
        return "map", Path(INPUT_CSV).expanduser().resolve(), {}
    if Path(CAMPAIGN_MANIFEST_JSON).is_file():
        metadata = (
            read_step_campaign_metadata(CAMPAIGN_MANIFEST_JSON)
            if USE_CAMPAIGN_FLUX_COORDINATES
            else {}
        )
        return (
            "rows",
            list(read_step_campaign_manifest(CAMPAIGN_MANIFEST_JSON)),
            metadata,
        )
    root = Path(repository.hardware["storage"]["quick_native_root"])
    index = NativeIndex(root).refresh()
    maps = index.select(quick_class="FluxStepSpectroscopy", n_axes=2)
    rows = index.select(
        quick_class="FluxStepSpectroscopy",
        n_axes=1,
        title_contains="FluxStepAdaptive",
    )
    newest_map = maps[-1] if maps else None
    newest_row = rows[-1] if rows else None
    if newest_row is not None and (
        newest_map is None or newest_row.mtime > newest_map.mtime
    ):
        selected = rows[-int(ADAPTIVE_ROW_COUNT):]
        return "rows", [record.csv_path for record in selected], {}
    if newest_map is not None:
        return "map", newest_map.csv_path, {}
    raise FileNotFoundError(
        "no FluxStepSpectroscopy data found; run 17a or set INPUT_CSV(S)"
    )


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


def _resolved_flux_coordinates(record, campaign_metadata=None):
    if (BASELINE_Z_OVERRIDE is None) != (COMMANDED_STEP_Z_OVERRIDE is None):
        raise ConfigError(
            "set both BASELINE_Z_OVERRIDE and COMMANDED_STEP_Z_OVERRIDE, or neither"
        )
    metadata = dict(campaign_metadata or {})
    if BASELINE_Z_OVERRIDE is None and {
        "baseline_z",
        "commanded_step_z",
    }.issubset(metadata):
        baseline_z = float(metadata["baseline_z"])
        commanded_step_z = float(metadata["commanded_step_z"])
    elif BASELINE_Z_OVERRIDE is None:
        baseline_z, commanded_step_z = flux_command_from_phi0_fractions(
            record,
            baseline_phi0=PAPER_BASELINE_PHI0,
            step_phi0=PAPER_STEP_PHI0,
        )
    else:
        baseline_z = float(BASELINE_Z_OVERRIDE)
        commanded_step_z = float(COMMANDED_STEP_Z_OVERRIDE)
    branch_z = (
        monotonic_branch_for_flux_step(
            record,
            baseline_z=baseline_z,
            commanded_step_z=commanded_step_z,
        )
        if MONOTONIC_BRANCH_Z_OVERRIDE is None
        else tuple(map(float, MONOTONIC_BRANCH_Z_OVERRIDE))
    )
    return baseline_z, commanded_step_z, branch_z


def _demo_fit():
    time_us = np.geomspace(0.025, 50.0, 70)
    response = (
        0.68 * np.exp(-time_us / 20.0)
        + 0.22 * np.exp(-time_us / 1.3)
        + 0.10 * np.exp(-time_us / 0.055)
    )
    response += np.random.default_rng(17).normal(0.0, 8e-4, time_us.size)
    return fit_step_response(
        time_us,
        response,
        model_orders=MODEL_ORDERS,
        dc_gain=FIT_DC_GAIN,
        minimum_time_us=FIT_MINIMUM_TIME_US,
        maximum_time_us=FIT_MAXIMUM_TIME_US,
        tau_bounds_us=TAU_BOUNDS_US,
        multistarts=MULTISTARTS,
        seed=17,
    )


def main():
    repository = load_repository(PROJECT_ROOT)
    trace = None
    if USE_SYNTHETIC_DEMO:
        fit = _demo_fit()
        source = "deterministic synthetic three-timescale response"
    else:
        kind, inputs, campaign_metadata = _candidate_inputs(repository)
        trace = (
            extract_step_spectroscopy(inputs)
            if kind == "map"
            else extract_step_spectroscopy_rows(inputs)
        )
        record = _qubit_flux_record(repository)
        baseline_z, commanded_step_z, branch_z = _resolved_flux_coordinates(
            record,
            campaign_metadata,
        )
        _measured_z, normalized, uncertainty = spectroscopy_to_normalized_step(
            trace,
            record,
            branch_z=branch_z,
            baseline_z=baseline_z,
            commanded_step_z=commanded_step_z,
            spectroscopy_offset_mhz=SPECTROSCOPY_OFFSET_MHZ,
            normalization=NORMALIZATION,
        )
        print(
            f"Local flux coordinates: baseline={baseline_z:+.8g}, "
            f"step={commanded_step_z:+.8g}, branch={branch_z}."
        )
        fit = fit_step_response(
            trace.time_us,
            normalized,
            uncertainty=uncertainty,
            model_orders=MODEL_ORDERS,
            dc_gain=FIT_DC_GAIN,
            minimum_time_us=FIT_MINIMUM_TIME_US,
            maximum_time_us=FIT_MAXIMUM_TIME_US,
            tau_bounds_us=TAU_BOUNDS_US,
            multistarts=MULTISTARTS,
            seed=17,
        )
        source = str(trace.source_csv)
    inverse = design_iir_inverse(
        fit,
        sample_interval_ns=FILTER_SAMPLE_INTERVAL_NS,
        leak_tau_us=LEAK_TAU_US,
    )
    print(f"Fit source: {source}")
    print(
        f"Selected {fit.model_order} exponentials by BIC; "
        f"R^2={fit.statistics['r_squared']:.7f}, "
        f"RMSE={fit.statistics['rmse']:.4g}."
    )
    for index, (alpha, tau) in enumerate(zip(fit.alphas, fit.taus_us), start=1):
        print(f"  alpha[{index}]={alpha:+.8g}, tau[{index}]={tau:.8g} us")
    stability = "stable" if inverse.stable else "marginal integrator at z=1"
    print(
        f"IIR: {inverse.sos.shape[0]} SOS at "
        f"{FILTER_SAMPLE_INTERVAL_NS:.6g} ns; {stability}."
    )
    if WRITE_CANDIDATE:
        path = write_filter_bundle(
            OUTPUT_JSON,
            step_fit=fit,
            iir=inverse,
            metadata={
                "source": source,
                "iteration": "long_time_candidate",
                "hardware_applied": False,
            },
        )
        print(f"Candidate (not applied to hardware) written to {path}")
    if SHOW_PLOT:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].semilogx(fit.time_us, fit.measured, "o", label="measured")
        dense = np.geomspace(max(fit.time_us.min(), 1e-9), fit.time_us.max(), 1000)
        axes[0].semilogx(dense, fit.response(dense), "-", label="fit")
        axes[0].set(xlabel="Time (us)", ylabel="Normalized step response")
        axes[0].legend()
        axes[1].semilogx(fit.time_us, fit.measured - fit.fitted, ".-")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set(xlabel="Time (us)", ylabel="Fit residual")
        for axis in axes:
            axis.grid(alpha=0.3)
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
