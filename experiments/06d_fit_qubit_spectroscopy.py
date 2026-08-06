"""Fit a one-dimensional fixed-flux qubit spectroscopy run."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.notch_fit import fit_spectroscopy_features
from quickexp_v3.native_fit import (
    accept_spectroscopy_fit,
    find_latest_native,
    fit_spectroscopy,
    plot_spectroscopy_fit,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# None selects the newest one-dimensional QubitSpectroscopy CSV/YML pair.
# To fit a particular run, paste its CSV path as a raw string:
# INPUT_CSV = r"Z:\Your\Data\folder\00001 - (QubitSpectroscopy)name.csv"
INPUT_CSV = None

# Choose amplitude, phase, I, Q, or IQ. IQ uses the principal measured axis.
FIT_SIGNAL = "IQ"
# Strongly recommended when a coarse scan contains multiple candidate peaks.
# Example: (5595.0, 5615.0). None fits the full acquired frequency axis.
FIT_WINDOW_MHZ = None
ENABLE_TWO_FEATURE_SELECTION = True

# R^2 is measured over the local window the feature is fitted in, not the whole
# sweep, so it now reflects how well the line itself is described. A shallow
# line on a long noisy trace can be fitted perfectly and still score low here,
# which is why the contrast and uncertainty gates carry most of the weight.
MINIMUM_R_SQUARED = 0.30
MINIMUM_CONTRAST_SNR = 3.0
MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM = 0.30

# Safety latch: inspect the plot/statistics first, then change this to True.
WRITE_ACCEPTED_FIT = False
# Independent manual override: writes even when acceptance gates fail.
FORCE_WRITE = False
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    data_directory = Path(
        repository.hardware["storage"]["quick_native_root"]
    )
    return find_latest_native(
        data_directory,
        quick_class="QubitSpectroscopy",
        axis_text="qubit pulse frequency",
    )


def main():
    source = _input_path()
    fit_function = (
        fit_spectroscopy_features
        if ENABLE_TWO_FEATURE_SELECTION
        else fit_spectroscopy
    )
    fit = fit_function(
        source,
        kind="qubit",
        signal=FIT_SIGNAL,
        window_mhz=FIT_WINDOW_MHZ,
    )
    passes = fit.passes(
        minimum_r_squared=MINIMUM_R_SQUARED,
        minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
        maximum_center_uncertainty_fraction_of_fwhm=(
            MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM
        ),
    )
    print(f"Fit source: {fit.source_csv}")
    print(f"Fit signal: {fit.signal_label}")
    print(f"Detected feature polarity: {fit.parameters['feature_polarity']}")
    print(
        f"Qubit frequency: {fit.center_mhz:.9f} +/- "
        f"{fit.parameters['center_uncertainty_mhz']:.3g} MHz"
    )
    print(f"FWHM: {fit.parameters['fwhm_mhz']:.6g} MHz")
    print(
        f"R^2: {fit.statistics['r_squared']:.6f}; "
        f"contrast SNR: {fit.statistics['contrast_snr']:.3f}; "
        f"RMSE: {fit.statistics['rmse']:.6g}"
    )
    if "detection_basis" in fit.statistics:
        print(
            "Detection: "
            f"{fit.statistics['detection_prominence_snr']:.1f} sigma "
            f"({fit.statistics['detection_basis']}); "
            f"I/Q geometry {fit.statistics['detection_geometry_score']:.2f}; "
            f"{fit.statistics.get('detected_candidates', 1)} candidate(s)"
        )
        per_channel = fit.statistics.get("detection_channel_snr", {})
        if per_channel:
            print(
                "  per channel: "
                + ", ".join(
                    f"{name} {value:.1f} sigma"
                    for name, value in per_channel.items()
                )
            )
    if ENABLE_TWO_FEATURE_SELECTION:
        print(
            "Feature selection: "
            f"{'multiple features' if fit.statistics['multi_feature'] else 'single feature'}; "
            f"Delta BIC(2 vs 1)={fit.statistics['delta_bic_two_vs_one']:.3f}; "
            f"ripple suspected={fit.statistics['ripple_suspected']}"
        )
    gates = fit.acceptance_gates(
        minimum_r_squared=MINIMUM_R_SQUARED,
        minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
        maximum_center_uncertainty_fraction_of_fwhm=(
            MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM
        ),
    )
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    for name, (ok, measured, threshold) in gates.items():
        limit = "" if not np.isfinite(threshold) else f" (limit {threshold:g})"
        print(f"  [{'ok' if ok else 'XX'}] {name}: {measured:.4g}{limit}")
    figure = plot_spectroscopy_fit(fit)
    if WRITE_ACCEPTED_FIT or FORCE_WRITE:
        if FORCE_WRITE and not passes:
            print("WARNING: FORCE_WRITE=True is bypassing failed acceptance gates.")
        calibration_path = accept_spectroscopy_fit(
            PROJECT_ROOT,
            fit,
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
            maximum_center_uncertainty_fraction_of_fwhm=(
                MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM
            ),
            force_write=FORCE_WRITE,
        )
        action = "Force-written" if FORCE_WRITE else "Accepted"
        print(f"{action} q_freq written atomically to {calibration_path}")
    else:
        print(
            "WRITE_ACCEPTED_FIT=False: calibration.yml was not changed. "
            "Inspect the diagnostics, then enable the latch to accept this fit."
        )
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
