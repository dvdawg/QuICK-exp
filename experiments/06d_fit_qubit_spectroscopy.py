"""Fit a one-dimensional fixed-flux qubit spectroscopy run."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

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
# INPUT_CSV = r"Z:\David\Data\folder\00001 - (QubitSpectroscopy)name.csv"
INPUT_CSV = None

# Choose amplitude, phase, I, Q, or IQ. IQ uses the principal measured axis.
FIT_SIGNAL = "IQ"
# Strongly recommended when a coarse scan contains multiple candidate peaks.
# Example: (5595.0, 5615.0). None fits the full acquired frequency axis.
FIT_WINDOW_MHZ = None
ENABLE_TWO_FEATURE_SELECTION = True

MINIMUM_R_SQUARED = 0.50
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
    if ENABLE_TWO_FEATURE_SELECTION:
        print(
            "Feature selection: "
            f"{'multiple features' if fit.statistics['multi_feature'] else 'single feature'}; "
            f"Delta BIC(2 vs 1)={fit.statistics['delta_bic_two_vs_one']:.3f}; "
            f"ripple suspected={fit.statistics['ripple_suspected']}"
        )
    print(
        "Acceptance gates: "
        f"{'PASS' if passes else 'FAIL'} "
        f"(R^2 >= {MINIMUM_R_SQUARED}, "
        f"contrast SNR >= {MINIMUM_CONTRAST_SNR}, "
        "center uncertainty / FWHM <= "
        f"{MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM})"
    )
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
