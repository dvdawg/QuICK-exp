"""Fit a native Quick loopback edge and optionally accept r_offset."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.native_fit import (
    accept_loopback_fit,
    find_latest_native,
    fit_loopback,
    plot_loopback_fit,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# None selects the newest one-dimensional LoopBack CSV/YML pair.
# To fit a particular run, paste its CSV path as a raw string:
# INPUT_CSV = r"Z:\Your\Data\folder\00001 - (LoopBack)name.csv"
INPUT_CSV = None

# Run 02 with READOUT_OFFSET_US = 0 before fitting so the rising pulse edge is
# inside the ADC record. An already aligned trace may have its edge clipped.
SMOOTH_SIGMA_BINS = 5.0
MINIMUM_EDGE_SNR = 5.0
MINIMUM_R_SQUARED = 0.85
MAXIMUM_EDGE_UNCERTAINTY_US = 0.02

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
        quick_class="LoopBack",
        axis_text="time",
    )


def main():
    source = _input_path()
    fit = fit_loopback(source, smooth_sigma_bins=SMOOTH_SIGMA_BINS)
    passes = fit.passes(
        minimum_edge_snr=MINIMUM_EDGE_SNR,
        minimum_r_squared=MINIMUM_R_SQUARED,
        maximum_edge_uncertainty_us=MAXIMUM_EDGE_UNCERTAINTY_US,
    )
    print(f"Fit source: {fit.source_csv}")
    print(
        f"Edge in ADC trace: {fit.parameters['edge_in_trace_us']:.9g} "
        f"+/- {fit.parameters['edge_uncertainty_us']:.3g} us"
    )
    print(
        f"Current r_offset: {fit.parameters['current_r_offset_us']:.9g} us; "
        f"recommended r_offset: {fit.recommended_r_offset_us:.9g} us"
    )
    print(
        f"10-90% rise time: "
        f"{fit.parameters['rise_10_to_90_us']:.6g} us; "
        f"edge SNR: {fit.statistics['edge_snr']:.3f}; "
        f"R^2: {fit.statistics['r_squared']:.6f}"
    )
    print(
        "Acceptance gates: "
        f"{'PASS' if passes else 'FAIL'} "
        f"(edge SNR >= {MINIMUM_EDGE_SNR}, "
        f"R^2 >= {MINIMUM_R_SQUARED}, "
        f"edge uncertainty <= {MAXIMUM_EDGE_UNCERTAINTY_US} us)"
    )
    figure = plot_loopback_fit(fit)
    if WRITE_ACCEPTED_FIT or FORCE_WRITE:
        if FORCE_WRITE and not passes:
            print("WARNING: FORCE_WRITE=True is bypassing failed acceptance gates.")
        calibration_path = accept_loopback_fit(
            PROJECT_ROOT,
            fit,
            minimum_edge_snr=MINIMUM_EDGE_SNR,
            minimum_r_squared=MINIMUM_R_SQUARED,
            maximum_edge_uncertainty_us=MAXIMUM_EDGE_UNCERTAINTY_US,
            force_write=FORCE_WRITE,
        )
        action = "Force-written" if FORCE_WRITE else "Accepted"
        print(f"{action} r_offset written atomically to {calibration_path}")
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
