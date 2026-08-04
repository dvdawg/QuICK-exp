"""Fit a native Quick T1 run and optionally save the derived lifetime."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.native_fit import (
    accept_t1_fit,
    find_latest_native,
    fit_t1,
    plot_t1_fit,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# None selects the newest one-dimensional T1 CSV/YML pair.
# To fit a particular run, paste its CSV path as a raw string:
# INPUT_CSV = r"Z:\Your\Data\folder\00001 - (T1)name.csv"
INPUT_CSV = None

# Choose amplitude, phase, I, Q, or IQ. IQ follows the relaxation direction.
FIT_SIGNAL = "IQ"

MINIMUM_R_SQUARED = 0.70
MINIMUM_SPAN_OVER_T1 = 0.75
MAXIMUM_RELATIVE_T1_UNCERTAINTY = 0.25

# Safety latch: writes records.derived.t1, not a pulse parameter.
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
        quick_class="T1",
        axis_text="delay time",
    )


def main():
    source = _input_path()
    fit = fit_t1(source, signal=FIT_SIGNAL)
    passes = fit.passes(
        minimum_r_squared=MINIMUM_R_SQUARED,
        minimum_span_over_t1=MINIMUM_SPAN_OVER_T1,
        maximum_relative_t1_uncertainty=(
            MAXIMUM_RELATIVE_T1_UNCERTAINTY
        ),
    )
    print(f"Fit source: {fit.source_csv}")
    print(f"Fit signal: {fit.signal_label}")
    print(
        f"T1: {fit.t1_us:.9g} +/- "
        f"{fit.parameters['t1_uncertainty_us']:.3g} us"
    )
    print(
        f"R^2: {fit.statistics['r_squared']:.6f}; "
        f"RMSE: {fit.statistics['rmse']:.6g}; "
        f"measured span / T1: {fit.statistics['span_over_t1']:.3f}; "
        "relative T1 uncertainty: "
        f"{fit.statistics['relative_t1_uncertainty']:.2%}"
    )
    print(
        "Acceptance gates: "
        f"{'PASS' if passes else 'FAIL'} "
        f"(R^2 >= {MINIMUM_R_SQUARED}, "
        f"span / T1 >= {MINIMUM_SPAN_OVER_T1}, "
        "relative uncertainty <= "
        f"{MAXIMUM_RELATIVE_T1_UNCERTAINTY:.1%})"
    )
    figure = plot_t1_fit(fit)
    if WRITE_ACCEPTED_FIT or FORCE_WRITE:
        if FORCE_WRITE and not passes:
            print("WARNING: FORCE_WRITE=True is bypassing failed acceptance gates.")
        calibration_path = accept_t1_fit(
            PROJECT_ROOT,
            fit,
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_span_over_t1=MINIMUM_SPAN_OVER_T1,
            maximum_relative_t1_uncertainty=(
                MAXIMUM_RELATIVE_T1_UNCERTAINTY
            ),
            force_write=FORCE_WRITE,
        )
        action = "Force-written" if FORCE_WRITE else "Accepted"
        print(f"{action} T1 written atomically to {calibration_path}")
    else:
        print(
            "WRITE_ACCEPTED_FIT=False: calibration.yml was not changed. "
            "Inspect the diagnostics, then enable the latch to save this fit."
        )
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
