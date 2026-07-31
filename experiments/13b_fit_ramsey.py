"""Fit a native Quick Ramsey run and report T2* and q-frequency estimates."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.native_fit import (
    accept_ramsey_fit,
    find_latest_native,
    fit_ramsey,
    plot_ramsey_fit,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# None selects the newest one-dimensional T2Ramsey CSV/YML pair.
# To fit a particular run, paste its CSV path as a raw string:
# INPUT_CSV = r"Z:\David\Data\folder\00001 - (T2Ramsey)name.csv"
INPUT_CSV = None

# Choose amplitude, phase, I, Q, or IQ.
FIT_SIGNAL = "IQ"

MINIMUM_R_SQUARED = 0.70
MINIMUM_OSCILLATIONS = 1.0
MAXIMUM_RELATIVE_T2_UNCERTAINTY = 0.30

# WRITE_ACCEPTED_FIT saves records.derived.t2_ramsey.
WRITE_ACCEPTED_FIT = False
# Independent manual override: writes even when acceptance gates fail.
FORCE_WRITE = False
# This separate latch additionally replaces records.defaults.q_freq.
UPDATE_Q_FREQUENCY = False
# +1 is the Quick/notebook convention; -1 selects the reported alternative.
# A scalar Ramsey trace cannot prove this sign. Confirm it with another run.
Q_FREQUENCY_CORRECTION_SIGN = +1
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
        quick_class="T2Ramsey",
        axis_text="delay time",
    )


def main():
    source = _input_path()
    fit = fit_ramsey(source, signal=FIT_SIGNAL)
    passes = fit.passes(
        minimum_r_squared=MINIMUM_R_SQUARED,
        minimum_oscillations=MINIMUM_OSCILLATIONS,
        maximum_relative_t2_uncertainty=(
            MAXIMUM_RELATIVE_T2_UNCERTAINTY
        ),
    )
    parameters = fit.parameters
    print(f"Fit source: {fit.source_csv}")
    print(f"Fit signal: {fit.signal_label}")
    print(
        f"T2*: {fit.t2_star_us:.9g} +/- "
        f"{parameters['t2_star_uncertainty_us']:.3g} us"
    )
    print(
        f"Fitted fringe: {parameters['fitted_fringe_mhz']:.9g} +/- "
        f"{parameters['fitted_fringe_uncertainty_mhz']:.3g} MHz; "
        f"programmed fringe: {parameters['programmed_fringe_mhz']:.9g} MHz"
    )
    print(
        f"Current q_freq: {parameters['drive_frequency_mhz']:.9f} MHz; "
        f"detuning magnitude/sign from fit: "
        f"{parameters['detuning_mhz']:+.9g} MHz"
    )
    print(
        "Quick-convention q_freq estimate: "
        f"{parameters['predicted_drive_mhz']:.9f} MHz; "
        "sign-check alternative: "
        f"{parameters['alternate_drive_mhz']:.9f} MHz"
    )
    print(
        f"R^2: {fit.statistics['r_squared']:.6f}; "
        f"oscillations: {fit.statistics['oscillations']:.3f}; "
        "relative T2* uncertainty: "
        f"{fit.statistics['relative_t2_uncertainty']:.2%}"
    )
    print(
        "Acceptance gates: "
        f"{'PASS' if passes else 'FAIL'} "
        f"(R^2 >= {MINIMUM_R_SQUARED}, "
        f"oscillations >= {MINIMUM_OSCILLATIONS}, "
        "relative T2* uncertainty <= "
        f"{MAXIMUM_RELATIVE_T2_UNCERTAINTY:.1%})"
    )
    figure = plot_ramsey_fit(fit)
    if WRITE_ACCEPTED_FIT or FORCE_WRITE:
        if FORCE_WRITE and not passes:
            print("WARNING: FORCE_WRITE=True is bypassing failed acceptance gates.")
        calibration_path = accept_ramsey_fit(
            PROJECT_ROOT,
            fit,
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_oscillations=MINIMUM_OSCILLATIONS,
            maximum_relative_t2_uncertainty=(
                MAXIMUM_RELATIVE_T2_UNCERTAINTY
            ),
            update_q_frequency=UPDATE_Q_FREQUENCY,
            correction_sign=Q_FREQUENCY_CORRECTION_SIGN,
            force_write=FORCE_WRITE,
        )
        action = "Force-written" if FORCE_WRITE else "Accepted"
        print(f"{action} Ramsey result written atomically to {calibration_path}")
    else:
        print(
            "WRITE_ACCEPTED_FIT=False: calibration.yml was not changed. "
            "Inspect the diagnostics, then enable the latch to save this fit."
        )
    if UPDATE_Q_FREQUENCY and not (WRITE_ACCEPTED_FIT or FORCE_WRITE):
        print(
            "UPDATE_Q_FREQUENCY=True has no effect until "
            "WRITE_ACCEPTED_FIT or FORCE_WRITE is also True."
        )
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
