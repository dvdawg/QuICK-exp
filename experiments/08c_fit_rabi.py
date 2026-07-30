"""Fit a native Quick Rabi run and optionally accept its pi-pulse value."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.fit_stats import pi_consistency
from quickexp_v3.rabi_fit import (
    accept_rabi_fit,
    find_latest_rabi,
    fit_rabi,
    plot_rabi_fit,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# None selects the newest matching native Quick CSV/YML pair.
INPUT_CSV = r"Z:\David\Data\2026-07-21_MET_ver191_qubit3\00073 - (Rabi)Zp0p1400_rabi_q3959p000.csv"
# q_length fits Time Rabi; q_gain fits Power Rabi; auto uses the newest of both.
FIT_VARIABLE = "q_length"

MINIMUM_R_SQUARED = 0.70
MINIMUM_OSCILLATIONS = 1.0
MAXIMUM_RELATIVE_PI_UNCERTAINTY = 0.25

# Safety latch: inspect the plot/statistics first, then change this to True.
WRITE_ACCEPTED_FIT = False
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    data_directory = Path(
        repository.hardware["storage"]["quick_native_root"]
    )
    return find_latest_rabi(
        data_directory,
        variable=FIT_VARIABLE,
    )


def main():
    source = _input_path()
    print(source)
    fit = fit_rabi(source, variable=FIT_VARIABLE)
    consistency = pi_consistency(fit)
    passes = fit.passes(
        minimum_r_squared=MINIMUM_R_SQUARED,
        minimum_oscillations=MINIMUM_OSCILLATIONS,
        maximum_relative_pi_uncertainty=MAXIMUM_RELATIVE_PI_UNCERTAINTY,
    )

    print(f"Fit source: {fit.source_csv}")
    print(f"Rabi variable: {fit.variable} ({fit.unit})")
    print(
        f"Pi recommendation: {fit.pi_value:.9g} {fit.unit} "
        f"+/- {fit.parameters['pi_uncertainty']:.3g} {fit.unit}"
    )
    print(
        f"Half period: {fit.parameters['half_period']:.9g} {fit.unit}; "
        f"frequency: {fit.parameters['frequency']:.9g} /{fit.unit}"
    )
    print(
        f"R^2: {fit.statistics['r_squared']:.6f}; "
        f"RMSE: {fit.statistics['rmse']:.6g}; "
        f"oscillations: {fit.statistics['oscillations']:.3f}; "
        "relative pi uncertainty: "
        f"{fit.statistics['relative_pi_uncertainty']:.2%}"
    )
    print(
        "Pi consistency: measured contrast "
        f"{consistency['measured_contrast_at_pi']:.3f}; "
        "pi/half-period "
        f"{consistency['pi_to_half_period_ratio']:.3f}; "
        f"odd-multiple check={'PASS' if consistency['odd_multiple_consistent'] else 'FAIL'}"
    )
    print(
        "Acceptance gates: "
        f"{'PASS' if passes else 'FAIL'} "
        f"(R^2 >= {MINIMUM_R_SQUARED}, "
        f"oscillations >= {MINIMUM_OSCILLATIONS}, "
        "relative pi uncertainty <= "
        f"{MAXIMUM_RELATIVE_PI_UNCERTAINTY:.1%})"
    )

    figure = plot_rabi_fit(fit)
    if WRITE_ACCEPTED_FIT:
        calibration_path = accept_rabi_fit(
            PROJECT_ROOT,
            fit,
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_oscillations=MINIMUM_OSCILLATIONS,
            maximum_relative_pi_uncertainty=(
                MAXIMUM_RELATIVE_PI_UNCERTAINTY
            ),
        )
        print(
            f"Accepted {fit.variable} written atomically to "
            f"{calibration_path}"
        )
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
