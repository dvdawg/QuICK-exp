"""Fit a native readout-power punchout map and optionally accept r_power."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.native_index import NativeIndex
from quickexp_v3.punchout_fit import (
    accept_punchout_fit,
    fit_punchout,
    plot_punchout_fit,
)


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
INPUT_CSV = None
SMOOTH_SIGMA_BINS = 1.5
PRIOR_LINEWIDTH_MHZ = 2.0
MINIMUM_PLATEAU_ROWS = 2
MINIMUM_SHIFT_OVER_STEP = 2.0
MAXIMUM_TRANSITION_WIDTH_DB = 15.0
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
    return NativeIndex(data_directory).refresh().latest(
        quick_class="ResonatorSpectroscopy",
        n_axes=2,
    ).csv_path


def main():
    fit = fit_punchout(
        _input_path(),
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
        prior_linewidth_mhz=PRIOR_LINEWIDTH_MHZ,
    )
    print(f"Fit source: {fit.source_csv}")
    print(f"Punchout verdict: {fit.status}")
    if fit.status == "resolved":
        print(
            f"Low-power center: {fit.parameters['f_low_mhz']:.6f} MHz; "
            f"bare center: {fit.parameters['f_bare_mhz']:.6f} MHz; "
            f"shift: {fit.parameters['punchout_shift_mhz']:.6f} MHz"
        )
        print(
            f"Transition: {fit.parameters['transition_power_db']:.3f} dB "
            f"(width {fit.parameters['transition_width_db']:.3f} dB); "
            f"recommended r_power={fit.parameters['recommended_r_power_db']:.3f} dB"
        )
    else:
        print(
            "Insufficient resolution. Recommended maximum frequency step: "
            f"{fit.recommendation['frequency_step_mhz_max']:.6g} MHz. "
            f"{fit.recommendation['power_scan']}"
        )
    passes = fit.passes(
        minimum_plateau_rows=MINIMUM_PLATEAU_ROWS,
        minimum_shift_over_step=MINIMUM_SHIFT_OVER_STEP,
        maximum_transition_width_db=MAXIMUM_TRANSITION_WIDTH_DB,
    )
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    figure = plot_punchout_fit(fit)
    if WRITE_ACCEPTED_FIT:
        path = accept_punchout_fit(
            PROJECT_ROOT,
            fit,
            minimum_plateau_rows=MINIMUM_PLATEAU_ROWS,
            minimum_shift_over_step=MINIMUM_SHIFT_OVER_STEP,
            maximum_transition_width_db=MAXIMUM_TRANSITION_WIDTH_DB,
        )
        print(f"Accepted r_power written atomically to {path}")
    else:
        print("WRITE_ACCEPTED_FIT=False: calibration.yml was not changed.")
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
