"""Fit a saved Ramsey chevron and determine the detuning-sign convention."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.chevron_fit import fit_ramsey_chevron, plot_ramsey_chevron
from quickexp_v3.ide import load_repository
from quickexp_v3.native_index import NativeIndex


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
INPUT_CSV = None
ARTIFICIAL_FRINGE_MHZ = None
EXPECTED_Q_FREQUENCY_MHZ = None
MINIMUM_ROW_R_SQUARED = 0.60
MINIMUM_VALID_COLUMNS = 5
MAXIMUM_VERTEX_UNCERTAINTY_MHZ = 0.50
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    root = Path(repository.hardware["storage"]["quick_native_root"])
    return NativeIndex(root).refresh().latest(
        quick_class="T2Ramsey",
        n_axes=2,
    ).csv_path


def main():
    fit = fit_ramsey_chevron(
        _input_path(),
        artificial_fringe_mhz=ARTIFICIAL_FRINGE_MHZ,
        expected_q_frequency_mhz=EXPECTED_Q_FREQUENCY_MHZ,
        minimum_row_r_squared=MINIMUM_ROW_R_SQUARED,
    )
    passes = fit.passes(
        minimum_valid_columns=MINIMUM_VALID_COLUMNS,
        maximum_vertex_uncertainty_mhz=MAXIMUM_VERTEX_UNCERTAINTY_MHZ,
    )
    print(f"Fit source: {fit.source_csv}")
    print(
        f"Qubit frequency={fit.qubit_frequency_mhz:.9f} +/- "
        f"{fit.parameters['qubit_frequency_uncertainty_mhz']:.3g} MHz; "
        "detuning sign="
        f"{fit.parameters['detuning_sign_convention']:+d}"
    )
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    figure = plot_ramsey_chevron(fit)
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
