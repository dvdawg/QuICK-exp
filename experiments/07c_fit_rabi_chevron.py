"""Fit a saved Rabi chevron; amplitude chevrons are diagnostic-only."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.chevron_fit import fit_rabi_chevron, plot_rabi_chevron
from quickexp_v3.ide import load_repository
from quickexp_v3.native_index import NativeIndex


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
INPUT_CSV = None
MINIMUM_ROW_R_SQUARED = 0.60
MINIMUM_VALID_COLUMNS = 5
MINIMUM_PARABOLA_R_SQUARED = 0.90
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    root = Path(repository.hardware["storage"]["quick_native_root"])
    return NativeIndex(root).refresh().latest(
        quick_class="Rabi",
        n_axes=2,
    ).csv_path


def main():
    fit = fit_rabi_chevron(
        _input_path(),
        minimum_row_r_squared=MINIMUM_ROW_R_SQUARED,
    )
    passes = fit.passes(
        minimum_valid_columns=MINIMUM_VALID_COLUMNS,
        minimum_r_squared_parabola=MINIMUM_PARABOLA_R_SQUARED,
    )
    print(f"Fit source: {fit.source_csv}")
    print(f"Chevron mode: {fit.mode}")
    print(
        f"Resonance f0={fit.parameters['f0_mhz']:.9f} MHz; "
        f"Omega0={fit.parameters['omega0_mhz']:.6g} MHz; "
        f"pi time={fit.parameters['pi_time_us']:.6g} us"
    )
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    figure = plot_rabi_chevron(fit)
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
