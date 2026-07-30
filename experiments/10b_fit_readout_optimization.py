"""Score a saved dispersive scan by noise-normalized IQ separation."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.iq_gmm import (
    fit_readout_optimization,
    plot_readout_optimization,
)
from quickexp_v3.native_index import NativeIndex


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
INPUT_CSV = None
NOTCH_FREQUENCY_MHZ = None
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    root = Path(repository.hardware["storage"]["quick_native_root"])
    return NativeIndex(root).refresh().latest(
        quick_class="DispersiveSpectroscopy",
        n_axes=1,
    ).csv_path


def main():
    fit = fit_readout_optimization(
        _input_path(),
        notch_frequency_mhz=NOTCH_FREQUENCY_MHZ,
    )
    print(f"Fit source: {fit.source_csv}")
    print(
        f"Readout optimum={fit.optimum_frequency_mhz:.9f} MHz; "
        f"SNR={fit.snr_at_optimum:.6g}"
    )
    if fit.optimum_offset_from_notch_mhz is not None:
        print(
            "Optimum offset from notch="
            f"{fit.optimum_offset_from_notch_mhz:+.6g} MHz"
        )
    figure = plot_readout_optimization(fit)
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
