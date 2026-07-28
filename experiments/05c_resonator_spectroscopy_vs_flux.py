"""Resonator spectroscopy rows versus held Z bias with row-level runs."""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import run_flux_sweep


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
Z_GAIN = np.linspace(-0.4, 0.4, 21)
R_FREQUENCY_MHZ = np.arange(6875.0, 6890.0, 0.1)
R_POWER_DB = -30.0
R_LENGTH_US = 10.0
HARD_AVG = 1000
SOFT_AVG = 1
SHOW_PLOT = True
# ============================================================================


def main():
    return run_flux_sweep(
        PROJECT_ROOT,
        experiment="resonator_spectroscopy",
        preset="resonator_fine",
        title="ResVsZ_held_bias",
        live_hardware=LIVE_HARDWARE,
        flux_values=Z_GAIN,
        overrides={
            "r_freq": R_FREQUENCY_MHZ,
            "r_power": R_POWER_DB,
            "r_length": R_LENGTH_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULTS = main()

