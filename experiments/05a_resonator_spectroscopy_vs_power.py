"""Native Quick 2D resonator power-versus-frequency punchout."""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import run_experiment
from quickexp_v3.naming import number_tag, z_tag


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = True
Z_GAIN = 0.0
R_FREQUENCY_MHZ = np.arange(6800.0, 6830.0, 1.0)
R_POWER_DB = np.arange(-40.0, 0.0, 5.0)
R_LENGTH_US = 50.0
HARD_AVG = 2000
SOFT_AVG = 1
SHOW_PLOT = True
# ============================================================================


def main():
    return run_experiment(
        PROJECT_ROOT,
        experiment="resonator_spectroscopy",
        preset="resonator_power",
        title=f"Punchout_{z_tag(Z_GAIN)}_r{number_tag(R_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": R_FREQUENCY_MHZ,
            "r_power": R_POWER_DB,
            "r_length": R_LENGTH_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        analyze=False,
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

