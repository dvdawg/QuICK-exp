"""Optimize readout frequency from ground/excited resonator separation."""

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
R_FREQUENCY_MHZ = np.arange(6873.36, 6893.36, 0.02)
R_POWER_DB = -35.0
R_LENGTH_US = 1.0
Q_FREQUENCY_MHZ = 5603.910
Q_GAIN = 0.4
Q_LENGTH_US = 0.115
HARD_AVG = 5000
SOFT_AVG = 10
SHOW_PLOT = True
# ============================================================================


def main():
    return run_experiment(
        PROJECT_ROOT,
        experiment="dispersive_spectroscopy",
        preset="dispersive",
        title=f"{z_tag(Z_GAIN)}_dispersive_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": R_FREQUENCY_MHZ,
            "r_power": R_POWER_DB,
            "r_length": R_LENGTH_US,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

