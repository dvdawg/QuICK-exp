"""Optimize readout frequency from ground/excited resonator separation."""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import resolve_readout_frequency, run_experiment
from quickexp_v3.naming import number_tag, z_tag


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = True
Z_GAIN = 0.0
USE_ACCEPTED_RESONATOR_FLUX_FIT = True
MANUAL_CENTER_FREQUENCY_MHZ = 6883.36
R_FREQUENCY_OFFSET_MHZ = np.arange(-10.0, 10.0, 0.02)
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
    center_frequency_mhz = resolve_readout_frequency(
        PROJECT_ROOT,
        Z_GAIN,
        use_accepted_fit=USE_ACCEPTED_RESONATOR_FLUX_FIT,
        fixed_frequency_mhz=MANUAL_CENTER_FREQUENCY_MHZ,
    )
    r_frequency_mhz = center_frequency_mhz + R_FREQUENCY_OFFSET_MHZ
    return run_experiment(
        PROJECT_ROOT,
        experiment="dispersive_spectroscopy",
        preset="dispersive",
        title=f"{z_tag(Z_GAIN)}_dispersive_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": r_frequency_mhz,
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

