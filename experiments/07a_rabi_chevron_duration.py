"""Rabi chevron: drive frequency versus pulse duration."""

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
READOUT_FREQUENCY_MHZ = 6883.11
Q_FREQUENCY_MHZ = np.arange(5593.91, 5613.91, 0.25)
Q_GAIN = 0.4
Q_LENGTH_US = np.arange(0.02, 1.0, 0.02)
HARD_AVG = 1000
SOFT_AVG = 1
SHOW_PLOT = True
# ============================================================================


def main():
    return run_experiment(
        PROJECT_ROOT,
        experiment="rabi_chevron",
        preset="rabi_chevron_duration",
        title=f"RabiChevronDuration_{z_tag(Z_GAIN)}_r{number_tag(READOUT_FREQUENCY_MHZ)}_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": READOUT_FREQUENCY_MHZ,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        analyze=False,
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

