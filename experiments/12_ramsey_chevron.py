"""Ramsey chevron: qubit-drive frequency versus idle time."""

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
Q_FREQUENCY_MHZ = np.arange(5601.5, 5611.5, 0.25)
DELAY_US = np.arange(0.0, 5.0, 0.02)
FRINGE_FREQUENCY_MHZ = 0.0
HARD_AVG = 1000
SOFT_AVG = 1
SHOW_PLOT = True
# ============================================================================


def main():
    return run_experiment(
        PROJECT_ROOT,
        experiment="ramsey_chevron",
        preset="ramsey_chevron",
        title=f"{z_tag(Z_GAIN)}_RamseyChevron_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": READOUT_FREQUENCY_MHZ,
            "q_freq": Q_FREQUENCY_MHZ,
            "delay": DELAY_US,
            "fringe_frequency_mhz": FRINGE_FREQUENCY_MHZ,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        analyze=False,
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

