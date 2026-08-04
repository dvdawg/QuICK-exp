"""Hahn Echo / fixed-cycle CPMG measurement."""

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
# Used only when the accepted-fit option above is False.
READOUT_FREQUENCY_MHZ = 6884.0
Q_FREQUENCY_MHZ = 5600.0
DELAY_US = np.arange(0.0, 50.0, 1.0)
# Quick calls this "extra cycles"; 0 is the notebook's Hahn-Echo setting.
EXTRA_CYCLES = 0
FRINGE_FREQUENCY_MHZ = 0.0
HARD_AVG = 1000
SOFT_AVG = 1
SHOW_PLOT = True
# ============================================================================


def main():
    readout_frequency_mhz = resolve_readout_frequency(
        PROJECT_ROOT,
        Z_GAIN,
        use_accepted_fit=USE_ACCEPTED_RESONATOR_FLUX_FIT,
        fixed_frequency_mhz=READOUT_FREQUENCY_MHZ,
    )
    return run_experiment(
        PROJECT_ROOT,
        experiment="echo",
        preset="echo",
        title=f"{z_tag(Z_GAIN)}_T2Echo_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": readout_frequency_mhz,
            "q_freq": Q_FREQUENCY_MHZ,
            "delay": DELAY_US,
            "pulse_count": EXTRA_CYCLES,
            "fringe_frequency_mhz": FRINGE_FREQUENCY_MHZ,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

