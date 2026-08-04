"""Ramsey dephasing with an artificial fringe frequency."""

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
Q_GAIN = 0.4
Q_LENGTH_US = 0.1
Q_GAIN_2 = 0.2
Q_LENGTH_2_US = 0.1
DELAY_US = np.arange(0.0, 5.0, 0.01)
FRINGE_FREQUENCY_MHZ = 5.0
HARD_AVG = 10000
SOFT_AVG = 5
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
        experiment="ramsey",
        preset="ramsey",
        title=f"{z_tag(Z_GAIN)}_T2Ramsey_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": readout_frequency_mhz,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "q_gain_2": Q_GAIN_2,
            "q_length_2": Q_LENGTH_2_US,
            "delay": DELAY_US,
            "fringe_frequency_mhz": FRINGE_FREQUENCY_MHZ,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
            "rep": 1,
        },
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

