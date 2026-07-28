"""High-power q_gain-versus-frequency search for the |0> to |2> transition."""

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
READOUT_FREQUENCY_MHZ = 6883.110
Q_FREQUENCY_MHZ = np.arange(5440.0, 5510.0001, 0.10)
Q_GAIN = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
Q_LENGTH_US = 5.0
HARD_AVG = 2000
SOFT_AVG = 2
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
        experiment="qubit_spectroscopy",
        preset="qubit_fine",
        title=f"{z_tag(Z_GAIN)}_two_photon_search_r{number_tag(readout_frequency_mhz)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": readout_frequency_mhz,
            "r_power": -30.0,
            "r_length": 5.0,
            "r_relax": 30.0,
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
