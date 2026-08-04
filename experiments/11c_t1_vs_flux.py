"""Native T1 delay versus finite Z-pulse bias using authored T1_zpa."""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import resolve_readout_frequency, run_experiment
from quickexp_v3.naming import number_tag


# ============================ EDIT THESE ====================================
# Keep false until the authored-program hardware gates have passed.
LIVE_HARDWARE = False
PARKED_Z_GAIN = 0.0
USE_ACCEPTED_RESONATOR_FLUX_FIT = True
READOUT_FREQUENCY_MHZ = 6884.0
Z_GAIN = np.linspace(-0.2, 0.2, 9)
Q_FREQUENCY_MHZ = 5600.0
Q_GAIN = 0.4
Q_LENGTH_US = 0.1
DELAY_US = np.linspace(0.1, 30.0, 61)
READOUT_RELAX_US = 60.0
HARD_AVG = 1000
SOFT_AVG = 1
SHOW_PLOT = True
# ============================================================================


def main():
    readout_frequency_mhz = resolve_readout_frequency(
        PROJECT_ROOT,
        PARKED_Z_GAIN,
        use_accepted_fit=USE_ACCEPTED_RESONATOR_FLUX_FIT,
        fixed_frequency_mhz=READOUT_FREQUENCY_MHZ,
    )
    return run_experiment(
        PROJECT_ROOT,
        experiment="t1_zpa",
        preset="t1_zpa",
        title=f"T1ZPA_r{number_tag(readout_frequency_mhz)}_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        overrides={
            "r_freq": readout_frequency_mhz,
            "r_relax": READOUT_RELAX_US,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "z_gain": Z_GAIN,
            "delay": DELAY_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
            "rep": 1,
        },
        run_options={"population": False},
        analyze=False,
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()
