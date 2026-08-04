"""Native q-frequency versus Z-pulse spectroscopy using authored TwoTone_ZPA."""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import resolve_readout_frequency, run_experiment
from quickexp_v3.naming import number_tag


# ============================ EDIT THESE ====================================
# Keep false until hardware gates G1-G3 in plans/20-pulses.md are completed.
LIVE_HARDWARE = False
PARKED_Z_GAIN = 0.0
USE_ACCEPTED_RESONATOR_FLUX_FIT = True
READOUT_FREQUENCY_MHZ = 6884.0
Z_GAIN = np.linspace(-0.2, 0.2, 21)
Z_LENGTH_US = 20.0
Z_SETTLE_US = 5.0
Q_FREQUENCY_MHZ = np.linspace(5596.5, 5616.5, 101)
Q_GAIN = 0.3
Q_LENGTH_US = 10.0
R_POWER_DB = -35.0
R_LENGTH_US = 2.0
R_RELAX_US = 20.0
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
        experiment="two_tone_zpa",
        preset="two_tone_zpa",
        title=(
            "TwoToneZPA"
            f"_r{number_tag(readout_frequency_mhz)}"
            f"_q{number_tag(Q_FREQUENCY_MHZ)}"
        ),
        live_hardware=LIVE_HARDWARE,
        overrides={
            "r_freq": readout_frequency_mhz,
            "r_power": R_POWER_DB,
            "r_length": R_LENGTH_US,
            "r_relax": R_RELAX_US,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "z_gain": Z_GAIN,
            "z_length": Z_LENGTH_US,
            "z_settle": Z_SETTLE_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
            "rep": 0,
        },
        run_options={"population": False},
        analyze=False,
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()
