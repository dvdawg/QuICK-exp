"""Time Rabi: sweep qubit pulse length at fixed drive gain/frequency."""

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
Z_GAIN = 0.14
Z_LENGTH_US = 0.2
Z_SETTLE_US = 5.0
USE_ACCEPTED_RESONATOR_FLUX_FIT = True
READOUT_FREQUENCY_MHZ = 6884.0 # Used only when the accepted-fit option above is False.
Q_FREQUENCY_MHZ = 5625
Q_GAIN = 0.4
Q_LENGTH_US = np.arange(0.02, 1.70, 0.01)
EXTRA_PI_PULSES = 0
R_POWER_DB = -30.0
R_LENGTH_US = 2.0
R_OFFSET_US = 0.5
R_PHASE_DEG = 0.0
R_RELAX_US = 20.0
HARD_AVG = 500
SOFT_AVG = 1
REP = 1000
POPULATION = False
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
        experiment="rabi",
        preset="rabi_length",
        title=f"{z_tag(Z_GAIN)}_rabi_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": readout_frequency_mhz,
            "r_power": R_POWER_DB,
            "r_length": R_LENGTH_US,
            "r_offset": R_OFFSET_US,
            "r_phase": R_PHASE_DEG,
            "r_relax": R_RELAX_US,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "cycle": EXTRA_PI_PULSES,
            "z_length": Z_LENGTH_US,
            "z_settle": Z_SETTLE_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
            "rep": REP,
        },
        run_options={"population": POPULATION},
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()
