"""One-dimensional resonator frequency sweep at a held Z bias."""

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
MANUAL_CENTER_FREQUENCY_MHZ = 6883.0
R_FREQUENCY_OFFSET_MHZ = np.arange(-5.0, 5.0, 0.1)
R_POWER_DB = -35.0
R_LENGTH_US = 2.0
HARD_AVG = 3000
SOFT_AVG = 1
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
        experiment="resonator_spectroscopy",
        preset="resonator_fine",
        title=f"{z_tag(Z_GAIN)}_resonator_spec",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": r_frequency_mhz,
            "r_power": R_POWER_DB,
            "r_length": R_LENGTH_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

