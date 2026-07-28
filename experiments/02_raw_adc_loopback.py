"""Raw/decimated loopback used to establish the readout trigger offset."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import run_experiment
from quickexp_v3.naming import number_tag, z_tag


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = True
READOUT_FREQUENCY_MHZ = 5500.0
READOUT_POWER_DB = 0.0
READOUT_LENGTH_US = 10.0
READOUT_OFFSET_US = 0
SHOW_PLOT = True
# ============================================================================


def main():
    return run_experiment(
        PROJECT_ROOT,
        experiment="loopback",
        preset="loopback",
        title=f"readout_loopback_r{number_tag(READOUT_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        overrides={
            "r_freq": READOUT_FREQUENCY_MHZ,
            "r_power": READOUT_POWER_DB,
            "r_length": READOUT_LENGTH_US,
            "r_offset": READOUT_OFFSET_US,
            "soft_avg": 100,
        },
        fixed_z_gain=None,
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

