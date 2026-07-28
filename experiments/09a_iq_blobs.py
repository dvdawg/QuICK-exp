"""Ground/excited IQ blobs and readout-threshold/phase calibration."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import run_experiment
from quickexp_v3.naming import number_tag, z_tag


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = True
Z_GAIN = 0.0
READOUT_FREQUENCY_MHZ = 6883.11
Q_FREQUENCY_MHZ = 5606.5
Q_GAIN = 0.4
Q_LENGTH_US = 0.115
SHOTS = 10000
SHOW_PLOT = True
# ============================================================================


def main():
    return run_experiment(
        PROJECT_ROOT,
        experiment="iq_blobs",
        preset="iq_blobs",
        title=f"{z_tag(Z_GAIN)}_IQScatter_q{number_tag(Q_FREQUENCY_MHZ)}",
        live_hardware=LIVE_HARDWARE,
        fixed_z_gain=Z_GAIN,
        overrides={
            "r_freq": READOUT_FREQUENCY_MHZ,
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "shots": SHOTS,
        },
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULT = main()

