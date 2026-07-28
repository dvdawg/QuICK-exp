"""Qubit spectroscopy versus held Z with optional fitted readout tracking."""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import (
    load_repository,
    resonator_frequency_from_flux,
    run_flux_sweep,
)
from quickexp_v3.naming import number_tag


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = True
Z_GAIN = np.linspace(-0.2, 0.2, 41)
Q_FREQUENCY_MHZ = np.arange(4000.0, 6000.0, 1.0)
Q_GAIN = 0.3
Q_LENGTH_US = 10.0
TRACK_READOUT_FROM_ACCEPTED_FLUX_FIT = True
FIXED_READOUT_FREQUENCY_MHZ = 6883.11
HARD_AVG = 5000
SOFT_AVG = 5
SHOW_PLOT = True
# ============================================================================


def main():
    if TRACK_READOUT_FROM_ACCEPTED_FLUX_FIT:
        repository = load_repository(PROJECT_ROOT)
        readout = lambda z: float(resonator_frequency_from_flux(repository, z))
        readout_metadata = repository.calibration["records"]["lookups"][
            "resonator_vs_flux"
        ]
        title = "QubitSpecVsZ_fitted_readout"
    else:
        readout = lambda z: FIXED_READOUT_FREQUENCY_MHZ
        readout_metadata = {
            "model": "fixed",
            "frequency_mhz": float(FIXED_READOUT_FREQUENCY_MHZ),
        }
        title = (
            "QubitSpecVsZ_fixed_readout_r"
            + number_tag(FIXED_READOUT_FREQUENCY_MHZ)
        )
    return run_flux_sweep(
        PROJECT_ROOT,
        experiment="qubit_spectroscopy",
        preset="qubit_fine",
        title=title,
        live_hardware=LIVE_HARDWARE,
        flux_values=Z_GAIN,
        readout_frequency=readout,
        readout_metadata=readout_metadata,
        overrides={
            "q_freq": Q_FREQUENCY_MHZ,
            "q_gain": Q_GAIN,
            "q_length": Q_LENGTH_US,
            "hard_avg": HARD_AVG,
            "soft_avg": SOFT_AVG,
        },
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    RESULTS = main()

