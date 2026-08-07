"""Long-timescale flux-step spectroscopy on this installation's Z line.

Steps the flux line, holds it, and measures the qubit transition at 70
logarithmic observation times. 17b fits the decay of that step and designs the
inverse IIR. Everything is expressed in this installation's own Z-gain units
and read out through the accepted flux fits; nothing is converted from the
reference paper's flux quanta.
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.flux_step_campaign import (  # noqa: E402
    LongTimescaleFluxParameters,
    run_long_timescale_flux_step,
)


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False

# Operating point, chosen from the accepted 06e arch
# (f_max 4829 MHz, sweet spot Z=-0.1354, period 0.447):
#
#   baseline Z=-0.080   f01 ~ 4644 MHz   0.055 from the sweet spot
#   stepped  Z=-0.065   f01 ~ 4529 MHz   0.070 from the sweet spot
#
# Both sit on one monotonic branch, well clear of the sweet spot where the
# qubit stops responding to flux, and inside the measured [-0.20, -0.05] domain
# with margin at each end. The step is 1.9% of the +/-0.8 DAC range and moves
# the qubit ~115 MHz, which is a deliberately small commissioning amplitude:
# confirm the response scales linearly before increasing it.
BASELINE_Z = -0.080
STEP_Z = +0.015

PARAMETERS = LongTimescaleFluxParameters(
    live_hardware=LIVE_HARDWARE,
    # One DC-coupled RF line (z = DAC15) both holds the operating point and
    # delivers the step, so every Z command below is an absolute level and
    # there is no external DC source to set.
    baseline_on_fast_line=True,
    baseline_z=BASELINE_Z,
    commanded_step_z=STEP_Z,
    # 70 logarithmic times. The 25 ns start is set by the 430.08 MHz tProc
    # timing clock: below it, adjacent logarithmic points land on the same
    # 2.33 ns tick. Shorter timescales belong to 17c.
    probe_times_us=np.geomspace(0.025, 100.0, 70),
    # None places every window from the accepted qubit_vs_flux fit. +/-125 MHz
    # exceeds the 115 MHz excursion, so the line stays in frame at every probe
    # time even where the placement model is wrong. 201 points is 1.25 MHz,
    # far finer than the tens-of-MHz linewidth a 40 ns Gaussian probe produces.
    q_frequency_offsets_mhz=np.linspace(-125.0, 125.0, 201),
    q_frequency_centers_mhz=None,
    fallback_q_frequency_mhz=4644.0,  # Synthetic/offline runs only.
    # 4404-4769 MHz stays below the 4792.32 MHz first-zone edge of DAC1.
    q_nyquist_zone=1,
    q_gain=0.15,
    q_length_us=0.040,
    # Readout frequency comes from the accepted resonator_vs_flux fit evaluated
    # at the baseline, where the line rests during every readout.
    use_accepted_resonator_flux_fit=True,
    readout_frequency_mhz=5879.9,  # Only used if the fit is switched off.
    readout_power_db=-35.0,
    readout_length_us=2.0,
    # Long enough for qubit reset and for the line to settle after the longest
    # 100 us hold. Shorten it once 17b has reported the actual time constants.
    readout_relax_us=300.0,
    hard_avg=2048,
    soft_avg=1,
    # No bias tee on a DC-coupled line, so there is no droop to cancel during
    # readout; the line simply returns to the baseline level.
    first_uncompensated_pass=False,
    post_probe_to_return_us=0.070,
    return_to_readout_us=0.110,
    # Driven from baseline_z while baseline_on_fast_line is set.
    z_idle_gain=0.0,
    # Three-plus clocks on the Z generator's 430.08 MHz fabric (6.98 ns).
    z_set_length_us=0.008,
    show_plot=True,
)
# ============================================================================


def main(parameters=None):
    if parameters is None:
        parameters = PARAMETERS
    return run_long_timescale_flux_step(PROJECT_ROOT, parameters)


if __name__ == "__main__":
    RESULTS = main()
