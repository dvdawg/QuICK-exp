"""Run the paper's long-timescale flux-step spectroscopy with your parameters."""

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

PARAMETERS = LongTimescaleFluxParameters(
    live_hardware=LIVE_HARDWARE,
    # 17a does not control the paper's separate DC source. For a live run, set
    # and read back the resolved baseline first, then release this latch.
    external_baseline_confirmed=False,
    # Default: convert these Phi0 coordinates with the accepted 06e fit.
    # To use local units instead, set BOTH baseline_z and commanded_step_z.
    baseline_phi0=-0.127,
    step_phi0=-0.217,
    step_scale=0.20,
    baseline_z=None,
    commanded_step_z=None,
    # Paper protocol: 70 logarithmic times and a +/-100 MHz moving window.
    probe_times_us=np.geomspace(0.010, 100.0, 70),
    q_frequency_offsets_mhz=np.linspace(-100.0, 100.0, 201),
    # None predicts each center from the accepted 06e fit. You may instead
    # supply one fixed center or one center for every probe time.
    q_frequency_centers_mhz=None,
    fallback_q_frequency_mhz=5200.0,  # Synthetic/offline runs only.
    q_nyquist_zone=2,
    q_gain=0.15,
    q_length_us=0.040,
    # Set False to use readout_frequency_mhz directly.
    use_accepted_resonator_flux_fit=True,
    readout_frequency_mhz=6884.0,
    readout_power_db=-35.0,
    readout_length_us=2.0,
    readout_relax_us=300.0,
    hard_avg=2048,
    soft_avg=1,
    # Keep True for the initial unfiltered pass (paper Appendix F). Set False
    # only after the dominant long-time IIR correction is active.
    first_uncompensated_pass=True,
    bias_tee_tau_us=20.0,
    post_probe_to_return_us=0.070,
    return_to_readout_us=0.110,
    z_idle_gain=0.0,
    z_set_length_us=0.004,
    show_plot=True,
)
# ============================================================================


def main(parameters=None):
    if parameters is None:
        parameters = PARAMETERS
    return run_long_timescale_flux_step(PROJECT_ROOT, parameters)


if __name__ == "__main__":
    RESULTS = main()
