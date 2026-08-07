"""Use the readout resonator as a slow flux sensor for flux-line distortion.

Steps the Z line and probes the resonator at a set of observation times while
the line is still held at the stepped level, then inverts the static f_r(z)
calibration to recover the effective flux trajectory z_hat(t). The tail is fitted
with the same normalized sum of exponentials and matched-z IIR inverse that 17b
uses, so the candidate filter is directly comparable to the pulsed result.

Compared with 17a this trades a full qubit spectrum per observation time for one
short resonator mini-spectrum: minutes instead of hours. What it gives up is
short-time reach. The resonator field time constant tau_r = 1/(pi*FWHM) sets a
floor, and observation times below a few tau_r are masked out of the fit rather
than fitted and believed. Use it for multi-microsecond tails such as bias-tee
droop; use 17c for the nanosecond edge.
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.resonator_transient import (  # noqa: E402
    ResonatorFluxCalibration,
    ResonatorTransientParameters,
    fit_transient_tail,
    invert_transient,
    run_resonator_flux_transient,
)


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False

# --- Static resonator flux calibration --------------------------------------
# Read off the 2026-08 cosine fit of the 5879 MHz resonator. The accepted
# lookups.resonator_vs_flux record in calibration.yml is a DIFFERENT resonator
# (6884 MHz) and must not be used here, so use_accepted_resonator_flux_fit stays
# False until that record is replaced.
#
# Replace these four numbers with the exact fitted parameters rather than the
# values below, which were transcribed from the fit plot and are good to about
# the last digit shown. The campaign prints them back before acquiring.
CALIBRATION = ResonatorFluxCalibration(
    center_frequency_mhz=5879.22,
    amplitude_mhz=0.825,
    period_z=0.36,
    peak_bias_z=-0.12,
    # 493 kHz as fitted, but two rows (z ~ -0.21 and z ~ +0.09) failed peak
    # extraction and dominate it; excluding them the scatter is about 200 kHz.
    # This scales the fitted amplitudes. It does not limit the time constants,
    # because the step response is normalized before fitting.
    rmse_mhz=0.493,
    domain_z=(-0.30, 0.30),
)

# --- Operating point --------------------------------------------------------
# Deliberately the same point as 17a, so this campaign and the pulsed one
# measure the same line at the same bias and can be compared directly.
#
#   z = -0.080 -> f_r = 5879.852 MHz
#   z = -0.065 -> f_r = 5879.693 MHz     excursion -159 kHz
#
# Both endpoints sit to the right of the extremum at z = -0.12, so f_r(z) is
# monotonic across the step. The local slope is 9.3 MHz/z, 64% of the 14.4
# MHz/z maximum.
#
# For more transduction, the steepest in-domain biases are z = -0.21, -0.03 and
# +0.15 (full 14.4 MHz/z). z = -0.03 is the clean one, but it sits outside the
# qubit's characterized [-0.20, -0.05] domain. That is harmless here because
# this measurement never drives the qubit, but it does break comparability with
# 17a. A larger step is the other lever: at z = -0.080, a step of +0.030 gives
# -350 kHz and +0.040 gives -489 kHz, both still on the monotonic branch.
BASELINE_Z = -0.080
STEP_Z = +0.015

PARAMETERS = ResonatorTransientParameters(
    live_hardware=LIVE_HARDWARE,
    baseline_z=BASELINE_Z,
    commanded_step_z=STEP_Z,
    use_accepted_resonator_flux_fit=False,
    calibration=CALIBRATION,
    # The 17a time grid, row for row.
    probe_times_us=np.geomspace(0.025, 100.0, 70),
    # Mini-spectrum at each observation time. Complex S21 is steepest exactly on
    # resonance, so the grid is centred there rather than detuned. Five points
    # over two linewidths make the per-time f_r estimate well conditioned; the
    # baseline and coupling terms are shared with the reference fit, so each
    # time costs only one free parameter.
    transient_probe_points=5,
    transient_span_linewidths=2.0,
    # Wide sweeps that fix the line shape and the settled level.
    reference_probe_points=41,
    reference_span_mhz=5.0,
    reference_hard_avg=3000,
    # Matches 17a. r_relax doubles as the baseline settle: the line rests at
    # BASELINE_Z for its whole duration, so it must exceed several times the
    # longest time constant being measured. Shorten it only after the fit
    # reports the actual poles.
    readout_power_db=-35.0,
    readout_length_us=2.0,
    readout_relax_us=300.0,
    hard_avg=2048,
    soft_avg=1,
    z_set_length_us=0.008,
    # Ignore the first five cavity lifetimes. With a 2 us readout the readout
    # window is the binding constraint unless Q_L is very high.
    cavity_lifetimes_to_mask=5.0,
    # None fits the settled level. This line is DC-coupled -- 17a sets
    # first_uncompensated_pass=False for exactly that reason -- so the step
    # response settles to a finite value rather than drooping to zero. Forcing
    # dc_gain=0.0 here would assert a bias-tee high-pass that this line does not
    # have, and would put an integrator pole at z=1 in the inverse. Use 0.0 only
    # if a bias tee is actually installed.
    fit_dc_gain=None,
    filter_sample_interval_ns=1.669,
    show_plot=True,
)
# ============================================================================


def main(parameters=None):
    if parameters is None:
        parameters = PARAMETERS

    campaign = run_resonator_flux_transient(PROJECT_ROOT, parameters)
    if campaign.reference_csv is None:
        print(
            "\nAcquisition finished, but no native CSV was written, so the "
            "line shape cannot be fitted and the inversion is skipped. This is "
            "expected offline: the synthetic backend models neither the notch "
            "nor the flux response. Set LIVE_HARDWARE = True to invert."
        )
        return campaign, None, None, None

    trace = invert_transient(campaign, parameters)
    print(
        f"\nLinewidth {trace.linewidth_mhz * 1e3:.0f} kHz -> "
        f"tau_r = {trace.cavity_tau_us * 1e3:.0f} ns; "
        f"masking observation times below {trace.mask_threshold_us:.3g} us "
        f"({int(np.count_nonzero(trace.mask))} of {trace.mask.size} rows kept)"
    )
    if np.any(trace.clipped):
        print(
            f"WARNING: {int(np.count_nonzero(trace.clipped))} rows hit the edge "
            "of the monotonic branch. The excursion is leaving the invertible "
            "interval; reduce the step or move the operating point."
        )

    fit, inverse = fit_transient_tail(trace, parameters)
    print(f"\nSelected {fit.model_order} exponentials by BIC:")
    for amplitude, tau in zip(fit.alphas, fit.taus_us):
        print(f"  alpha={amplitude:+.5f}  tau={tau:.5g} us")
    print(
        f"  dc gain {fit.dc_gain:.5g}; inverse is "
        f"{'stable' if inverse.stable else 'marginal' if inverse.marginal else 'UNSTABLE'}"
        f" at {inverse.sample_interval_ns:.4g} ns"
    )
    return campaign, trace, fit, inverse


if __name__ == "__main__":
    RESULTS = main()
