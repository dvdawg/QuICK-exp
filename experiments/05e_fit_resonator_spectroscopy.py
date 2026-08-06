"""Fit a one-dimensional fixed-flux resonator spectroscopy run."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.notch_fit import (
    accept_notch_fit,
    fit_notch,
    plot_notch_fit,
)
from quickexp_v3.native_fit import (
    accept_spectroscopy_fit,
    find_latest_native,
    fit_spectroscopy,
    plot_spectroscopy_fit,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# None selects the newest one-dimensional ResonatorSpectroscopy CSV/YML pair.
# To fit a particular run, paste its CSV path as a raw string:
# INPUT_CSV = r"Z:\Your\Data\folder\00001 - (ResonatorSpectroscopy)name.csv"
INPUT_CSV = None

# Only used when USE_COMPLEX_NOTCH is False. Choose amplitude, phase, I, Q, or
# IQ. Amplitude is the right choice for a resonator: crossing a notch sweeps
# the I/Q point around a circle, so any single projection of it mixes the
# absorptive and dispersive quadratures into an asymmetric shape that a
# symmetric Lorentzian cannot describe. |S21| is the absorptive part alone.
FIT_SIGNAL = "amplitude"
# Optional inclusive frequency window, for example (6882.0, 6886.0).
FIT_WINDOW_MHZ = None

USE_COMPLEX_NOTCH = True
MINIMUM_R_SQUARED = 0.90
MINIMUM_CONTRAST_SNR = 5.0
MINIMUM_EDGE_DISTANCE_OVER_FWHM = 1.0
MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM = 0.25

# Safety latch: inspect the plot/statistics first, then change this to True.
WRITE_ACCEPTED_FIT = False
# Independent manual override: writes even when acceptance gates fail.
FORCE_WRITE = False
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    data_directory = Path(
        repository.hardware["storage"]["quick_native_root"]
    )
    return find_latest_native(
        data_directory,
        quick_class="ResonatorSpectroscopy",
        axis_text="readout pulse frequency",
    )


def main():
    source = _input_path()
    if USE_COMPLEX_NOTCH:
        fit = fit_notch(source)
        passes = fit.passes(
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
            minimum_edge_distance_over_fwhm=(
                MINIMUM_EDGE_DISTANCE_OVER_FWHM
            ),
        )
    else:
        fit = fit_spectroscopy(
            source,
            kind="resonator",
            signal=FIT_SIGNAL,
            window_mhz=FIT_WINDOW_MHZ,
        )
        passes = fit.passes(
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
            maximum_center_uncertainty_fraction_of_fwhm=(
                MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM
            ),
        )
    print(f"Fit source: {fit.source_csv}")
    print(
        f"Fit model: {fit.model}"
        if USE_COMPLEX_NOTCH
        else f"Fit signal: {fit.signal_label}"
    )
    print(f"Detected feature polarity: {fit.parameters['feature_polarity']}")
    print(
        f"Resonator frequency: {fit.center_mhz:.9f} +/- "
        f"{fit.parameters['center_uncertainty_mhz']:.3g} MHz"
    )
    print(
        f"FWHM: {fit.parameters['fwhm_mhz']:.6g} MHz; "
        f"loaded Q: {fit.parameters['loaded_q']:.6g}"
    )
    print(
        f"R^2: "
        f"{fit.statistics.get('r_squared_complex', fit.statistics.get('r_squared')):.6f}; "
        f"contrast SNR: {fit.statistics['contrast_snr']:.3f}; "
        f"RMSE: {fit.statistics['rmse']:.6g}"
    )
    if USE_COMPLEX_NOTCH:
        gates = fit.acceptance_gates(
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
            minimum_edge_distance_over_fwhm=(
                MINIMUM_EDGE_DISTANCE_OVER_FWHM
            ),
        )
    else:
        gates = fit.acceptance_gates(
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
            maximum_center_uncertainty_fraction_of_fwhm=(
                MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM
            ),
        )
        if "detection_basis" in fit.statistics:
            print(
                "Detection: "
                f"{fit.statistics['detection_prominence_snr']:.1f} sigma "
                f"({fit.statistics['detection_basis']}); "
                f"I/Q arc {fit.statistics['detection_geometry_score']:.2f}; "
                f"cable delay {fit.statistics['delay_ns']:.1f} ns"
                + (" removed" if fit.statistics["delay_applied"] else " not removed")
            )
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    for name, (ok, measured, threshold) in gates.items():
        limit = "" if not np.isfinite(threshold) else f" (limit {threshold:g})"
        print(f"  [{'ok' if ok else 'XX'}] {name}: {measured:.4g}{limit}")
    figure = (
        plot_notch_fit(fit)
        if USE_COMPLEX_NOTCH
        else plot_spectroscopy_fit(fit)
    )
    if WRITE_ACCEPTED_FIT or FORCE_WRITE:
        if FORCE_WRITE and not passes:
            print("WARNING: FORCE_WRITE=True is bypassing failed acceptance gates.")
        if USE_COMPLEX_NOTCH:
            calibration_path = accept_notch_fit(
                PROJECT_ROOT,
                fit,
                minimum_r_squared=MINIMUM_R_SQUARED,
                minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
                minimum_edge_distance_over_fwhm=(
                    MINIMUM_EDGE_DISTANCE_OVER_FWHM
                ),
                force_write=FORCE_WRITE,
            )
        else:
            calibration_path = accept_spectroscopy_fit(
                PROJECT_ROOT,
                fit,
                minimum_r_squared=MINIMUM_R_SQUARED,
                minimum_contrast_snr=MINIMUM_CONTRAST_SNR,
                maximum_center_uncertainty_fraction_of_fwhm=(
                    MAXIMUM_CENTER_UNCERTAINTY_FRACTION_OF_FWHM
                ),
                force_write=FORCE_WRITE,
            )
        action = "Force-written" if FORCE_WRITE else "Accepted"
        print(f"{action} r_freq written atomically to {calibration_path}")
    else:
        print(
            "WRITE_ACCEPTED_FIT=False: calibration.yml was not changed. "
            "Inspect the diagnostics, then enable the latch to accept this fit."
        )
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
