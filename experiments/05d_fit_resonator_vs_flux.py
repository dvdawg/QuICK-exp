"""Fit the 05c resonator-vs-Z map and optionally accept it for later runs."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.resonator_flux import (
    DEFAULT_SCAN_NAME,
    accept_fit,
    find_latest_scan,
    fit_resonator_flux,
    plot_resonator_flux_fit,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# None selects the newest *ResVsZ_held_bias*.csv in the configured Quick folder.
INPUT_CSV = None
SCAN_NAME = DEFAULT_SCAN_NAME
# "fit" = refined notch centers plus the cosine fit.
# "min"/"max" = select that smoothed frequency bin per Z row and interpolate.
# Use min/max when the cosine fit is unreliable; set SMOOTH_SIGMA_BINS = 0 to
# take the extrema from the raw rows.
FIT_METHOD = "fit"
SMOOTH_SIGMA_BINS = 2.0
PERIOD_MIN = None
PERIOD_MAX = None

# Acceptance gates for FIT_METHOD="fit"; min/max lookups ignore them.
MINIMUM_R_SQUARED = 0.95
MAXIMUM_RMSE_MHZ = 0.20

# Safety latch: inspect the plot/statistics first, then change this to True.
WRITE_ACCEPTED_FIT = False
PREVIEW_Z_GAIN = 0.0
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    data_directory = Path(
        repository.hardware["storage"]["quick_native_root"]
    )
    return find_latest_scan(data_directory, SCAN_NAME)


def main():
    source = _input_path()
    fit = fit_resonator_flux(
        source,
        fit_method=FIT_METHOD,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
        period_min=PERIOD_MIN,
        period_max=PERIOD_MAX,
    )

    preview_frequency = fit.frequency(PREVIEW_Z_GAIN)
    passes = fit.passes(
        minimum_r_squared=MINIMUM_R_SQUARED,
        maximum_rmse_mhz=MAXIMUM_RMSE_MHZ,
    )
    print(f"Calibration source: {fit.source_csv}")
    print(f"Calibration method: {fit.fit_method}")
    if fit.fit_method == "fit":
        parameters = fit.parameters
        statistics = fit.statistics
        print(
            "Fit: r_freq(z) = "
            f"{parameters['center_frequency']:.6f} + "
            f"{parameters['amplitude']:.6f} * "
            "cos(2*pi*(z - "
            f"{parameters['peak_bias']:.6f})/"
            f"{parameters['period']:.6f}) MHz"
        )
        print(
            f"RMSE: {1000.0 * statistics['rmse_mhz']:.1f} kHz; "
            f"R^2: {statistics['r_squared']:.6f}; "
            f"fit domain: [{fit.z_gain.min():+.6f}, "
            f"{fit.z_gain.max():+.6f}]"
        )
        print(
            "Acceptance gates: "
            f"{'PASS' if passes else 'FAIL'} "
            f"(R^2 >= {MINIMUM_R_SQUARED}, "
            f"RMSE <= {MAXIMUM_RMSE_MHZ} MHz)"
        )
    else:
        print(
            f"Lookup: selected the smoothed amplitude {fit.fit_method} "
            f"in each of {len(fit.z_gain)} Z rows; linear interpolation "
            "is used between rows."
        )
        print(
            "Frequency-bin spacing: "
            f"{fit.statistics['frequency_bin_mhz']:.6f} MHz; "
            f"lookup domain: [{fit.z_gain.min():+.6f}, "
            f"{fit.z_gain.max():+.6f}]"
        )
        print(
            "R^2/RMSE gates do not apply to sampled min/max lookups; "
            "WRITE_ACCEPTED_FIT remains the explicit acceptance latch."
        )
    print(
        f"Calibrated readout at Z={PREVIEW_Z_GAIN:+.6f}: "
        f"{preview_frequency:.6f} MHz"
    )
    if fit.dropped_z_gain.size:
        print(f"Dropped incomplete/non-finite Z rows: {fit.dropped_z_gain}")

    figure = plot_resonator_flux_fit(fit)
    if WRITE_ACCEPTED_FIT:
        calibration_path = accept_fit(
            PROJECT_ROOT,
            fit,
            minimum_r_squared=MINIMUM_R_SQUARED,
            maximum_rmse_mhz=MAXIMUM_RMSE_MHZ,
        )
        print(f"Accepted calibration written atomically to {calibration_path}")
    else:
        print(
            "WRITE_ACCEPTED_FIT=False: calibration.yml was not changed. "
            "Inspect the diagnostics, then enable the latch to accept it."
        )
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
