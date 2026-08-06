"""Fit a QubitSpecVsZ native map with a constrained transmon model."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.native_index import NativeIndex
from quickexp_v3.qubit_flux_fit import (
    accept_qubit_flux_fit,
    fit_qubit_flux,
    plot_qubit_flux_fit,
)


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
INPUT_CSV = None
EC_MHZ = 180.0
USE_RESONATOR_PERIOD_HINT = True
FIT_FREQUENCY_WINDOW_MHZ = None # None uses every acquired custom-path band. Bounds are inclusive and may be entered in either order. Example: (4500.0, 4800.0).
FIT_FLUX_WINDOW_Z = None # None uses every acquired flux row. Values are Z-gain units and bounds are inclusive. Example: (-0.25, 0.15).
MINIMUM_RIDGE_ROWS = 6
MINIMUM_R_SQUARED = 0.95
MAXIMUM_RMSE_MHZ = 5.0
WRITE_ACCEPTED_FIT = False
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
    return NativeIndex(data_directory).refresh().latest(
        title_contains="QubitSpecVsZ",
        n_axes=2,
    ).csv_path


def _period_hint():
    if not USE_RESONATOR_PERIOD_HINT:
        return None
    repository = load_repository(PROJECT_ROOT)
    try:
        return float(
            repository.calibration["records"]["lookups"][
                "resonator_vs_flux"
            ]["value"]["parameters"]["period"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _sweet_spot_hint():
    if not USE_RESONATOR_PERIOD_HINT:
        return None
    repository = load_repository(PROJECT_ROOT)
    try:
        return float(
            repository.calibration["records"]["lookups"][
                "resonator_vs_flux"
            ]["value"]["parameters"]["peak_bias"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def main():
    fit = fit_qubit_flux(
        _input_path(),
        ec_mhz=EC_MHZ,
        period_hint=_period_hint(),
        sweet_spot_hint=_sweet_spot_hint(),
        frequency_window_mhz=FIT_FREQUENCY_WINDOW_MHZ,
        flux_window_z=FIT_FLUX_WINDOW_Z,
    )
    passes = fit.passes(
        minimum_ridge_rows=MINIMUM_RIDGE_ROWS,
        minimum_r_squared=MINIMUM_R_SQUARED,
        maximum_rmse_mhz=MAXIMUM_RMSE_MHZ,
    )
    print(f"Fit source: {fit.source_csv}")
    print(
        f"Fit window: frequency [{fit.frequencies_mhz.min():.6f}, "
        f"{fit.frequencies_mhz.max():.6f}] MHz; flux "
        f"[{fit.map_z_gain.min():+.6g}, {fit.map_z_gain.max():+.6g}] Z"
    )
    print(
        f"Ridge rows: {len(fit.ridge_rows)}; "
        f"R^2={fit.statistics['r_squared']:.6f}; "
        f"RMSE={fit.statistics['rmse_mhz']:.6g} MHz"
    )
    if fit.is_ragged:
        counts = fit.statistics["row_point_counts"]
        print(
            "Custom-path map: "
            f"{len(counts)} rows with {min(counts)}-{max(counts)} points; "
            f"{fit.statistics['disjoint_frequency_rows']} rows contain "
            "disjoint frequency bands"
        )
    print(
        f"f_max={fit.parameters['f_max_mhz']:.6f} MHz; "
        f"period={fit.parameters['period_z']:.6g}; "
        f"sweet spot Z={fit.parameters['sweet_spot_z']:+.6g}; "
        f"asymmetry={fit.parameters['asymmetry']:.6g}; EC={EC_MHZ:.6g} MHz"
    )
    print(f"Identifiability: {dict(fit.identifiable)}")
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    figure = plot_qubit_flux_fit(fit)
    if WRITE_ACCEPTED_FIT or FORCE_WRITE:
        if FORCE_WRITE and not passes:
            print("WARNING: FORCE_WRITE=True is bypassing failed acceptance gates.")
        path = accept_qubit_flux_fit(
            PROJECT_ROOT,
            fit,
            minimum_ridge_rows=MINIMUM_RIDGE_ROWS,
            minimum_r_squared=MINIMUM_R_SQUARED,
            maximum_rmse_mhz=MAXIMUM_RMSE_MHZ,
            force_write=FORCE_WRITE,
        )
        action = "Force-written" if FORCE_WRITE else "Accepted"
        print(f"{action} qubit-flux records written atomically to {path}")
    else:
        print("WRITE_ACCEPTED_FIT=False: calibration.yml was not changed.")
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
