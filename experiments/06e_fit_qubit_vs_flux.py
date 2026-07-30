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
MINIMUM_RIDGE_ROWS = 6
MINIMUM_R_SQUARED = 0.95
MAXIMUM_RMSE_MHZ = 5.0
WRITE_ACCEPTED_FIT = False
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


def main():
    fit = fit_qubit_flux(
        _input_path(),
        ec_mhz=EC_MHZ,
        period_hint=_period_hint(),
    )
    passes = fit.passes(
        minimum_ridge_rows=MINIMUM_RIDGE_ROWS,
        minimum_r_squared=MINIMUM_R_SQUARED,
        maximum_rmse_mhz=MAXIMUM_RMSE_MHZ,
    )
    print(f"Fit source: {fit.source_csv}")
    print(
        f"Ridge rows: {len(fit.ridge_rows)}; "
        f"R^2={fit.statistics['r_squared']:.6f}; "
        f"RMSE={fit.statistics['rmse_mhz']:.6g} MHz"
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
    if WRITE_ACCEPTED_FIT:
        path = accept_qubit_flux_fit(
            PROJECT_ROOT,
            fit,
            minimum_ridge_rows=MINIMUM_RIDGE_ROWS,
            minimum_r_squared=MINIMUM_R_SQUARED,
            maximum_rmse_mhz=MAXIMUM_RMSE_MHZ,
        )
        print(f"Accepted qubit-flux records written atomically to {path}")
    else:
        print("WRITE_ACCEPTED_FIT=False: calibration.yml was not changed.")
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
