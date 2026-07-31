"""Fit saved ground/excited IQ shots with a full-covariance GMM."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.iq_gmm import (
    accept_iq_gmm,
    fit_iq_gmm,
    load_iq_shots,
    plot_iq_gmm,
)
from quickexp_v3.native_index import NativeIndex


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
INPUT_CSV = None
MINIMUM_FIDELITY = 0.80
MINIMUM_SHOTS_PER_STATE = 2000
MAXIMUM_ANGLE_BOOTSTRAP_STD = 0.20
WRITE_ACCEPTED_FIT = False
FORCE_WRITE = False
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    root = Path(repository.hardware["storage"]["quick_native_root"])
    return NativeIndex(root).refresh().latest(
        quick_class="IQScatter",
        n_axes=0,
    ).csv_path


def main():
    source, metadata, columns = load_iq_shots(_input_path())
    fit = fit_iq_gmm(*columns)
    passes = fit.passes(
        minimum_fidelity=MINIMUM_FIDELITY,
        minimum_shots_per_state=MINIMUM_SHOTS_PER_STATE,
        maximum_angle_bootstrap_std=MAXIMUM_ANGLE_BOOTSTRAP_STD,
    )
    print(f"Fit source: {source}")
    print(
        f"Assignment fidelity={fit.assignment_fidelity:.3%}; "
        f"centroid baseline={fit.baseline_fidelity:.3%}; "
        f"cross-validated GMM/baseline="
        f"{fit.cross_validated_fidelity:.3%}/"
        f"{fit.cross_validated_baseline_fidelity:.3%}"
    )
    print(
        f"Threshold={fit.threshold:.9g}; angle={fit.angle_rad:.6g} rad; "
        f"thermal population={fit.thermal_population:.3%}; "
        f"leakage={dict(fit.leakage)}"
    )
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    figure = plot_iq_gmm(fit, *columns)
    if WRITE_ACCEPTED_FIT or FORCE_WRITE:
        if FORCE_WRITE and not passes:
            print("WARNING: FORCE_WRITE=True is bypassing failed acceptance gates.")
        path = accept_iq_gmm(
            PROJECT_ROOT,
            fit,
            source,
            minimum_fidelity=MINIMUM_FIDELITY,
            minimum_shots_per_state=MINIMUM_SHOTS_PER_STATE,
            maximum_angle_bootstrap_std=MAXIMUM_ANGLE_BOOTSTRAP_STD,
            force_write=FORCE_WRITE,
        )
        action = "Force-written" if FORCE_WRITE else "Accepted"
        print(f"{action} IQ records written atomically to {path}")
    else:
        print("WRITE_ACCEPTED_FIT=False: calibration.yml was not changed.")
    if SHOW_PLOT:
        plt.show()
    _ = metadata
    return fit


if __name__ == "__main__":
    FIT = main()
