"""Fit a native T2Echo run with exponential/stretched-exponential selection."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.native_fit_ext import (
    accept_echo_fit,
    fit_echo,
    plot_decay_fit,
)
from quickexp_v3.native_index import NativeIndex


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
INPUT_CSV = None
FIT_SIGNAL = "IQ"
BOOTSTRAP_RESAMPLES = 100
MINIMUM_R_SQUARED = 0.70
MINIMUM_SPAN_OVER_T = 0.75
MAXIMUM_RELATIVE_T_UNCERTAINTY = 0.25
WRITE_ACCEPTED_FIT = False
SHOW_PLOT = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    root = Path(repository.hardware["storage"]["quick_native_root"])
    return NativeIndex(root).refresh().latest(
        quick_class="T2Echo",
        axis_text="delay time",
        n_axes=1,
    ).csv_path


def main():
    fit = fit_echo(
        _input_path(),
        signal=FIT_SIGNAL,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    passes = fit.passes(
        minimum_r_squared=MINIMUM_R_SQUARED,
        minimum_span_over_t=MINIMUM_SPAN_OVER_T,
        maximum_relative_t_uncertainty=MAXIMUM_RELATIVE_T_UNCERTAINTY,
    )
    print(f"Fit source: {fit.source_csv}")
    print(
        f"T2 echo={fit.decay_us:.9g} +/- "
        f"{fit.parameters['decay_uncertainty_us']:.3g} us; "
        f"stretch exponent={fit.parameters['exponent']:.6g} +/- "
        f"{fit.parameters['exponent_uncertainty']:.3g}"
    )
    print(
        f"Model={'stretched' if fit.statistics['selected_stretched'] else 'single exponential'}; "
        f"Delta BIC={fit.statistics['delta_bic_stretched_vs_exponential']:.3f}; "
        f"R^2={fit.statistics['r_squared']:.6f}"
    )
    print(f"Acceptance gates: {'PASS' if passes else 'FAIL'}")
    figure = plot_decay_fit(fit)
    if WRITE_ACCEPTED_FIT:
        path = accept_echo_fit(
            PROJECT_ROOT,
            fit,
            minimum_r_squared=MINIMUM_R_SQUARED,
            minimum_span_over_t=MINIMUM_SPAN_OVER_T,
            maximum_relative_t_uncertainty=MAXIMUM_RELATIVE_T_UNCERTAINTY,
        )
        print(f"Accepted echo record written atomically to {path}")
    else:
        print("WRITE_ACCEPTED_FIT=False: calibration.yml was not changed.")
    if SHOW_PLOT:
        plt.show()
    return fit


if __name__ == "__main__":
    FIT = main()
