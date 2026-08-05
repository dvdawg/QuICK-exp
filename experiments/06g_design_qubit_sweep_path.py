"""Design a reusable row-dependent sweep path from any two-axis native map."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import load_repository
from quickexp_v3.native_index import NativeIndex
from quickexp_v3.native_map import load_native_map
from quickexp_v3.sweep_path import (
    design_sweep_path_ui,
    plot_sweep_path,
    save_sweep_path,
    sweep_path_from_native_ridge,
)


# ============================ EDIT THESE ====================================
# Analysis only: this file never connects to QICK.
LIVE_HARDWARE = False

# A prior two-dimensional native CSV/YML map used as background and source.
# None selects the newest compatible two-dimensional map.
INPUT_CSV = None

# "fit_margin" fits the feature and builds a corridor around the model.
# "ui_polygon" lets you draw the desired region directly over the prior map.
PATH_METHOD = "ui_polygon"
OUTPUT_YML = (
    PROJECT_ROOT / "analysis_cache" / "qubit_flux_frequency_sweep_path.yml"
)
BACKGROUND_SIGNAL = "I"  # Amplitude, I, Q, Phase

# None preserves the background sweep's frequency resolution. Set a positive
# MHz value to override that spacing while retaining the selected region.
FREQUENCY_RESOLUTION_MHZ = None

# A scalar fit margin applies the same margin above and below; a pair makes it
# asymmetric. PATH_OUTER_VALUES applies to both modes; None uses background rows.
FIT_MARGIN_MHZ = 10.0
PATH_OUTER_VALUES = None
FIT_FREQUENCY_WINDOW_MHZ = None

SHOW_PREVIEW = True
# ============================================================================


def _input_path() -> Path:
    if INPUT_CSV is not None:
        return Path(INPUT_CSV).expanduser().resolve()
    repository = load_repository(PROJECT_ROOT)
    data_directory = Path(repository.hardware["storage"]["quick_native_root"])
    records = NativeIndex(data_directory).refresh().select(
        n_axes=2,
    )
    compatible = [
        record
        for record in records
        if len(record.independent) == 2
    ]
    if not compatible:
        raise FileNotFoundError(
            f"No two-dimensional native map found in {data_directory}"
        )
    return max(compatible, key=lambda record: record.mtime).csv_path


def main():
    source = _input_path()
    background = load_native_map(source)
    resolution = (
        None
        if FREQUENCY_RESOLUTION_MHZ is None
        else float(FREQUENCY_RESOLUTION_MHZ)
    )
    if resolution is not None and (
        not np.isfinite(resolution) or resolution <= 0
    ):
        raise ValueError("FREQUENCY_RESOLUTION_MHZ must be positive and finite")
    method = str(PATH_METHOD).strip().lower()
    if method == "fit_margin":
        sweep_path = sweep_path_from_native_ridge(
            source,
            margin=FIT_MARGIN_MHZ,
            inner_resolution=resolution,
            outer_values=PATH_OUTER_VALUES,
            frequency_window=FIT_FREQUENCY_WINDOW_MHZ,
        )
    elif method == "ui_polygon":
        sweep_path = design_sweep_path_ui(
            source,
            signal=BACKGROUND_SIGNAL,
            inner_resolution=resolution,
            outer_values=PATH_OUTER_VALUES,
        )
    else:
        raise ValueError("PATH_METHOD must be 'fit_margin' or 'ui_polygon'")
    output = save_sweep_path(sweep_path, OUTPUT_YML)
    inner_rows = sweep_path.inner_sweeps()
    finest_spacing = min(
        float(np.min(np.diff(row)))
        for row in inner_rows
    )
    rectangular_points = int(
        sweep_path.outer_values.size
        * (
            np.floor(
                (
                    float(sweep_path.upper_inner_values.max())
                    - float(sweep_path.lower_inner_values.min())
                )
                / finest_spacing
            )
            + 1
        )
    )
    print(f"Background map: {source}")
    print(f"Saved sweep path: {output}")
    print(
        f"Path: {sweep_path.outer_values.size} "
        f"{sweep_path.outer_name} rows x "
        f"{sweep_path.point_counts.min()}-{sweep_path.point_counts.max()} "
        f"{sweep_path.inner_name} values = "
        f"{sweep_path.total_points} acquired points"
    )
    disjoint_rows = (
        0
        if sweep_path.inner_segments is None
        else sum(len(row) > 1 for row in sweep_path.inner_segments)
    )
    if disjoint_rows:
        print(
            f"Disjoint polygon slices: {disjoint_rows} rows; values in the "
            "gaps are excluded from acquisition"
        )
    print(
        f"Inner-axis resolution: {finest_spacing:g} "
        f"{sweep_path.inner_unit or 'units'} "
        + (
            "(preserved from background)"
            if FREQUENCY_RESOLUTION_MHZ is None
            else "(manual override)"
        )
    )
    print(
        "Approximate full-window equivalent at the finest selected spacing: "
        f"{rectangular_points} points"
    )
    figure = plot_sweep_path(
        sweep_path,
        source,
        signal=BACKGROUND_SIGNAL,
    )
    if SHOW_PREVIEW:
        plt.show()
    return sweep_path


if __name__ == "__main__":
    SWEEP_PATH = main()
