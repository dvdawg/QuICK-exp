from pathlib import Path
import runpy
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_launchers import EXPECTED


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = ROOT / "experiments"


def prepare_project(tmp_path):
    for name in (
        "hardware.example.yml",
        "calibration.example.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)


def test_every_numbered_file_runs_to_completion_offline(tmp_path):
    prepare_project(tmp_path)
    for filename in EXPECTED:
        namespace = runpy.run_path(
            str(LAUNCHER_DIR / filename),
            run_name=f"offline_{filename[:-3]}",
        )
        module_globals = namespace["main"].__globals__
        module_globals["PROJECT_ROOT"] = tmp_path
        module_globals["LIVE_HARDWARE"] = False
        if "SHOW_PLOT" in module_globals:
            module_globals["SHOW_PLOT"] = False
        if "SHOTS" in module_globals:
            module_globals["SHOTS"] = 40
        for name, value in list(module_globals.items()):
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 3:
                module_globals[name] = value[:3]

        result = namespace["main"]()
        if filename == "00_connect_and_ports.py":
            assert result is None
        elif filename == "01_configure_experiment.py":
            assert result is False
        elif isinstance(result, list):
            assert len(result) == 3
            assert all(row.status.startswith("completed") for row in result)
        else:
            assert result.status.startswith("completed")
        plt.close("all")
