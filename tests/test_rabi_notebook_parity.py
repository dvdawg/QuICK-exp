from pathlib import Path

import pytest

from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError
from quickexp_v3.runtime import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


@pytest.mark.parametrize(
    "preset",
    (
        "rabi_length",
        "rabi_amplitude",
        "rabi_chevron_duration",
        "rabi_chevron_amplitude",
    ),
)
def test_rabi_family_matches_working_notebook_controls(preset):
    planned = ExperimentRunner(repository(), backend=None).plan(
        "rabi_chevron" if "chevron" in preset else "rabi",
        preset,
    )

    variables = planned.plan.variables
    assert variables["rep"] == 1000
    assert variables["r_power"] == -30.0
    assert variables["r_length"] == 2.0
    assert variables["r_offset"] == 0.5
    assert variables["r_relax"] == 20.0
    assert variables["cycle"] == 0
    assert variables["z_length"] == 0.2
    assert variables["z_settle"] == 5.0
    assert planned.plan.run_options["population"] is False


def test_launcher_run_option_can_explicitly_control_population():
    runner = ExperimentRunner(repository(), backend=None)

    planned = runner.plan(
        "rabi",
        "rabi_length",
        run_options={"population": True},
    )

    assert planned.plan.run_options["population"] is True
    assert planned.plan.signal_names[0] == "population"


def test_launcher_rejects_non_quick_run_options():
    with pytest.raises(ConfigError, match="unsupported run options: retry"):
        ExperimentRunner(repository(), backend=None).plan(
            "rabi",
            "rabi_length",
            run_options={"retry": 1},
        )
