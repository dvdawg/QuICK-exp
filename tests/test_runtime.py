from copy import deepcopy
from pathlib import Path

import pytest

from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.config import ConfigRepository
from quickexp_v3.runtime import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


def repository(tmp_path):
    source = ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )
    hardware = deepcopy(source.hardware)
    hardware["storage"]["quick_native_root"] = str(tmp_path / "quick-data")
    return ConfigRepository(
        hardware,
        source.calibration,
        {"schema_version": 3, "presets": source.presets},
    )


@pytest.mark.parametrize(
    "experiment,preset",
    [
        ("loopback", "loopback"),
        ("resonator_spectroscopy", "resonator_fine"),
        ("qubit_spectroscopy", "qubit_fine"),
        ("rabi", "rabi_amplitude"),
        ("iq_blobs", "iq_blobs"),
        ("t1", "t1"),
        ("ramsey", "ramsey"),
        ("echo", "echo"),
    ],
)
def test_synthetic_runs_stay_in_memory(tmp_path, experiment, preset):
    repo = repository(tmp_path)
    completed = ExperimentRunner(repo, SyntheticBackend()).run(experiment, preset)
    assert completed.status in {"completed", "completed_with_analysis_error"}
    assert completed.data.points > 0
    assert not (tmp_path / "quick-data").exists()
    assert not (tmp_path / "runs").exists()


def test_retry_recovers_exact_plan_and_finishes(tmp_path):
    repo = repository(tmp_path)
    backend = SyntheticBackend(fail_attempts=2)
    completed = ExperimentRunner(repo, backend).run("t1", "t1")
    assert completed.status == "completed"
    assert backend.calls == 3
    assert backend.recoveries == 2


def test_title_override_reaches_backend_plan(tmp_path):
    repo = repository(tmp_path)
    runner = ExperimentRunner(repo, SyntheticBackend())
    planned = runner.plan(
        "qubit_spectroscopy",
        "qubit_fine",
        title="QubitSpec_Zp0p0000_r6884p000",
    )
    assert planned.plan.title == "QubitSpec_Zp0p0000_r6884p000"
