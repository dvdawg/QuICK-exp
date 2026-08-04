from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def test_resolution_precedence_and_sweep_expansion():
    repo = repository()
    resolved = repo.resolve(
        "resonator_fine",
        experiment="resonator_spectroscopy",
        overrides={"hard_avg": 17},
    )
    assert resolved.parameters["r_freq"]["center_from"] == "defaults.r_freq"
    assert resolved.parameters["hard_avg"] == 17
    axis = resolved.expanded_parameters()["r_freq"]
    assert isinstance(axis, np.ndarray)
    assert axis.size == 101
    assert np.mean(axis) == pytest.approx(6884.0)


def test_rejected_calibration_never_enters_parameters():
    repo = repository()
    calibration = deepcopy(repo.calibration)
    calibration["records"]["defaults"]["q_gain"] = {
        "value": 0.9,
        "status": "rejected",
    }
    changed = ConfigRepository(
        repo.hardware,
        calibration,
        {"schema_version": 3, "presets": repo.presets},
    )
    assert changed.resolve("t1").parameters["q_gain"] == 0.4


def test_runtime_cannot_override_hardware_roots():
    repo = repository()
    with pytest.raises(ConfigError, match="hardware-controlled"):
        repo.resolve("t1", overrides={"channels": {"q": {"gen": 99}}})


def test_expanded_sweep_must_stay_inside_hardware_limits():
    repo = repository()
    with pytest.raises(ConfigError, match="outside"):
        repo.resolve(
            "resonator_fine",
            overrides={
                "r_freq": {"center": 8999.0, "span": 10.0, "points": 11}
            },
        )


def test_preset_experiment_mismatch_is_rejected():
    repo = repository()
    with pytest.raises(ConfigError, match="targets"):
        repo.resolve("t1", experiment="ramsey")


def test_example_flux_calibration_is_explicitly_synthetic():
    record = repository().calibration["records"]["lookups"]["resonator_vs_flux"]
    assert record["status"] == "accepted"
    assert record["quality"]["r_squared"] == pytest.approx(0.99)
    assert record["provenance"]["source"] == "synthetic-resonator-vs-flux.csv"


def test_all_example_presets_build_with_generic_channels():
    from quickexp_v3.experiments.registry import get

    repo = repository()
    for name in repo.preset_names():
        target = repo.presets[name]["experiment"]
        experiment = get(target)
        plan = experiment.build(repo.resolve(name, experiment=target))
        assert plan.name == target
        assert plan.variables["q"] == 1
        assert plan.variables["r"] == 0
        assert plan.variables["rr"] == 0
        assert plan.variables["z"] == 2
