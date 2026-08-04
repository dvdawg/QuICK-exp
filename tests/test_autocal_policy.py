from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from quickexp_v3.autocal.policy import load_autocal_policy
from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError


ROOT = Path(__file__).resolve().parents[1]


def _repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def _proposal(address, value, *, node="N5", analysis="test"):
    return {
        "record": address,
        "value": value,
        "valid_domain": {"z_gain": [0.0, 0.2]},
        "provenance": {
            "autocal_node": node,
            "analysis": analysis,
        },
    }


def test_policy_levels_tolerances_hard_stops_and_special_cases():
    policy = load_autocal_policy(_repository().hardware)
    q_frequency = _proposal("defaults.q_freq", 5600.3)

    assert not policy.promotion_decision(
        autonomy_level=0,
        proposal=q_frequency,
        current_value=5600.0,
        gates_pass=True,
        working_z_gain=0.1,
    ).promote
    assert policy.promotion_decision(
        autonomy_level=1,
        proposal=q_frequency,
        current_value=5600.0,
        gates_pass=True,
        working_z_gain=0.1,
    ).promote
    assert not policy.promotion_decision(
        autonomy_level=1,
        proposal=_proposal("defaults.q_freq", 5601.5),
        current_value=5600.0,
        gates_pass=True,
        working_z_gain=0.1,
    ).promote
    assert not policy.promotion_decision(
        autonomy_level=2,
        proposal=_proposal("lookups.resonator_vs_flux", 1.0),
        current_value=0.0,
        gates_pass=True,
    ).promote
    assert not policy.promotion_decision(
        autonomy_level=2,
        proposal=_proposal(
            "defaults.r_freq",
            6884.1,
            node="N10r",
        ),
        current_value=6884.0,
        gates_pass=True,
    ).promote
    ramsey = _proposal(
        "defaults.q_freq",
        5600.2,
        node="N12",
        analysis="quickexp_v3.native_fit.fit_ramsey",
    )
    assert not policy.promotion_decision(
        autonomy_level=1,
        proposal=ramsey,
        current_value=5600.0,
        gates_pass=True,
    ).promote
    assert policy.promotion_decision(
        autonomy_level=1,
        proposal=ramsey,
        current_value=5600.0,
        gates_pass=True,
        ramsey_sign_confirmed=True,
    ).promote
    assert policy.promotion_decision(
        autonomy_level=1,
        proposal=_proposal("derived.t2_echo.cycle_0", 5.1, node="N13"),
        current_value=None,
        gates_pass=True,
    ).promote


def test_policy_caps_clamp_not_reject():
    policy = load_autocal_policy(_repository().hardware)
    clamped = policy.clamp_overrides(
        {
            "q_gain": np.asarray([-0.9, 0.4, 0.95]),
            "r_power": np.asarray([-35.0, -10.0]),
            "q_freq": 5600.0,
        }
    )
    assert np.allclose(clamped["q_gain"], [-0.8, 0.4, 0.8])
    assert np.allclose(clamped["r_power"], [-35.0, -20.0])
    assert clamped["q_freq"] == 5600.0


def test_hypothesis_adaptive_and_backtracking_policy_defaults_are_safe():
    policy = load_autocal_policy(_repository().hardware)
    assert policy.hypothesis_nodes == frozenset()
    assert policy.adaptive_nodes == frozenset()
    assert policy.margin_threshold > 0.0
    assert policy.probe_budget_seconds > 0.0
    assert policy.max_backtracks_per_address <= policy.max_backtracks_per_session
    assert policy.advisor_mode == "null"


def test_missing_hardware_policy_is_proposal_only_at_every_level():
    repository = _repository()
    hardware = deepcopy(repository.hardware)
    hardware.pop("autocal")
    policy = load_autocal_policy(hardware)
    decision = policy.promotion_decision(
        autonomy_level=2,
        proposal=_proposal("defaults.q_freq", 5600.1),
        current_value=5600.0,
        gates_pass=True,
    )
    assert not decision.promote
    assert "hardware.autocal is absent" in decision.reason


def test_mandatory_hard_stops_cannot_be_removed_from_policy():
    repository = _repository()
    hardware = deepcopy(repository.hardware)
    hardware["autocal"]["hard_stop_records"] = []
    policy = load_autocal_policy(hardware)
    for address in (
        "defaults.r_offset",
        "lookups.resonator_vs_flux",
        "lookups.qubit_vs_flux",
    ):
        decision = policy.promotion_decision(
            autonomy_level=2,
            proposal=_proposal(address, 1.0),
            current_value=0.0,
            gates_pass=True,
        )
        assert not decision.promote
        assert "hard-stop record" in decision.reason


@pytest.mark.parametrize(
    "autocal, message",
    (
        (
            {"auto_accept": {"defaults.q_freq": {"absolute_mhz": 1, "relative": 1}}},
            "exactly one tolerance",
        ),
        (
            {"auto_accept": {"defaults.q_freq": {"mystery": 1}}},
            "unknown tolerance",
        ),
        (
            {"budgets": {"max_total_runs": 0}},
            "must be positive",
        ),
        (
            {"budgets": {"max_node_attempts": 1.5}},
            "must be an integer",
        ),
        (
            {"caps": {"q_gain_max": 0}},
            "q_gain_max must be positive",
        ),
        (
            {"caps": {"r_power_max_db": 1}},
            "r_power_max_db cannot be positive",
        ),
        (
            {"caps": {"r_power_max_db": -100}},
            "r_power_max_db is outside",
        ),
        (
            {"hypothesis_nodes": "N5"},
            "hypothesis_nodes must be a list",
        ),
        (
            {"hypothesis_nodes": ["N4"]},
            "unsupported node ids",
        ),
        (
            {"adaptive_nodes": ["N5"]},
            "unsupported node ids",
        ),
        (
            {"hypothesis": {"margin_threshold": 0}},
            "margin_threshold must be positive",
        ),
        (
            {"backtracking": {"max_backtracks_per_address": 4,
                               "max_backtracks_per_session": 3}},
            "per-address cap cannot exceed",
        ),
        (
            {"advisor": {"mode": "mystery"}},
            "advisor.mode must be",
        ),
    ),
)
def test_malformed_policy_is_rejected_during_hardware_validation(
    autocal,
    message,
):
    repository = _repository()
    hardware = deepcopy(repository.hardware)
    hardware["autocal"] = autocal
    with pytest.raises(ConfigError, match=message):
        ConfigRepository(
            hardware,
            repository.calibration,
            {"schema_version": 3, "presets": repository.presets},
        )


def test_q_gain_policy_cap_must_fit_both_hardware_bounds():
    repository = _repository()
    hardware = deepcopy(repository.hardware)
    hardware["limits"]["q_gain"] = [-0.5, 1.0]
    hardware["autocal"]["caps"]["q_gain_max"] = 0.8
    with pytest.raises(ConfigError, match="does not fit symmetrically"):
        ConfigRepository(
            hardware,
            repository.calibration,
            {"schema_version": 3, "presets": repository.presets},
        )


def test_autocal_policy_root_cannot_be_weakened_by_run_overrides():
    with pytest.raises(ConfigError, match="hardware-controlled roots.*autocal"):
        _repository().resolve(
            "t1",
            overrides={"autocal": {"caps": {"q_gain_max": 1.0}}},
        )
