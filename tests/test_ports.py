from copy import deepcopy

import pytest

from quickexp_v3.ports import errors, resolve_channels


MET_CHANNELS = {
    "r": {
        "gen": 0,
        "physical_port": "DAC0",
        "generator_type": "axis_signal_gen_v6",
    },
    "rr": {
        "ro": 0,
        "physical_port": "ADC4",
        "readout_type": "axis_dyn_readout_v1",
    },
    "q": {
        "gen": 1,
        "physical_port": "DAC1",
        "generator_type": "axis_signal_gen_v6",
    },
    "z": {
        "gen": 15,
        "physical_port": "DAC15",
        "generator_type": "axis_sg_int4_v2",
    },
}


def met_soccfg_parts():
    gens = [
        {"dac": (index // 4, index % 4), "type": "unused"}
        for index in range(16)
    ]
    gens[0]["type"] = "axis_signal_gen_v6"
    gens[1]["type"] = "axis_signal_gen_v6"
    gens[15]["type"] = "axis_sg_int4_v2"
    readouts = [
        {"adc": (2, 0), "ro_type": "axis_dyn_readout_v1"},
    ]
    return gens, readouts


def test_met_logical_indices_resolve_to_notebook_physical_ports():
    gens, readouts = met_soccfg_parts()
    resolved = resolve_channels(gens, readouts, "", MET_CHANNELS)
    assert errors(resolved) == []
    assert resolved["r"].port == 0
    assert resolved["q"].port == 1
    assert resolved["rr"].port == 4
    assert resolved["z"].port == 15


@pytest.mark.parametrize(
    "role,field,value,fragment",
    [
        ("q", "physical_port", "DAC8", "resolves to DAC1"),
        ("rr", "readout_type", "wrong_readout", "uses axis_dyn_readout_v1"),
    ],
)
def test_connected_port_or_type_mismatch_is_reported(
    role, field, value, fragment
):
    gens, readouts = met_soccfg_parts()
    channels = deepcopy(MET_CHANNELS)
    channels[role][field] = value
    problems = errors(resolve_channels(gens, readouts, "", channels))
    assert any(fragment in problem for problem in problems)
