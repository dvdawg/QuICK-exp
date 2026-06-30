"""Tests for the pure logical->physical port-mapping layer (labreadout.ports).

These run offline with no quick/soccfg: we feed the same shapes a live soccfg
exposes -- a list of generators with a (tile, block) ``dac`` index, a list of
readouts with an ``adc`` index + ``ro_type``, and the ``str(soccfg)`` card
description that lists which physical ports carry an RF daughter card.
"""

import pytest

from labreadout import ports


# A small synthetic board: 12 gens (DAC ports 0..11), 8 readouts (ADC ports 0..7).
# RF Out daughter card covers DAC ports 8-11; RF In card covers ADC ports 4-7.
DESCRIPTION = (
    "QICK configuration:\n"
    "\tRF Out card has ports [8, 9, 10, 11]\n"
    "\tRF In card has ports [4, 5, 6, 7]\n"
)


def _gens():
    # gen index -> (tile, block); dac_port = 4*tile + block
    return [{"dac": (i // 4, i % 4)} for i in range(12)]


def _readouts():
    # Deliberately non-identity (like the real board): the logical readout index
    # is NOT the physical ADC port. adc_port = 4*(tile-1) + block.
    return [
        {"adc": (1, 0), "ro_type": "axis_readout_v2"},  # index 0 -> ADC 0 (direct)
        {"adc": (2, 0), "ro_type": "axis_readout_v2"},  # index 1 -> ADC 4 (RF In)
        {"adc": (2, 1), "ro_type": "axis_readout_v2"},  # index 2 -> ADC 5
        {"adc": (2, 2), "ro_type": "axis_readout_v2"},  # index 3 -> ADC 6
    ]


def test_dac_port_arithmetic():
    assert ports.dac_port(2, 2) == 10
    assert ports.dac_port(0, 0) == 0


def test_adc_port_arithmetic():
    # ADCs start at tile 1, so (tile=2, block=0) is physical ADC port 4.
    assert ports.adc_port(2, 0) == 4
    assert ports.adc_port(1, 0) == 0


def test_parse_card_ports_reads_rf_out_and_in():
    assert ports.parse_card_ports(DESCRIPTION, "RF Out") == {8, 9, 10, 11}
    assert ports.parse_card_ports(DESCRIPTION, "RF In") == {4, 5, 6, 7}


def test_resolve_maps_logical_index_to_physical_port():
    resolved = ports.resolve_channels(
        _gens(), _readouts(), DESCRIPTION, {"q": 8, "r": 10, "rr": 1}
    )
    assert resolved["r"].port == 10  # gen index 10 -> DAC 10
    assert resolved["r"].kind == "DAC"
    assert resolved["rr"].port == 4  # ro index 1 -> ADC 4
    assert resolved["rr"].kind == "ADC"


def test_resolve_flags_rf_card_presence():
    resolved = ports.resolve_channels(
        _gens(), _readouts(), DESCRIPTION, {"q": 8, "r": 10, "rr": 1}
    )
    assert resolved["r"].has_rf_card is True   # DAC 10 is on the RF Out card
    assert resolved["rr"].has_rf_card is True   # ADC 4 is on the RF In card
    assert resolved["q"].has_rf_card is True    # DAC 8 is on the RF Out card


def test_direct_no_rf_card_channel_produces_warning():
    # rr -> ro index 0 -> ADC port 0, which is NOT on the RF In card.
    resolved = ports.resolve_channels(
        _gens(), _readouts(), DESCRIPTION, {"q": 8, "r": 10, "rr": 0}
    )
    issues = ports.check(resolved)
    assert any(i.severity == "warn" and "rr" in i.message for i in issues)


def test_declared_port_match_is_clean():
    resolved = ports.resolve_channels(
        _gens(), _readouts(), DESCRIPTION, {"q": 8, "r": 10, "rr": 1},
        expected_ports={"r": 10, "rr": 4},
    )
    assert resolved["r"].mismatch is False
    assert ports.errors(resolved) == []


def test_declared_port_mismatch_is_an_error():
    # Operator believes r=10 is DAC 8, but it actually resolves to DAC 10.
    resolved = ports.resolve_channels(
        _gens(), _readouts(), DESCRIPTION, {"q": 8, "r": 10, "rr": 1},
        expected_ports={"r": 8},
    )
    assert resolved["r"].mismatch is True
    errs = ports.errors(resolved)
    assert len(errs) == 1
    assert "r" in errs[0].message and "10" in errs[0].message and "8" in errs[0].message


def test_report_is_a_readable_string():
    resolved = ports.resolve_channels(
        _gens(), _readouts(), DESCRIPTION, {"q": 8, "r": 10, "rr": 1}
    )
    text = ports.report(resolved)
    assert "DAC 10" in text and "ADC 4" in text
    assert "r" in text and "rr" in text
