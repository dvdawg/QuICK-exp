import textwrap
import pytest

from labreadout import config


def _write(tmp_path, text):
    p = tmp_path / "hardware.yml"
    p.write_text(textwrap.dedent(text))
    return str(p)


VALID = """
    ip: 192.168.1.123
    data_path: "Z:/David/Data/run/"
    channels:
      q: 8
      r: 10
      rr: 1
    rf_board:
      gen:
        - {ch: 10, atten1: 10, atten2: 10, filter: {type: bypass, fc: 0}}
      ro:
        - {ch: 0, atten1: 0, atten2: 0}
    bias:
      channel: 0
      default: -0.2
    var:
      r_freq: 6584.5
      r_power: -10
"""


def test_loads_core_fields(tmp_path):
    cfg = config.load_config(_write(tmp_path, VALID))
    assert cfg.ip == "192.168.1.123"
    assert cfg.data_path == "Z:/David/Data/run/"
    assert cfg.channels == {"q": 8, "r": 10, "rr": 1}
    assert cfg.bias["default"] == -0.2


def test_missing_required_section_raises(tmp_path):
    bad = "ip: 1.2.3.4\ndata_path: x\n"  # no channels
    with pytest.raises(ValueError, match="channels"):
        config.load_config(_write(tmp_path, bad))


def test_missing_channel_key_raises(tmp_path):
    bad = VALID.replace("      rr: 1\n", "")
    with pytest.raises(ValueError, match="rr"):
        config.load_config(_write(tmp_path, bad))


def test_build_var_merges_channel_map_and_var(tmp_path):
    cfg = config.load_config(_write(tmp_path, VALID))
    var = config.build_var(cfg)
    assert var["q"] == 8 and var["r"] == 10 and var["rr"] == 1
    assert var["r_freq"] == 6584.5 and var["r_power"] == -10


class _RecordingSoc:
    def __init__(self):
        self.calls = []

    def reset_gens(self):
        self.calls.append(("reset_gens",))

    def rfb_set_gen_rf(self, ch, a1, a2):
        self.calls.append(("gen_rf", ch, a1, a2))

    def rfb_set_gen_filter(self, ch, fc, ftype, bw=None):
        self.calls.append(("gen_filter", ch, fc, ftype, bw))

    def rfb_set_ro_rf(self, ch, a1, a2):
        self.calls.append(("ro_rf", ch, a1, a2))

    def rfb_set_bias(self, ch, v):
        self.calls.append(("bias", ch, v))


def test_apply_to_soc_emits_expected_driver_calls(tmp_path):
    cfg = config.load_config(_write(tmp_path, VALID))
    soc = _RecordingSoc()
    config.apply_to_soc(cfg, soc)
    assert ("gen_rf", 10, 10, 10) in soc.calls
    assert ("gen_filter", 10, 0, "bypass", None) in soc.calls
    assert ("ro_rf", 0, 0, 0) in soc.calls
    assert ("bias", 0, -0.2) in soc.calls
