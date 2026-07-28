from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

from quickexp_v3.config import ConfigRepository
from quickexp_v3.lab import connect_quick


ROOT = Path(__file__).resolve().parents[1]


class FakeSocCfg(dict):
    def __str__(self):
        return "QICK test configuration"


class FakeSoc:
    def reset_gens(self):
        pass

    def clear_interrupts(self, **kwargs):
        return True


def fake_soccfg():
    generators = [
        {"dac": (index // 4, index % 4), "type": "unused"}
        for index in range(16)
    ]
    generators[0]["type"] = "axis_signal_gen_v6"
    generators[1]["type"] = "axis_signal_gen_v6"
    generators[15]["type"] = "axis_sg_int4_v2"
    return FakeSocCfg(
        gens=generators,
        readouts=[
            {"adc": (2, 0), "ro_type": "axis_dyn_readout_v1"},
        ],
    )


def repository(tmp_path, *, show_progress=True):
    source = ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )
    hardware = deepcopy(source.hardware)
    hardware["storage"] = {
        "quick_native_output": True,
        "quick_native_root": str(tmp_path / "quick-data"),
    }
    hardware["qick"]["show_progress"] = show_progress
    return ConfigRepository(
        hardware,
        source.calibration,
        {"schema_version": 3, "presets": source.presets},
    )


def test_live_backend_uses_only_configured_quick_native_root(
    tmp_path, monkeypatch
):
    config = fake_soccfg()
    fake_quick = SimpleNamespace(
        __version__="0.7.2",
        connect=lambda *args, **kwargs: (config, FakeSoc()),
    )
    monkeypatch.setitem(sys.modules, "quick", fake_quick)

    connection = connect_quick(repository(tmp_path))
    expected = (tmp_path / "quick-data").resolve()
    assert Path(connection.backend.data_path) == expected
    assert expected.is_dir()
    assert connection.backend.show_progress is True
    assert not (tmp_path / "runs").exists()
    connection.close()


def test_progress_can_be_disabled_in_hardware_config(tmp_path, monkeypatch):
    config = fake_soccfg()
    fake_quick = SimpleNamespace(
        __version__="0.7.2",
        connect=lambda *args, **kwargs: (config, FakeSoc()),
    )
    monkeypatch.setitem(sys.modules, "quick", fake_quick)

    connection = connect_quick(repository(tmp_path, show_progress=False))
    assert connection.backend.show_progress is False
    connection.close()
