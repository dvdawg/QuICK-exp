from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from quickexp_v3.backend import QuickBackend
from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError
from quickexp_v3.lab import connect_quick


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def test_live_connection_refuses_unexpected_quick_version(monkeypatch):
    called = {"connect": False}

    def connect(*args, **kwargs):
        called["connect"] = True
        raise AssertionError("version mismatch must fail before connection")

    fake_quick = SimpleNamespace(__version__="9.9.9", connect=connect)
    monkeypatch.setitem(sys.modules, "quick", fake_quick)
    with pytest.raises(ConfigError, match="expects Quick 0.7.2"):
        connect_quick(repository())
    assert not called["connect"]


class DoneFlag:
    def __init__(self, calls):
        self.calls = calls

    def wait(self, timeout):
        self.calls.append(("wait", timeout))


class Streamer:
    def __init__(self, calls):
        self.calls = calls
        self.done_flag = DoneFlag(calls)

    def stop_readout(self):
        self.calls.append("stop_readout")


class RecoverSoc:
    def __init__(self):
        self.calls = []
        self.streamer = Streamer(self.calls)

    def stop_tproc(self):
        self.calls.append("stop_tproc")

    def poll_data(self, *, totaltime, timeout):
        self.calls.append(("poll_data", totaltime, timeout))

    def reset_gens(self):
        self.calls.append("reset_gens")

    def clear_interrupts(self):
        self.calls.append("clear_interrupts")


def test_recovery_follows_notebook_stop_drain_flush_reset_order():
    soc = RecoverSoc()
    quick = SimpleNamespace(__version__="0.7.2", experiment=SimpleNamespace())
    backend = QuickBackend(soc=soc, soccfg=object(), quick_module=quick)
    details = backend.recover(RuntimeError("transient"), attempt=1)
    assert soc.calls == [
        "stop_tproc",
        "stop_readout",
        ("wait", 1.0),
        ("poll_data", -1, 0.05),
        "reset_gens",
        "clear_interrupts",
    ]
    assert details["warnings"] == []
