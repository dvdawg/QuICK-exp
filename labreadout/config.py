"""Hardware configuration: the single source of truth for board setup.

``hardware.yml`` (static, user-edited) declares the QICK board IP, data path,
channel map (``q``/``r``/``rr``), per-channel RF-board setup, bias, and base
``var`` defaults. This module loads and validates it, builds the base ``var``
dict, and -- on the lab PC -- translates the declared setup into ``soc`` driver
calls. It replaces the copy-pasted setup blocks scattered through the notebook.

Loading, validation, and ``build_var`` are pure and unit-tested offline. The
``apply_to_soc`` translator is tested against a recording stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yaml

REQUIRED_SECTIONS = ("ip", "data_path", "channels")
REQUIRED_CHANNELS = ("q", "r", "rr")


@dataclass
class HardwareConfig:
    ip: str
    data_path: str
    channels: Dict[str, int]
    rf_board: Dict[str, Any]
    bias: Dict[str, Any]
    var: Dict[str, Any]


def load_config(path: str) -> HardwareConfig:
    """Load and validate ``hardware.yml`` into a HardwareConfig."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    for section in REQUIRED_SECTIONS:
        if section not in data:
            raise ValueError(f"hardware config missing required section: '{section}'")

    channels = data["channels"] or {}
    for key in REQUIRED_CHANNELS:
        if key not in channels:
            raise ValueError(f"hardware config 'channels' missing required key: '{key}'")

    return HardwareConfig(
        ip=str(data["ip"]),
        data_path=str(data["data_path"]),
        channels={k: int(channels[k]) for k in REQUIRED_CHANNELS},
        rf_board=data.get("rf_board") or {},
        bias=data.get("bias") or {},
        var=data.get("var") or {},
    )


def build_var(cfg: HardwareConfig) -> Dict[str, Any]:
    """Base ``var`` dict: declared var defaults plus the channel map."""
    var = dict(cfg.var)
    var.update(cfg.channels)
    return var


def apply_to_soc(cfg: HardwareConfig, soc) -> None:
    """Translate the RF-board + bias config into driver calls on a live ``soc``.

    Only invoked on the lab PC where a real ``soc`` exists.
    """
    soc.reset_gens()

    for gen in cfg.rf_board.get("gen", []):
        ch = int(gen["ch"])
        soc.rfb_set_gen_rf(ch, int(gen.get("atten1", 0)), int(gen.get("atten2", 0)))
        filt = gen.get("filter")
        if filt:
            soc.rfb_set_gen_filter(
                ch,
                fc=filt.get("fc", 0),
                ftype=filt.get("type", "bypass"),
                bw=filt.get("bw"),
            )

    for ro in cfg.rf_board.get("ro", []):
        ch = int(ro["ch"])
        soc.rfb_set_ro_rf(ch, int(ro.get("atten1", 0)), int(ro.get("atten2", 0)))

    if "channel" in cfg.bias and "default" in cfg.bias:
        soc.rfb_set_bias(int(cfg.bias["channel"]), float(cfg.bias["default"]))
