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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import yaml

REQUIRED_SECTIONS = ("ip", "data_path", "channels")
REQUIRED_CHANNELS = ("q", "r", "rr")

# Keys accepted for the logical index and the declared physical port when a
# channel is written in the long form, e.g. ``r: {gen: 1, dac_port: 10}``.
_INDEX_KEYS = ("index", "gen", "ro", "ch")
_PORT_KEYS = ("port", "dac_port", "adc_port")


@dataclass
class HardwareConfig:
    ip: str
    data_path: str
    channels: Dict[str, int]
    rf_board: Dict[str, Any]
    bias: Dict[str, Any]
    var: Dict[str, Any]
    expected_ports: Dict[str, int] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, Any] = field(default_factory=dict)


def _parse_channel(role: str, value: Any) -> Tuple[int, Optional[int]]:
    """Return ``(logical_index, expected_physical_port_or_None)`` for a channel.

    Accepts the plain form (``r: 1``) and the documented long form
    (``r: {gen: 1, dac_port: 10}``) which additionally declares the physical
    port the operator expects, so a wiring/index mismatch is caught at startup.
    """
    if isinstance(value, dict):
        index = next((value[k] for k in _INDEX_KEYS if k in value), None)
        if index is None:
            raise ValueError(
                f"channel '{role}' is a mapping but declares no logical index "
                f"(one of {_INDEX_KEYS})"
            )
        port = next((value[k] for k in _PORT_KEYS if k in value), None)
        return int(index), (int(port) if port is not None else None)
    return int(value), None


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

    indices: Dict[str, int] = {}
    expected_ports: Dict[str, int] = {}
    for key in REQUIRED_CHANNELS:
        index, port = _parse_channel(key, channels[key])
        indices[key] = index
        if port is not None:
            expected_ports[key] = port

    return HardwareConfig(
        ip=str(data["ip"]),
        data_path=str(data["data_path"]),
        channels=indices,
        rf_board=data.get("rf_board") or {},
        bias=data.get("bias") or {},
        var=data.get("var") or {},
        expected_ports=expected_ports,
        expected=data.get("expected") or {},
        limits=data.get("limits") or {},
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
        # Unlike generator channels (two attenuators), an RF-board readout
        # channel has one attenuator: rfb_set_ro_rf(ro_ch, att).
        # Accept the old ``atten1`` spelling so existing configs still load.
        atten = ro.get("atten", ro.get("atten1", 0))
        soc.rfb_set_ro_rf(ch, int(atten))
        filt = ro.get("filter")
        if filt:
            soc.rfb_set_ro_filter(
                ch,
                fc=filt.get("fc", 0),
                ftype=filt.get("type", "bypass"),
                bw=filt.get("bw", 1),
            )

    if "channel" in cfg.bias and "default" in cfg.bias:
        soc.rfb_set_bias(int(cfg.bias["channel"]), float(cfg.bias["default"]))

    # RFQickSoc.prepare_round() raises a board-only ADCInterruptError if an
    # interrupt flag was already set.  Over Pyro/pickle, a Windows client then
    # tries to import qick.rfboard -> pynq and hides the useful hardware error.
    # Clear stale flags here and convert a persistent condition to a portable
    # built-in exception before the first experiment starts.
    interrupts_cleared = soc.clear_interrupts(
        max_attempts=5, error_on_interrupt=False, error_on_persist=False
    )
    if not interrupts_cleared:
        raise RuntimeError(
            "ADC interrupt flags persist after clearing; reduce the signal or "
            "amplification into the ADC and check the RF input path."
        )
