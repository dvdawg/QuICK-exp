"""Pure logical->physical port mapping for QICK channels.

``quick``/``soccfg`` address generators and readouts by a *logical index*
(``v['r']``, ``v['rr']``, ``v['q']``). That index is **not** the physical
QICK-box port number printed on the chassis, which is a recurring source of
confusion in the lab ("is r=1 really DAC 10?"). This module resolves each
logical index to its physical DAC/ADC port, says whether that port carries an
RF daughter card, and -- when the operator declares the port they expect in
``hardware.yml`` -- flags a mismatch loudly.

Everything here is pure: it takes the same shapes a live ``soccfg`` exposes
(``soccfg['gens']``, ``soccfg['readouts']``, ``str(soccfg)``) so it is unit
-tested offline with plain dicts. The thin ``from_soccfg`` adapter is the only
part that touches a live board, and it does no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set


def dac_port(tile: int, block: int) -> int:
    """Physical DAC port for a generator's (tile, block) index."""
    return 4 * int(tile) + int(block)


def adc_port(tile: int, block: int) -> int:
    """Physical ADC port for a readout's (tile, block) index (ADC tiles start at 1)."""
    return 4 * (int(tile) - 1) + int(block)


def parse_card_ports(description: str, kind: str) -> Set[int]:
    """Physical ports that carry an RF daughter card, from ``str(soccfg)``.

    ``kind`` is ``"RF Out"`` (generators/DACs) or ``"RF In"`` (readouts/ADCs).
    """
    pattern = rf"{re.escape(kind)} card has ports \[([^\]]*)\]"
    found: Set[int] = set()
    for match in re.finditer(pattern, description):
        body = match.group(1).strip()
        if not body:
            continue
        found.update(int(x.strip()) for x in body.split(",") if x.strip())
    return found


@dataclass
class ResolvedChannel:
    role: str               # 'q' | 'r' | 'rr'
    index: int              # logical generator/readout index
    port: int               # physical DAC/ADC port
    kind: str               # 'DAC' (generator) | 'ADC' (readout)
    has_rf_card: bool
    expected_port: Optional[int] = None

    @property
    def mismatch(self) -> bool:
        return self.expected_port is not None and self.port != self.expected_port


@dataclass
class Issue:
    severity: str           # 'warn' | 'error'
    message: str


_GENERATOR_ROLES = ("q", "r")
_READOUT_ROLES = ("rr",)


def _tile_block(entry: Any, key: str) -> tuple[int, int]:
    """Extract a (tile, block) pair from a soccfg gen/readout entry."""
    tile, block = entry[key]
    return int(tile), int(block)


def resolve_channels(
    gens: Sequence[Any],
    readouts: Sequence[Any],
    description: str,
    channels: Dict[str, int],
    expected_ports: Optional[Dict[str, int]] = None,
) -> Dict[str, ResolvedChannel]:
    """Resolve each logical channel index to its physical port + RF-card status."""
    expected_ports = expected_ports or {}
    rf_out = parse_card_ports(description, "RF Out")
    rf_in = parse_card_ports(description, "RF In")

    resolved: Dict[str, ResolvedChannel] = {}
    for role, index in channels.items():
        index = int(index)
        if role in _READOUT_ROLES:
            tile, block = _tile_block(readouts[index], "adc")
            port = adc_port(tile, block)
            resolved[role] = ResolvedChannel(
                role, index, port, "ADC", port in rf_in, expected_ports.get(role)
            )
        else:
            tile, block = _tile_block(gens[index], "dac")
            port = dac_port(tile, block)
            resolved[role] = ResolvedChannel(
                role, index, port, "DAC", port in rf_out, expected_ports.get(role)
            )
    return resolved


def check(resolved: Dict[str, ResolvedChannel]) -> List[Issue]:
    """All issues: declared-port mismatches (error) + direct/no-RF-card paths (warn)."""
    issues: List[Issue] = []
    for ch in resolved.values():
        if ch.mismatch:
            issues.append(
                Issue(
                    "error",
                    f"channel '{ch.role}' (index {ch.index}) resolves to "
                    f"{ch.kind} {ch.port}, but hardware.yml declares {ch.kind} "
                    f"{ch.expected_port}. Wiring or the channel index is wrong.",
                )
            )
    # Readout/readout-drive paths should normally go through an RF card.
    for role in ("r", "rr"):
        ch = resolved.get(role)
        if ch is not None and not ch.has_rf_card:
            issues.append(
                Issue(
                    "warn",
                    f"channel '{role}' uses {ch.kind} {ch.port}, a direct "
                    f"output with no RF daughter card.",
                )
            )
    return issues


def errors(resolved: Dict[str, ResolvedChannel]) -> List[Issue]:
    """Just the blocking issues (declared-port mismatches)."""
    return [i for i in check(resolved) if i.severity == "error"]


def report(resolved: Dict[str, ResolvedChannel]) -> str:
    """Human-readable logical->physical map with tags, for printing at startup."""
    lines = ["Channel map (logical index -> physical port):"]
    for role in ("q", "r", "rr"):
        ch = resolved.get(role)
        if ch is None:
            continue
        tags = [f"{ch.kind} {ch.port}"]
        tags.append("RF-card" if ch.has_rf_card else "DIRECT/no-RF-card")
        if ch.expected_port is not None:
            tags.append("OK" if not ch.mismatch else f"MISMATCH (expected {ch.expected_port})")
        lines.append(f"  {role:>2} = index {ch.index:<2} -> {', '.join(tags)}")
    for issue in check(resolved):
        lines.append(f"  [{issue.severity.upper()}] {issue.message}")
    return "\n".join(lines)


def from_soccfg(
    soccfg: Any,
    channels: Dict[str, int],
    expected_ports: Optional[Dict[str, int]] = None,
) -> Dict[str, ResolvedChannel]:
    """Adapter: pull gens/readouts/description off a live soccfg, then resolve."""
    return resolve_channels(
        soccfg["gens"], soccfg["readouts"], str(soccfg), channels, expected_ports
    )
