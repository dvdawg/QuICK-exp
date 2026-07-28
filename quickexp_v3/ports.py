"""Resolve Quick logical channel indices to physical QICK-box ports."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .errors import ConfigError


def dac_port(tile: int, block: int) -> int:
    return 4 * int(tile) + int(block)


def adc_port(tile: int, block: int) -> int:
    # RFSoC ADC tiles are numbered from one in QICK configuration.
    return 4 * (int(tile) - 1) + int(block)


def parse_card_ports(description: str, kind: str) -> Set[int]:
    pattern = rf"{re.escape(kind)} card has ports \[([^\]]*)\]"
    result: Set[int] = set()
    for match in re.finditer(pattern, description):
        result.update(
            int(value.strip())
            for value in match.group(1).split(",")
            if value.strip()
        )
    return result


def declared_port(value: Any, kind: str) -> Optional[int]:
    if value is None:
        return None
    match = re.fullmatch(rf"\s*{kind}\s*(\d+)\s*", str(value), re.IGNORECASE)
    if not match:
        raise ConfigError(
            f"physical_port must look like {kind}0, got {value!r}"
        )
    return int(match.group(1))


@dataclass(frozen=True)
class ResolvedChannel:
    role: str
    index: int
    kind: str
    port: int
    implementation: str
    has_rf_card: bool
    expected_port: Optional[int]
    expected_implementation: Optional[str]

    @property
    def port_mismatch(self) -> bool:
        return (
            self.expected_port is not None
            and self.expected_port != self.port
        )

    @property
    def implementation_mismatch(self) -> bool:
        expected = self.expected_implementation
        if not expected or expected.lower() in {"full", "direct", "any"}:
            return False
        return expected != self.implementation


def _tile_block(entry: Mapping[str, Any], key: str) -> Tuple[int, int]:
    try:
        tile, block = entry[key]
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"soccfg entry has no valid {key!r} tile/block") from error
    return int(tile), int(block)


def _implementation(entry: Mapping[str, Any]) -> str:
    return str(
        entry.get("type")
        or entry.get("gen_type")
        or entry.get("ro_type")
        or "unknown"
    )


def resolve_channels(
    gens: Sequence[Mapping[str, Any]],
    readouts: Sequence[Mapping[str, Any]],
    description: str,
    hardware_channels: Mapping[str, Any],
) -> Dict[str, ResolvedChannel]:
    rf_out = parse_card_ports(description, "RF Out")
    rf_in = parse_card_ports(description, "RF In")
    resolved: Dict[str, ResolvedChannel] = {}
    for role in ("r", "rr", "q", "z"):
        configured = hardware_channels.get(role)
        if not isinstance(configured, Mapping):
            raise ConfigError(f"hardware.channels.{role} must be a mapping")
        if role == "rr":
            index = int(configured["ro"])
            if index < 0 or index >= len(readouts):
                raise ConfigError(
                    f"readout index rr={index} is outside soccfg readouts"
                )
            entry = readouts[index]
            tile, block = _tile_block(entry, "adc")
            kind = "ADC"
            port = adc_port(tile, block)
            expected_type = configured.get("readout_type")
            has_card = port in rf_in
        else:
            index = int(configured["gen"])
            if index < 0 or index >= len(gens):
                raise ConfigError(
                    f"generator index {role}={index} is outside soccfg generators"
                )
            entry = gens[index]
            tile, block = _tile_block(entry, "dac")
            kind = "DAC"
            port = dac_port(tile, block)
            expected_type = configured.get("generator_type")
            has_card = port in rf_out
        resolved[role] = ResolvedChannel(
            role=role,
            index=index,
            kind=kind,
            port=port,
            implementation=_implementation(entry),
            has_rf_card=has_card,
            expected_port=declared_port(configured.get("physical_port"), kind),
            expected_implementation=(
                str(expected_type) if expected_type is not None else None
            ),
        )
    return resolved


def from_soccfg(
    soccfg: Any,
    hardware_channels: Mapping[str, Any],
) -> Dict[str, ResolvedChannel]:
    return resolve_channels(
        soccfg["gens"],
        soccfg["readouts"],
        str(soccfg),
        hardware_channels,
    )


def errors(resolved: Mapping[str, ResolvedChannel]) -> List[str]:
    result = []
    for channel in resolved.values():
        if channel.port_mismatch:
            result.append(
                f"{channel.role} index {channel.index} resolves to "
                f"{channel.kind}{channel.port}, expected "
                f"{channel.kind}{channel.expected_port}"
            )
        if channel.implementation_mismatch:
            result.append(
                f"{channel.role} index {channel.index} uses "
                f"{channel.implementation}, expected "
                f"{channel.expected_implementation}"
            )
    return result


def report(resolved: Mapping[str, ResolvedChannel]) -> str:
    lines = ["Quick logical-to-physical channel map:"]
    for role in ("r", "rr", "q", "z"):
        channel = resolved[role]
        tags = [
            f"{channel.kind}{channel.port}",
            channel.implementation,
            "RF-card" if channel.has_rf_card else "direct/no detected RF card",
        ]
        if channel.expected_port is not None:
            tags.append("port OK" if not channel.port_mismatch else "PORT MISMATCH")
        if channel.expected_implementation:
            tags.append(
                "type OK"
                if not channel.implementation_mismatch
                else "TYPE MISMATCH"
            )
        lines.append(
            f"  {role:>2} = index {channel.index:<2} -> " + ", ".join(tags)
        )
    for message in errors(resolved):
        lines.append(f"  [ERROR] {message}")
    return "\n".join(lines)

