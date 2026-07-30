"""Explicit dependency graph for the single-device calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from ..errors import ConfigError


@dataclass(frozen=True)
class NodeSpec:
    """One auditable calibration step and the records it can produce."""

    node_id: str
    name: str
    dependencies: Tuple[str, ...] = ()
    products: Tuple[str, ...] = ()
    optional: bool = False
    invalidates: Tuple[str, ...] = ()
    invalidation_threshold: Optional[float] = None
    invalidation_relative: bool = False
    invalidation_products: Tuple[str, ...] = ()


NODE_REGISTRY: Mapping[str, NodeSpec] = {
    "N0": NodeSpec("N0", "connect and verify ports", products=("session.facts",)),
    "N1": NodeSpec(
        "N1",
        "loopback timing",
        dependencies=("N0",),
        products=("defaults.r_offset",),
        invalidates=(
            "N2",
            "N3",
            "N4",
            "N5",
            "N8",
            "N9",
            "N11",
            "N12",
            "N13",
        ),
    ),
    "N2": NodeSpec(
        "N2",
        "readout punchout",
        dependencies=("N1",),
        products=("defaults.r_power", "session.f_r_plateau"),
        invalidates=("N4", "N9"),
        invalidation_threshold=3.0,
    ),
    "N3": NodeSpec(
        "N3",
        "resonator versus flux",
        dependencies=("N2",),
        products=("lookups.resonator_vs_flux",),
        invalidates=("N4",),
    ),
    "N4": NodeSpec(
        "N4",
        "readout frequency at working flux",
        dependencies=("N2", "N3"),
        products=("defaults.r_freq",),
        invalidates=("N5", "N8", "N9"),
        invalidation_threshold=0.5,
    ),
    "N5": NodeSpec(
        "N5",
        "qubit spectroscopy coarse to fine",
        dependencies=("N4",),
        products=("defaults.q_freq",),
        invalidates=("N7", "N8", "N11", "N12", "N13"),
        invalidation_threshold=1.0,
    ),
    "N6": NodeSpec(
        "N6",
        "qubit versus flux",
        dependencies=("N5",),
        products=("lookups.qubit_vs_flux",),
        optional=True,
    ),
    "N7": NodeSpec(
        "N7",
        "Rabi chevron",
        dependencies=("N5",),
        products=("session.rabi_chevron",),
        optional=True,
    ),
    "N8": NodeSpec(
        "N8",
        "pi pulse calibration",
        dependencies=("N5",),
        products=("defaults.q_length",),
        invalidates=("N9", "N11", "N12", "N13"),
        invalidation_threshold=0.15,
        invalidation_relative=True,
    ),
    "N9": NodeSpec(
        "N9",
        "IQ discrimination",
        dependencies=("N8",),
        products=("defaults.r_threshold", "derived.readout_fidelity"),
    ),
    "N10r": NodeSpec(
        "N10r",
        "readout Bayesian optimization",
        dependencies=("N9",),
        products=("defaults.r_freq", "defaults.r_power"),
        optional=True,
        invalidates=("N9",),
    ),
    "N11": NodeSpec(
        "N11",
        "T1",
        dependencies=("N8",),
        products=("derived.t1",),
    ),
    "N12": NodeSpec(
        "N12",
        "Ramsey and detuning sign",
        dependencies=("N8",),
        products=("derived.t2_ramsey", "defaults.q_freq"),
        invalidates=("N13",),
        invalidation_products=("defaults.q_freq",),
    ),
    "N13": NodeSpec(
        "N13",
        "echo",
        dependencies=("N12",),
        products=("derived.t2_echo.cycle_0",),
    ),
    "N14": NodeSpec(
        "N14",
        "two-photon spectroscopy",
        dependencies=("N8",),
        products=("derived.anharmonicity",),
        optional=True,
    ),
}


TARGETS: Mapping[str, Tuple[str, ...]] = {
    "full_cold_start": (
        "N0",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N8",
        "N9",
        "N11",
        "N12",
        "N13",
    ),
    "flux_point": ("N4", "N5", "N8", "N9", "N11", "N12", "N13"),
    "readout_only": ("N4", "N9", "N10r"),
    "coherence_only": ("N11", "N12", "N13"),
}


def validate_graph(registry: Mapping[str, NodeSpec] = NODE_REGISTRY) -> None:
    """Reject missing dependencies and cycles at import/test time."""
    for node_id, spec in registry.items():
        if node_id != spec.node_id:
            raise ConfigError(
                f"autocal registry key {node_id!r} does not match {spec.node_id!r}"
            )
        missing = set(spec.dependencies).difference(registry)
        if missing:
            raise ConfigError(
                f"autocal node {node_id} has unknown dependencies: "
                + ", ".join(sorted(missing))
            )

    visiting = set()
    visited = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ConfigError(f"autocal graph contains a cycle at {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in registry[node_id].dependencies:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in registry:
        visit(node_id)


def target_nodes(target: str) -> Tuple[NodeSpec, ...]:
    """Return a target in registry/topological order."""
    name = str(target)
    if name not in TARGETS:
        raise ConfigError(
            f"unknown autocal target {name!r}; choose from "
            + ", ".join(sorted(TARGETS))
        )
    return tuple(NODE_REGISTRY[node_id] for node_id in TARGETS[name])


def downstream(node_id: str) -> Tuple[str, ...]:
    """Return all transitive consumers of ``node_id`` in registry order."""
    if node_id not in NODE_REGISTRY:
        raise ConfigError(f"unknown autocal node {node_id!r}")
    found = set()
    changed = True
    while changed:
        changed = False
        for candidate, spec in NODE_REGISTRY.items():
            if candidate in found:
                continue
            if node_id in spec.dependencies or found.intersection(spec.dependencies):
                found.add(candidate)
                changed = True
    return tuple(candidate for candidate in NODE_REGISTRY if candidate in found)


def invalidated_nodes(node_id: str) -> Tuple[str, ...]:
    """Expand one node's explicit invalidation roots transitively."""
    if node_id not in NODE_REGISTRY:
        raise ConfigError(f"unknown autocal node {node_id!r}")
    affected = set()
    for root in NODE_REGISTRY[node_id].invalidates:
        affected.add(root)
        affected.update(downstream(root))
    return tuple(candidate for candidate in NODE_REGISTRY if candidate in affected)


def change_exceeds_invalidation_threshold(
    node_id: str,
    previous_value: object,
    current_value: object,
) -> bool:
    """Return whether an accepted move requires downstream recalibration."""
    if node_id not in NODE_REGISTRY:
        raise ConfigError(f"unknown autocal node {node_id!r}")
    spec = NODE_REGISTRY[node_id]
    if spec.invalidation_threshold is None:
        return True
    try:
        previous = float(previous_value)
        current = float(current_value)
    except (TypeError, ValueError):
        return True
    delta = abs(current - previous)
    if spec.invalidation_relative:
        scale = max(abs(previous), 1.0e-15)
        delta /= scale
    return delta > float(spec.invalidation_threshold)


validate_graph()
