"""Shared atomic writer for accepted records and inert fit proposals."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import yaml

from .config import ConfigRepository, SCHEMA_VERSION
from .errors import ConfigError
from .util import to_builtin


_ADDRESS_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _atomic_yaml(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            yaml.safe_dump(
                to_builtin(document),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _project_paths(project_root: Path) -> tuple:
    root = Path(project_root).expanduser().resolve()
    target = root / "calibration.yml"
    source = target if target.exists() else root / "calibration.example.yml"
    return root, target, source


def _load_document(project_root: Path) -> tuple:
    root, target, source = _project_paths(project_root)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{source} must contain a YAML mapping")
    document = deepcopy(dict(loaded))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"calibration schema_version must be {SCHEMA_VERSION}"
        )

    return root, target, document


def _address_parts(address: str) -> tuple:
    parts = tuple(str(address).split("."))
    if len(parts) < 2 or not all(
        part and _ADDRESS_PART.fullmatch(part) for part in parts
    ):
        raise ConfigError(
            "calibration record addresses must contain at least two safe "
            "dot-separated names"
        )
    return parts


def _record_parent(records: dict, address: str) -> tuple:
    parts = _address_parts(address)
    node = records
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(
                f"calibration records.{'.'.join(parts[:-1])} must be a mapping"
            )
        node = child
    return node, parts[-1]


def _install_records(
    document: dict,
    updates: Mapping[str, Mapping[str, Any]],
    *,
    superseded_at: str,
) -> None:
    records = document.setdefault("records", {})
    if not isinstance(records, dict):
        raise ConfigError("calibration records must be a mapping")
    history = document.setdefault("history", [])
    if not isinstance(history, list):
        raise ConfigError("calibration history must be a list")
    for address, record in updates.items():
        if not isinstance(record, Mapping) or "value" not in record:
            raise ConfigError(
                f"calibration update {address!r} must be a record with value"
            )
        parent, record_name = _record_parent(records, str(address))
        previous = deepcopy(parent.get(record_name))
        parent[record_name] = deepcopy(dict(record))
        if previous is not None:
            history.append(
                {
                    "record": address,
                    "superseded_at": superseded_at,
                    "previous": previous,
                }
            )


def _validate_and_write(root: Path, target: Path, document: dict) -> Path:
    hardware_path = (
        root / "hardware.yml"
        if (root / "hardware.yml").exists()
        else root / "hardware.example.yml"
    )
    presets_path = (
        root / "presets.yml"
        if (root / "presets.yml").exists()
        else root / "presets.example.yml"
    )
    repository = ConfigRepository.from_files(
        hardware_path,
        None,
        presets_path,
    )
    ConfigRepository(
        repository.hardware,
        document,
        {"schema_version": SCHEMA_VERSION, "presets": repository.presets},
    )
    _atomic_yaml(target, document)
    return target


def write_calibration_records(
    project_root: Path,
    updates: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Atomically write accepted records at arbitrary dotted addresses."""
    from .util import utc_now

    root, target, document = _load_document(project_root)
    now = utc_now()
    _install_records(document, updates, superseded_at=now)
    document["revision"] = int(document.get("revision", 0)) + 1
    document["updated_at"] = date.today().isoformat()
    return _validate_and_write(root, target, document)


def _proposal_identity(proposal: Mapping[str, Any]) -> tuple:
    provenance = proposal.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    return (
        proposal.get("record"),
        provenance.get("autocal_session"),
        provenance.get("autocal_node"),
    )


def write_calibration_proposals(
    project_root: Path,
    proposals: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Atomically add inert proposals, replacing retakes from the same node."""
    from .util import utc_now

    root, target, document = _load_document(project_root)
    open_proposals = document.setdefault("proposals", {})
    if not isinstance(open_proposals, dict):
        raise ConfigError("calibration proposals must be a mapping")
    for proposal_id, raw in proposals.items():
        identifier = str(proposal_id).strip()
        if not identifier or "\n" in identifier:
            raise ConfigError("proposal ids must be non-empty single-line strings")
        if not isinstance(raw, Mapping):
            raise ConfigError(f"proposal {identifier!r} must be a mapping")
        proposal = deepcopy(dict(raw))
        address = proposal.get("record")
        _address_parts(str(address))
        if "value" not in proposal:
            raise ConfigError(f"proposal {identifier!r} has no value")
        if proposal.get("status", "proposed") != "proposed":
            raise ConfigError(
                f"proposal {identifier!r} status must be proposed"
            )
        proposal["status"] = "proposed"
        proposal["proposal_id"] = identifier
        proposal.setdefault("created_at", utc_now())
        identity = _proposal_identity(proposal)
        if identity[1] is not None and identity[2] is not None:
            for existing_id, existing in tuple(open_proposals.items()):
                if (
                    existing_id != identifier
                    and isinstance(existing, Mapping)
                    and _proposal_identity(existing) == identity
                ):
                    del open_proposals[existing_id]
        open_proposals[identifier] = proposal
    # Proposal-only changes deliberately do not bump the accepted-calibration
    # revision. ResolvedConfig fingerprints therefore remain stable.
    return _validate_and_write(root, target, document)


def list_open_proposals(project_root: Path) -> list:
    """Return open proposals as sorted ``(proposal_id, mapping)`` pairs."""
    _root, _target, document = _load_document(project_root)
    proposals = document.get("proposals", {})
    if proposals is None:
        return []
    if not isinstance(proposals, Mapping):
        raise ConfigError("calibration proposals must be a mapping")
    return [
        (str(proposal_id), deepcopy(dict(proposal)))
        for proposal_id, proposal in sorted(proposals.items())
        if isinstance(proposal, Mapping)
        and proposal.get("status", "proposed") == "proposed"
    ]


def _validate_proposal_domain(proposal: Mapping[str, Any]) -> None:
    provenance = proposal.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    working_z = provenance.get(
        "working_z_gain",
        provenance.get("autocal_z_gain"),
    )
    domain = proposal.get("valid_domain")
    z_domain = domain.get("z_gain") if isinstance(domain, Mapping) else None
    if working_z is None or z_domain is None:
        return
    try:
        minimum, maximum = map(float, z_domain)
        value = float(working_z)
    except (TypeError, ValueError) as error:
        raise ConfigError(
            "proposal valid_domain.z_gain must contain two finite values"
        ) from error
    if (
        not all(math.isfinite(item) for item in (minimum, maximum, value))
        or minimum > maximum
    ):
        raise ConfigError(
            "proposal valid_domain.z_gain must contain two ordered finite values"
        )
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"proposal working z_gain {value} is outside its measured "
            f"domain [{minimum}, {maximum}]"
        )


def promote_proposal(
    project_root: Path,
    proposal_id: str,
    *,
    accepted_by: str,
) -> Path:
    """Atomically promote one proposal through the accepted-record machinery."""
    from .util import utc_now

    operator = str(accepted_by).strip()
    if not operator:
        raise ConfigError("accepted_by is required to promote a proposal")
    root, target, document = _load_document(project_root)
    proposals = document.get("proposals", {})
    if not isinstance(proposals, dict) or proposal_id not in proposals:
        raise ConfigError(f"unknown open proposal {proposal_id!r}")
    proposal = deepcopy(dict(proposals[proposal_id]))
    if proposal.get("status", "proposed") != "proposed":
        raise ConfigError(f"proposal {proposal_id!r} is not open")
    _validate_proposal_domain(proposal)
    address = str(proposal.pop("record"))
    next_revision = int(document.get("revision", 0)) + 1
    now = utc_now()
    record = {
        key: value
        for key, value in proposal.items()
        if key not in {"rejected_at", "rejection_reason"}
    }
    record.update(
        {
            "status": "accepted",
            "proposal_id": str(proposal_id),
            "created_at": proposal.get("created_at", now),
            "accepted_at": now,
            "accepted_by": operator,
            "accepted_revision": next_revision,
        }
    )
    _install_records(
        document,
        {address: record},
        superseded_at=now,
    )
    del proposals[proposal_id]
    document["revision"] = next_revision
    document["updated_at"] = date.today().isoformat()
    return _validate_and_write(root, target, document)


def reject_proposal(
    project_root: Path,
    proposal_id: str,
    *,
    reason: str,
) -> Path:
    """Archive an open proposal as rejected without touching accepted records."""
    from .util import utc_now

    explanation = str(reason).strip()
    if not explanation:
        raise ConfigError("a rejection reason is required")
    root, target, document = _load_document(project_root)
    proposals = document.get("proposals", {})
    if not isinstance(proposals, dict) or proposal_id not in proposals:
        raise ConfigError(f"unknown open proposal {proposal_id!r}")
    proposal = deepcopy(dict(proposals.pop(proposal_id)))
    history = document.setdefault("history", [])
    if not isinstance(history, list):
        raise ConfigError("calibration history must be a list")
    history.append(
        {
            "proposal_id": str(proposal_id),
            "rejected_at": utc_now(),
            "reason": explanation,
            "rejected": proposal,
        }
    )
    return _validate_and_write(root, target, document)
