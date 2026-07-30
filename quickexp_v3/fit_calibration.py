"""Shared atomic writer for explicitly accepted analysis results."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml

from .config import ConfigRepository, SCHEMA_VERSION
from .errors import ConfigError
from .util import to_builtin


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


def write_calibration_records(
    project_root: Path,
    updates: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Atomically write accepted records addressed as ``section.name``.

    Examples are ``defaults.r_freq``, ``defaults.r_offset``, and
    ``derived.t1``. Existing records are copied to calibration history before
    they are superseded.
    """
    root = Path(project_root).expanduser().resolve()
    target = root / "calibration.yml"
    source = target if target.exists() else root / "calibration.example.yml"
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{source} must contain a YAML mapping")
    document = deepcopy(dict(loaded))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"calibration schema_version must be {SCHEMA_VERSION}"
        )

    records = document.setdefault("records", {})
    if not isinstance(records, dict):
        raise ConfigError("calibration records must be a mapping")
    history = document.setdefault("history", [])
    if not isinstance(history, list):
        raise ConfigError("calibration history must be a list")

    from .util import utc_now

    for address, record in updates.items():
        parts = str(address).split(".")
        if len(parts) != 2 or not all(parts):
            raise ConfigError(
                "calibration record addresses must have the form section.name"
            )
        section_name, record_name = parts
        section = records.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ConfigError(
                f"calibration records.{section_name} must be a mapping"
            )
        previous = deepcopy(section.get(record_name))
        section[record_name] = deepcopy(dict(record))
        if previous is not None:
            history.append(
                {
                    "record": address,
                    "superseded_at": utc_now(),
                    "previous": previous,
                }
            )

    document["revision"] = int(document.get("revision", 0)) + 1
    document["updated_at"] = date.today().isoformat()

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
