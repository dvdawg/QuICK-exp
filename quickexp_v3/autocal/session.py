"""Atomic resumable session state and append-only decision events."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Optional

import yaml

from ..errors import ConfigError
from ..util import to_builtin, utc_now


_SAFE_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _atomic_state(path: Path, state: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(
                to_builtin(dict(state)),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _default_session_id(target: str) -> str:
    compact = re.sub(r"[^0-9A-Za-z]", "", utc_now())
    return f"{compact}_{target}"


class AutocalSession:
    """Own one session directory and its two small control files."""

    def __init__(self, directory: Path, state: Mapping[str, Any]):
        self.directory = Path(directory).resolve()
        self.state_path = self.directory / "state.yml"
        self.decisions_path = self.directory / "decisions.jsonl"
        self.native_directory = self.directory / "native"
        self.state = deepcopy(dict(state))

    @classmethod
    def create_or_resume(
        cls,
        project_root: Path,
        *,
        target: str,
        autonomy_level: int,
        z_gain: float,
        node_ids: Iterable[str],
        calibration_revision: int,
        session_name: Optional[str] = None,
    ) -> "AutocalSession":
        root = Path(project_root).expanduser().resolve()
        sessions = root / "autocal_runs"
        sessions.mkdir(parents=True, exist_ok=True)
        identifier = str(session_name or _default_session_id(target))
        if not _SAFE_SESSION.fullmatch(identifier):
            raise ConfigError(
                "SESSION_NAME must use only letters, numbers, '.', '-', and '_'"
            )
        directory = sessions / identifier
        state_path = directory / "state.yml"
        if state_path.exists():
            loaded = yaml.safe_load(state_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"{state_path} must contain a mapping")
            if loaded.get("session_id") != identifier:
                raise ConfigError("autocal state session_id does not match its directory")
            if loaded.get("target") != target:
                raise ConfigError(
                    f"session {identifier} targets {loaded.get('target')!r}, not {target!r}"
                )
            if int(loaded.get("autonomy_level", -1)) != int(autonomy_level):
                raise ConfigError(
                    f"session {identifier} has autonomy level "
                    f"{loaded.get('autonomy_level')}, not {autonomy_level}"
                )
            if abs(float(loaded.get("z_gain", 0.0)) - float(z_gain)) > 1.0e-12:
                raise ConfigError(
                    f"session {identifier} has z_gain {loaded.get('z_gain')}, "
                    f"not {z_gain}"
                )
            return cls(directory, loaded)

        directory.mkdir(parents=True, exist_ok=False)
        (directory / "native").mkdir()
        state = {
            "schema_version": 1,
            "session_id": identifier,
            "target": str(target),
            "autonomy_level": int(autonomy_level),
            "z_gain": float(z_gain),
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "calibration_revision": int(calibration_revision),
            "nodes": {
                str(node_id): {
                    "status": "pending",
                    "attempts": 0,
                    "last_csv": None,
                    "last_values": {},
                    "reason": "",
                }
                for node_id in node_ids
            },
            "working_values": {},
            "budget": {"spent_seconds": 0.0, "total_runs": 0},
        }
        session = cls(directory, state)
        session.save()
        session.event(
            "session_started",
            decision="start",
            reason="new session",
            target=str(target),
            autonomy_level=int(autonomy_level),
            z_gain=float(z_gain),
            calibration_revision=int(calibration_revision),
        )
        return session

    @classmethod
    def load(cls, directory: Path) -> "AutocalSession":
        location = Path(directory).expanduser().resolve()
        state_path = location / "state.yml"
        loaded = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"{state_path} must contain a mapping")
        return cls(location, loaded)

    @property
    def session_id(self) -> str:
        return str(self.state["session_id"])

    @property
    def stop_path(self) -> Path:
        return self.directory.parent / "STOP"

    def stop_requested(self) -> bool:
        return self.stop_path.is_file()

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        _atomic_state(self.state_path, self.state)

    def event(self, event: str, *, node: Optional[str] = None, **details: Any) -> dict:
        payload = {
            "time": utc_now(),
            "event": str(event),
        }
        if node is not None:
            payload["node"] = str(node)
        payload.update(to_builtin(details))
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.decisions_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def node(self, node_id: str) -> dict:
        nodes = self.state.setdefault("nodes", {})
        if node_id not in nodes:
            nodes[node_id] = {
                "status": "pending",
                "attempts": 0,
                "last_csv": None,
                "last_values": {},
                "reason": "",
            }
        return nodes[node_id]

    def update_node(self, node_id: str, **changes: Any) -> None:
        self.node(node_id).update(to_builtin(changes))
        self.save()

    def set_working_values(self, values: Mapping[str, Any]) -> None:
        working = self.state.setdefault("working_values", {})
        working.update(to_builtin(dict(values)))
        self.save()

    def set_budget(self, budget: Mapping[str, Any]) -> None:
        self.state["budget"] = to_builtin(dict(budget))
        self.save()

    def finish(self, status: str) -> None:
        self.state["status"] = str(status)
        self.state["finished_at"] = utc_now()
        self.save()

    def events(self) -> list:
        if not self.decisions_path.exists():
            return []
        result = []
        for line in self.decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(json.loads(line))
        return result


def replay_decisions(directory: Path) -> list:
    """Read an audit log without touching hardware or calibration state."""
    return AutocalSession.load(directory).events()
