"""Small dependency-free helpers shared by configuration and safety."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(*documents: Optional[Mapping[str, Any]]) -> dict:
    result: dict = {}
    for document in documents:
        if document:
            _merge_into(result, document)
    return result


def _merge_into(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if (
            isinstance(value, Mapping)
            and isinstance(target.get(key), Mapping)
        ):
            child = deepcopy(dict(target[key]))
            _merge_into(child, value)
            target[key] = child
        else:
            target[key] = deepcopy(value)


def dotted_get(document: Mapping[str, Any], path: str, default: Any = None) -> Any:
    node: Any = document
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return deepcopy(node)


def dotted_set(document: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = document
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = deepcopy(value)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_builtin(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

