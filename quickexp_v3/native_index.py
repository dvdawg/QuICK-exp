"""Incremental metadata index for native Quick CSV/YML pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Optional

import yaml

from .errors import AnalysisError


INDEX_VERSION = 1
_INDEX_PATTERN = re.compile(r"^(\d+)\s*-\s*(.*)$")


@dataclass(frozen=True)
class NativeRecord:
    csv_path: Path
    yml_path: Path
    index: Optional[int]
    title: str
    quick_class: Optional[str]
    independent: tuple
    dependent: tuple
    n_axes: int
    csv_rows: int
    csv_columns: int
    mtime: float
    var: dict
    has_config: bool
    z_gain: Optional[float]


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _pair_signature(csv_path: Path, yml_path: Path) -> tuple:
    csv_stat = csv_path.stat()
    yml_stat = yml_path.stat()
    return (
        int(csv_stat.st_size),
        int(csv_stat.st_mtime_ns),
        int(yml_stat.st_size),
        int(yml_stat.st_mtime_ns),
    )


def _csv_shape(path: Path) -> tuple:
    rows = 0
    columns = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            if columns == 0:
                columns = (
                    len(stripped.split(","))
                    if "," in stripped
                    else len(stripped.split())
                )
    return rows, columns


def _entries(value: Any) -> tuple:
    if not isinstance(value, list):
        return ()
    result = []
    for entry in value:
        if isinstance(entry, (list, tuple)) and entry:
            result.append(
                (
                    str(entry[0]),
                    str(entry[1]) if len(entry) > 1 else "",
                )
            )
    return tuple(result)


def _record_from_cache(data: Mapping[str, Any]) -> NativeRecord:
    return NativeRecord(
        csv_path=Path(data["csv_path"]),
        yml_path=Path(data["yml_path"]),
        index=data.get("index"),
        title=str(data.get("title", "")),
        quick_class=data.get("quick_class"),
        independent=tuple(tuple(item) for item in data.get("independent", [])),
        dependent=tuple(tuple(item) for item in data.get("dependent", [])),
        n_axes=int(data.get("n_axes", 0)),
        csv_rows=int(data.get("csv_rows", 0)),
        csv_columns=int(data.get("csv_columns", 0)),
        mtime=float(data.get("mtime", 0.0)),
        var=dict(data.get("var", {})),
        has_config=bool(data.get("has_config", False)),
        z_gain=data.get("z_gain"),
    )


def _record_to_cache(record: NativeRecord) -> dict:
    result = asdict(record)
    result["csv_path"] = str(record.csv_path)
    result["yml_path"] = str(record.yml_path)
    return result


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


class NativeIndex:
    """Versioned, fail-open cache of one native Quick data directory."""

    def __init__(
        self,
        data_directory: Path,
        cache_root: Optional[Path] = None,
    ):
        self.data_directory = Path(data_directory).expanduser().resolve()
        default_root = Path(__file__).resolve().parents[1] / "analysis_cache" / "native_index"
        self.cache_root = (
            Path(cache_root).expanduser().resolve()
            if cache_root is not None
            else default_root
        )
        digest = hashlib.sha256(
            str(self.data_directory).encode("utf-8")
        ).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.data_directory.name).strip("-")
        self.cache_path = self.cache_root / f"{slug or 'data'}-{digest}.json"
        self._records = ()
        self.warnings = ()
        self.skipped = ()

    def _load_cache(self) -> dict:
        try:
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return {}
        if (
            not isinstance(loaded, Mapping)
            or loaded.get("version") != INDEX_VERSION
            or loaded.get("data_directory") != str(self.data_directory)
        ):
            return {}
        entries = loaded.get("entries")
        return dict(entries) if isinstance(entries, Mapping) else {}

    def _parse(self, csv_path: Path, yml_path: Path) -> NativeRecord:
        metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise AnalysisError("YML does not contain a mapping")
        rows, columns = _csv_shape(csv_path)
        if rows < 2 or columns < 2:
            raise AnalysisError("CSV is empty or an aborted stub")
        independent = _entries(metadata.get("independent"))
        dependent = _entries(metadata.get("dependent"))
        raw_independent = metadata.get("independent")
        if not independent and raw_independent not in (None, []):
            raise AnalysisError("YML has invalid independent axes")
        parameters = metadata.get("parameters")
        parameters = parameters if isinstance(parameters, Mapping) else {}
        var = parameters.get("var")
        if not isinstance(var, Mapping):
            var = parameters
        var = dict(var) if isinstance(var, Mapping) else {}
        z_gain = var.get("z_gain")
        if isinstance(z_gain, bool) or isinstance(z_gain, (list, tuple, dict)):
            z_gain = None
        elif z_gain is not None:
            try:
                z_gain = float(z_gain)
            except (TypeError, ValueError):
                z_gain = None
        stem_match = _INDEX_PATTERN.match(csv_path.stem)
        index = int(stem_match.group(1)) if stem_match else None
        title = (
            str(metadata.get("title"))
            if metadata.get("title") is not None
            else (stem_match.group(2) if stem_match else csv_path.stem)
        )
        return NativeRecord(
            csv_path=csv_path.resolve(),
            yml_path=yml_path.resolve(),
            index=index,
            title=title,
            quick_class=(
                str(parameters.get("quick_experiment"))
                if parameters.get("quick_experiment") is not None
                else None
            ),
            independent=independent,
            dependent=dependent,
            n_axes=len(independent),
            csv_rows=rows,
            csv_columns=columns,
            mtime=max(csv_path.stat().st_mtime, yml_path.stat().st_mtime),
            var=var,
            has_config=isinstance(parameters.get("config"), Mapping),
            z_gain=z_gain,
        )

    def refresh(self) -> "NativeIndex":
        if not self.data_directory.exists():
            raise FileNotFoundError(
                f"Quick data directory does not exist: {self.data_directory}"
            )
        cache = self._load_cache()
        entries = {}
        records = []
        warnings = []
        paired_yml = set()
        for csv_path in sorted(self.data_directory.glob("*.csv")):
            yml_path = csv_path.with_suffix(".yml")
            if not csv_path.is_file() or not yml_path.is_file():
                warnings.append(f"{csv_path.name}: missing paired YML")
                continue
            paired_yml.add(yml_path.resolve())
            signature = None
            try:
                signature = _pair_signature(csv_path, yml_path)
                cached = cache.get(str(csv_path.resolve()))
                if (
                    isinstance(cached, Mapping)
                    and tuple(cached.get("signature", ())) == signature
                ):
                    if "record" not in cached:
                        warnings.append(
                            f"{csv_path.name}: {cached.get('skipped', 'invalid pair')}"
                        )
                        entries[str(csv_path.resolve())] = dict(cached)
                        continue
                    record = _record_from_cache(cached["record"])
                else:
                    record = self._parse(csv_path, yml_path)
                entries[str(csv_path.resolve())] = {
                    "signature": list(signature),
                    "record": _record_to_cache(record),
                }
                records.append(record)
            except (AnalysisError, OSError, ValueError, TypeError, yaml.YAMLError) as error:
                warnings.append(f"{csv_path.name}: {error}")
                if signature is not None:
                    entries[str(csv_path.resolve())] = {
                        "signature": list(signature),
                        "skipped": str(error),
                    }
        for yml_path in sorted(self.data_directory.glob("*.yml")):
            if yml_path.resolve() not in paired_yml:
                warnings.append(f"{yml_path.name}: no paired CSV")
        records.sort(
            key=lambda record: (
                record.index is None,
                record.index if record.index is not None else 0,
                record.mtime,
            )
        )
        _atomic_json(
            self.cache_path,
            {
                "version": INDEX_VERSION,
                "data_directory": str(self.data_directory),
                "entries": entries,
            },
        )
        self._records = tuple(records)
        self.warnings = tuple(warnings)
        self.skipped = self.warnings
        return self

    def records(self) -> tuple:
        return self._records

    def select(
        self,
        *,
        quick_class=None,
        axis_text=None,
        n_axes=None,
        title_contains=None,
        min_rows=2,
    ) -> tuple:
        selected = []
        for record in self._records:
            if quick_class is not None and record.quick_class != quick_class:
                continue
            if n_axes is not None and record.n_axes != int(n_axes):
                continue
            if record.csv_rows < int(min_rows):
                continue
            if title_contains is not None and str(title_contains).lower() not in record.title.lower():
                continue
            if axis_text is not None:
                needle = str(axis_text).lower()
                if not any(needle in label.lower() for label, _ in record.independent):
                    continue
            selected.append(record)
        return tuple(selected)

    def latest(self, **select_kwargs) -> NativeRecord:
        candidates = self.select(**select_kwargs)
        if not candidates:
            criteria = ", ".join(
                f"{key}={value!r}" for key, value in sorted(select_kwargs.items())
            )
            raise FileNotFoundError(
                f"No complete native Quick CSV/YML pair matching {criteria or 'selection'} "
                f"was found in {self.data_directory}."
            )
        return max(candidates, key=lambda record: record.mtime)


def find_latest_native_indexed(
    data_directory: Path,
    *,
    quick_class: str,
    axis_text: str,
) -> Path:
    """Accelerated equivalent of :func:`native_fit.find_latest_native`."""
    index = NativeIndex(data_directory).refresh()
    try:
        return index.latest(
            quick_class=quick_class,
            axis_text=axis_text,
            n_axes=1,
        ).csv_path
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"No complete one-dimensional {quick_class} CSV/YML pair "
            f"with axis {axis_text!r} was found in {index.data_directory}."
        ) from error
