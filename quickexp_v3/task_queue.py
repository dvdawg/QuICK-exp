"""Sequential execution of numbered IDE experiment launchers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import runpy
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional

from .errors import ConfigError, ExperimentError


@dataclass(frozen=True)
class QueueTask:
    """One numbered launcher plus optional edits to its ``EDIT THESE`` values."""

    file: str
    label: Optional[str] = None
    enabled: bool = True
    settings: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", str(self.file))
        object.__setattr__(self, "settings", dict(self.settings or {}))

    @property
    def display_name(self) -> str:
        return str(self.label or Path(self.file).stem)


@dataclass(frozen=True)
class QueueOutcome:
    task: QueueTask
    status: str
    elapsed_seconds: float
    result: Any = None
    error: Optional[BaseException] = None


@dataclass(frozen=True)
class QueueResult:
    outcomes: tuple[QueueOutcome, ...]
    elapsed_seconds: float

    @property
    def status(self) -> str:
        return (
            "completed"
            if all(outcome.status == "completed" for outcome in self.outcomes)
            else "completed_with_errors"
        )

    @property
    def completed(self) -> int:
        return sum(outcome.status == "completed" for outcome in self.outcomes)

    @property
    def failed(self) -> int:
        return sum(outcome.status == "failed" for outcome in self.outcomes)


def _coerce_task(value: Any) -> QueueTask:
    if isinstance(value, QueueTask):
        return value
    if isinstance(value, str):
        return QueueTask(file=value)
    if not isinstance(value, Mapping):
        raise ConfigError("queue tasks must be filenames or mappings")
    allowed = {"file", "label", "enabled", "settings"}
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(
            "unknown queue task fields: " + ", ".join(sorted(unknown))
        )
    if "file" not in value:
        raise ConfigError("each queue task mapping requires 'file'")
    settings = value.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ConfigError("queue task settings must be a mapping")
    return QueueTask(
        file=str(value["file"]),
        label=value.get("label"),
        enabled=bool(value.get("enabled", True)),
        settings=dict(settings),
    )


def _resolve_launcher(
    experiments_directory: Path,
    filename: str,
    forbidden_files: set[str],
) -> Path:
    root = Path(experiments_directory).expanduser().resolve()
    requested = Path(filename)
    if requested.name != str(requested) or requested.suffix.lower() != ".py":
        raise ConfigError(
            f"queued file must be one Python filename in {root}: {filename!r}"
        )
    if requested.name in forbidden_files:
        raise ConfigError(f"queue cannot recursively run {requested.name}")
    candidate = (root / requested.name).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise FileNotFoundError(f"queued experiment does not exist: {candidate}")
    return candidate


def run_measurement_queue(
    experiments_directory: Path,
    tasks: Iterable[Any],
    *,
    shared_settings: Optional[Mapping[str, Any]] = None,
    stop_on_error: bool = True,
    forbidden_files: Iterable[str] = (),
) -> QueueResult:
    """Run enabled numbered launchers in order using each launcher's ``main``."""
    parsed = tuple(task for task in map(_coerce_task, tasks) if task.enabled)
    if not parsed:
        raise ConfigError("measurement queue has no enabled tasks")
    common = dict(shared_settings or {})
    forbidden = {str(name) for name in forbidden_files}
    outcomes = []
    queue_started = perf_counter()

    for index, task in enumerate(parsed, start=1):
        path = _resolve_launcher(
            experiments_directory,
            task.file,
            forbidden,
        )
        prefix = f"Queue {index}/{len(parsed)}"
        print(f"{prefix} START: {task.display_name} ({path.name})")
        task_started = perf_counter()
        try:
            namespace = runpy.run_path(
                str(path),
                run_name=f"_quickexp_queue_{index}_{path.stem}",
            )
            main = namespace.get("main")
            if not callable(main):
                raise ConfigError(f"{path.name} does not expose callable main()")
            module_globals = main.__globals__
            unknown = set(task.settings) - set(module_globals)
            if unknown:
                raise ConfigError(
                    f"{path.name} has no editable setting(s): "
                    + ", ".join(sorted(unknown))
                )
            settings = {
                key: value
                for key, value in common.items()
                if key in module_globals
            }
            settings.update(task.settings)
            module_globals.update(settings)
            result = main()
        except Exception as error:
            elapsed = perf_counter() - task_started
            outcomes.append(
                QueueOutcome(
                    task=task,
                    status="failed",
                    elapsed_seconds=elapsed,
                    error=error,
                )
            )
            print(
                f"{prefix} FAILED after {elapsed:.1f} s: "
                f"{type(error).__name__}: {error}"
            )
            if stop_on_error:
                raise ExperimentError(
                    f"measurement queue stopped at {task.display_name!r}"
                ) from error
            continue

        elapsed = perf_counter() - task_started
        outcomes.append(
            QueueOutcome(
                task=task,
                status="completed",
                elapsed_seconds=elapsed,
                result=result,
            )
        )
        print(f"{prefix} DONE in {elapsed:.1f} s: {task.display_name}")

    total = perf_counter() - queue_started
    queue_result = QueueResult(tuple(outcomes), total)
    print(
        f"Queue {queue_result.status}: {queue_result.completed} completed, "
        f"{queue_result.failed} failed, {total:.1f} s total."
    )
    return queue_result
