from pathlib import Path

import pytest

from quickexp_v3.errors import ExperimentError
from quickexp_v3.task_queue import run_measurement_queue


SCRIPT = """
from pathlib import Path

LIVE_HARDWARE = True
SHOW_PLOT = True
LOG_PATH = None
VALUE = ""
FAIL = False

def main():
    path = Path(LOG_PATH)
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(previous + VALUE + "\\n", encoding="utf-8")
    if FAIL:
        raise RuntimeError(VALUE)
    return VALUE
"""


def write_script(directory: Path, name: str) -> None:
    (directory / name).write_text(SCRIPT, encoding="utf-8")


def test_queue_runs_launchers_in_order_with_settings(tmp_path):
    write_script(tmp_path, "01_first.py")
    write_script(tmp_path, "02_second.py")
    log = tmp_path / "order.txt"

    result = run_measurement_queue(
        tmp_path,
        [
            {
                "file": "01_first.py",
                "settings": {"LOG_PATH": log, "VALUE": "first"},
            },
            {
                "file": "02_second.py",
                "settings": {"LOG_PATH": log, "VALUE": "second"},
            },
        ],
        shared_settings={"LIVE_HARDWARE": False, "SHOW_PLOT": False},
    )

    assert result.status == "completed"
    assert result.completed == 2
    assert log.read_text(encoding="utf-8").splitlines() == ["first", "second"]


def test_queue_can_continue_after_failure(tmp_path):
    write_script(tmp_path, "01_first.py")
    write_script(tmp_path, "02_second.py")
    log = tmp_path / "order.txt"

    result = run_measurement_queue(
        tmp_path,
        [
            {
                "file": "01_first.py",
                "settings": {
                    "LOG_PATH": log,
                    "VALUE": "failed",
                    "FAIL": True,
                },
            },
            {
                "file": "02_second.py",
                "settings": {"LOG_PATH": log, "VALUE": "continued"},
            },
        ],
        stop_on_error=False,
    )

    assert result.status == "completed_with_errors"
    assert result.failed == 1
    assert result.completed == 1
    assert log.read_text(encoding="utf-8").splitlines() == [
        "failed",
        "continued",
    ]


def test_queue_stops_on_failure_by_default(tmp_path):
    write_script(tmp_path, "01_first.py")
    log = tmp_path / "order.txt"

    with pytest.raises(ExperimentError, match="stopped"):
        run_measurement_queue(
            tmp_path,
            [
                {
                    "file": "01_first.py",
                    "settings": {
                        "LOG_PATH": log,
                        "VALUE": "failed",
                        "FAIL": True,
                    },
                }
            ],
        )
