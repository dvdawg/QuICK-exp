"""Run selected numbered measurement files sequentially from the IDE."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.task_queue import run_measurement_queue


# ============================ EDIT THESE ====================================
# This value is applied to every queued measurement that defines LIVE_HARDWARE.
LIVE_HARDWARE = True
# False lets the next task start without waiting for a plot window to close.
SHOW_PLOT = False

# Tasks run from top to bottom. Edit the numbered files themselves, or place
# one-run edits in a task's "settings" mapping. Duplicate files are allowed.
TASKS = [
    {
        "file": "08a_time_rabi.py",
        "label": "Time Rabi",
        "enabled": True,
        "settings": {"Q_FREQUENCY_MHZ" : 3950, "Q_GAIN" : 0.2,"EXTRA_PI_PULSES" : 0,"R_POWER_DB" : -30.0,"R_LENGTH_US" : 2.0,"R_OFFSET_US" : 0.495,"R_PHASE_DEG" : 0.0,"R_RELAX_US" : 20.0,"HARD_AVG" : 500,"SOFT_AVG" : 1,"REP" : 1000,"POPULATION" : False},
    },
    {
        "file": "08b_time_rabi.py",
        "label": "Time Rabi",
        "enabled": True,
        "settings": {"Q_FREQUENCY_MHZ" : 5625, "Q_GAIN" : 0.4,"EXTRA_PI_PULSES" : 0,"R_POWER_DB" : -30.0,"R_LENGTH_US" : 2.0,"R_OFFSET_US" : 0.495,"R_PHASE_DEG" : 0.0,"R_RELAX_US" : 20.0,"HARD_AVG" : 500,"SOFT_AVG" : 1,"REP" : 1000,"POPULATION" : False},
    },
]

# True stops immediately on a failed measurement. False continues the queue
# and reports completed_with_errors at the end.
STOP_ON_ERROR = False
# ============================================================================

"""
Example of how to use the task queue. Use file names!! label is just for the internal queue (doesn't really matter)
TASKS = [
    {
        "file": "08a_time_rabi.py",
        "label": "Time Rabi",
        "enabled": True,
        "settings": {"Q_FREQUENCY_MHZ": 5625,
            "Q_GAIN": 0.4,
            "R_LENGTH_US": 2.0,
        },
    },
    {
        "file": "08b_time_rabi.py",
        "label": "Time Rabi",
        "enabled": True,
        "settings": {},
    },
]

"""

def main():
    return run_measurement_queue(
        PROJECT_ROOT / "experiments",
        TASKS,
        shared_settings={
            "LIVE_HARDWARE": LIVE_HARDWARE,
            "SHOW_PLOT": SHOW_PLOT,
        },
        stop_on_error=STOP_ON_ERROR,
        forbidden_files={Path(__file__).name},
    )


if __name__ == "__main__":
    QUEUE_RESULT = main()
