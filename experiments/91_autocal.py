"""Run or resume the policy-governed automated calibration graph."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.autocal import run_autocal


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
SESSION_NAME = None
TARGET = "full_cold_start"
Z_GAIN = 0.0
AUTONOMY_LEVEL = 0
MAX_WALL_CLOCK_HOURS = 8.0
REPLAY_SESSION = None
SHOW_PLOT = False
# ============================================================================


def main():
    result = run_autocal(
        PROJECT_ROOT,
        target=TARGET,
        autonomy_level=AUTONOMY_LEVEL,
        z_gain=Z_GAIN,
        session_name=SESSION_NAME,
        max_wall_clock_hours=MAX_WALL_CLOCK_HOURS,
        live_hardware=LIVE_HARDWARE,
        replay_session=REPLAY_SESSION,
    )
    if hasattr(result, "as_dict"):
        print("Autocal summary:")
        for key, value in result.as_dict().items():
            print(f"  {key}: {value}")
    else:
        print(
            f"Replayed {len(result.events)} decision events and verified "
            f"{len(result.verified_fits)} fit verdicts from "
            f"{result.session_directory}."
        )
    return result


if __name__ == "__main__":
    AUTOCAL_RESULT = main()
