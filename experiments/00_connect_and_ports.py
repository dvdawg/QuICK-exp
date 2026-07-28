"""Connect to MET v191 and verify Quick logical channels before experiments."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.ide import inspect_connection


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = True
# None follows hardware.yml; True forces reviewed RF-board settings to apply.
APPLY_RF_BOARD = None
# ============================================================================


def main():
    connection = inspect_connection(
        PROJECT_ROOT,
        live_hardware=LIVE_HARDWARE,
        apply_rf_board=APPLY_RF_BOARD,
    )
    if connection is not None:
        try:
            print("\nFull connected soccfg:\n")
            print(connection.soccfg)
        finally:
            connection.close()
            print("Connection inspection complete; generators reset.")
    return connection


if __name__ == "__main__":
    CONNECTION = main()

