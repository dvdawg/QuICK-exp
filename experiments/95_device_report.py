"""Generate a local Markdown device-state report from calibration and native data."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.device_report import generate_device_report


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
LATEST_RUNS = 20
REPORT_DATE = None
# ============================================================================


def main():
    artifacts = generate_device_report(
        PROJECT_ROOT,
        latest_runs=LATEST_RUNS,
        report_date=REPORT_DATE,
    )
    print(f"Device report: {artifacts.markdown_path}")
    for figure in artifacts.figure_paths:
        print(f"Figure: {figure}")
    if artifacts.warnings:
        print(f"Warnings: {len(artifacts.warnings)} (listed in the report)")
    return artifacts


if __name__ == "__main__":
    ARTIFACTS = main()
