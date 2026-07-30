from datetime import date
from pathlib import Path
import shutil

from quickexp_v3.device_report import generate_device_report


ROOT = Path(__file__).resolve().parents[1]


def test_device_report_contains_calibration_trends_qc_and_flux_figure(tmp_path):
    for name in (
        "hardware.example.yml",
        "calibration.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    (tmp_path / "data").mkdir()

    artifacts = generate_device_report(
        tmp_path,
        output_directory=tmp_path / "out",
        trend_directory=tmp_path / "cache",
        report_date=date(2026, 7, 29),
    )
    report = artifacts.markdown_path.read_text(encoding="utf-8")

    assert artifacts.markdown_path.name == "report_2026-07-29.md"
    assert "## Current accepted calibration" in report
    assert "lookups.resonator_vs_flux" in report
    assert "## Trends" in report
    assert "## Recent native-data QC" in report
    assert "## Open calibration proposals" in report
    assert len(artifacts.figure_paths) == 1
    assert artifacts.figure_paths[0].is_file()
