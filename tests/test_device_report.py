from datetime import date
from pathlib import Path
import shutil
import yaml

from quickexp_v3.device_report import generate_device_report


ROOT = Path(__file__).resolve().parents[1]


def _copy_project(tmp_path):
    shutil.copy2(
        ROOT / "hardware.example.yml",
        tmp_path / "hardware.example.yml",
    )
    shutil.copy2(
        ROOT / "calibration.example.yml",
        tmp_path / "calibration.yml",
    )
    shutil.copy2(
        ROOT / "presets.example.yml",
        tmp_path / "presets.example.yml",
    )


def test_device_report_contains_calibration_trends_qc_and_flux_figure(tmp_path):
    _copy_project(tmp_path)
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


def test_device_report_renders_top_level_proposal_records(tmp_path):
    _copy_project(tmp_path)
    document = yaml.safe_load(
        (tmp_path / "calibration.yml").read_text(encoding="utf-8")
    )
    document["proposals"] = {
        "proposal-1": {
            "proposal_id": "proposal-1",
            "record": "defaults.q_freq",
            "value": 5606.75,
            "status": "proposed",
        }
    }
    (tmp_path / "calibration.yml").write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    artifacts = generate_device_report(
        tmp_path,
        output_directory=tmp_path / "out",
        trend_directory=tmp_path / "cache",
        report_date=date(2026, 7, 29),
    )
    report = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "| defaults.q_freq | proposal-1 | proposed | 5606.75 |" in report
