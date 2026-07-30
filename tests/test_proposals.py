from pathlib import Path
import shutil

import pytest
import yaml

from quickexp_v3.config import ConfigRepository, accepted_calibration_values
from quickexp_v3.errors import ConfigError
from quickexp_v3.fit_calibration import (
    list_open_proposals,
    promote_proposal,
    reject_proposal,
    write_calibration_proposals,
    write_calibration_records,
)


ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path):
    for name in (
        "hardware.example.yml",
        "calibration.example.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copy2(
        ROOT / "calibration.example.yml",
        tmp_path / "calibration.yml",
    )


def _repository(root):
    calibration = (
        root / "calibration.yml"
        if (root / "calibration.yml").exists()
        else root / "calibration.example.yml"
    )
    return ConfigRepository.from_files(
        root / "hardware.example.yml",
        calibration,
        root / "presets.example.yml",
    )


def _proposal(value=5606.8, *, source="first.csv"):
    return {
        "record": "defaults.q_freq",
        "value": value,
        "unit": "MHz",
        "uncertainty": {"center_mhz": 0.04},
        "provenance": {
            "source": source,
            "fitted_at": "2026-07-29T03:15:00+00:00",
            "analysis": "test",
            "autocal_session": "session-a",
            "autocal_node": "N5",
            "working_z_gain": 0.14,
        },
        "quality": {"r_squared": 0.94},
        "valid_domain": {"z_gain": [0.14, 0.14]},
        "model": "signed_lorentzian",
        "status": "proposed",
    }


def test_proposals_are_inert_to_resolution_and_retake_replaces(tmp_path):
    _project(tmp_path)
    before = _repository(tmp_path).resolve("qubit_fine")
    original_revision = before.calibration["revision"]

    write_calibration_proposals(
        tmp_path,
        {"old-proposal": _proposal()},
    )
    after = _repository(tmp_path).resolve("qubit_fine")
    assert after.config_fingerprint == before.config_fingerprint
    assert after.parameters["q_freq"] == before.parameters["q_freq"]
    assert after.calibration["revision"] == original_revision

    write_calibration_proposals(
        tmp_path,
        {"retake-proposal": _proposal(5606.9, source="retake.csv")},
    )
    proposals = list_open_proposals(tmp_path)
    assert [proposal_id for proposal_id, _ in proposals] == ["retake-proposal"]
    assert proposals[0][1]["value"] == 5606.9
    assert proposals[0][1]["proposal_id"] == "retake-proposal"


def test_promote_stamps_lifecycle_bumps_revision_and_versions_previous(tmp_path):
    _project(tmp_path)
    initial = _repository(tmp_path).calibration["revision"]
    write_calibration_proposals(
        tmp_path,
        {"qfreq-proposal": _proposal()},
    )

    path = promote_proposal(
        tmp_path,
        "qfreq-proposal",
        accepted_by="dk",
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    record = document["records"]["defaults"]["q_freq"]

    assert record["status"] == "accepted"
    assert record["proposal_id"] == "qfreq-proposal"
    assert record["created_at"]
    assert record["accepted_at"]
    assert record["accepted_by"] == "dk"
    assert record["accepted_revision"] == initial + 1
    assert document["revision"] == initial + 1
    assert document["proposals"] == {}
    assert document["history"][-1]["record"] == "defaults.q_freq"
    assert _repository(tmp_path).resolve("qubit_fine").references[
        "q_freq"
    ] == pytest.approx(5606.8)


def test_reject_archives_without_touching_accepted_record_or_revision(tmp_path):
    _project(tmp_path)
    original = _repository(tmp_path).calibration
    accepted = original["records"]["defaults"]["q_freq"]["value"]
    write_calibration_proposals(
        tmp_path,
        {"reject-me": _proposal()},
    )
    path = reject_proposal(
        tmp_path,
        "reject-me",
        reason="ambiguous second line",
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert document["records"]["defaults"]["q_freq"]["value"] == accepted
    assert document["revision"] == original["revision"]
    assert document["proposals"] == {}
    archived = document["history"][-1]
    assert archived["proposal_id"] == "reject-me"
    assert archived["reason"] == "ambiguous second line"
    assert archived["rejected"]["record"] == "defaults.q_freq"


def test_arbitrary_depth_records_resolve_and_domain_extrapolation_blocks_promotion(
    tmp_path,
):
    _project(tmp_path)
    path = write_calibration_records(
        tmp_path,
        {
            "derived.t2_echo.cycle_1": {
                "value": 12.5,
                "unit": "us",
                "uncertainty": {"decay_us": 0.4},
                "provenance": {"source": "echo.csv"},
                "quality": {"r_squared": 0.9},
                "valid_domain": {},
                "model": "stretched_exponential",
                "status": "accepted",
            }
        },
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = accepted_calibration_values(document)
    assert values["derived"]["t2_echo"]["cycle_1"] == 12.5

    invalid = _proposal()
    invalid["valid_domain"] = {"z_gain": [-0.1, 0.1]}
    write_calibration_proposals(tmp_path, {"outside": invalid})
    with pytest.raises(ConfigError, match="outside its measured domain"):
        promote_proposal(tmp_path, "outside", accepted_by="dk")

    malformed = _proposal()
    malformed["valid_domain"] = {"z_gain": [float("nan"), 0.2]}
    write_calibration_proposals(tmp_path, {"nonfinite": malformed})
    with pytest.raises(ConfigError, match="ordered finite"):
        promote_proposal(tmp_path, "nonfinite", accepted_by="dk")
