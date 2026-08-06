import json
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from quickexp_v3.autocal import run_autocal
from quickexp_v3.autocal import nodes as autocal_nodes
from quickexp_v3.autocal import orchestrator as autocal_orchestrator
from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.errors import AcquisitionError, AnalysisError, ConfigError
from quickexp_v3.fit_calibration import (
    list_open_proposals,
    promote_proposal,
    write_calibration_proposals,
)
from quickexp_v3.synthetic_device import DeviceModel


ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path):
    for name in (
        "hardware.example.yml",
        "calibration.example.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)


def test_full_simulated_cold_start_completes_through_real_ide_path(tmp_path):
    _project(tmp_path)
    result = run_autocal(
        tmp_path,
        target="full_cold_start",
        autonomy_level=0,
        max_wall_clock_hours=8.0,
    )
    assert result.status == "completed"
    assert set(result.node_status.values()) == {"done"}
    assert result.total_runs >= 20

    proposals = dict(list_open_proposals(tmp_path))
    addresses = {proposal["record"] for proposal in proposals.values()}
    assert {
        "defaults.r_offset",
        "defaults.r_power",
        "lookups.resonator_vs_flux",
        "defaults.r_freq",
        "defaults.q_freq",
        "defaults.q_length",
        "defaults.r_threshold",
        "derived.readout_fidelity",
        "derived.t1",
        "derived.t2_ramsey",
        "derived.t2_echo.cycle_0",
    } <= addresses
    calibration = yaml.safe_load(
        (tmp_path / "calibration.yml").read_text(encoding="utf-8")
    )
    assert calibration["revision"] == 2
    assert all(
        proposal["provenance"]["autocal_session"] == result.session_id
        for proposal in proposals.values()
    )
    assert all(
        proposal["quality"]["autocal_gates_passed"] is True
        and proposal["quality"]["autocal_gates"]
        for proposal in proposals.values()
    )

    events = [
        json.loads(line)
        for line in (result.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event"] == "session_completed"
    assert all("signals" not in event for event in events)
    assert any(event["event"] == "fit_evaluated" for event in events)
    replay = run_autocal(
        tmp_path,
        replay_session=result.session_directory,
    )
    # One verified fit per fitting node, plus one for any node the coverage
    # check legitimately retook. Pinning an exact total made this brittle: a
    # narrower, more honest linewidth makes the resolution check ask for a
    # finer sweep, which is the intended behaviour and adds a fit.
    assert {
        verification.node_id for verification in replay.verified_fits
    } == {"N1", "N2", "N3", "N4", "N5", "N8", "N9", "N11", "N12", "N13"}
    assert 10 <= len(replay.verified_fits) <= 12
    assert all(verification.matches for verification in replay.verified_fits)


def test_n5_hypothesis_path_is_opt_in_audited_and_writes_ledgers(tmp_path):
    _project(tmp_path)
    hardware_path = tmp_path / "hardware.example.yml"
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware["autocal"]["hypothesis_nodes"] = ["N5"]
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False),
        encoding="utf-8",
    )
    result = run_autocal(
        tmp_path,
        target="flux_point",
        autonomy_level=0,
        backend=SyntheticBackend(
            seed=17,
            device=autocal_orchestrator._default_device(),
        ),
    )
    state = yaml.safe_load(
        (result.session_directory / "state.yml").read_text(encoding="utf-8")
    )
    assert state["nodes"]["N5"]["status"] == "done"
    assert state["hypothesis_ledger"]["addresses"]["defaults.q_freq"]
    entries = state["discrepancy_ledger"]["entries"]
    assert {entry["prediction_id"] for entry in entries} == {
        "f_q_band",
        "chi",
        "flux_period_agreement",
        "anharmonicity",
        "t2_bound",
        "rabi_gain_linearity",
        "readout_fidelity_vs_snr",
    }
    assert next(
        entry for entry in entries if entry["prediction_id"] == "t2_bound"
    )["verdict"] == "consistent"
    assert (result.session_directory / "discrepancy-report.md").is_file()
    events = [
        json.loads(line)
        for line in (result.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    hp_event = next(event for event in events if event["event"] == "hypothesis_adjudicated")
    assert hp_event["decision"] == "accept"
    assert hp_event["adjudication"]["hypothesis_id"] == "qubit_01"
    assert "r_squared" not in json.dumps(hp_event["scorecard"])
    replay = run_autocal(tmp_path, replay_session=result.session_directory)
    assert all(item.matches for item in replay.verified_fits)
    assert any(item.node_id == "N5" for item in replay.verified_fits)


def test_failed_rabi_revises_joint_frequency_gain_choice_before_blocking(
    tmp_path,
    monkeypatch,
):
    _project(tmp_path)
    hardware_path = tmp_path / "hardware.example.yml"
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware["autocal"]["hypothesis_nodes"] = ["N5"]
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False),
        encoding="utf-8",
    )
    original_fit_rabi = autocal_nodes.fit_rabi
    calls = {"count": 0}

    def fail_first_rabi_attempts(csv_path, *args, **kwargs):
        fit = original_fit_rabi(csv_path, *args, **kwargs)
        calls["count"] += 1
        if calls["count"] <= int(
            hardware["autocal"]["budgets"]["max_node_attempts"]
        ):
            statistics = dict(fit.statistics)
            statistics["r_squared"] = -1.0
            return replace(fit, statistics=statistics)
        return fit

    monkeypatch.setattr(autocal_nodes, "fit_rabi", fail_first_rabi_attempts)
    result = run_autocal(
        tmp_path,
        target="flux_point",
        autonomy_level=0,
    )
    state = yaml.safe_load(
        (result.session_directory / "state.yml").read_text(encoding="utf-8")
    )
    assert state["nodes"]["N8"]["status"] == "done"
    entry = state["hypothesis_ledger"]["addresses"]["defaults.q_freq"]
    assert entry["joint_retune_attempted"] is True
    assert state["hypothesis_ledger"]["total_backtracks"] == 0
    assert state["working_values"]["defaults.q_gain"] != pytest.approx(0.3)
    events = [
        json.loads(line)
        for line in (result.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recovery = next(event for event in events if event["event"] == "upstream_doubt")
    assert recovery["decision"] == "retune_joint_operating_point"


def test_n2_n3_adaptive_paths_fit_with_at_most_seven_rows(tmp_path):
    _project(tmp_path)
    hardware_path = tmp_path / "hardware.example.yml"
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware["autocal"]["adaptive_nodes"] = ["N2", "N3"]
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False),
        encoding="utf-8",
    )
    result = run_autocal(
        tmp_path,
        target="full_cold_start",
        autonomy_level=0,
    )
    state = yaml.safe_load(
        (result.session_directory / "state.yml").read_text(encoding="utf-8")
    )
    assert state["nodes"]["N2"]["status"] == "done"
    assert state["nodes"]["N3"]["status"] == "done"
    for node_id, fixed_rows in (("N2", 11), ("N3", 13)):
        schedule = state["adaptive_sweeps"][node_id]
        assert len(schedule["rows"]) <= 7
        assert len(schedule["rows"]) <= 0.60 * fixed_rows
        assert schedule["aborted"] is False
    replay = run_autocal(tmp_path, replay_session=result.session_directory)
    assert all(item.matches for item in replay.verified_fits)


def test_stop_resume_and_read_only_replay(tmp_path):
    _project(tmp_path)
    stop = tmp_path / "autocal_runs" / "STOP"
    stop.parent.mkdir()
    stop.touch()
    stopped = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="stop-resume",
    )
    assert stopped.status == "stopped"
    assert stopped.total_runs == 0

    stop.unlink()
    completed = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="stop-resume",
    )
    assert completed.status == "completed"
    state_before = (completed.session_directory / "state.yml").read_bytes()
    calibration_before = (tmp_path / "calibration.yml").read_bytes()
    replay = run_autocal(
        tmp_path,
        replay_session=completed.session_directory,
    )
    assert replay.status == "replayed"
    assert len(replay.verified_fits) == 3
    assert all(verification.matches for verification in replay.verified_fits)
    assert tuple(replay.events) == tuple(
        json.loads(line)
        for line in (completed.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert (completed.session_directory / "state.yml").read_bytes() == state_before
    assert (tmp_path / "calibration.yml").read_bytes() == calibration_before

    events = list(replay.events)
    fit_event = next(
        event for event in events if event.get("event") == "fit_evaluated"
    )
    fit_event["gates"].pop(next(iter(fit_event["gates"])))
    (completed.session_directory / "decisions.jsonl").write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    with pytest.raises(AnalysisError, match="missing recorded gates"):
        run_autocal(
            tmp_path,
            replay_session=completed.session_directory,
        )


def test_stop_created_during_acquisition_is_honored_before_next_run(
    tmp_path,
):
    _project(tmp_path)
    stop = tmp_path / "autocal_runs" / "STOP"

    def create_stop(_kind, check):
        if check == 1:
            stop.touch()
        return False

    device = DeviceModel(failure_hook=create_stop)
    backend = SyntheticBackend(seed=12, device=device)
    stopped = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="mid-run-stop",
        backend=backend,
    )
    assert stopped.status == "stopped"
    assert stopped.total_runs == 1
    assert stopped.node_status["N11"] == "done"
    assert stopped.node_status["N12"] == "pending"

    stop.unlink()
    completed = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="mid-run-stop",
        backend=backend,
    )
    assert completed.status == "completed"


def test_budget_exhaustion_stops_before_first_acquisition(tmp_path):
    _project(tmp_path)
    result = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        max_wall_clock_hours=1.0e-9,
    )
    assert result.status == "budget_exceeded"
    assert result.total_runs == 0
    assert "budget_exceeded" in {
        json.loads(line)["event"]
        for line in (result.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }


def test_flux_target_fails_closed_outside_lookup_domain(tmp_path):
    _project(tmp_path)
    result = run_autocal(
        tmp_path,
        target="flux_point",
        autonomy_level=0,
        z_gain=0.6,
    )
    assert result.status == "completed_with_escalations"
    assert result.total_runs == 0
    assert result.node_status["N4"] == "blocked"
    state = yaml.safe_load(
        (result.session_directory / "state.yml").read_text(encoding="utf-8")
    )
    assert "in-domain resonator lookup" in state["nodes"]["N4"]["reason"]


def test_run_cap_exhaustion_stops_mid_graph_and_preserves_resume_state(
    tmp_path,
):
    _project(tmp_path)
    hardware_path = tmp_path / "hardware.example.yml"
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware["autocal"]["budgets"]["max_total_runs"] = 1
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False),
        encoding="utf-8",
    )
    result = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
    )
    assert result.status == "budget_exceeded"
    assert result.total_runs == 1
    state = yaml.safe_load(
        (result.session_directory / "state.yml").read_text(encoding="utf-8")
    )
    assert state["nodes"]["N11"]["status"] == "done"
    assert state["nodes"]["N12"]["status"] == "pending"


def test_persistent_device_failure_escalates_only_its_dependents(tmp_path):
    _project(tmp_path)
    device = DeviceModel()
    device.fail_next("t1", count=20)
    result = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        backend=SyntheticBackend(seed=9, device=device),
    )
    assert result.status == "completed_with_escalations"
    assert result.node_status["N11"] == "blocked"
    assert result.node_status["N12"] == "done"
    assert result.node_status["N13"] == "done"


def test_configuration_errors_are_not_retried(tmp_path, monkeypatch):
    _project(tmp_path)
    calls = {}

    def invalid_configuration(_context, spec, *, attempt):
        calls[spec.node_id] = calls.get(spec.node_id, 0) + 1
        raise ConfigError(f"{spec.node_id} invalid")

    monkeypatch.setattr(
        autocal_orchestrator,
        "run_node",
        invalid_configuration,
    )
    result = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
    )
    assert result.status == "completed_with_escalations"
    assert calls == {"N11": 1, "N12": 1}
    events = [
        json.loads(line)
        for line in (result.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(event["event"] == "retake" for event in events)


def test_shifted_qubit_outside_expected_window_uses_widening_ladder(
    tmp_path,
):
    _project(tmp_path)
    hardware_path = tmp_path / "hardware.example.yml"
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware["expected"]["q_freq_mhz"] = [5400.0, 5800.0]
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False),
        encoding="utf-8",
    )
    device = DeviceModel(
        resonator_base_mhz=6884.0,
        resonator_flux_amplitude_mhz=0.5,
        resonator_flux_period_z=0.2,
        resonator_flux_peak_z=0.0,
        resonator_linewidth_mhz=0.45,
        punchout_transition_power_db=-30.0,
        punchout_width_db=3.0,
        qubit_max_frequency_mhz=6200.0,
        qubit_power_broadening_mhz_per_gain=4.0,
    )
    result = run_autocal(
        tmp_path,
        target="flux_point",
        autonomy_level=0,
        backend=SyntheticBackend(seed=17, device=device),
    )
    state = yaml.safe_load(
        (result.session_directory / "state.yml").read_text(encoding="utf-8")
    )
    assert state["nodes"]["N5"]["status"] == "done"
    assert state["nodes"]["N5"]["attempts"] == 2
    events = [
        json.loads(line)
        for line in (result.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    n5_retakes = [
        event
        for event in events
        if event["event"] == "retake" and event.get("node") == "N5"
    ]
    assert n5_retakes
    assert all("classification" in event for event in n5_retakes)


def test_low_snr_retake_doubles_node_averaging(
    tmp_path,
    monkeypatch,
):
    _project(tmp_path)
    original_fit_t1 = autocal_nodes.fit_t1
    calls = {"count": 0}

    def noisy_first_fit(csv_path, *args, **kwargs):
        calls["count"] += 1
        source = Path(csv_path)
        if calls["count"] == 1:
            matrix = np.loadtxt(source, delimiter=",")
            rng = np.random.default_rng(91)
            matrix[:, -2:] += rng.normal(
                0.0,
                0.45,
                size=matrix[:, -2:].shape,
            )
            iq = matrix[:, -2] + 1j * matrix[:, -1]
            matrix[:, 1] = np.abs(iq)
            matrix[:, 2] = np.angle(iq)
            np.savetxt(source, matrix, delimiter=",")
        return original_fit_t1(source, *args, **kwargs)

    monkeypatch.setattr(autocal_nodes, "fit_t1", noisy_first_fit)
    result = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
    )
    state = yaml.safe_load(
        (result.session_directory / "state.yml").read_text(encoding="utf-8")
    )
    assert state["nodes"]["N11"]["status"] == "done"
    assert state["nodes"]["N11"]["attempts"] == 2
    events = [
        json.loads(line)
        for line in (result.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    t1_sources = [
        Path(event["csv"])
        for event in events
        if event.get("event") == "acquisition_completed"
        and event.get("node") == "N11"
    ]
    hard_averages = [
        yaml.safe_load(source.with_suffix(".yml").read_text(encoding="utf-8"))[
            "parameters"
        ]["var"]["hard_avg"]
        for source in t1_sources
    ]
    assert hard_averages == [10000, 20000]


def test_exhausted_live_acquisition_errors_are_critical_aborts(
    tmp_path,
    monkeypatch,
):
    _project(tmp_path)

    def disconnected(*_args, **_kwargs):
        raise AcquisitionError("Pyro link unavailable")

    monkeypatch.setattr(autocal_orchestrator, "run_node", disconnected)
    result = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        live_hardware=True,
    )
    assert result.status == "critical_abort"
    assert result.total_runs == 0
    events = (
        result.session_directory / "decisions.jsonl"
    ).read_text(encoding="utf-8")
    assert '"event":"critical_abort"' in events
    assert "cannot prove that an RF-held Z line was parked" in events


def test_l1_promotes_bounded_coherence_records_but_keeps_audit_events(tmp_path):
    _project(tmp_path)
    result = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=1,
    )
    assert result.status == "completed"
    assert list_open_proposals(tmp_path) == []
    calibration = yaml.safe_load(
        (tmp_path / "calibration.yml").read_text(encoding="utf-8")
    )
    assert calibration["revision"] == 6
    assert (
        calibration["records"]["derived"]["t2_echo"]["cycle_0"]["accepted_by"]
        == "autocal-L1"
    )
    events = (result.session_directory / "decisions.jsonl").read_text(
        encoding="utf-8"
    )
    assert events.count('"event":"auto_promoted"') == 4


def test_resume_requeues_only_downstream_nodes_after_external_promotion(
    tmp_path,
):
    _project(tmp_path)
    first = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="invalidate-on-resume",
    )
    q_frequency_id = next(
        proposal_id
        for proposal_id, proposal in list_open_proposals(tmp_path)
        if proposal["record"] == "defaults.q_freq"
    )
    promote_proposal(tmp_path, q_frequency_id, accepted_by="test")

    resumed = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="invalidate-on-resume",
    )
    assert resumed.status == "completed"
    assert resumed.total_runs == first.total_runs + 1
    events = [
        json.loads(line)
        for line in (resumed.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    revision_event = next(
        event
        for event in reversed(events)
        if event["event"] == "calibration_revision_changed"
    )
    assert revision_event["invalidated_nodes"] == ["N13"]


def test_resume_uses_cumulative_change_since_session_revision(tmp_path):
    _project(tmp_path)
    first = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="cumulative-invalidation",
    )
    calibration = yaml.safe_load(
        (tmp_path / "calibration.example.yml").read_text(encoding="utf-8")
    )
    initial = float(calibration["records"]["defaults"]["q_freq"]["value"])

    for index, increment in enumerate((0.6, 1.2), start=1):
        proposal_id = f"external-qfreq-{index}"
        write_calibration_proposals(
            tmp_path,
            {
                proposal_id: {
                    "record": "defaults.q_freq",
                    "value": initial + increment,
                    "unit": "MHz",
                    "uncertainty": {"center_mhz": 0.05},
                    "provenance": {
                        "source": f"external-{index}.csv",
                        "analysis": "external review",
                        "autocal_session": f"external-{index}",
                        "autocal_node": "N5",
                        "working_z_gain": 0.0,
                    },
                    "quality": {"r_squared": 0.99},
                    "valid_domain": {"z_gain": [0.0, 0.0]},
                    "model": "reviewed_external_fit",
                    "status": "proposed",
                }
            },
        )
        promote_proposal(
            tmp_path,
            proposal_id,
            accepted_by="test",
        )

    resumed = run_autocal(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        session_name="cumulative-invalidation",
    )
    assert resumed.total_runs == first.total_runs + 4
    events = [
        json.loads(line)
        for line in (resumed.session_directory / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    revision_event = next(
        event
        for event in reversed(events)
        if event["event"] == "calibration_revision_changed"
    )
    assert revision_event["invalidated_nodes"] == ["N11", "N12", "N13"]
