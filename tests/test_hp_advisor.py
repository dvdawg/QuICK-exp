import json
from pathlib import Path

import pytest

from quickexp_v3.autocal.hp.advisor import (
    AdvisoryRequest,
    AdvisoryResponse,
    AdvisorValidationError,
    ClaudeAdvisor,
    NullAdvisor,
    ReplayAdvisor,
    request_hash,
    validate_response,
)


def _request(images=()):
    return AdvisoryRequest(
        trigger="unresolved_scorecard",
        node_id="N5",
        candidates=({"candidate_id": "c1", "center_mhz": 5600.0},),
        probe_responses=({"probe_id": "P1", "exponent": 1.4},),
        scorecard={"margin": 0.2},
        discrepancies=(),
        device_context={"q_delta_mhz": -180.0},
        images=tuple(images),
    )


def _response():
    return AdvisoryResponse(
        hypothesis_label="qubit_01",
        proposed_action={"experiment": "rabi", "preset": "rabi_length", "overrides": {}},
        confidence=0.7,
        rationale="A two-gain coherence check separates the tied signatures.",
        discrepancy_notes=(),
        novel_program_sketch=None,
    )


def test_null_advisor_always_escalates_without_accepting():
    response = NullAdvisor().advise(_request())
    assert response.proposed_action == {"action": "escalate"}
    assert response.confidence == 0.0


def test_request_hash_is_stable_and_path_sensitive(tmp_path):
    first = _request((tmp_path / "a.png",))
    second = _request((tmp_path / "a.png",))
    third = _request((tmp_path / "b.png",))
    assert request_hash(first) == request_hash(second)
    assert request_hash(first) != request_hash(third)


def test_replay_matches_logged_response_and_never_calls_out(tmp_path):
    audit = tmp_path / "advisory.jsonl"
    record = {
        "request_hash": request_hash(_request()),
        "request": _request().as_dict(),
        "response": _response().as_dict(),
        "model": "recorded",
        "latency_seconds": 0.1,
        "policy_accepted": False,
    }
    audit.write_text(json.dumps(record) + "\n", encoding="utf-8")
    advisor = ReplayAdvisor(audit)
    assert advisor.advise(_request()) == _response()
    missing = advisor.advise(
        AdvisoryRequest(**{**_request().as_dict(), "node_id": "N4"})
    )
    assert missing.proposed_action == {"action": "escalate"}
    assert advisor.divergences


def test_response_validation_rejects_unknown_program_and_limit_violation():
    with pytest.raises(AdvisorValidationError, match="registered"):
        validate_response(
            AdvisoryResponse(
                "qubit_01",
                {"experiment": "invented", "preset": "new", "overrides": {}},
                0.5,
                "test",
                (),
            ),
            hypothesis_ids=("qubit_01",),
            experiment_presets={"rabi": ("rabi_length",)},
            limits={"q_gain": (-1.0, 1.0)},
            remaining_budget_seconds=100.0,
            estimated_action_seconds=1.0,
        )
    with pytest.raises(AdvisorValidationError, match="limits"):
        validate_response(
            AdvisoryResponse(
                "qubit_01",
                {"experiment": "rabi", "preset": "rabi_length", "overrides": {"q_length": 5.0}},
                0.5,
                "test",
                (),
            ),
            hypothesis_ids=("qubit_01",),
            experiment_presets={"rabi": ("rabi_length",)},
            limits={"q_length": (0.01, 2.0)},
            remaining_budget_seconds=100.0,
            estimated_action_seconds=1.0,
        )


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_claude_advisor_uses_messages_shape_images_and_audit(tmp_path):
    image = tmp_path / "overlay.png"
    image.write_bytes(b"png-bytes")
    captured = {}

    def opener(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            {
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": json.dumps(_response().as_dict())}],
                "stop_reason": "end_turn",
            }
        )

    audit = tmp_path / "advisory.jsonl"
    advisor = ClaudeAdvisor(
        model="claude-sonnet-5",
        audit_path=audit,
        environment={"ANTHROPIC_API_KEY": "test-key"},
        opener=opener,
    )
    response = advisor.advise(_request((image,)))
    assert response == _response()
    payload = json.loads(captured["request"].data)
    assert payload["model"] == "claude-sonnet-5"
    assert payload["messages"][0]["content"][0]["type"] == "image"
    assert captured["request"].headers["X-api-key"] == "test-key"
    logged = json.loads(audit.read_text(encoding="utf-8"))
    assert logged["request_hash"] == request_hash(_request((image,)))
    assert "png-bytes" not in audit.read_text(encoding="utf-8")

    advisor.audit_policy_decision(
        _request((image,)),
        policy_accepted=True,
    )
    records = [
        json.loads(line)
        for line in audit.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["record_type"] == "policy_validation"
    assert records[-1]["policy_accepted"] is True
