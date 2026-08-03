"""Typed, auditable, out-of-band calibration advisory interface."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

from ...util import to_builtin, utc_now


ADVISORY_TRIGGERS = frozenset(
    {
        "session_start",
        "unresolved_scorecard",
        "signature_mismatch",
        "session_end",
    }
)
AUTHORIZED_OVERRIDE_KNOBS = frozenset(
    {
        "hard_avg",
        "soft_avg",
        "r_relax",
        "q_length",
        "r_length",
        "r_power",
        "q_freq",
        "r_freq",
        "z_gain",
    }
)


class AdvisorError(RuntimeError):
    """An advisor could not produce a well-formed proposal."""


class AdvisorValidationError(AdvisorError):
    """A typed advisory proposal failed deterministic policy validation."""


@dataclass(frozen=True)
class AdvisoryRequest:
    trigger: str
    node_id: str
    candidates: Sequence[Mapping[str, Any]]
    probe_responses: Sequence[Mapping[str, Any]]
    scorecard: Mapping[str, Any]
    discrepancies: Sequence[Mapping[str, Any]]
    device_context: Mapping[str, Any]
    images: Sequence[Path]

    def __post_init__(self) -> None:
        if str(self.trigger) not in ADVISORY_TRIGGERS:
            raise ValueError("unknown advisory trigger: " + str(self.trigger))
        if not str(self.node_id):
            raise ValueError("advisory node_id cannot be empty")

    def as_dict(self) -> dict:
        return {
            "trigger": str(self.trigger),
            "node_id": str(self.node_id),
            "candidates": to_builtin(list(self.candidates)),
            "probe_responses": to_builtin(list(self.probe_responses)),
            "scorecard": to_builtin(dict(self.scorecard)),
            "discrepancies": to_builtin(list(self.discrepancies)),
            "device_context": to_builtin(dict(self.device_context)),
            "images": [
                str(Path(image).expanduser().resolve(strict=False))
                for image in self.images
            ],
        }


@dataclass(frozen=True)
class AdvisoryResponse:
    hypothesis_label: str
    proposed_action: Optional[Mapping[str, Any]]
    confidence: float
    rationale: str
    discrepancy_notes: Sequence[Mapping[str, Any]]
    novel_program_sketch: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "hypothesis_label": str(self.hypothesis_label),
            "proposed_action": (
                None
                if self.proposed_action is None
                else to_builtin(dict(self.proposed_action))
            ),
            "confidence": float(self.confidence),
            "rationale": str(self.rationale),
            "discrepancy_notes": to_builtin(list(self.discrepancy_notes)),
            "novel_program_sketch": self.novel_program_sketch,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdvisoryResponse":
        required = {
            "hypothesis_label",
            "proposed_action",
            "confidence",
            "rationale",
            "discrepancy_notes",
        }
        missing = required.difference(raw)
        if missing:
            raise AdvisorValidationError(
                "advisory response is missing: " + ", ".join(sorted(missing))
            )
        action = raw.get("proposed_action")
        if action is not None and not isinstance(action, Mapping):
            raise AdvisorValidationError("proposed_action must be a mapping or null")
        notes = raw.get("discrepancy_notes")
        if (
            not isinstance(notes, Sequence)
            or isinstance(notes, (str, bytes))
            or any(not isinstance(note, Mapping) for note in notes)
        ):
            raise AdvisorValidationError("discrepancy_notes must be mappings")
        sketch = raw.get("novel_program_sketch")
        if sketch is not None and not isinstance(sketch, str):
            raise AdvisorValidationError("novel_program_sketch must be text or null")
        return cls(
            str(raw["hypothesis_label"]),
            None if action is None else dict(action),
            float(raw["confidence"]),
            str(raw["rationale"]),
            tuple(dict(note) for note in notes),
            sketch,
        )


def request_hash(request: AdvisoryRequest) -> str:
    encoded = json.dumps(
        request.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _null_response(reason: str) -> AdvisoryResponse:
    return AdvisoryResponse(
        "novel",
        {"action": "escalate"},
        0.0,
        str(reason),
        (),
        None,
    )


class NullAdvisor:
    model = "null"

    def advise(self, request: AdvisoryRequest) -> AdvisoryResponse:
        return _null_response(
            "No external advisor is configured; deterministic escalation required."
        )


class ReplayAdvisor:
    model = "replay"

    def __init__(self, audit_path: Path):
        self.audit_path = Path(audit_path).expanduser().resolve()
        self._responses = {}
        self.divergences = []
        if self.audit_path.exists():
            for line in self.audit_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = str(record.get("request_hash", ""))
                response = record.get("response")
                if key and isinstance(response, Mapping):
                    self._responses.setdefault(key, AdvisoryResponse.from_mapping(response))

    def advise(self, request: AdvisoryRequest) -> AdvisoryResponse:
        key = request_hash(request)
        if key in self._responses:
            return self._responses[key]
        self.divergences.append(
            {
                "event": "advisory_divergence",
                "request_hash": key,
                "node_id": request.node_id,
                "trigger": request.trigger,
            }
        )
        return _null_response(
            "Replay has no advisory response for request hash " + key
        )


def _finite_override_values(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AdvisorValidationError("advisor override must be numeric") from exc
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise AdvisorValidationError("advisor override must contain finite values")
    return result


def validate_response(
    response: AdvisoryResponse,
    *,
    hypothesis_ids: Sequence[str],
    experiment_presets: Mapping[str, Sequence[str]],
    limits: Mapping[str, Sequence[float]],
    remaining_budget_seconds: float,
    estimated_action_seconds: float,
    authorized_override_knobs: Sequence[str] = tuple(AUTHORIZED_OVERRIDE_KNOBS),
) -> AdvisoryResponse:
    """Reject, rather than repair, any advisory proposal outside policy."""
    if response.hypothesis_label not in set(hypothesis_ids).union({"novel"}):
        raise AdvisorValidationError("advisor hypothesis is not in the taxonomy")
    if not np.isfinite(float(response.confidence)) or not 0.0 <= float(
        response.confidence
    ) <= 1.0:
        raise AdvisorValidationError("advisor confidence must be between 0 and 1")
    if not response.rationale.strip():
        raise AdvisorValidationError("advisor rationale cannot be empty")
    action = response.proposed_action
    if action is None:
        return response
    if action.get("action") == "escalate":
        if set(action).difference({"action"}):
            raise AdvisorValidationError("escalation action has unknown fields")
        return response
    if "action" in action:
        raise AdvisorValidationError("advisor cannot accept, reject, or promote")
    experiment = action.get("experiment")
    preset = action.get("preset")
    if experiment not in experiment_presets:
        raise AdvisorValidationError("advisor experiment is not registered")
    if preset not in set(experiment_presets[experiment]):
        raise AdvisorValidationError("advisor preset is not registered for experiment")
    overrides = action.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise AdvisorValidationError("advisor overrides must be a mapping")
    authorized = set(authorized_override_knobs)
    for name, value in overrides.items():
        if str(name) not in authorized:
            raise AdvisorValidationError(
                "advisor override is not an authorized knob: " + str(name)
            )
        bounds = limits.get(name)
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise AdvisorValidationError("advisor override has no hardware limits: " + str(name))
        lower, upper = sorted(float(item) for item in bounds)
        values = _finite_override_values(value)
        if np.any(values < lower) or np.any(values > upper):
            raise AdvisorValidationError("advisor override violates hardware limits: " + str(name))
    estimate = max(float(estimated_action_seconds), 0.0)
    remaining = max(float(remaining_budget_seconds), 0.0)
    if estimate > remaining:
        raise AdvisorValidationError("advisor action exceeds the remaining budget")
    return response


def _media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "")


def _response_prompt(request: AdvisoryRequest) -> str:
    schema = {
        "hypothesis_label": "taxonomy id or novel",
        "proposed_action": {
            "experiment": "registered id",
            "preset": "registered preset",
            "overrides": {},
        },
        "confidence": 0.0,
        "rationale": "text",
        "discrepancy_notes": [],
        "novel_program_sketch": None,
    }
    return (
        "Review this calibration evidence. Propose a diagnostic only; never accept, "
        "reject, promote, or claim hardware was changed. Return exactly one JSON object "
        "with this shape and no markdown: "
        + json.dumps(schema, sort_keys=True)
        + "\nEvidence:\n"
        + json.dumps(request.as_dict(), sort_keys=True, separators=(",", ":"))
    )


class ClaudeAdvisor:
    """Small stdlib client for Anthropic's synchronous Messages API."""

    endpoint = "https://api.anthropic.com/v1/messages"
    api_version = "2023-06-01"

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-5",
        audit_path: Optional[Path] = None,
        timeout_seconds: float = 60.0,
        max_tokens: int = 2048,
        environment: Optional[Mapping[str, str]] = None,
        opener: Any = None,
    ):
        self.model = str(model)
        self.audit_path = (
            None if audit_path is None else Path(audit_path).expanduser().resolve()
        )
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        self.environment = os.environ if environment is None else environment
        self.opener = urllib_request.urlopen if opener is None else opener

    def _content(self, advisory_request: AdvisoryRequest) -> list:
        content = []
        for raw in advisory_request.images:
            path = Path(raw).expanduser().resolve()
            media_type = _media_type(path)
            if not media_type:
                raise AdvisorValidationError(
                    "advisor image must be PNG, JPEG, GIF, or WebP: " + str(path)
                )
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )
        content.append({"type": "text", "text": _response_prompt(advisory_request)})
        return content

    def _audit(
        self,
        advisory_request: AdvisoryRequest,
        response: AdvisoryResponse,
        *,
        model: str,
        latency_seconds: float,
    ) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": utc_now(),
            "request_hash": request_hash(advisory_request),
            "request": advisory_request.as_dict(),
            "response": response.as_dict(),
            "model": str(model),
            "latency_seconds": float(latency_seconds),
            "policy_accepted": False,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def audit_policy_decision(
        self,
        advisory_request: AdvisoryRequest,
        *,
        policy_accepted: bool,
    ) -> None:
        """Append the deterministic policy result after response validation."""
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": utc_now(),
            "record_type": "policy_validation",
            "request_hash": request_hash(advisory_request),
            "policy_accepted": bool(policy_accepted),
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def advise(self, advisory_request: AdvisoryRequest) -> AdvisoryResponse:
        api_key = self.environment.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AdvisorError("ANTHROPIC_API_KEY is not set")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": (
                "You are an out-of-band circuit-QED calibration advisor. "
                "You may propose only diagnostics from the supplied registry."
            ),
            "messages": [
                {"role": "user", "content": self._content(advisory_request)}
            ],
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        http_request = urllib_request.Request(
            self.endpoint,
            data=encoded,
            headers={
                "content-type": "application/json",
                "x-api-key": str(api_key),
                "anthropic-version": self.api_version,
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with self.opener(http_request, timeout=self.timeout_seconds) as opened:
                raw = json.loads(opened.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise AdvisorError(
                "Anthropic Messages API returned HTTP "
                + str(exc.code)
                + ": "
                + details[:500]
            ) from exc
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            raise AdvisorError("Anthropic Messages API request failed: " + str(exc)) from exc
        latency = time.monotonic() - started
        if raw.get("stop_reason") in {"max_tokens", "refusal"}:
            raise AdvisorError(
                "Anthropic response stopped with " + str(raw.get("stop_reason"))
            )
        texts = [
            str(block.get("text", ""))
            for block in raw.get("content", ())
            if isinstance(block, Mapping) and block.get("type") == "text"
        ]
        text = "\n".join(texts).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdvisorValidationError("advisor did not return one JSON object") from exc
        if not isinstance(parsed, Mapping):
            raise AdvisorValidationError("advisor response JSON must be an object")
        response = AdvisoryResponse.from_mapping(parsed)
        self._audit(
            advisory_request,
            response,
            model=str(raw.get("model", self.model)),
            latency_seconds=latency,
        )
        return response
