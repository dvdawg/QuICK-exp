"""Hypothesis-and-probe calibration decision primitives."""

from .candidates import Candidate, extract_candidates
from .adaptive import (
    AdaptiveRow,
    AdaptiveRowScheduler,
    spanning_rows,
    tracked_frequency_axis,
)
from .advisor import (
    AdvisoryRequest,
    AdvisoryResponse,
    AdvisorError,
    AdvisorValidationError,
    ClaudeAdvisor,
    NullAdvisor,
    ReplayAdvisor,
    request_hash,
    validate_response,
)
from .coverage import CoverageAssessment, assess_coverage
from .engine import EngineResult, HypothesisNodeSpec, run
from .ledger import (
    BacktrackDecision,
    BacktrackLimitExceeded,
    DiscrepancyEntry,
    DiscrepancyLedger,
    HypothesisLedger,
    render_discrepancy_report,
)
from .probes import Probe, expand_probe_runs, get_probe, probe_ids
from .remediation import RemediationStep, next_remediation
from .scorecard import Adjudication, Scorecard, adjudicate, build_scorecard
from .taxonomy import Hypothesis, Signature, hypotheses_for, hypothesis_ids

__all__ = [
    "Candidate",
    "AdaptiveRow",
    "AdaptiveRowScheduler",
    "spanning_rows",
    "tracked_frequency_axis",
    "AdvisoryRequest",
    "AdvisoryResponse",
    "AdvisorError",
    "AdvisorValidationError",
    "NullAdvisor",
    "ReplayAdvisor",
    "ClaudeAdvisor",
    "request_hash",
    "validate_response",
    "extract_candidates",
    "CoverageAssessment",
    "assess_coverage",
    "HypothesisNodeSpec",
    "EngineResult",
    "run",
    "BacktrackDecision",
    "BacktrackLimitExceeded",
    "HypothesisLedger",
    "DiscrepancyEntry",
    "DiscrepancyLedger",
    "render_discrepancy_report",
    "Probe",
    "probe_ids",
    "get_probe",
    "expand_probe_runs",
    "RemediationStep",
    "next_remediation",
    "Signature",
    "Hypothesis",
    "hypotheses_for",
    "hypothesis_ids",
    "Scorecard",
    "Adjudication",
    "build_scorecard",
    "adjudicate",
]
