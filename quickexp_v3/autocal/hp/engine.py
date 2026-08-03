"""Deterministic hypothesis-and-probe node runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from .candidates import Candidate
from .coverage import CoverageAssessment, assess_coverage
from .probes import Probe, expand_probe_runs, get_probe
from .scorecard import Adjudication, Scorecard, adjudicate, build_scorecard
from .taxonomy import get_hypothesis


_PREDICTION_COMPONENTS = {
    "rabi_gain_linearity": ("rabi_ping:rabi_gain_exponent",),
    "flux_period_agreement": (
        "flux_nudge:flux_slope_mhz_per_z",
        "flux_nudge:flux_curvature_mhz_per_z2",
    ),
    "dispersive_shift": ("dispersive_response:dispersive_shift_mhz",),
    "chi": ("dispersive_response:dispersive_shift_mhz",),
}


@dataclass(frozen=True)
class HypothesisNodeSpec:
    node_id: str
    acquire: Callable[[Any, int], Path]
    extract: Callable[[Path], Sequence[Candidate]]
    hypotheses: Sequence[str]
    wanted: str
    probes: Sequence[str]
    predictions: Sequence[str]
    product_address: str
    family: str = "qubit"
    assess: Optional[Callable[[Sequence[Candidate], Any], CoverageAssessment]] = None
    probe_runner: Optional[
        Callable[[Any, Probe, Candidate, Sequence[Mapping[str, Any]]], Mapping[str, float]]
    ] = None


@dataclass(frozen=True)
class EngineResult:
    node_id: str
    source_csv: Path
    candidates: Tuple[Candidate, ...]
    coverage: CoverageAssessment
    responses: Mapping[str, Mapping[str, Mapping[str, float]]]
    scorecard: Scorecard
    adjudication: Adjudication
    probes_run: Tuple[str, ...]
    probe_seconds: float


def _get(context: Any, name: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _device_context(context: Any) -> Mapping[str, Any]:
    explicit = _get(context, "device_context")
    if callable(explicit):
        explicit = explicit()
    if isinstance(explicit, Mapping):
        result = dict(explicit)
    elif isinstance(context, Mapping):
        result = dict(context)
    else:
        result = {}
    # Conservative scale defaults keep the engine usable with a minimal
    # context.  Production integration supplies measured device scales.
    defaults = {
        "power_exponent_tolerance": 0.25,
        "flux_slope_tolerance_mhz_per_z": 100.0,
        "flux_curvature_tolerance_mhz_per_z2": 1000.0,
        "rabi_exponent_tolerance": 0.25,
        "rabi_contrast_tolerance": 0.15,
        "dispersive_shift_tolerance_mhz": 0.25,
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    return result


def _default_coverage(
    candidates: Sequence[Candidate],
    context: Any,
) -> CoverageAssessment:
    explicit = _get(context, "coverage")
    if isinstance(explicit, CoverageAssessment):
        return explicit
    real = [candidate for candidate in candidates if not candidate.is_null]
    if not real:
        window = (0.0, 1.0)
        width = 1.0
        contrast = 0.0
        points = 2
    else:
        window = real[0].window_mhz
        width = real[0].fwhm_mhz
        contrast = abs(real[0].contrast)
        points = int(real[0].statistics.get("points", 101))
    prior = _get(context, "prior_window", window)
    return assess_coverage(
        candidates,
        prior_window=tuple(prior),
        scan_window=tuple(window),
        points=points,
        expected_fwhm_mhz=width,
        expected_contrast=contrast,
    )


def _selected_candidates(
    candidates: Sequence[Candidate],
    context: Any,
) -> Tuple[Candidate, ...]:
    real = sorted(
        (candidate for candidate in candidates if not candidate.is_null),
        key=lambda candidate: int(candidate.rank),
    )
    if not real:
        return tuple(candidates[:1])
    top_k = max(int(_get(context, "top_k_candidates", 3)), 1)
    ratio = max(float(_get(context, "candidate_prominence_ratio", 0.5)), 0.0)
    leader = max(abs(float(real[0].contrast)), 1.0e-15)
    selected = [
        candidate
        for candidate in real
        if int(candidate.rank) < top_k
        or abs(float(candidate.contrast)) >= ratio * leader
    ]
    return tuple(selected)


def _escalation(
    scorecard: Scorecard,
    reason: str,
) -> Adjudication:
    leader = scorecard.leader
    return Adjudication(
        "escalate",
        reason,
        "B",
        leader.candidate_id,
        leader.hypothesis_id,
        float(scorecard.margin),
    )


def consistency_passes(
    predictions: Sequence[str],
    wanted: str,
    family: str,
    scorecard: Scorecard,
) -> bool:
    """Check completed physical predictions at their declared tolerance."""
    leader = scorecard.leader
    if leader.hypothesis_id != str(wanted):
        return True
    hypothesis = get_hypothesis(str(wanted), family=str(family))
    weights = {
        signature.probe_id + ":" + signature.observable: float(signature.weight)
        for signature in hypothesis.signatures
    }
    for prediction_id in predictions:
        for component_id in _PREDICTION_COMPONENTS.get(
            str(prediction_id),
            (),
        ):
            if component_id not in leader.components:
                continue
            # score = -0.5 * normalized_residual**2 * weight. A completed
            # prediction is consistent exactly when it is within its declared
            # (scale-aware) tolerance; this is a physics check, not a GOF gate.
            if float(leader.components[component_id]) < -0.5 * weights[
                component_id
            ]:
                return False
    return True


def _run_probe(
    spec: HypothesisNodeSpec,
    context: Any,
    probe: Probe,
    candidate: Candidate,
    runs: Sequence[Mapping[str, Any]],
) -> Mapping[str, float]:
    runner = spec.probe_runner or _get(context, "probe_runner")
    if callable(runner):
        response = runner(context, probe, candidate, runs)
        if not isinstance(response, Mapping):
            raise ValueError("probe runner must return a response mapping")
        return {str(key): float(value) for key, value in response.items()}
    acquire_probe = _get(context, "acquire_probe")
    if not callable(acquire_probe):
        raise ValueError("hypothesis engine context has no probe runner")
    paths = acquire_probe(context, probe, candidate, runs)
    return {
        str(key): float(value)
        for key, value in probe.extract_response(tuple(Path(path) for path in paths)).items()
    }


def run(
    spec: HypothesisNodeSpec,
    context: Any,
    *,
    attempt: int = 1,
) -> EngineResult:
    """Acquire, probe in declared order, and return a margin-only verdict."""
    source = Path(spec.acquire(context, int(attempt)))
    candidates = tuple(spec.extract(source))
    if not candidates:
        raise ValueError("hypothesis extraction returned no candidates")
    coverage = (
        spec.assess(candidates, context)
        if spec.assess is not None
        else _default_coverage(candidates, context)
    )
    device_context = _device_context(context)
    responses = {}
    probes_run = []
    probe_seconds = 0.0
    scorecard = build_scorecard(
        candidates,
        spec.hypotheses,
        responses,
        device_context,
        family=spec.family,
    )
    verdict = adjudicate(
        scorecard,
        coverage,
        wanted=spec.wanted,
        margin_threshold=float(_get(context, "margin_threshold", 2.0)),
        probes_remaining=bool(spec.probes),
        consistency_passes=consistency_passes(
            spec.predictions,
            spec.wanted,
            spec.family,
            scorecard,
        ),
    )
    if verdict.action == "remediate":
        return EngineResult(
            spec.node_id,
            source,
            candidates,
            coverage,
            responses,
            scorecard,
            verdict,
            (),
            0.0,
        )

    selected = _selected_candidates(candidates, context)
    budget = max(float(_get(context, "probe_budget_seconds", 600.0)), 0.0)
    for probe_index, probe_id in enumerate(spec.probes):
        if verdict.action != "probe":
            break
        probe = get_probe(probe_id)
        predicted = probe.estimated_seconds(device_context) * len(selected)
        if probe_seconds + predicted > budget:
            verdict = _escalation(
                scorecard,
                "probe budget would be exceeded before " + probe.probe_id,
            )
            break
        for candidate in selected:
            runs = expand_probe_runs(probe, candidate, device_context)
            response = _run_probe(spec, context, probe, candidate, runs)
            responses.setdefault(candidate.candidate_id, {})[probe.probe_id] = response
        probe_seconds += predicted
        probes_run.append(probe.probe_id)
        scorecard = build_scorecard(
            candidates,
            spec.hypotheses,
            responses,
            device_context,
            family=spec.family,
        )
        verdict = adjudicate(
            scorecard,
            coverage,
            wanted=spec.wanted,
            margin_threshold=float(_get(context, "margin_threshold", 2.0)),
            probes_remaining=probe_index + 1 < len(spec.probes),
            consistency_passes=consistency_passes(
                spec.predictions,
                spec.wanted,
                spec.family,
                scorecard,
            ),
        )
    return EngineResult(
        spec.node_id,
        source,
        candidates,
        coverage,
        responses,
        scorecard,
        verdict,
        tuple(probes_run),
        float(probe_seconds),
    )
