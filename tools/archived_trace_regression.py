"""Reality-check candidate extraction and replay against operator labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import yaml

from quickexp_v3.autocal.hp.candidates import extract_candidates
from quickexp_v3.autocal.replay import verify_session_replay
from quickexp_v3.autocal.session import AutocalSession
from quickexp_v3.errors import AnalysisError
from quickexp_v3.notch_fit import fit_spectroscopy_features


@dataclass(frozen=True)
class ArchivedTraceResult:
    csv_path: Path
    correct_value_mhz: float
    matched_candidate_mhz: float
    correct_hypothesis: str
    identity_replayed: bool


def load_manifest(
    manifest_path: Path,
    *,
    require_labels: bool = False,
) -> Tuple[Mapping, ...]:
    source = Path(manifest_path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or int(raw.get("schema_version", 0)) != 1:
        raise AnalysisError("archived-trace manifest must have schema_version: 1")
    traces = raw.get("traces", ())
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
        raise AnalysisError("archived-trace manifest traces must be a list")
    if require_labels and not traces:
        raise AnalysisError(
            "archived-trace manifest has no operator-labeled traces"
        )
    required = {"csv", "correct_value", "correct_hypothesis", "notes"}
    normalized = []
    for index, entry in enumerate(traces):
        if not isinstance(entry, Mapping) or required.difference(entry):
            raise AnalysisError(
                "archived trace {0} must define {1}".format(
                    index,
                    ", ".join(sorted(required)),
                )
            )
        normalized.append(dict(entry))
    return tuple(normalized)


def _identity_matches_session(
    session_path: Path,
    csv_path: Path,
    expected_hypothesis: str,
) -> bool:
    session = AutocalSession.load(session_path)
    verify_session_replay(session)
    source = csv_path.expanduser().resolve()
    for event in session.events():
        if event.get("event") != "hypothesis_adjudicated":
            continue
        candidates = event.get("candidates", ())
        candidate_sources = {
            Path(str(candidate.get("source_csv", ""))).expanduser().resolve()
            for candidate in candidates
            if isinstance(candidate, Mapping)
        }
        if source in candidate_sources:
            adjudication = event.get("adjudication", {})
            return str(adjudication.get("hypothesis_id")) == str(
                expected_hypothesis
            )
    raise AnalysisError(
        "labeled CSV has no hypothesis event in session " + str(session_path)
    )


def verify_archived_traces(
    manifest_path: Path,
) -> Tuple[ArchivedTraceResult, ...]:
    source = Path(manifest_path).expanduser().resolve()
    results = []
    for entry in load_manifest(source, require_labels=True):
        csv_path = (source.parent / str(entry["csv"])).resolve()
        fit = fit_spectroscopy_features(
            csv_path,
            kind=str(entry.get("kind", "qubit")),
            signal=str(entry.get("signal", "amplitude")),
        )
        candidates = tuple(
            candidate
            for candidate in extract_candidates(fit)
            if not candidate.is_null
        )
        if not candidates:
            raise AnalysisError("no candidate was extracted from " + str(csv_path))
        expected = float(entry["correct_value"])
        tolerance = abs(float(entry.get("tolerance_mhz", 1.0)))
        matched = min(
            candidates,
            key=lambda candidate: abs(candidate.center_mhz - expected),
        )
        if abs(matched.center_mhz - expected) > tolerance:
            raise AnalysisError(
                "archived trace candidate changed for {0}: expected {1}, got {2}".format(
                    csv_path,
                    expected,
                    matched.center_mhz,
                )
            )
        identity_replayed = False
        if entry.get("session"):
            session_path = (source.parent / str(entry["session"])).resolve()
            identity_replayed = _identity_matches_session(
                session_path,
                csv_path,
                str(entry["correct_hypothesis"]),
            )
            if not identity_replayed:
                raise AnalysisError(
                    "archived trace hypothesis changed for " + str(csv_path)
                )
        results.append(
            ArchivedTraceResult(
                csv_path,
                expected,
                float(matched.center_mhz),
                str(entry["correct_hypothesis"]),
                identity_replayed,
            )
        )
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        default=Path("tests/fixtures/labeled/manifest.yml"),
        nargs="?",
    )
    arguments = parser.parse_args()
    results = verify_archived_traces(arguments.manifest)
    replayed = sum(result.identity_replayed for result in results)
    print(
        "verified={0} identity_replayed={1}".format(
            len(results),
            replayed,
        )
    )


if __name__ == "__main__":
    main()
