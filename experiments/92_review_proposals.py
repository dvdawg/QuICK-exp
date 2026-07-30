"""Review, promote, or reject inert calibration proposals."""

from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.config import accepted_calibration_values
from quickexp_v3.fit_calibration import (
    list_open_proposals,
    promote_proposal,
    reject_proposal,
)
from quickexp_v3.ide import load_repository
from quickexp_v3.util import dotted_get


# ============================ EDIT THESE ====================================
LIVE_HARDWARE = False
PROMOTE = []
REJECT = {}
ACCEPTED_BY = ""
# ============================================================================


def _display_value(value):
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _age(created_at):
    try:
        created = datetime.fromisoformat(
            str(created_at).replace("Z", "+00:00")
        )
        created = created.replace(
            tzinfo=created.tzinfo or timezone.utc
        ).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return "unknown"
    hours = max(
        0.0,
        (datetime.now(timezone.utc) - created).total_seconds() / 3600.0,
    )
    return f"{hours:.1f}h"


def _gate_summary(proposal):
    quality = proposal.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    gates = quality.get("autocal_gates")
    gates = gates if isinstance(gates, dict) else {}
    if not gates:
        return "not recorded"
    passed = sum(
        bool(value.get("passed"))
        for value in gates.values()
        if isinstance(value, dict)
    )
    return f"{passed}/{len(gates)} pass"


def main():
    for proposal_id, reason in dict(REJECT).items():
        reject_proposal(
            PROJECT_ROOT,
            str(proposal_id),
            reason=str(reason),
        )
        print(f"Rejected {proposal_id}: {reason}")
    if PROMOTE and not str(ACCEPTED_BY).strip():
        raise ValueError("ACCEPTED_BY initials are required when PROMOTE is non-empty")
    for proposal_id in list(PROMOTE):
        promote_proposal(
            PROJECT_ROOT,
            str(proposal_id),
            accepted_by=str(ACCEPTED_BY),
        )
        print(f"Promoted {proposal_id} by {ACCEPTED_BY}")

    repository = load_repository(PROJECT_ROOT)
    accepted = accepted_calibration_values(repository.calibration)
    proposals = list_open_proposals(PROJECT_ROOT)
    print("Open calibration proposals:")
    if not proposals:
        print("  (none)")
        return proposals
    print("  id | target | proposed | current | delta | age | gates | source")
    for proposal_id, proposal in proposals:
        address = str(proposal["record"])
        current = dotted_get(accepted, address)
        proposed = proposal.get("value")
        try:
            delta = f"{float(proposed) - float(current):+.6g}"
        except (TypeError, ValueError):
            delta = "n/a"
        provenance = proposal.get("provenance", {})
        source = provenance.get("source", "") if isinstance(provenance, dict) else ""
        print(
            f"  {proposal_id} | {address} | {_display_value(proposed)} | "
            f"{_display_value(current)} | {delta} | "
            f"{_age(proposal.get('created_at'))} | "
            f"{_gate_summary(proposal)} | {source}"
        )
    return proposals


if __name__ == "__main__":
    PROPOSALS = main()
