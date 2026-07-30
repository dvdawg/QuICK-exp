# Automated calibration operator runbook

Automated calibration uses the same configuration resolution, IDE acquisition
helpers, Quick retry/recovery, held-Z cleanup, native CSV/YML pairs, and fit
gates as the numbered manual workflow. It decides which measurement to run
next; it does not introduce a second acquisition stack.

## Start offline

Open `experiments/91_autocal.py` and begin with:

```python
LIVE_HARDWARE = False
SESSION_NAME = None
TARGET = "full_cold_start"
Z_GAIN = 0.0
AUTONOMY_LEVEL = 0
MAX_WALL_CLOCK_HOURS = 8.0
REPLAY_SESSION = None
```

The supported targets are:

| Target | Nodes |
|---|---|
| `full_cold_start` | ports, timing, punchout, resonator map/working point, qubit frequency, pi pulse, IQ threshold, T1, Ramsey, Echo |
| `flux_point` | resonator/qubit working point, pi pulse, IQ threshold, coherence |
| `readout_only` | resonator working point and IQ threshold; optimization escalates only when fidelity is below target |
| `coherence_only` | T1, two-point Ramsey sign check, Echo |

Offline runs use one persistent `DeviceModel`, so a frequency found by one node
is the ground truth consumed by downstream acquisitions. Synthetic signal data
is materialized under the session's `native/` directory in native Quick
CSV/YML form and fitted from those files.

## Autonomy

- L0 always proposes and never promotes. This is the required level for initial
  hardware rollout.
- L1 promotes only allowlisted records within the hardware policy tolerance.
- L2 promotes all passing records except hard stops; configured tolerances still
  apply.

`defaults.r_offset`, `lookups.resonator_vs_flux`, and
`lookups.qubit_vs_flux` are hard stops at every level. Ramsey can never promote
a qubit-frequency correction unless its two-point detuning-sign test passes.
Readout optimization node N10r can never promote its frequency result.

The policy lives under `hardware.autocal`. It is protected from presets and
one-run overrides. The lower of the launcher and hardware wall-clock limits
wins. If the policy root is absent, every requested autonomy level is forced
to proposal-only behavior. The three structural hard stops cannot be removed
from a configured policy.

## Stop and resume

Create this empty file while a session is running:

```text
autocal_runs/STOP
```

The scheduler notices it before the next acquisition, records
`stopped_by_operator`, atomically preserves state, and exits. Remove the file,
copy the existing directory name into `SESSION_NAME`, and run 91 again.
Completed nodes are not repeated unless accepted calibration changed while the
session was stopped. In that case, only products whose changes exceed their
node invalidation thresholds—and their affected downstream nodes—are marked
stale and reacquired.

Each session contains:

- `state.yml`: target, working values, node attempts/statuses, and budget;
- `decisions.jsonl`: append-only decisions and native-file references, with no
  signals or arrays;
- `native/`: simulation-only native pairs. Live data remains in
  `storage.quick_native_root`.

Set `REPLAY_SESSION` to a session directory for a read-only audit replay. Replay
re-runs every logged fitter from its referenced native CSV/YML pair and fails
if a gate or verdict changes. It does not connect, acquire, or change
calibration/session files.

## Review proposals

Open `experiments/92_review_proposals.py`. Run once with empty latches to print
the table, then fill:

```python
PROMOTE = ["proposal-id"]
REJECT = {"another-id": "ambiguous second feature"}
ACCEPTED_BY = "initials"
```

Promotion is one atomic mutation: it versions the previous accepted record,
stamps proposal/creator/revision metadata, increments calibration revision, and
removes the open proposal. Rejection archives the proposal without changing
accepted records or revision. A retake by the same session/node/record replaces
its older open proposal.

## Live rollout gates

Offline completion establishes control-flow coherence, not hardware validity.
Before unattended use:

1. complete the authored-program hardware gates in `plans/20-pulses.md`;
2. run `readout_only`, then `flux_point`, then `full_cold_start` at L0 with an
   operator watching;
3. review every native trace and decision event through 92;
4. tune thresholds only from observed repeatability;
5. enable L1 first for `coherence_only`, then expand cautiously.

The missing bitfile hash, reference clock, and ADC full-scale values remain
explicitly visible in session facts. A lost hardware link can prevent software
from commanding a held RF Z line to park. Exhausted connection or acquisition
retries end the session with `critical_abort`; treat that condition as a
physical operator escalation, not as proof of a safe zero.
