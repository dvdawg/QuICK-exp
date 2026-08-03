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


| Target            | Nodes                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| `full_cold_start` | ports, timing, punchout, resonator map/working point, qubit frequency, pi pulse, IQ threshold, T1, Ramsey, Echo |
| `flux_point`      | resonator/qubit working point, pi pulse, IQ threshold, coherence                                                |
| `readout_only`    | resonator working point and IQ threshold; optimization escalates only when fidelity is below target             |
| `coherence_only`  | T1, two-point Ramsey sign check, Echo                                                                           |


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

## Hypothesis-and-probe decisions

Identity-critical decisions are opt-in per node. The shipped hardware example
keeps both migrations disabled so copying it cannot silently change an existing
lab workflow. Enable the reviewed paths deliberately:

```yaml
autocal:
  hypothesis_nodes: [N5]
  adaptive_nodes: [N2, N3]
```

With N5 enabled, qubit spectroscopy emits every ranked feature plus a null
candidate. It checks whether the scan could answer the question, then probes
candidate identity in cost order: drive-power ladder, held-flux nudge,
dispersive response, and two-gain Rabi. Every acquisition still goes through
the registered experiment/preset and existing IDE safety path. The held-flux
probe uses the existing flux-sweep parking and recovery behavior.

The identity verdict depends only on the score margin between physical
hypotheses. Absolute goodness-of-fit thresholds are not part of that verdict.
The example `margin_threshold: 2.0` is the conservative zoo-validated setting:
the 210-chip run at that threshold has zero false accepts. The harness never
claims that a lower threshold is safe from a higher-threshold run, because a
lower cutoff can stop the probe battery earlier. Rerun
`python -m tools.baseline_hypothesis` after changing the threshold, device
model, or probe signatures. A coverage failure is class A and changes
acquisition parameters. A competing physical identity is class B and runs the
next probe. An imprecise estimate is class C and refines the measurement.

If Rabi later fails, the scheduler revisits the joint `(q_freq, q_gain)` choice
before blocking downstream nodes. It tries a bounded gain retune first, then
demotes the current N5 candidate and promotes the next ledger entry. The hard
caps under `autocal.backtracking` prevent loops; a demoted candidate cannot be
promoted again in the same session.

## Adaptive maps and physics report

N2 and N3 can acquire rows adaptively. Both start with five spanning rows,
abort a map that has fewer than three trackable features at the reviewed abort
point, and use at most the configured seven rows. N3 recenters each frequency
window on the previous fitted notch. The combined result remains one native
CSV/YML pair, including when row frequency axes differ.

Hypothesis sessions add these control artifacts:

- `discrepancy-report.md`: all seven declared circuit-QED checks, including
  explicit `untestable` rows when evidence was not acquired;
- `state.yml` → `hypothesis_ledger`: ranked identities, evidence, demotions,
  and backtrack counters;
- `state.yml` → `discrepancy_ledger`: predicted/measured values, residuals in
  sigma, assumptions, sources, and verdicts;
- `advisory_images/`: rendered fit overlays only when an unresolved scorecard
  needs advisory review.

The report covers qubit design band, chi, resonator/qubit flux agreement,
anharmonicity, the `T2 <= 2*T1` bound, Rabi-rate linearity, and readout fidelity
versus IQ separation. Missing measurements are not silently treated as passes.

## Optional advisor

`autocal.advisor.mode` accepts `null`, `replay`, or `claude`. `null` is the
default and deterministically escalates; calibration remains complete and
correct without network access. `claude` uses a small standard-library HTTP
client and reads `ANTHROPIC_API_KEY` from the environment. The key is never
read from YAML.

The advisor is called only at session start, an unresolved scorecard, a novel
signature mismatch, or session end. Requests may contain candidate and fit
statistics, configuration context, discrepancy entries, and rendered PNG
overlays. They do not contain raw signal arrays. The deterministic policy
rejects unknown programs, unknown knobs, out-of-limit values, and over-budget
actions. A validated suggestion is stored as `validated_not_executed`; it is
never acquired automatically. A proposed novel pulse program remains report
text for manual implementation.

External advisory use therefore needs two deliberate lab decisions: outbound
access to the configured API endpoint and approval for the listed metadata and
PNG overlays to leave the lab network. Keep `mode: null` otherwise.

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
re-runs every logged fitter from its referenced native CSV/YML pair, rebuilds
probe responses and scorecards, and verifies advisor responses by request hash.
It fails if a gate, identity verdict, response, or policy decision changes. It
does not connect, acquire, call an external advisor, or change calibration and
session files.

## Validation commands

Run the same seeded harness used to select the identity margin:

```powershell
python -m tools.baseline_legacy
python -m tools.baseline_hp
python -m tools.baseline_hypothesis --count 210 --margin-threshold 2.0
python -m tools.adaptive_zoo
```

The first two tables are retained in `docs/autocal-baseline.md`; the third
reports the production hypothesis path and all five decision metrics per
defect class. `tools.adaptive_zoo` compares lookup error against the fixed
13-row map and reports the row fraction.

Real cooldown traces are the remaining external validation input. Add
operator-reviewed entries to `tests/fixtures/labeled/manifest.yml`, then run:

```powershell
python -m tools.archived_trace_regression
```

Each label must name the correct frequency and physical hypothesis. Pointing an
entry at a complete hypothesis session additionally replays its probes and
identity verdict. Do not invent labels from fit quality; they require operator
knowledge of the chip.

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
3. review every native trace, scorecard, and discrepancy entry through 92;
4. enable N5 at L0, then N2/N3 adaptivity, and compare the recorded zoo and
   supervised-hardware metrics;
5. tune margins only from labeled zoo/archive outcomes and repeatability;
6. enable L1 first for `coherence_only`, then expand cautiously.

The missing bitfile hash, reference clock, and ADC full-scale values remain
explicitly visible in session facts. A lost hardware link can prevent software
from commanding a held RF Z line to park. Exhausted connection or acquisition
retries end the session with `critical_abort`; treat that condition as a
physical operator escalation, not as proof of a safe zero.
