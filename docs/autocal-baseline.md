# Automated-calibration decision baseline

This reference snapshot is generated from the repository root with:

```console
python -m tools.baseline_legacy
```

These results are deterministic synthetic-device benchmarks, not hardware
acceptance thresholds. Regenerate them after changing the policy, simulator, or
probe signatures, and validate deployment thresholds against labeled data.

```text
class                   n  false_acc  false_rej  propagate   median_s   escalate    error
clean                  30      0.000      0.000      0.000    596.504      0.000    0.000
f02_shadow             30      0.800      0.200      0.000    596.504      0.200    0.000
low_snr                30      0.000      1.000      0.000        nan      1.000    0.000
neighbor_qubit         30      0.133      0.667      0.000    596.504      0.667    0.000
package_mode           30      0.000      0.000      0.000    596.504      0.000    0.000
tls                    30      0.000      1.000      0.000        nan      1.000    0.000
wrong_prior            30      0.000      0.000      0.000        nan      1.000    0.000
overall               210      0.133      0.410      0.000    596.504      0.552    0.000
```

The legacy gate falsely accepts 24 of 30 f02-shadow cases and 4 of 30
neighbor-qubit cases. It falsely rejects every low-SNR and TLS case, 20
neighbor-qubit cases, and 6 f02-shadow cases. Wrong-prior cases escalate but do
not count as false rejects because the answer was outside the acquired window.
Wrong-value propagation is zero here because this Phase 0 runner measures the
N5 decision in isolation and does not execute downstream nodes.

## Phase 1 — candidates and coverage

Measured on the same 210 synthetic devices with:

```console
python -m tools.baseline_hp
```

```text
class                   n  false_acc  false_rej  propagate   median_s   escalate    error
clean                  30      0.000      0.167      0.000    596.504      0.167    0.000
f02_shadow             30      0.867      0.133      0.000    596.504      0.133    0.000
low_snr                30      0.000      0.267      0.000    596.504      0.267    0.000
neighbor_qubit         30      0.733      0.067      0.000    596.504      0.067    0.000
package_mode           30      0.000      0.000      0.000    596.504      0.000    0.000
tls                    30      0.000      0.800      0.000    596.504      0.800    0.000
wrong_prior            30      0.000      0.000      0.000        nan      1.000    0.000
overall               210      0.229      0.205      0.000    596.504      0.348    0.000
```

| Overall metric | Legacy | Phase 1 | Change |
|---|---:|---:|---:|
| False-accept rate | 0.133 | 0.229 | +0.096 |
| False-reject rate | 0.410 | 0.205 | -0.205 |
| Escalation rate | 0.552 | 0.348 | -0.204 |

Phase 1 cuts false rejects by 20.5 percentage points, satisfying its coverage
and stall-reduction goal. It also exposes why candidate extraction is not an
identity decision: because weak alternatives are deliberately retained rather
than thresholded away, prominence alone falsely chooses 86.7% of f02 shadows
and 73.3% of neighbor qubits. The resulting 9.6-point false-accept regression is
reported rather than hidden; Phase 2 perturbation probes are required before
this path can promote a value.

## Phase 2 + Phase 3 — probes, scorecard, and ledgers

Measured through the production opt-in N5 path, including native CSV/YML
materialization, at the configured margin of 2.0:

```console
python -m tools.baseline_hypothesis --count 210 --seed 0 --margin-threshold 2.0
```

```text
class                   n  false_acc  false_rej  propagate   median_s   escalate    error
clean                  30      0.000      0.000      0.000   1278.162      0.000    0.000
f02_shadow             30      0.000      0.967      0.000   2156.368      0.967    0.000
low_snr                30      0.000      0.233      0.000   1278.162      0.233    0.000
neighbor_qubit         30      0.000      0.467      0.000   2156.368      0.467    0.000
package_mode           30      0.000      0.000      0.000   1278.162      0.000    0.000
tls                    30      0.000      0.767      0.000   2156.368      0.767    0.000
wrong_prior            30      0.000      0.000      0.000   5798.922      0.000    0.000
overall               210      0.000      0.348      0.000   1278.162      0.348    0.000
suggested_margin_threshold=2
```

| Overall metric | Legacy | Phase 1 | Phase 2 + 3 |
|---|---:|---:|---:|
| False-accept rate | 0.133 | 0.229 | **0.000** |
| False-reject rate | 0.410 | 0.205 | 0.348 |
| Wrong-value propagation | 0.000 | 0.000 | **0.000** |
| Median accepted time (s) | 596.504 | 596.504 | 1278.162 |
| Escalation rate | 0.552 | 0.348 | 0.348 |
| Error rate | 0.000 | 0.000 | 0.000 |

The perturbation path eliminates all 28 legacy wrong accepts and permits no
wrong value to proceed downstream. It also reduces false rejects from 86 to 73
synthetic cases versus the legacy gate, while deliberately escalating more
ambiguity than Phase 1. The extra time is the measured cost of identity
evidence; it is reported as a monitor rather than traded against the primary
safety metric.

The 2.0 cutoff is the least restrictive threshold validated by this synthetic
run; it is not a hardware-ready default. The calibrator will not infer a lower
cutoff from these results, because lowering the margin can stop the probe
sequence earlier and change the evidence. A different cutoff requires a new
full run.

## Phase 4 — adaptive two-dimensional acquisition

Measured over the same 210 seeded synthetic devices against the fixed 13-row
resonator map:

```console
python -m tools.adaptive_zoo
```

```text
count=210
maximum_row_fraction=0.538461538
median_adaptive_rmse_mhz=1.16176032e-08
median_fixed_rmse_mhz=8.17818851e-09
noninferior_fraction=1
```

The adaptive scheduler uses at most 7 of 13 rows (53.85%). Every synthetic
device is non-inferior within the declared 1e-6 MHz numerical tolerance; the
two median errors differ only at floating-point scale.

## Archived-trace reality check

`tests/fixtures/labeled/manifest.yml` remains intentionally empty until an
operator supplies the correct identity and value for real cooldown traces.
`python -m tools.archived_trace_regression` validates candidate extraction and,
when a session is supplied, the complete probe/adjudication replay. Synthetic
labels are not substituted for this external ground truth.
