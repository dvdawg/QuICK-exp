# Architecture

## Flow

```text
numbered IDE launcher
        |
        | experiment + preset + explicit overrides + native title
        v
hardware/calibration/presets YAML -> ConfigRepository -> ResolvedConfig
                                                    |
                                                    v
                                      experiment adapter -> ExperimentPlan
                                                    |
                                  SyntheticBackend / QuickBackend
                                                    |
                                  Quick native Saver -> CSV + YML
                                                    |
                                           decode -> analysis/plot
```

Top-level `experiments/` files are operator entry points. Modules under
`quickexp_v3/experiments/` translate v3 parameter names and result columns to
installed Quick 0.7.2. Library modules are not runnable entry points.

## Ownership

- A numbered launcher owns today's axes, fixed values, live/simulated mode,
  descriptive native title, and plot choice.
- `01_configure_experiment.py` is the central IDE editor for shared YAML values.
- `hardware.yml` owns connection identity, routing, limits, output directory,
  RF-board policy, and flux safety.
- `calibration.yml` owns accepted values, valid domains, fit quality, and
  provenance.
- `task_queue.py` runs existing numbered launchers sequentially without bypassing
  their normal safety, persistence, or cleanup paths.
- `autocal/` owns the explicit calibration DAG, search ladders, policy decisions,
  budgets, atomic session state, append-only audit events, and the node scheduler.
  Node acquisition still flows through `ide.run_experiment`; synthetic results
  are materialized into native Quick pairs before fitting.
- `fit_calibration.py` is the single atomic accepted-record and proposal
  lifecycle writer. Proposal-only writes do not increment accepted calibration
  revision and never participate in resolved configuration.
- `rabi_fit.py` identifies native Time/Power Rabi axes from paired Quick YML,
  fits rotated IQ, quality-gates the pi recommendation, and atomically versions
  accepted `q_length`/`q_gain` defaults.
- `resonator_flux.py` reproduces the notebook notch extraction/cosine fit,
  quality-gates acceptance, and atomically versions the lookup calibration. It
  also builds a sampled `min`/`max` piecewise-linear lookup for maps where the
  cosine fit is unreliable; both models resolve through `flux_lookup.py`.
- `presets.yml` owns reusable starting scans, averaging, retry, and analysis.
- An adapter owns the Quick class, axis mapping, output columns, and analysis.
- `runtime.py` owns exact retry, decode, analysis, and cleanup; it does not
  implement a second persistence format.
- `lab.py` owns Pyro connection, connected-port verification, optional RF-board
  calls, and notebook-compatible held-Z setup.
- Quick's `Saver` exclusively owns live CSV/YML persistence.

Resolution order is deterministic:

```text
hardware defaults < accepted calibration < preset parameters < launcher overrides
```

Automated calibration adds a control plane around, not beneath, that flow:

```text
91_autocal.py -> explicit target DAG -> IDE acquisition helper -> native pair
                                  |                         |
                                  v                         v
                         fitter passes() gates       decisions.jsonl
                                  |
                                  v
                   inert proposal -> policy -> optional promotion
```

`hardware.autocal` is a protected hardware root. A preset or run override
cannot widen its drive caps, promotion allowlist, hard stops, attempt cap, run
cap, or wall-clock cap. `autocal_runs/STOP` is checked before every acquisition.

## Quick 0.7.2 boundary

Quick registers a sweep only when an iterable is passed as a constructor
keyword. V3 passes each declared axis as a constructor sweep. `hard_avg`,
`soft_avg`, `rep`, and `p0_nqz`-`p3_nqz` remain Mercator configuration
overrides. The Nyquist keys let a launcher select the physical DAC image
without editing the installed Quick package; a zone other than 1 or 2 is
rejected before acquisition.

`QuickBackend` forces `silent=False` when `qick.show_progress` is enabled, which
activates Quick's native `quick.Sweep` progress display. It also returns the
exact Saver CSV/YML paths as acquisition metadata.

Before held flux is established, `QuickBackend` validates uploaded envelopes
against the connected generator's waveform-memory metadata. Built-in classes
use the verified legacy table; authored programs attach a style-aware
`MercatorProgram.preflight` callback covering memory, fabric-clock
granularity, clipping, mixer continuity, and a sequence-duty warning. The
sample calculation intentionally matches Quick 0.7.2's Mercator conversion.
Deterministic `ConfigError` failures bypass acquisition recovery and retry.

Native Cartesian sweeps power the resonator punchout, qubit-gain scan, and
Rabi/Ramsey chevrons. Flux remains an outer loop because generator 15 must be
held and reapplied after recovery. Row acquisitions disable their individual
Savers; one direct Quick `Saver` writes the assembled Z-by-inner-axis table,
matching the historical `ResVsZ_held_bias` and
`QubitSpecVsZ_fitted_readout` files.

The numbered `08c` analysis launcher reads a one-dimensional native Rabi
CSV/YML pair, infers `q_length` or `q_gain` from Quick metadata, and writes a
history-preserving scalar calibration only through an explicit acceptance latch.

The numbered `05d` analysis launcher reads the native `ResVsZ_held_bias` CSV,
extracts one resonator notch per Z row, and fits the accepted cosine lookup.
Later fixed-Z launchers evaluate that lookup once; `06b` evaluates it for each
outer Z row. The calibration record carries its source, quality, uncertainty,
and measured domain, and out-of-domain use fails before hardware acquisition.

`run_flux_sweep` also accepts a per-Z override callback. It resolves every
row's overrides before the first acquisition, so the preview plan, the saved
`row_overrides_by_z` metadata, and the acquisition loop cannot diverge. A
callback may move the inner sweep itself; the combined plot then uses the full
2D coordinate grid, and one combined native Quick CSV/YML pair is still
written.

Before the auxiliary held-Z program is compiled, every numeric array in its
variables is reduced to a safe scalar (frequency center; minimum
power/gain/length; first value otherwise). This shared boundary covers every
native 2D experiment rather than special-casing resonator power.

## Adding a measurement

1. Add or reuse an adapter in `quickexp_v3/experiments/` and register it.
2. Add a conservative reusable preset if needed.
3. Add a numbered launcher with an `EDIT THESE` block, descriptive native
   title, `main()`, and a normal `__main__` guard.
4. Add synthetic plan/decode coverage and a fake Quick constructor test for
   custom behavior.
5. Compile and run tests with the Python 3.9 `qcodes` interpreter before
   enabling hardware.

## Adding an authored Mercator program

`mercator.py` is deliberately a thin, key-validating builder for the observed
Quick schema; it is not a second pulse compiler. `TwoTone_ZPA` and `T1_zpa`
are the worked examples.

1. Define `PROGRAM` under `quickexp_v3/programs/` with declared variables,
   labels, pulses/readout/steps, and every non-constant envelope term.
2. Export it from `programs/__init__.py`. That registers its constructor
   variables and envelope terms without mutating the seed tables. Configuration
   controls `hard_avg`, `soft_avg`, and `rep` are rejected as program variables.
3. Add a dedicated adapter, set `quick_class = PROGRAM.name`, attach
   `PROGRAM.preflight` to plan metadata, and register the adapter explicitly.
4. `lab.install_authored_programs` installs all authored templates immediately
   after the Quick version check. Installation is additive and idempotent.
5. Add matching conservative presets, a numbered launcher, a coherent
   `SyntheticBackend` branch, fake-Quick sweep/config routing coverage, and an
   offline launcher test.
6. Run the parity and full test suites. On the lab machine, perform a three-point
   smoke and diff the native sidecar's resolved config against the rendered
   template before scaling a sweep.

The current offline implementation reconstructs the held-Z template and seven
real Quick sidecars key-for-key. Live enablement remains intentionally off in
the two authored launchers until hardware gates G1–G4 in
`plans/20-pulses.md` are completed. In particular, the flat-top RAM cost still
uses the conservative observed `8σ` ramp assumption and reports that assumption
at preflight.
