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

## Quick 0.7.2 boundary

Quick registers a sweep only when an iterable is passed as a constructor
keyword. V3 passes each declared axis as a constructor sweep. `hard_avg`,
`soft_avg`, and `rep` remain Mercator configuration overrides.

`QuickBackend` forces `silent=False` when `qick.show_progress` is enabled, which
activates Quick's native `quick.Sweep` progress display. It also returns the
exact Saver CSV/YML paths as acquisition metadata.

Native Cartesian sweeps power the resonator punchout, qubit-gain scan, and
Rabi/Ramsey chevrons. Flux remains an outer loop because generator 15 must be
held and reapplied after recovery. Row acquisitions disable their individual
Savers; one direct Quick `Saver` writes the assembled Z-by-inner-axis table,
matching the historical `ResVsZ_held_bias` and
`QubitSpecVsZ_fitted_readout` files.

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

Custom notebook programs such as `T1_zpa` and `TwoTone_ZPA` require dedicated
adapters/templates and must not be routed through an unrelated Quick class.
