# Improved QICK Lab-Readout Workflow — Design

**Date:** 2026-06-28
**Status:** Approved (architecture)

## Problem

The current workflow lives in an exploratory Jupyter notebook (`2026-06-24 QICK2.ipynb`,
~86 code cells) driving the `quick` framework (v0.7.0) on a lab control PC at
`192.168.1.123`. It walks the full calibration chain — bringup → resonator spectroscopy →
qubit spectroscopy → dispersive/IQ readout → Rabi → T1/T2 — writing numbered `CSV`+`YAML`
result files to a dated data folder (e.g. `2026-06-28_MET_ver191/`).

Pain points in the notebook:

- Hardware setup (IP, channel map `q`/`r`/`rr`, bias, RF-board atten/filter) is copy-pasted
  and hand-edited in many cells (e.g. cells 4, 5, 11).
- Fitting is ad-hoc: inline `savgol`/SVD snippets (cells 9, 13) except `quick.fitT1`/`fitT2`.
- Sweep windows (coarse vs. fine resonator/qubit scans) are typed by hand each run.
- Calibrated values (`q_freq`, `r_freq`, thresholds, π-pulse, T1/T2) live only in cell state
  and are lost on kernel restart.

## Goals

1. A **config file** for hardware setup, replacing the copy-pasted setup blocks.
2. **Fitting** routines for every stage, factored out of the notebook.
3. **Adaptive window sweeps** (coarse → fine) driven by fits.
4. A clear, **logical progression** of calibration and experiments.
5. **Robust outputs** — without changing the raw CSV format.

## Constraints

- **Raw data CSV format must stay byte-for-byte identical** to the original notebook output.
  It is consumed by separate plotting software. All new outputs are written alongside, never
  by modifying the CSV.
- The `quick` package only exists on the lab PC and is not importable elsewhere. Therefore all
  non-hardware logic must be testable offline against the existing result CSVs
  (columns: `freq, amplitude(dB), phase(rad), I, Q`).

## Key Decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Deliverable form | Python package **and** a thin driver notebook on top (both testable and usable) |
| Scope (v1) | **Full chain**: readout → qubit spec → Rabi → T1/T2/Ramsey |
| Adaptivity | **Fit-and-suggest, operator in the loop** (default); per-step coarse→fine auto-narrow is opt-in |
| Outputs added | **Persistent calibration state** + **fit results saved** (sidecar). No PNG export, no run log. |
| Raw CSV | **Unchanged** (external plotting software depends on it) |
| Architecture | **A — Layered library + thin "step" wrappers** |

## Architecture (Approach A)

Pure, hardware-free modules plus one thin `steps` layer that is the *only* code touching the
QICK hardware. The notebook calls `steps`. Fitting and adaptive logic are unit-tested offline
against the existing result CSVs; the one untestable part (hardware I/O) is quarantined in
`steps.py`.

```
labreadout/
  config.py      # load + validate hardware YAML, apply to soc, build base `var`
  state.py       # load/save persistent calibration state, merge into `var`
  fitting.py     # resonator dip, qubit peak, Rabi, T1, T2, IQ-threshold fits
  adaptive.py    # given a fit, propose a tightened coarse→fine window
  steps.py       # the ONLY hardware layer: run quick.experiment → fit → plot → suggest
  results.py     # write fit sidecar YAML next to each CSV (CSV untouched)
hardware.yml     # hardware config file (user-edited, static)
calibration.yml  # persistent calibration state (code-managed, evolving)
driver.ipynb     # thin notebook walking the full chain
tests/           # offline tests against existing result CSVs
```

### Module responsibilities

- **config.py** — Loads and validates `hardware.yml`. On session start, applies the RF-board
  setup (`soc.rfb_set_gen_rf`, `rfb_set_gen_filter`, `rfb_set_ro_*`) and bias
  (`soc.rfb_set_bias`) to the connected `soc`, and seeds the base `var` dict. Replaces the
  copy-pasted setup blocks (cells 4/5/11). What does it do / how used / depends on: turns a
  YAML file into a configured `soc` + base `var`; depends on a connected `soc`, `quick.experiment.var`.
- **state.py** — Loads `calibration.yml` at startup and merges fitted values over the base
  `var` defaults so calibrated numbers survive kernel restarts. Writes back only when a fit is
  accepted. Pure file + dict logic, fully offline-testable.
- **fitting.py** — One function per stage, each returning `params + uncertainties +
  goodness-of-fit + matplotlib figure`. Consumes the CSV column arrays directly, so testable
  against the 40 existing result files. Replaces inline savgol/SVD snippets.
- **adaptive.py** — Pure functions: given a fit result and detected feature, propose a tightened
  scan window (center ± span, step). No hardware, no I/O. Offline-testable.
- **steps.py** — The fit-and-suggest layer and the only hardware-touching code. Each step:
  apply hardware/`var` → run the `quick` experiment (CSV saved by `quick` exactly as today) →
  fit → plot with overlay + extracted value → print a suggested next window/param → return a
  result object. Nothing commits until the operator calls `.accept()`. Opt-in `auto_narrow=True`
  performs the coarse→fine re-scan in one call.
- **results.py** — Writes a small fit-sidecar YAML next to each result CSV containing fit
  params, uncertainties, and GoF. Never modifies the CSV.

## Config split (the important boundary)

- **`hardware.yml` (static, user-edited):** IP, data path, channel map (`q`, `r`, `rr`),
  per-channel RF-board setup (atten1/atten2, filter type/fc/bw), bias channel + default bias,
  and base `var` defaults.
- **`calibration.yml` (evolving, code-managed):** fitted values — `r_freq`, `q_freq`,
  π-pulse length/gain, `r_threshold`, T1, T2, etc. Loaded at startup, merged over defaults,
  updated only on accepted fits.

Per-call experiment overrides still win over both, preserving the notebook's ability to scan.

## Fit-and-suggest step flow

```
steps.<stage>(...)            # apply hardware + var, run quick.experiment (CSV saved as today)
  → fitting.<stage>(...)      # fit the returned data
  → plot with fit overlay + extracted value
  → adaptive.<stage>(...)     # print suggested next window/param
  → return result object      # not yet committed
result.accept()               # writes accepted values to calibration.yml + fit sidecar
```

Default is operator-in-the-loop: review the plot and suggestion, then `.accept()` to advance.
`auto_narrow=True` is an opt-in convenience that runs coarse→fine within a single call.

## Fitting module (offline-testable today)

| Fit | Input | Output |
| --- | --- | --- |
| `resonator` | freq, amp/I,Q | f₀, Q, depth (notch/Lorentzian dip) |
| `qubit_peak` | freq, amp/rotated-IQ | f₀, linewidth (Lorentzian) |
| `rabi` | gain/length, signal | π length/gain (decaying sinusoid) |
| `T1` | time, signal | T1 (wraps `quick.fitT1`) |
| `T2` | time, signal | T2 (wraps `quick.fitT2`) |
| `iq_threshold` | I, Q (two states) | readout threshold + fidelity (two-blob) |

Each returns params + uncertainties + goodness-of-fit + a matplotlib figure.

## Outputs

- **Raw CSV:** unchanged, still written by `quick`'s `Saver`.
- **Fit sidecar:** `results.py` writes a small YAML next to each CSV with fit params,
  uncertainties, and GoF.
- **`calibration.yml`:** updated on accepted fits.
- No PNG export, no run log (explicitly out of scope for v1).

## Chain progression in `driver.ipynb`

Loopback bringup → resonator spectroscopy (coarse→fine) → dispersive/power sweep →
IQ readout calibration → qubit spectroscopy (coarse→fine, gain) → Rabi → T1 →
T2 Ramsey/Echo. Each section: one `steps.*` call, review plot + suggestion, `.accept()` to
advance — mirroring the existing notebook order.

## Testing strategy

- `fitting.py`, `adaptive.py`, `state.py`, `results.py`, and `config.py` validation are
  unit-tested offline. Fitting/adaptive tests run against the real CSVs in
  `2026-06-28_MET_ver191/` (and equivalents), asserting recovered features match known values.
- `steps.py` hardware interaction is exercised only on the lab PC; its non-hardware glue is
  kept thin enough to read by inspection.

## Out of scope (v1)

- PNG/figure export and human-readable run logs.
- Fully autonomous chain execution (no operator confirmation).
- Any change to the raw CSV format or the `quick` framework itself.
