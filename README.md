# labreadout

An improved, testable QICK calibration & readout workflow layered over the
`quick` framework. It replaces the hand-edited exploratory notebook with a small
Python package plus a thin driver notebook.

## What it adds

| Module | Responsibility |
| --- | --- |
| `config` | Load/validate `hardware.yml`, apply RF-board + bias setup to `soc`, build base `var` |
| `state` | Persist fitted values in `calibration.yml` so they survive kernel restarts |
| `fitting` | Resonator (complex notch), qubit peak, Rabi, T1, T2, IQ-threshold fits (params + uncertainties + GoF + plot) |
| `adaptive` | Coarse→fine sweep-window logic |
| `ports` | Resolve logical `q`/`r`/`rr` indices to physical DAC/ADC ports; flag direct/no-RF-card paths and declared-port mismatches |
| `diagnostics` | Advisory health report: SNR/noise, ADC over-range, known-value sanity, a decision-tree verdict, and recommended knobs |
| `loopback` | Loopback bring-up: find the readout-window delay (`r_offset`) and a safe drive level |
| `results` | Write fit-result sidecars (`*.fit.yml`) **next to** the raw CSV |
| `steps` | The fit-and-suggest `Session` that drives the hardware |

## Design principles

- **Raw CSVs are never touched.** `quick` writes them in the original format
  (external plotting software depends on it); we only add `*.fit.yml` sidecars.
- **Operator in the loop.** Each step runs, fits, plots, and *suggests* the next
  scan. Nothing reaches `calibration.yml` until you call `.accept()`.
- **Testable off the lab PC.** Only `steps.py` imports `quick`, lazily. Fitting,
  adaptive, config, state, and results are unit-tested offline — fitters against
  synthetic data with known ground truth and against the real result CSVs.

## Configuration

- **`hardware.yml`** (static, edit by hand): board IP, data path, channel map
  (`q`/`r`/`rr`), per-channel RF-board atten/filter, bias, base `var` defaults.
- **`calibration.yml`** (code-managed): latest fitted values, merged over the
  `var` defaults at session start and updated on `.accept()`.

Precedence at run time: **per-call overrides > calibration state > config defaults**.

## Usage

```python
import numpy as np
import labreadout as lr

sess = lr.Session.connect("hardware.yml", "calibration.yml")   # lab PC only

res = sess.resonator_spectroscopy(r_freq=np.arange(6580, 6590, 0.01))
res.plot()
print(res.suggestion)          # e.g. "resonance at 6584.5 MHz (Q≈...) → fine scan ..."
res.accept()                   # commit r_freq to calibration.yml
```

See `driver.ipynb` for the full chain: bringup → resonator → dispersive/power →
IQ readout → qubit spec → Rabi → T1 → T2.

## Bring-up & diagnostics

- **Ports.** `Session.connect()` prints the logical→physical DAC/ADC map and
  warns on direct/no-RF-card paths. Declare the expected port in `hardware.yml`
  (`r: {gen: 1, dac_port: 10}`) and startup *refuses to run* on a mismatch.
  Re-check any time with `sess.check_ports()`, or from the shell:

  ```bash
  python diagnose_ports.py            # read-only map
  python diagnose_ports.py --test 1 1 # safe low-power loopback on one pair
  ```

- **Loopback calibration.** `sess.loopback_calibrate()` starts at `r_offset=0`,
  measures the readout-window delay, then sweeps power to pick a level above
  noise but below ADC over-range. `.accept()` commits `r_offset` + `r_power`.

- **Result diagnosis (advisory).** Every step attaches a `.diagnosis` and prints
  a short report — SNR/noise, over-range, and a known-value sanity check against
  the `expected:` block in `hardware.yml` — with a likely cause and next action.
  It never blocks; the `.accept()` invalid-fit guard is the only hard gate.

- **Recommend-then-override.** Each result exposes `.recommendations` (e.g. Rabi →
  `q_length`/`q_gain` for the π pulse). Stage them with `sess.apply(recs)`;
  explicit per-call kwargs always win. `hardware.yml` documents knob ranges under
  `limits:`.

## Tests

```bash
python -m pytest        # runs offline; no hardware or `quick` required
```
