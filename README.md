# labreadout

An improved, testable QICK calibration & readout workflow layered over the
`quick` framework. It replaces the hand-edited exploratory notebook with a small
Python package plus a thin driver notebook.

## What it adds

| Module | Responsibility |
| --- | --- |
| `config` | Load/validate `hardware.yml`, apply RF-board + bias setup to `soc`, build base `var` |
| `state` | Persist fitted values in `calibration.yml` so they survive kernel restarts |
| `fitting` | Resonator dip, qubit peak, Rabi, T1, T2, IQ-threshold fits (params + uncertainties + GoF + plot) |
| `adaptive` | Coarse→fine sweep-window logic |
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

## Tests

```bash
python -m pytest        # runs offline; no hardware or `quick` required
```
