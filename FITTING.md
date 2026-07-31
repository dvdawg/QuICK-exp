# Native Quick fitting

The fitting launchers are analysis-only Python files. They do not connect to
QICK and can be run directly with the IDE Run button.

## Selecting input data

Every fitter has the same `INPUT_CSV` setting:

```python
# Newest matching one-dimensional native Quick CSV/YML pair:
INPUT_CSV = None

# Or one specific run:
INPUT_CSV = (
    r"Z:\David\Data\2026-07-21_MET_ver191_qubit3"
    r"\00054 - (ResonatorSpectroscopy)Zp0p0000_resonator_spec.csv"
)
```

Point to the CSV itself, not its directory. The paired YML must have the same
base filename and remain beside it. Use a raw string (`r"..."`), forward
slashes, or a raw UNC path:

```python
INPUT_CSV = (
    r"\\136.152.250.235\QDG\David\Data"
    r"\2026-07-21_MET_ver191_qubit3\00054 - example.csv"
)
```

When `INPUT_CSV = None`, two-dimensional files and files belonging to another
Quick experiment class are ignored.

## Fit launchers

| File | Result | Optional accepted record |
|---|---|---|
| `02b_fit_loopback.py` | pulse arrival and recommended trigger offset | `defaults.r_offset` |
| `05e_fit_resonator_spectroscopy.py` | fixed-flux resonator center and linewidth | `defaults.r_freq` |
| `06d_fit_qubit_spectroscopy.py` | fixed-flux qubit center and linewidth | `defaults.q_freq` |
| `11b_fit_t1.py` | T1 and uncertainty | `derived.t1` |
| `13b_fit_ramsey.py` | T2*, fringe, and two q-frequency estimates | `derived.t2_ramsey`; optionally `defaults.q_freq` |

Spectroscopy, T1, and Ramsey can analyze `amplitude`, `phase`, `I`, `Q`, or
`IQ`. `IQ` uses the measured principal axis; for T1 it uses the early-to-late
relaxation direction. Resonator spectroscopy defaults to the amplitude notch,
matching the resonator-vs-flux extraction.

`05d_fit_resonator_vs_flux.py` additionally accepts `FIT_METHOD`. `"fit"` is
the refined-notch cosine fit gated on R^2 and RMSE. `"min"`/`"max"` select that
smoothed amplitude bin in every Z row and store a piecewise-linear lookup
instead; use them when the cosine fit is unreliable. The sampled lookup is
exact at the measured rows, so the numerical gates do not apply and
`WRITE_ACCEPTED_FIT` is the only acceptance step. Its resolution is the
frequency-bin spacing, reported as the record's uncertainty.

For a broad qubit spectroscopy scan with several candidate features, set
`FIT_WINDOW_MHZ` around the one feature that should be calibrated:

```python
FIT_WINDOW_MHZ = (5595.0, 5615.0)
```

## Accepting a fit

All write latches default to `False`. First inspect the plotted fit, residuals,
uncertainty, and printed quality gates. Then set:

```python
WRITE_ACCEPTED_FIT = True
```

The update is quality-gated and written atomically to `calibration.yml`; a
superseded record is retained in `history`.

Ramsey always reports both possible drive-frequency corrections because a
single scalar Ramsey trace does not independently determine the detuning sign.
Writing `q_freq` therefore requires both:

```python
WRITE_ACCEPTED_FIT = True
UPDATE_Q_FREQUENCY = True
Q_FREQUENCY_CORRECTION_SIGN = +1  # Quick/notebook convention
```

Repeat Ramsey at the proposed frequency and verify that the fitted fringe moves
toward the programmed artificial fringe before treating the sign as confirmed.

For loopback, acquire the calibration trace with
`READOUT_OFFSET_US = 0`. This leaves the rising DAC-to-ADC edge inside the
record. A trace already aligned near the accepted offset can clip the edge and
will fail the quality gates.
