# Native Quick fitting

The fitting launchers are analysis-only Python files. They do not connect to
QICK and can be run directly with Python or an editor's run command.

## Selecting input data

Every fitter has the same `INPUT_CSV` setting:

```python
# Newest matching one-dimensional native Quick CSV/YML pair:
INPUT_CSV = None

# Or one specific run:
INPUT_CSV = r"/path/to/data/run.csv"
```

Point to the CSV itself, not its directory. The paired YML must have the same
base filename and remain beside it. Use a raw string (`r"..."`), forward
slashes, or a raw UNC path:

```python
INPUT_CSV = r"\\file-server\share\path\to\run.csv"
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
FIT_WINDOW_MHZ = (MIN_MHZ, MAX_MHZ)  # replace with numeric bounds
```

For `06e_fit_qubit_vs_flux.py`, the frequency and flux axes can be restricted
independently. `None` uses the full acquired axis:

```python
FIT_FREQUENCY_WINDOW_MHZ = (MIN_MHZ, MAX_MHZ)
FIT_FLUX_WINDOW_Z = (MIN_Z_GAIN, MAX_Z_GAIN)
```

## Following a qubit feature with a custom sweep path

Use `06g_design_qubit_sweep_path.py` after a broad two-dimensional scan. It
infers generic parameter names from the background axes and stores one or more
selected inner-axis intervals for each outer-axis row. For a
Z-by-qubit-frequency map, it creates a `z_gain`/`q_freq` path that 06b can run.

Two design modes are available:

```python
# Fit the spectroscopy ridge and add a margin on both sides.
PATH_METHOD = "fit_margin"
FIT_MARGIN_MHZ = 10.0  # or (below_mhz, above_mhz)

# Draw a polygon directly over the phase/amplitude map from the prior sweep.
PATH_METHOD = "ui_polygon"
BACKGROUND_SIGNAL = "phase"
```

In UI mode, click around the feature to close a polygon, inspect the white
previewed region, then choose **Use region**. The prior sweep remains visible
as the background throughout editing.

Concave polygons are retained as one or more disjoint intervals at each outer
row. If a vertical slice crosses two selected frequency bands with a gap
between them, the saved path jumps between those bands; it does not sweep the
unused frequencies inside the overall lower/upper envelope.

By default, each selected row keeps the frequency spacing of the background
map. Narrow polygon rows therefore contain fewer points without changing the
MHz resolution. To choose a different spacing, set:

```python
FREQUENCY_RESOLUTION_MHZ = None  # preserve the background resolution
# FREQUENCY_RESOLUTION_MHZ = 0.5  # optional manual override
```

To acquire a Z-by-frequency path, set this in
`06b_qubit_spectroscopy_vs_flux.py`:

```python
SWEEP_PATH_YML = (
    PROJECT_ROOT / "analysis_cache" / "qubit_flux_frequency_sweep_path.yml"
)
```

06b then maps each saved `z_gain` through the held-flux controller, runs that
row's saved `q_freq` values, and retains fitted readout-frequency tracking.
Setting
`SWEEP_PATH_YML = None` retains the original rectangular native sweep.

Analyze either kind of result with `06e_fit_qubit_vs_flux.py`. With
`INPUT_CSV = None`, it selects the latest `QubitSpecVsZ` run. Custom-path rows
are kept at their acquired lengths and split at unswept gaps; the fitter finds
several local candidates in each acquired band and selects one globally
transmon-like ridge. The left plot displays the actual acquired points rather
than filling the holes. Leave both fit-window settings at `None` to use every
acquired point, or set either one to restrict the analysis further.

At the library level, `SweepPath` and `run_sweep_path` are axis-generic: the
YAML declares `outer.name` and `inner.name`, and the runner maps those names to
the corresponding experiment overrides. Launcher-specific code only validates
that a path contains axes appropriate for that experiment and selects any
required physical outer-axis control, such as held flux for `z_gain`.

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
# Choose a sign, then verify it experimentally.
Q_FREQUENCY_CORRECTION_SIGN = +1
```

Repeat Ramsey at the proposed frequency and verify that the fitted fringe moves
toward the programmed artificial fringe before treating the sign as confirmed.

For loopback, set `READOUT_OFFSET_US` so the complete rising DAC-to-ADC edge
stays inside the recorded window. A trace aligned too close to a record boundary
can clip the edge and fail the quality gates.
