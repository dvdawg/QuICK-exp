# Flux-line predistortion replication runbook

This runbook implements the calibration in C. Hellings *et al.*, “Calibrating
magnetic flux control in superconducting circuits by compensating distortions
on timescales from nanoseconds up to tens of microseconds,” *Physical Review
Research* **7**, 043142 (2025), DOI
[10.1103/1qhb-r4fb](https://doi.org/10.1103/1qhb-r4fb). The authors' figure data
are archived at
[10.3929/ethz-c-000783499](https://doi.org/10.3929/ethz-c-000783499).

The implementation is intentionally split into acquisition, candidate-filter
construction, waveform checks, and hardware application. The first three are
implemented. Hardware application remains blocked until the installed
Quick/QICK stack exposes and verifies an arbitrary Z-waveform uploader.
Candidate JSON files therefore carry `status: candidate_not_applied`.

## What is implemented

| Layer | Implementation | Status |
| --- | --- | --- |
| Long-time acquisition | `17a_flux_step_spectroscopy.py`, authored `FluxStepSpectroscopy` program | Implemented; adaptive or rectangular map |
| Long-time fit | normalized sum of exponentials with automatic BIC order selection | Implemented |
| IIR inverse | final-paper Appendix G roots, matched-z mapping, nearest SOS pairing | Implemented |
| Short-time acquisition | `17c_cryoscope.py`, authored `Cryoscope` program | Interior diagnostic on current fabric; exact-edge data supported when imported |
| Cryoscope analysis | sinusoidal phase fits and adaptive finite differences | Implemented |
| FIR forward model | nonlinear qubit-frequency objective with energy and exponential-tail regularization | Implemented; edge-coverage gated |
| FIR inverse | Gaussian target and derivative/Sobolev regularization | Implemented |
| Candidate export | atomic JSON plus waveform amplitude/slew checks | Implemented |
| Arbitrary waveform upload | guarded `upload_predistorted_waveform(...)` site-adapter interface | No verified site adapter yet |

The final journal paper has `p0,i = -1/tau_i` in Eq. G6. The implementation
uses that negative sign. An earlier text extraction of the preprint can make
the minus sign easy to miss; using the opposite sign creates an unstable
filter.

## Hardware gap that must not be hidden

The paper used a 2.4 GSa/s AWG (`Ts = 0.4167 ns`) and Gaussian flux-pulse edges
with `sigma = 0.5 ns`. The configured generator in this repository has a
599.04 MHz fabric clock, so Mercator pulse timing is approximately 1.669 ns.
Its verified path cannot represent the paper's edge exactly. The safe
cryoscope defaults are therefore `Ts = 1.669 ns` and `sigma = 2 ns`; attempting
the exact 0.5 ns envelope fails preflight instead of being silently rounded.
Because the complete flat-top ramps also impose a minimum duration, the
default 20–95 ns center-time sweep is an **interior diagnostic** and does not
cover both edges within the paper's +/-2.4 ns window. `17d` refuses to export a
short-time inverse from that partial coverage unless
`ALLOW_PARTIAL_EDGE_FIT=True`; such an override produces a diagnostic candidate,
not a replicated FIR calibration. Full edge coverage requires the verified
faster arbitrary-waveform path or imported measurements from an equivalent AWG.
At 1.669 ns, phase sampling aliases detunings above about 299.5 MHz, well below
the paper's roughly 738 MHz excursion. `17c` blocks that live combination by
default. The preferred independent test is a smaller, demonstrably linear
flux step below the phase Nyquist limit. `ALLOW_GUIDED_PHASE_UNWRAP` enables a
model-prior cycle assignment for the full step, but that result is explicitly
prior-guided and must be cross-checked at the smaller amplitude.

The filter-design code can still target 0.4167 ns for export. Before applying
that waveform, a site adapter must:

1. upload arbitrary samples on the physical Z generator;
2. demonstrate the requested sample interval on a scope;
3. define waveform start, latency, gain, clipping, and generator-reset behavior;
4. demonstrate that the last sample and any integrator state are reset or
   intentionally preserved between experiments; and
5. set `supports_arbitrary_waveforms = True` and implement
   `upload_z_waveform(...)` for the guarded callback.

Do not insert SOS or FIR coefficients into an unknown firmware API by guessing
its coefficient order or state convention.

`17d` never resamples a measured cryoscope trace onto the paper's faster grid:
it reads the persisted schedule and designs at that actual sample interval. A
numeric `FILTER_SAMPLE_INTERVAL_NS` is accepted only when it matches the
schedule. At 1.669 ns, the default 50 ns forward support becomes 30 taps; at
0.4167 ns it becomes the paper's 120 taps.

## Prerequisites and bench gates

Complete these before the first distortion measurement:

1. Run `00_connect_and_ports.py`; verify that `z` is the intended physical DAC.
2. Run `05c`/`05d` so readout frequency can be evaluated at the return bias.
3. Run `06b`/`06e`; accept `records.lookups.qubit_vs_flux` only after the
   monotonic branch used by the pulse is well measured. The inverse
   frequency-to-flux conversion rejects a branch that crosses a sweet spot.
4. Verify Z polarity with a small spectroscopy step. The paper's 0.217 Phi0 is
   a magnitude, **not a raw DAC gain**. By default the launchers convert the
   paper's fractions through the accepted `period_z` and `sweet_spot_z`, moving
   from about `-0.127 Phi0` toward `-0.344 Phi0`. Change the fraction's sign if
   the site's convention is opposite. The example presets use a small local
   gain instead of embedding the paper's `0.217` as a hardware command.
   The paper's `-0.127 Phi0` baseline is supplied on a separate DC path while
   the fast line returns to zero. `17a` therefore requires the operator to set
   and read back the external bias and explicitly release
   `EXTERNAL_BASELINE_CONFIRMED`; the launcher does not silently command an
   unconfigured external instrument. If the DC and fast paths do not share the
   fitted Z coordinate, cross-calibrate them and set both local-unit overrides;
   never set only one. If this installation uses a single direct-coupled line,
   change the program to command total baseline and target levels and revalidate
   its state/parking behavior.
5. Scope the unfiltered Z pulse into a representative 50-ohm load. Record full
   scale, offset, 20–80% edge time, overshoot, and the relationship between
   requested gain and voltage.
6. Start at 10–25% of the intended step. Confirm linear frequency response and
   no unexplained qubit loss before increasing amplitude.
7. Use a trigger period at least 15 times the slow high-pass time constant for
   the uncompensated first pass. For the paper's 19.2 us bias tee this is about
   288 us; the launcher defaults to 300 us.

The paper reports waveforms up to 0.6 V on a +/-2.5 V AWG (24% of full scale)
and checks that the 20–80% rise exceeds its 0.8 ns slew specification. Those
are reference values, not authorization to use 0.6 V on another cryostat.

## Replication campaign

### 0. Flux-frequency model

Acquire a dense qubit-frequency-versus-Z map over one period, fit it with
`06e_fit_qubit_vs_flux.py`, and inspect residuals on the branch used for the
step. The paper additionally subtracts a 1.1 MHz offset between Ramsey and
pulsed-spectroscopy estimates; `17b` exposes this as
`SPECTROSCOPY_OFFSET_MHZ`. Remeasure it on this device rather than assuming the
paper's value.

Acceptance gate:

- the selected pulse branch is strictly monotonic;
- every frequency window lies within the accepted lookup domain;
- interpolation residuals are materially below the residual distortion being
  optimized (preferably below 0.1–0.2 MHz for the final pass).

### 1. Uncompensated long-time step response

Run `17a_flux_step_spectroscopy.py`.

Paper-faithful settings:

- baseline approximately `-0.127 Phi0`;
- step magnitude `0.217 Phi0`;
- 70 logarithmic observation times from 10 ns to 100 us;
- a 40 ns Gaussian qubit probe (`sigma = 10 ns`, truncated at `4 sigma`);
- a +/-100 MHz spectrum around the predicted transition at each time;
- 2048 averages;
- step truncation near `t+110 ns` and readout near `t+220 ns`; and
- trigger period near 300 us for a 19.2 us bias-tee time constant.

Adaptive-row mode is the default. It measures only the +/-100 MHz window at
each predicted center. A single rectangular Quick map must span the entire
approximately 738 MHz qubit excursion, which is several times slower at the
same spectral resolution. For 70 times, 201 frequencies, 2048 averages, and a
300 us trigger, the ideal lower bound is about 2.4 hours before software and
readout overhead. A 1001-bin rectangular map is about five times longer.

The first prediction only places windows. If a resonance approaches a window
edge, expand the offsets or update `PREDICTION_ALPHAS/TAUS` and repeat that row.
Do not fit an edge-pinned line.
The launcher writes one campaign manifest containing the exact native CSV path
for every row together with the resolved local baseline and step coordinates.
`17b` consumes that manifest before considering recent-file discovery,
preventing rows from two campaigns from being mixed silently or reinterpreted
after a default changes.

For the first uncompensated pass, Appendix F requires a time-dependent fast-line
level during readout,
`a_during_RO = a_step * (1 - exp(-(t + 110 ns)/20 us))`. The authored program
waits about 70 ns after the 40 ns probe, applies that per-row level, starts
readout about 110 ns later, and returns the fast line to zero after readout so
the bias tee can discharge for the 300 us repetition interval. Adaptive mode
computes the scalar separately for every row. A rectangular native map cannot
express this correlated level and is therefore blocked on the uncompensated
pass. After the dominant IIR is active, set `MODEL_READOUT_RETURN=False`, as in
the paper. Verify all four timing landmarks on a scope because Mercator delay
semantics and pulse-center conventions are installation-specific.

Acquire the long-time rows in increasing-time order, as Appendix F does. The
paper also interleaves reference calibration points between the largest and
smallest observation times. That state-population reference interleave is not
automated by `17a`; add it through the lab's normal readout-calibration path if
you use population rather than the fitted complex-IQ ridge center, and monitor
the first and last rows for readout drift in either case.

### 2. First long-time IIR

Run `17b_fit_flux_iir.py` with the adaptive CSV paths or the rectangular map.
The analysis:

1. fits complex Gaussian and Lorentzian candidates with a linear background,
   selecting the lower-residual equal-complexity model for each row;
2. subtracts the configured spectroscopy offset;
3. converts frequency to flux on the explicit monotonic branch;
4. normalizes by the largest measured flux excursion, following Eq. G1
   (`NORMALIZATION="commanded_step"` is available as an absolute-scale QA);
5. fits one through six exponentials and selects order by BIC; and
6. constructs the normalized matched-z inverse in second-order sections.

The paper manually used four exponentials and excluded times below about
25 ns from this fit. BIC selection is the general default here: it removes a
manual model-order choice and reduces noise-fitting when another line has
fewer physical time constants.

The exponential amplitudes are fitted freely. As in Eq. G6, the inverse is
normalized after fitting so its common high-frequency scale is one; the fit is
not biased by forcing the amplitudes to sum exactly to one.

With zero fixed DC gain, the exact inverse has an integrator pole at `z=1`.
This is faithful to the high-pass correction but retains pulse-history state.
Use only net-zero sequences or an explicit state-reset policy. The optional
`LEAK_TAU_US` moves this pole inside the unit circle for a bounded leaky
inverse; treat that as a different method and remeasure its long-time bias.

Inspect `analysis_cache/flux_compensation/iir_candidate.json`. It is not an
accepted calibration and has not been applied to hardware.

### 3. Second long-time iteration

After the candidate is loaded through a verified site uploader, rerun `17a`
with the filter enabled and `MODEL_READOUT_RETURN=False`. In `17b`, set
`FIT_DC_GAIN=1.0`, refit the residual, and cascade that inverse after the first
candidate. The paper used two IIR iterations: the first left errors below
roughly 2 MHz over most of the fitted interval and the second reduced the
remaining long-time error to about 0.5 MHz.

Keep the uncompensated, IIR-1, and IIR-2 datasets separate. A filter must be
traceable to the exact source data and prior filter state used to measure it.

### 4. Cryoscope and first short-time FIR

Run `17c_cryoscope.py` after the two IIR passes are active. The program applies
a Ramsey pi/2 pulse, a smoothed rectangular Z pulse, and a second pi/2 pulse
with swept phase. The persisted schedule records the exact `t-dt/2` and
`t+dt/2` endpoints, sample interval, local baseline/step coordinates, pulse
duration, edge sigma, and whether guided unwrapping was authorized. `17d`
prefers those acquisition coordinates over current defaults.

Before the full run, use a small step with a known spectroscopy shift to verify
the sign of accumulated phase. Set `PHASE_SIGN` in `17d` so the recovered
detuning has that known sign; an apparently well-fit sinusoid does not by itself
resolve the laboratory phase convention.

The paper used:

- a 100 ns rectangular flux pulse;
- 16 Ramsey phases;
- 65,536 repetitions per phase;
- `dt = Ts` within 2.4 ns of an edge, `3 Ts` from 2.4 to 30 ns, and `8 Ts` in
  the low-distortion interior; and
- operation at the upper sweet spot to reduce turn-off systematic error.

Run `17d_fit_flux_fir.py`. It first fits each phase sweep by linear sinusoidal
regression, unwraps accumulated phase, and evaluates

`delta_f(t) = [theta(t+dt/2) - theta(t-dt/2)] / (2*pi*dt)`.

It then fits a causal forward FIR through the nonlinear accepted qubit model,
using the integration interval as the statistical weight, an energy penalty,
and an exponentially increasing tail penalty. The inverse FIR is a separate
linear regularized solve with a 0.75 ns Gaussian target and a derivative
penalty. Automatic causal-latency selection replaces another manual tuning
step. Phase rows below the configured sinusoidal-fit quality or above the phase
uncertainty limit block the fit. The candidate is written to
`analysis_cache/flux_compensation/fir_candidate.json`.

### 5. Second short-time iteration

Apply the first FIR after both IIR filters, reacquire the cryoscope, and fit an
inverse to the residual. The paper used two FIR iterations because the first
measurement has larger edge systematics and the second provides a cleaner
linear residual. Preserve the order `IIR-1 -> IIR-2 -> FIR-1 -> FIR-2` unless
the site waveform engine has demonstrated an equivalent combined convolution.

### 6. Final verification

Rerun both measurements without changing filter coefficients. Evaluate at
least these gates:

- long-time absolute error and drift through 100 us;
- after 15 ns, maximum frequency error divided by full excursion below 0.1%;
- no unstable/marginal state growth over the longest intended sequence;
- no clipping and no scope-measured slew violation;
- the result repeats after a generator reset; and
- the result repeats for both pulse polarities if both are used in gates.

The paper's final verification used an approximately 738 MHz excursion. Its
0.1% criterion is therefore about 0.74 MHz. It observed a roughly 3 MHz
overshoot near 10 ns, before the stated 15 ns settling point. Use
`settling_metrics(...)` with this device's measured excursion rather than
copying 0.74 MHz.

The reported CZ error and leakage are device-specific follow-on results, not
acceptance criteria for this calibration. Only begin two-qubit gate validation
after single-line waveform verification passes.

## Generalizable holdout experiments

These are additional to the paper's minimum replication and should be run
before declaring the filter reusable across pulse shapes.

### Amplitude linearity

Repeat `17a` at signed amplitudes approximately
`[-1.0, -0.5, +0.5, +1.0]` times the intended magnitude, staying inside the
accepted flux branch and voltage limit. Normalize every measured response by
its command. A reusable LTI filter should give overlapping normalized traces;
plot the spread versus time and hold one amplitude out of filter fitting.

Fail if a resonance leaves its adaptive window, the frequency-to-flux branch
becomes ambiguous, or normalized residuals grow systematically with amplitude.
In that case use amplitude-binned filters or a nonlinear control model rather
than one global linear filter.

### Pulse-shape holdouts

Apply the frozen filter to shapes not used for fitting:

- a positive and negative step;
- a Gaussian and a flat-top pulse;
- the intended net-zero CZ envelope;
- two pulses separated by 20 ns, 1 us, and 50 us; and
- a bounded pseudorandom binary or multisine sequence.

Measure the output on a room-temperature scope first. On the qubit, use the
cryoscope or phase accumulation appropriate to the shape. Do not refit on the
holdouts. Report RMS error, peak error, settling time, and any history-dependent
offset. The paired-pulse tests reveal IIR state and bias-tee memory that a
single isolated step cannot.

### Time invariance and drift

Repeat a compact sentinel set at the beginning and end of a lab day and after
one week: three long-step times, both edges of the cryoscope, and one holdout
pulse. Version a new filter only when the change exceeds measurement
uncertainty and the same hardware configuration is still installed.

## Efficiency variants

The following methods are implemented without changing the scientific target:

- **Adaptive spectroscopy windows:** approximately fivefold fewer frequency
  points than a full rectangular map for a 1 GHz excursion at 1 MHz spacing.
- **BIC exponential order:** removes manual selection of four time constants.
- **Four-phase cryoscope iterations:** phases at 0, 90, 180, and 270 degrees
  are information-complete for the sinusoidal model. Use 16 phases on the
  first and final pass to diagnose non-sinusoidal systematics.
- **Shot budgeting:** `recommended_shots_per_phase(...)` computes the required
  per-point allocation and `17c` reports its maximum. The current rectangular
  Quick map still uses one scalar `HARD_AVG`; grouped per-duration execution is
  not automated. A uniform 65,536 shots remains the faithful reference.
- **Active reset:** the paper estimates roughly an order-of-magnitude time
  reduction. No active-reset sequence is implemented in this repository, so
  this remains a hardware-programming task rather than a launcher switch.
- **Leaky IIR:** bounded alternative for long random sequences. It trades exact
  DC inversion for state stability and requires separate verification.
- **Regularized direct FIR inverse:** `design_inverse_fir(...)` can invert any
  measured causal forward impulse response, not only a qubit-derived one. This
  is useful when a high-bandwidth scope/VNA measurement is representative of
  the cold line; validate the result on the qubit as a holdout.

Pulse-specific optimal control is another viable method when only one gate
shape matters. It can absorb line distortion into a gate waveform, but it does
not replace a pulse-shape-independent line calibration and must be repeated for
each gate family. Use the holdout suite to decide whether the reusable LTI
model is adequate before escalating to nonlinear or pulse-specific control.

## Candidate waveform use

The reusable API is in `quickexp_v3/flux_compensation.py`:

```python
predistorted = apply_filter_bundles(
    command,
    ["iir_candidate.json", "fir_candidate.json"],
)
# Or use in-memory designs:
predistorted = apply_predistortion(command, iir=iir, fir=inverse_fir)
check = validate_waveform(
    predistorted,
    sample_interval_ns=0.4167,
    full_scale=2.5,
    maximum_fraction_of_full_scale=0.24,
)
assert check.passes

# Dry-run is the default. This raises until a verified site adapter exists.
manifest = upload_predistorted_waveform(
    uploader,
    predistorted,
    channel=z_channel,
    sample_interval_ns=0.4167,
    name="q0_flux_comp_v1",
)
```

Passing numerical checks does not replace the scope check. Preserve the raw
command, predistorted waveform, filter bundle, uploader/firmware version, scope
trace, qubit verification dataset, and acceptance decision together.
