"""Fitting routines for the lab-readout calibration chain.

Every fitter consumes the same raw arrays the result CSVs hold
(``freq, amplitude, phase, I, Q``) and returns a small dataclass carrying the
fitted ``params``, their ``uncertainties``, a goodness-of-fit (``gof``,
r-squared), and a ``plot`` method that overlays the fit on the data.

These routines are pure NumPy/SciPy and contain no hardware calls, so they are
unit-tested offline against synthetic data and the existing result CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, least_squares


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _r_squared(y: np.ndarray, y_model: np.ndarray) -> float:
    """Coefficient of determination of a model against data."""
    y = np.asarray(y, float)
    ss_res = float(np.sum((y - y_model) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _apply_window(
    freq: np.ndarray, window: Optional[Tuple[float, float]], *arrays
):
    """Restrict freq + companion arrays to ``window`` (inclusive)."""
    freq = np.asarray(freq, float)
    if window is None:
        return (freq, *[np.asarray(a, float) for a in arrays])
    lo, hi = window
    mask = (freq >= lo) & (freq <= hi)
    return (freq[mask], *[np.asarray(a, float)[mask] for a in arrays])


def _perr(pcov: np.ndarray) -> np.ndarray:
    """1-sigma parameter errors from a covariance matrix (NaN-safe)."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.diag(pcov))


def _smooth(y: np.ndarray) -> np.ndarray:
    """Light Savitzky-Golay smoothing for robust initial-guess estimation."""
    from scipy.signal import savgol_filter

    y = np.asarray(y, float)
    n = len(y)
    if n < 7:
        return y
    win = min(31, n if n % 2 else n - 1)
    if win % 2 == 0:
        win -= 1
    return savgol_filter(y, win, 3)


# --------------------------------------------------------------------------- #
# Resonator: complex notch response
# --------------------------------------------------------------------------- #
def _complex_notch(f, amp, alpha, tau, f0, ql, qc, phi, f_ref=None):
    """Complex hanger/notch response; MHz and microseconds are paired units."""
    f = np.asarray(f, float)
    if f_ref is None:
        f_ref = float(np.mean(f))
    baseline = amp * np.exp(1j * alpha) * np.exp(-2j * np.pi * (f - f_ref) * tau)
    detuning = (f - f0) / f0
    notch = 1.0 - (ql / qc) * np.exp(1j * phi) / (1.0 + 2j * ql * detuning)
    return baseline * notch


def _complex_r_squared(z: np.ndarray, model: np.ndarray) -> float:
    """R-squared using the joint squared I/Q residual."""
    z = np.asarray(z, complex)
    ss_res = float(np.sum(np.abs(z - model) ** 2))
    ss_tot = float(np.sum(np.abs(z - np.mean(z)) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _resonator_initial_guess(freq: np.ndarray, z: np.ndarray):
    """Estimate baseline delay/gain and resonance parameters for complex fit."""
    n = len(freq)
    edge_n = max(3, n // 5)
    edge = np.r_[0:edge_n, n - edge_n:n]
    phase = np.unwrap(np.angle(z))
    slope, intercept = np.polyfit(freq[edge] - np.mean(freq), phase[edge], 1)
    tau = -slope / (2.0 * np.pi)
    derotated = z * np.exp(2j * np.pi * (freq - np.mean(freq)) * tau)
    baseline = np.mean(derotated[edge])
    amp = max(float(np.abs(baseline)), 1e-12)
    alpha = float(np.angle(baseline))

    mag = np.abs(derotated)
    f0 = float(freq[int(np.argmin(_smooth(mag)))])
    off = max(float(np.median(mag[edge])), 1e-12)
    floor = float(np.min(mag))
    half = floor + (off - floor) / 2.0
    smoothed = _smooth(mag)
    center = int(np.argmin(smoothed))
    fstep = float(np.median(np.diff(freq)))
    left = center
    right = center
    while left > 0 and smoothed[left - 1] <= half:
        left -= 1
    while right < n - 1 and smoothed[right + 1] <= half:
        right += 1
    if right > left:
        fwhm = max(float(freq[right] - freq[left]), fstep)
    else:
        fwhm = max(float(np.ptp(freq)) / 20.0, fstep)
    ql = max(f0 / fwhm, 10.0)
    coupling = np.clip(1.0 - floor / off, 1e-3, 0.999)
    qc = max(ql / coupling, 10.0)
    return amp, alpha, tau, f0, ql, qc, 0.0


@dataclass
class ResonatorFit:
    f0: float
    q: float
    depth_db: float
    hwhm: float
    params: dict
    uncertainties: dict
    gof: float
    valid: bool
    freq: np.ndarray = field(repr=False)
    iq: np.ndarray = field(repr=False)

    @property
    def mag(self) -> np.ndarray:
        return np.abs(self.iq)

    def model(self, freq=None) -> np.ndarray:
        if freq is None:
            freq = self.freq
        p = self.params
        return _complex_notch(
            freq, p["amp"], p["alpha"], p["tau"], p["f0"],
            p["ql"], p["qc"], p["phi"], p["f_ref"],
        )

    def suggestion(self) -> str:
        if not self.valid:
            return (
                f"complex resonator fit rejected (R²={self.gof:.3f}); "
                "select a window containing one isolated resonance and rescan"
            )
        span = max(8 * self.hwhm, 0.5)
        return (
            f"resonance at {self.f0:.4f} MHz (Q≈{self.q:.0f}, "
            f"depth {self.depth_db:.1f} dB) → suggest fine scan "
            f"{self.f0 - span / 2:.4f}–{self.f0 + span / 2:.4f} MHz"
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, (ax, ax_iq) = plt.subplots(1, 2, figsize=(10, 4))
        else:
            ax_iq = None
        ax.plot(self.freq, self.mag, ".", ms=3, label="data")
        ax.plot(self.freq, np.abs(self.model()), "-", lw=1.5, label="complex fit")
        ax.axvline(self.f0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("|S21| (linear)")
        ax.legend()
        if ax_iq is not None:
            model = self.model()
            ax_iq.plot(self.iq.real, self.iq.imag, ".", ms=3, label="data")
            ax_iq.plot(model.real, model.imag, "-", lw=1.5, label="complex fit")
            ax_iq.set_xlabel("I")
            ax_iq.set_ylabel("Q")
            ax_iq.set_aspect("equal", adjustable="datalim")
            ax_iq.legend()
        return ax


def fit_resonator(freq, I, Q, window=None) -> ResonatorFit:
    """Fit the full complex hanger response to I+iQ.

    The model includes complex gain, cable delay, loaded/coupling quality
    factors, and impedance-mismatch phase. Both I and Q residuals participate
    in the optimization; magnitude is used only to construct initial guesses.
    """
    freq, I, Q = _apply_window(freq, window, I, Q)
    if len(freq) < 8 or not np.all(np.isfinite(np.r_[freq, I, Q])):
        raise ValueError("complex resonator fit requires at least 8 finite IQ samples")
    order = np.argsort(freq)
    freq, I, Q = freq[order], I[order], Q[order]
    z = I + 1j * Q
    # A coarse sweep often contains multiple modes and standing-wave ripple.
    # When no explicit window is supplied, isolate roughly ten linewidths
    # around the deepest contiguous notch before fitting a one-pole model.
    if window is None:
        rough = _resonator_initial_guess(freq, z)
        rough_f0, rough_ql = rough[3], rough[4]
        rough_fwhm = rough_f0 / rough_ql
        fstep = max(float(np.median(np.diff(freq))), np.finfo(float).eps)
        half_span = max(5.0 * rough_fwhm, 20.0 * fstep)
        if 2.0 * half_span < 0.8 * np.ptp(freq):
            local = np.abs(freq - rough_f0) <= half_span
            if np.count_nonzero(local) >= 20:
                freq, z = freq[local], z[local]
                I, Q = z.real, z.imag
    f_ref = float(np.mean(freq))
    amp0, alpha0, tau0, f0_0, ql0, qc0, phi0 = _resonator_initial_guess(freq, z)
    x0 = np.array([np.log(amp0), alpha0, tau0, f0_0,
                   np.log(ql0), np.log(qc0), phi0])
    fspan = float(np.ptp(freq))
    fstep = max(float(np.median(np.diff(freq))), np.finfo(float).eps)
    max_q = max(float(freq.max()) / (2.0 * fstep), 100.0)
    lower = [np.log(amp0) - 5, alpha0 - 2*np.pi, tau0 - 5,
             float(freq.min()), np.log(10.0), np.log(10.0), -np.pi]
    upper = [np.log(amp0) + 5, alpha0 + 2*np.pi, tau0 + 5,
             float(freq.max()), np.log(max_q), np.log(1e10), np.pi]
    scale = max(float(np.median(np.abs(z))), 1e-12)

    def unpack(x):
        return np.exp(x[0]), x[1], x[2], x[3], np.exp(x[4]), np.exp(x[5]), x[6]

    def residual(x):
        delta = (_complex_notch(freq, *unpack(x), f_ref=f_ref) - z) / scale
        return np.r_[delta.real, delta.imag]

    result = least_squares(
        residual, x0, bounds=(lower, upper), x_scale="jac",
        loss="soft_l1", max_nfev=50000,
    )
    popt = unpack(result.x)
    amp, alpha, tau, f0, ql, qc, phi = popt
    model = _complex_notch(freq, *popt, f_ref=f_ref)
    hwhm = f0 / (2.0 * ql)
    center_mag = abs(_complex_notch(np.array([f0]), *popt, f_ref=f_ref)[0])
    depth_db = 20.0 * np.log10(amp / max(center_mag, 1e-15))

    names = ["amp", "alpha", "tau", "f0", "ql", "qc", "phi"]
    uncertainties = dict.fromkeys(names, float("nan"))
    if result.jac.shape[0] > result.jac.shape[1]:
        try:
            covariance = np.linalg.inv(result.jac.T @ result.jac)
            variance = 2.0 * result.cost / (result.jac.shape[0] - result.jac.shape[1])
            xerr = np.sqrt(np.diag(covariance) * variance)
            transformed = xerr.copy()
            transformed[0] *= amp
            transformed[4] *= ql
            transformed[5] *= qc
            uncertainties = dict(zip(names, map(float, transformed)))
        except np.linalg.LinAlgError:
            pass
    params = dict(zip(names, map(float, popt)))
    params["f_ref"] = f_ref
    uncertainties["f_ref"] = 0.0
    gof = _complex_r_squared(z, model)
    valid = bool(result.success and gof >= 0.5 and 10.0 < ql < 0.98 * max_q)
    return ResonatorFit(
        f0=float(f0),
        q=float(ql),
        depth_db=float(depth_db),
        hwhm=float(hwhm),
        params=params,
        uncertainties=uncertainties,
        gof=gof,
        valid=valid,
        freq=freq,
        iq=z,
    )


# --------------------------------------------------------------------------- #
# Qubit spectroscopy: signed Lorentzian on the most-varying IQ quadrature
# --------------------------------------------------------------------------- #
def _lorentzian_peak(f, offset, amp, f0, hwhm):
    return offset + amp * hwhm**2 / ((f - f0) ** 2 + hwhm**2)


def rotated_iq(I, Q):
    """Project demeaned (I, Q) onto the direction of greatest variation."""
    I = np.asarray(I, float)
    Q = np.asarray(Q, float)
    pts = np.column_stack([I - I.mean(), Q - Q.mean()])
    _u, _s, vh = np.linalg.svd(pts, full_matrices=False)
    return pts @ vh[0]


@dataclass
class QubitFit:
    f0: float
    linewidth: float
    params: dict
    uncertainties: dict
    gof: float
    freq: np.ndarray = field(repr=False)
    signal: np.ndarray = field(repr=False)

    def suggestion(self) -> str:
        span = max(8 * self.linewidth, 2.0)
        return (
            f"qubit feature at {self.f0:.3f} MHz "
            f"(linewidth {self.linewidth:.2f} MHz) → suggest fine scan "
            f"{self.f0 - span / 2:.3f}–{self.f0 + span / 2:.3f} MHz"
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()
        ax.plot(self.freq, self.signal, ".", ms=3, label="rotated IQ")
        model = _lorentzian_peak(
            self.freq,
            self.params["offset"],
            self.params["amp"],
            self.params["f0"],
            self.params["hwhm"],
        )
        ax.plot(self.freq, model, "-", lw=1.5, label="fit")
        ax.axvline(self.f0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Qubit-drive frequency (MHz)")
        ax.set_ylabel("Rotated IQ")
        ax.legend()
        return ax


def fit_qubit_peak(freq, I, Q, window=None) -> QubitFit:
    """Fit a signed Lorentzian (peak or dip) to the rotated-IQ qubit signal."""
    freq, I, Q = _apply_window(freq, window, I, Q)
    signal = rotated_iq(I, Q)

    fspan = float(freq.max() - freq.min())
    fstep = fspan / max(len(freq) - 1, 1)
    offset0 = float(np.median(signal))
    sm = _smooth(signal)
    # Choose peak vs dip by whichever excursion from baseline is larger.
    up = float(np.max(sm) - offset0)
    down = float(offset0 - np.min(sm))
    if up >= down:
        f0_0 = float(freq[int(np.argmax(sm))])
        amp0 = max(up, 1e-9)
    else:
        f0_0 = float(freq[int(np.argmin(sm))])
        amp0 = -max(down, 1e-9)
    hwhm0 = max(fspan / 20.0, fstep)
    p0 = [offset0, amp0, f0_0, hwhm0]
    bounds = (
        [-np.inf, -np.inf, float(freq.min()), fstep],
        [np.inf, np.inf, float(freq.max()), fspan],
    )

    try:
        popt, pcov = curve_fit(
            _lorentzian_peak, freq, signal, p0=p0, bounds=bounds, maxfev=20000
        )
    except (RuntimeError, ValueError):
        popt, pcov = np.array(p0), np.full((4, 4), np.nan)

    offset, amp, f0, hwhm = popt
    hwhm = abs(hwhm)
    names = ["offset", "amp", "f0", "hwhm"]
    errs = _perr(pcov)
    return QubitFit(
        f0=float(f0),
        linewidth=float(2.0 * hwhm),
        params=dict(zip(names, map(float, popt))),
        uncertainties=dict(zip(names, map(float, errs))),
        gof=_r_squared(signal, _lorentzian_peak(freq, *popt)),
        freq=freq,
        signal=signal,
    )


# --------------------------------------------------------------------------- #
# Rabi: exponentially-decaying cosine -> pi-pulse value
# --------------------------------------------------------------------------- #
def _decaying_cosine(x, offset, amp, period, tau, phi):
    return offset + amp * np.exp(-x / tau) * np.cos(2 * np.pi * x / period + phi)


@dataclass
class RabiFit:
    period: float
    pi_value: float
    tau: float
    params: dict
    uncertainties: dict
    gof: float
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    xlabel: str = "Drive (gain or length)"

    def suggestion(self) -> str:
        return (
            f"pi-pulse at {self.pi_value:.4g} "
            f"(Rabi period {self.period:.4g}, decay {self.tau:.3g}) → "
            f"set the pi value and proceed to coherence measurements"
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()
        ax.plot(self.x, self.y, ".", ms=3, label="data")
        ax.plot(self.x, _decaying_cosine(self.x, *self.params.values()), "-",
                lw=1.5, label="fit")
        ax.axvline(self.pi_value, color="k", ls="--", lw=0.8, label="pi")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("Signal")
        ax.legend()
        return ax


def _dominant_period(x, y):
    """Estimate the oscillation period from the FFT of the detrended signal."""
    y = np.asarray(y, float) - np.mean(y)
    x = np.asarray(x, float)
    if len(x) < 4:
        return (x.max() - x.min()) or 1.0
    dx = np.mean(np.diff(x))
    freqs = np.fft.rfftfreq(len(y), d=dx)
    spec = np.abs(np.fft.rfft(y))
    spec[0] = 0.0  # ignore DC
    fpeak = freqs[int(np.argmax(spec))]
    return 1.0 / fpeak if fpeak > 0 else (x.max() - x.min())


def fit_rabi(x, y, xlabel="Drive (gain or length)") -> RabiFit:
    """Fit a decaying cosine to a Rabi sweep and report the pi-pulse value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    offset0 = float(np.mean(y))
    amp0 = float((np.max(y) - np.min(y)) / 2.0) or 1e-9
    period0 = _dominant_period(x, y)
    tau0 = float(x.max() - x.min()) or 1.0
    # Sign of the first sample relative to baseline sets the starting phase.
    phi0 = 0.0 if (y[0] - offset0) >= 0 else np.pi
    p0 = [offset0, amp0, period0, tau0, phi0]

    try:
        popt, pcov = curve_fit(_decaying_cosine, x, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        popt, pcov = np.array(p0), np.full((5, 5), np.nan)

    offset, amp, period, tau, phi = popt
    period = abs(period)
    names = ["offset", "amp", "period", "tau", "phi"]
    errs = _perr(pcov)
    return RabiFit(
        period=float(period),
        pi_value=float(period / 2.0),
        tau=float(abs(tau)),
        params=dict(zip(names, map(float, popt))),
        uncertainties=dict(zip(names, map(float, errs))),
        gof=_r_squared(y, _decaying_cosine(x, *popt)),
        x=x,
        y=y,
        xlabel=xlabel,
    )


# --------------------------------------------------------------------------- #
# IQ single-shot readout: separation axis, threshold, fidelity
# --------------------------------------------------------------------------- #
@dataclass
class IQThresholdFit:
    threshold: float
    fidelity: float
    angle: float
    center_g: float
    center_e: float
    proj_g: np.ndarray = field(repr=False)
    proj_e: np.ndarray = field(repr=False)

    def suggestion(self) -> str:
        return (
            f"readout fidelity {self.fidelity:.4f}, threshold "
            f"{self.threshold:.4g} on the IQ axis at {np.degrees(self.angle):.1f}° → "
            f"set r_threshold; raise readout power/avg if fidelity is low"
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()
        bins = np.linspace(
            min(self.proj_g.min(), self.proj_e.min()),
            max(self.proj_g.max(), self.proj_e.max()),
            80,
        )
        ax.hist(self.proj_g, bins=bins, alpha=0.6, label="|g>")
        ax.hist(self.proj_e, bins=bins, alpha=0.6, label="|e>")
        ax.axvline(self.threshold, color="k", ls="--", lw=1.0, label="threshold")
        ax.set_xlabel("Projection on separation axis")
        ax.set_ylabel("Counts")
        ax.legend()
        return ax


def fit_iq_threshold(Ig, Qg, Ie, Qe) -> IQThresholdFit:
    """Project two IQ clouds onto their separation axis and threshold them."""
    g = np.column_stack([np.asarray(Ig, float), np.asarray(Qg, float)])
    e = np.column_stack([np.asarray(Ie, float), np.asarray(Qe, float)])

    direction = e.mean(axis=0) - g.mean(axis=0)
    norm = np.hypot(*direction)
    if norm == 0:
        direction = np.array([1.0, 0.0])
        norm = 1.0
    axis = direction / norm
    angle = float(np.arctan2(axis[1], axis[0]))

    proj_g = g @ axis
    proj_e = e @ axis
    center_g = float(proj_g.mean())
    center_e = float(proj_e.mean())

    # Threshold that maximizes assignment fidelity, searched over a fine grid.
    lo = min(proj_g.min(), proj_e.min())
    hi = max(proj_g.max(), proj_e.max())
    grid = np.linspace(lo, hi, 512)
    hi_is_excited = center_e >= center_g
    best_thr, best_fid = grid[0], -1.0
    for thr in grid:
        if hi_is_excited:
            correct = np.mean(proj_g < thr) + np.mean(proj_e >= thr)
        else:
            correct = np.mean(proj_g >= thr) + np.mean(proj_e < thr)
        fid = correct / 2.0
        if fid > best_fid:
            best_fid, best_thr = fid, thr

    return IQThresholdFit(
        threshold=float(best_thr),
        fidelity=float(best_fid),
        angle=angle,
        center_g=center_g,
        center_e=center_e,
        proj_g=proj_g,
        proj_e=proj_e,
    )


# --------------------------------------------------------------------------- #
# Coherence: T1 exponential decay and T2 decaying cosine
# --------------------------------------------------------------------------- #
def _exp_decay(t, offset, amp, t1):
    return offset + amp * np.exp(-t / t1)


@dataclass
class T1Fit:
    t1: float
    params: dict
    uncertainties: dict
    gof: float
    t: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)

    def suggestion(self) -> str:
        return f"T1 = {self.t1:.3g} us → set relaxation wait >~ 5*T1 ({5*self.t1:.3g} us)"

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()
        ax.plot(self.t, self.y, ".", ms=3, label="data")
        ax.plot(self.t, _exp_decay(self.t, *self.params.values()), "-",
                lw=1.5, label="fit")
        ax.set_xlabel("Delay (us)")
        ax.set_ylabel("Signal")
        ax.legend()
        return ax


def fit_t1(t, y) -> T1Fit:
    """Fit an exponential relaxation to a T1 delay sweep."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    offset0 = float(y[-5:].mean())
    amp0 = float(y[0] - offset0) or 1e-9
    t1_0 = float((t.max() - t.min()) / 3.0) or 1.0
    p0 = [offset0, amp0, t1_0]
    try:
        popt, pcov = curve_fit(_exp_decay, t, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        popt, pcov = np.array(p0), np.full((3, 3), np.nan)
    names = ["offset", "amp", "t1"]
    errs = _perr(pcov)
    return T1Fit(
        t1=float(abs(popt[2])),
        params=dict(zip(names, map(float, popt))),
        uncertainties=dict(zip(names, map(float, errs))),
        gof=_r_squared(y, _exp_decay(t, *popt)),
        t=t,
        y=y,
    )


@dataclass
class T2Fit:
    t2: float
    detuning: float
    params: dict
    uncertainties: dict
    gof: float
    t: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)

    def suggestion(self) -> str:
        return (
            f"T2 = {self.t2:.3g} us, detuning {self.detuning*1e3:.2f} kHz → "
            f"adjust qubit frequency by the detuning to null the fringes"
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()
        ax.plot(self.t, self.y, ".", ms=3, label="data")
        ax.plot(self.t, _decaying_cosine(self.t, *self.params.values()), "-",
                lw=1.5, label="fit")
        ax.set_xlabel("Delay (us)")
        ax.set_ylabel("Signal")
        ax.legend()
        return ax


def fit_t2(t, y) -> T2Fit:
    """Fit a decaying cosine to a Ramsey/Echo sweep -> T2 and detuning."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    offset0 = float(np.mean(y))
    amp0 = float((np.max(y) - np.min(y)) / 2.0) or 1e-9
    period0 = _dominant_period(t, y)
    tau0 = float(t.max() - t.min()) or 1.0
    phi0 = 0.0 if (y[0] - offset0) >= 0 else np.pi
    p0 = [offset0, amp0, period0, tau0, phi0]
    try:
        popt, pcov = curve_fit(_decaying_cosine, t, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        popt, pcov = np.array(p0), np.full((5, 5), np.nan)
    offset, amp, period, tau, phi = popt
    period = abs(period)
    names = ["offset", "amp", "period", "tau", "phi"]
    errs = _perr(pcov)
    return T2Fit(
        t2=float(abs(tau)),
        detuning=float(1.0 / period) if period > 0 else float("nan"),
        params=dict(zip(names, map(float, popt))),
        uncertainties=dict(zip(names, map(float, errs))),
        gof=_r_squared(y, _decaying_cosine(t, *popt)),
        t=t,
        y=y,
    )
