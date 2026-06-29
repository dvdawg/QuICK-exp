"""labreadout -- an improved QICK calibration & readout workflow.

A thin, testable layer over the ``quick`` framework that adds:

- ``config``   : hardware setup from ``hardware.yml`` (replaces copy-pasted cells)
- ``state``    : persistent calibration state across kernel restarts
- ``fitting``  : resonator / qubit / Rabi / T1 / T2 / IQ-threshold fits
- ``adaptive`` : coarse-to-fine sweep-window logic
- ``results``  : fit sidecars written *next to* the unchanged raw CSV
- ``steps``    : the fit-and-suggest Session that drives the hardware

Typical lab use::

    import labreadout as lr
    sess = lr.Session.connect("hardware.yml", "calibration.yml")
    res = sess.resonator_spectroscopy(r_freq=np.arange(6580, 6590, 0.01))
    res.plot(); print(res.suggestion)
    res.accept()   # commit the fitted r_freq to calibration.yml
"""

from . import adaptive, config, fitting, results, state, steps
from .steps import Session, StepResult

__all__ = [
    "adaptive",
    "config",
    "fitting",
    "results",
    "state",
    "steps",
    "Session",
    "StepResult",
]

__version__ = "0.1.0"
