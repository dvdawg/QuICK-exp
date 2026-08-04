"""Explicit registry of experiment modules supported by the local Quick API."""

from __future__ import annotations

from typing import Dict

from ..errors import ExperimentError
from .base import Experiment
from .cryoscope import EXPERIMENT as cryoscope
from .dispersive_spectroscopy import EXPERIMENT as dispersive_spectroscopy
from .echo import EXPERIMENT as echo
from .flux_step_spectroscopy import EXPERIMENT as flux_step_spectroscopy
from .iq_blobs import EXPERIMENT as iq_blobs
from .loopback import EXPERIMENT as loopback
from .qubit_spectroscopy import EXPERIMENT as qubit_spectroscopy
from .rabi import EXPERIMENT as rabi
from .rabi_chevron import EXPERIMENT as rabi_chevron
from .ramsey import EXPERIMENT as ramsey
from .ramsey_chevron import EXPERIMENT as ramsey_chevron
from .resonator_spectroscopy import EXPERIMENT as resonator_spectroscopy
from .t1 import EXPERIMENT as t1
from .t1_zpa import EXPERIMENT as t1_zpa
from .two_tone_zpa import EXPERIMENT as two_tone_zpa


EXPERIMENTS: Dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in (
        loopback,
        resonator_spectroscopy,
        qubit_spectroscopy,
        dispersive_spectroscopy,
        rabi,
        rabi_chevron,
        iq_blobs,
        t1,
        ramsey,
        ramsey_chevron,
        echo,
        two_tone_zpa,
        t1_zpa,
        flux_step_spectroscopy,
        cryoscope,
    )
}


def names() -> list:
    return sorted(EXPERIMENTS)


def get(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        raise ExperimentError(
            f"unknown experiment {name!r}; available: {', '.join(names())}"
        ) from error
