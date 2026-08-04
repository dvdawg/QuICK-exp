"""Authored Mercator programs installed alongside Quick's built-in classes."""

from __future__ import annotations

from ..backend import register_envelope_terms, register_program_variables
from .cryoscope import PROGRAM as CRYOSCOPE
from .flux_step_spectroscopy import PROGRAM as FLUX_STEP_SPECTROSCOPY
from .t1_zpa import PROGRAM as T1_ZPA
from .two_tone_zpa import PROGRAM as TWO_TONE_ZPA


PROGRAMS = (TWO_TONE_ZPA, T1_ZPA, FLUX_STEP_SPECTROSCOPY, CRYOSCOPE)

for _program in PROGRAMS:
    register_program_variables(
        _program.name,
        _program.default_variables().keys(),
    )
    register_envelope_terms(
        _program.name,
        _program.envelope_terms(),
    )


__all__ = [
    "CRYOSCOPE",
    "FLUX_STEP_SPECTROSCOPY",
    "PROGRAMS",
    "T1_ZPA",
    "TWO_TONE_ZPA",
]
