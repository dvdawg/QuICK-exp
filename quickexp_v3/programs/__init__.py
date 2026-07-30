"""Authored Mercator programs installed alongside Quick's built-in classes."""

from __future__ import annotations

from ..backend import register_envelope_terms, register_program_variables
from .t1_zpa import PROGRAM as T1_ZPA
from .two_tone_zpa import PROGRAM as TWO_TONE_ZPA


PROGRAMS = (TWO_TONE_ZPA, T1_ZPA)

for _program in PROGRAMS:
    register_program_variables(
        _program.name,
        _program.default_variables().keys(),
    )
    register_envelope_terms(
        _program.name,
        _program.envelope_terms(),
    )


__all__ = ["PROGRAMS", "T1_ZPA", "TWO_TONE_ZPA"]
