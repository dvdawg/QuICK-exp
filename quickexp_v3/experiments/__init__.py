"""One module per experiment type, inspired by the OPX experiment layout."""

from .base import Experiment, ExperimentPlan
from .registry import EXPERIMENTS, get, names

__all__ = ["EXPERIMENTS", "Experiment", "ExperimentPlan", "get", "names"]
