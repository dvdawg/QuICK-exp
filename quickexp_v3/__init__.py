"""QuICK-exp v3: explicit experiment modules over a shared safe runtime."""

from .config import ConfigRepository, ResolvedConfig, expand_sweep
from .errors import (
    AcquisitionError,
    AnalysisError,
    ConfigError,
    ExperimentError,
    QuickExpError,
    SafetyError,
)

__all__ = [
    "AcquisitionError",
    "AnalysisError",
    "ConfigError",
    "ConfigRepository",
    "ExperimentError",
    "QuickExpError",
    "ResolvedConfig",
    "SafetyError",
    "expand_sweep",
]

__version__ = "0.5.0"
