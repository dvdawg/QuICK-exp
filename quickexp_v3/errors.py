"""Public exception hierarchy for QuICK-exp v3."""


class QuickExpError(Exception):
    """Base class for errors raised by this package."""


class ConfigError(QuickExpError):
    """Configuration is missing, inconsistent, or unsafe."""


class ExperimentError(QuickExpError):
    """An experiment cannot be planned, acquired, or decoded."""


class AcquisitionError(ExperimentError):
    """A hardware acquisition failed after its recovery policy."""




class SafetyError(QuickExpError):
    """A requested hardware action is outside the reviewed safety policy."""


class AnalysisError(QuickExpError):
    """Analysis failed or produced a result that should not be accepted."""

