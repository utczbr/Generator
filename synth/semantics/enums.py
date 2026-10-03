"""
synth/semantics/enums.py

Shared enums for the semantic manifests. Kept in a dependency-free leaf module so that
loader.py (runtime, must not import pydantic) and schema.py (build/test-time validation)
can share one definition without schema.py importing loader.py's import-time registry.
"""
from enum import Enum


class ScaleType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    DISCRETE_COUNT = "discrete_count"
    PERCENTAGE = "percentage"
    TEMPORAL = "temporal"


class AxisRole(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    BIDIRECTIONAL = "bidirectional"


class UnknownDomainError(KeyError, ValueError):
    """Raised when an unknown or unsupported domain is requested in semantic sampling or tabular generation."""
    pass
