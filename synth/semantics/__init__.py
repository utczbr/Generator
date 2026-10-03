"""
synth.semantics package

Exposes semantic sampling functions for chart titles, axis label pairs,
secondary Y axes, and comparative pairs.
"""
from synth.semantics.enums import UnknownDomainError
from synth.semantics.sampler import (
    sample_chart_title,
    sample_axis_pair,
    sample_secondary_y,
    sample_comparative_pair,
)

__all__ = [
    "UnknownDomainError",
    "sample_chart_title",
    "sample_axis_pair",
    "sample_secondary_y",
    "sample_comparative_pair",
]
