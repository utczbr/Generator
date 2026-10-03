"""
synth/__init__.py

Synthetic tabular data modeling and multivariate copula generation.
Uses PEP 562 lazy loading to prevent eager loading of heavy dependencies (scipy/numpy)
when importing lightweight subpackages such as synth.semantics.
"""
from typing import Any

__all__ = [
    "DOMAIN_PRESETS",
    "sample_coherent_labels",
    "sample_domain_schema",
    "sample_dose_response_series",
    "sample_gaussian_copula",
    "sample_multivariate_table",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import synth.tabular as _tabular
        return getattr(_tabular, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
