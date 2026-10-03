"""
Chart rendering backends.
Supports default Matplotlib engine and Vega-Lite vector engine.
"""
from backends.vegalite_backend import (
    generate_single_vegalite_chart,
    extract_svg_annotations,
    build_vegalite_bar_spec,
    build_vegalite_line_spec,
    build_vegalite_scatter_spec,
)

__all__ = [
    "generate_single_vegalite_chart",
    "extract_svg_annotations",
    "build_vegalite_bar_spec",
    "build_vegalite_line_spec",
    "build_vegalite_scatter_spec",
]
