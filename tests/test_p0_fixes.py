"""Unit tests verifying Phase 2 Core P0 correctness fixes."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import generator
from chart import (
    _generate_bar_chart,
    _generate_area_chart,
    apply_axis_scaling,
    extract_bar_info,
)
from generator import (
    has_non_background_pixels,
    create_unified_annotation,
)


def test_bar_series_idx_discrimination():
    """Verify grouped and stacked bar charts export distinct series_idx values."""
    theme_config = {
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'background_color': '#ffffff',
        'grid_color': '#e0e0e0',
        'font_family': 'DejaVu Sans'
    }

    # 1. Test side_by_side (grouped)
    import random
    random.seed(42)
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        style_config = {'style': 'side_by_side', 'orientation': 'horizontal', 'is_scientific': False, 'pattern': 'none'}
        result = _generate_bar_chart(ax, 'default', theme_config, style_config, debug_mode=False)
        bar_info_list = result[2]

        # Verify bar_info_list returned from generator has both series 0 and 1
        series_indices = {b['series_idx'] for b in bar_info_list if 'series_idx' in b}
        assert series_indices == {0, 1}, f"Expected {{0, 1}}, got {series_indices}"

        # Verify extract_bar_info on this axis also preserves distinct series_idx
        extracted = extract_bar_info(ax, 'bar')
        extracted_series = {b['series_idx'] for b in extracted if 'series_idx' in b}
        assert extracted_series == {0, 1}, f"extract_bar_info failed: {extracted_series}"

        # Verify create_unified_annotation carries distinct series_idx into detailed_metadata
        chart_info_map = {
            ax: {
                'chart_type_str': 'bar',
                'bar_info': bar_info_list,
                'bar_info_list': bar_info_list,
                'orientation': 'horizontal',
            }
        }
        unified_json = create_unified_annotation(
            fig, chart_info_map,
            cls_map=generator.CHART_CLASS_MAPS['bar'],
            img_w=600, img_h=400,
            annotations=[]
        )
        bar_info_meta = unified_json.get('bar_info', [])
        meta_series = {b.get('series_idx') for b in bar_info_meta}
        assert meta_series == {0, 1}, f"Unified metadata lost series_idx: {meta_series}"
    finally:
        plt.close(fig)

    # 2. Test stacked style
    random.seed(42)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    try:
        style_config = {'style': 'stacked', 'orientation': 'vertical', 'is_scientific': False, 'pattern': 'none'}
        result = _generate_bar_chart(ax2, 'default', theme_config, style_config, debug_mode=False)
        bar_info_list = result[2]
        series_indices = {b['series_idx'] for b in bar_info_list if 'series_idx' in b}
        assert series_indices == {0, 1}, f"Stacked style expected {0, 1}, got {series_indices}"
    finally:
        plt.close(fig2)


def test_area_chart_non_positive_symlog_fallback():
    """Verify non-positive data_min forces symlog instead of log scale."""
    fig, ax = plt.subplots()
    try:
        # If data_min <= 0 and scale_type='log', apply_axis_scaling MUST fall back to 'symlog'
        apply_axis_scaling(ax, data_min=-1.5, orientation='vertical', scale_type='log')
        assert ax.get_yscale() == 'symlog'

        # If data_min > 0, log scale is permitted
        apply_axis_scaling(ax, data_min=5.0, orientation='vertical', scale_type='log')
        assert ax.get_yscale() == 'log'

        # If data_min == 0, falls back to symlog
        apply_axis_scaling(ax, data_min=0.0, orientation='vertical', scale_type='log')
        assert ax.get_yscale() == 'symlog'
    finally:
        plt.close(fig)


def test_has_non_background_pixels_analytical():
    """Verify analytical rejection of hidden, empty, or degenerate text artists."""
    fig, ax = plt.subplots()
    try:
        fig.canvas.draw()

        # 1. Normal visible text
        t_valid = ax.text(0.5, 0.5, "Valid Text")
        fig.canvas.draw()
        assert has_non_background_pixels(t_valid, fig, ax) is True

        # 2. Empty text
        t_empty = ax.text(0.5, 0.5, "   ")
        assert has_non_background_pixels(t_empty, fig, ax) is False

        # 3. Invisible text
        t_hidden = ax.text(0.5, 0.5, "Hidden Text")
        t_hidden.set_visible(False)
        assert has_non_background_pixels(t_hidden, fig, ax) is False

        # 4. Out-of-bounds text (placed far outside figure)
        t_oob = ax.text(100.0, 100.0, "OOB Text")
        fig.canvas.draw()
        assert has_non_background_pixels(t_oob, fig, ax) is False
    finally:
        plt.close(fig)


def test_baseline_linkage_distance_cutoff():
    """Verify FR-006 / T072: Implausibly distant baselines are rejected."""
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)
        fig.canvas.draw()
        img_h = 400

        # Bar 1: Normal bar near baseline (bottom=0, top=5, x=1)
        # Bar 2: Implausibly distant bar (bottom=5000, top=5010, x=2)
        detailed_metadata = {
            "bar_info": [
                {"bar_idx": 1, "width": 0.8, "bottom": 0.0, "top": 5.0, "value": 5.0},
                {"bar_idx": 2, "width": 0.8, "bottom": 5000.0, "top": 5010.0, "value": 10.0}
            ]
        }

        from generator import add_graph_topology_metadata
        detailed_metadata = add_graph_topology_metadata(fig, detailed_metadata, img_h)

        bars_with_baseline = detailed_metadata.get("bars_with_baseline", [])
        # Only the plausible bar should be linked
        assert len(bars_with_baseline) == 1
        assert bars_with_baseline[0]["bar_index"] == 0
        assert bars_with_baseline[0]["baseline_id"] == "baseline_0"
    finally:
        plt.close(fig)


def test_config_defaults_fallback():
    """Verify FR-023 / T070: config_defaults provides complete required keys."""
    import config_defaults
    cfg = config_defaults.OCR_TRAINING_CONFIG
    assert isinstance(cfg, dict)
    for required in ["CLASS_MAP_BAR", "CLASS_MAP_LINE_OBJ", "CLASS_MAP_SCATTER", "realism_effects"]:
        assert required in cfg


def test_hollow_bar_pattern_visible_edge():
    """Verify that hollow bars in scientific and standard charts have visible edges and are not white-on-white."""
    theme_config = {
        'colors': ['#1f77b4', '#ff7f0e'],
        'background_color': '#ffffff',
        'grid_color': '#e0e0e0',
        'font_family': 'DejaVu Sans'
    }

    # Case 1: Scientific chart with hollow pattern (previously produced invisible white borders)
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        style_config = {
            'style': 'compare_side_by_side',
            'orientation': 'vertical',
            'is_scientific': True,
            'pattern': 'hollow',
            'force_dual_axis': False
        }
        data_artists, _, _, _, _, _, _, _ = _generate_bar_chart(ax, 'default', theme_config, style_config, debug_mode=False)
        assert len(data_artists) > 0
        for bar in data_artists:
            fc = bar.get_facecolor()
            ec = bar.get_edgecolor()
            # Facecolor should be transparent
            assert fc[3] == 0.0, f"Hollow bar facecolor should be transparent, got {fc}"
            # Edgecolor must NOT be white and must be visible
            assert ec[3] > 0.0, f"Hollow bar edgecolor must be visible, got {ec}"
            assert not all(c >= 0.95 for c in ec[:3]), f"Hollow bar edgecolor must not be white, got {ec}"
            # Hatch should be cleared
            assert bar.get_hatch() is None
    finally:
        plt.close(fig)

    # Case 2: Standard chart with hollow pattern
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        style_config = {
            'style': 'default',
            'orientation': 'vertical',
            'is_scientific': False,
            'pattern': 'hollow',
            'force_dual_axis': False
        }
        data_artists, _, _, _, _, _, _, _ = _generate_bar_chart(ax, 'default', theme_config, style_config, debug_mode=False)
        assert len(data_artists) > 0
        for bar in data_artists:
            fc = bar.get_facecolor()
            ec = bar.get_edgecolor()
            assert fc[3] == 0.0
            assert ec[3] > 0.0
            assert not all(c >= 0.95 for c in ec[:3])
    finally:
        plt.close(fig)


