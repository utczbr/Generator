"""
tests/test_chart_semantic_migration.py

Verification tests for Phase 5 (Tasks T116-T124, T128):
- Dual Y-axis bar chart domain re-resolution at line 1322 before table generation.
- Standard & grouped bar chart axis sampling immediately after orientation resolution.
- Treatment key support with domain_category and business path activation.
- Line, Area, Scatter, Box, and Histogram semantic sampling and domain write-backs.
- Bar series info write-back to theme_config.
- Additive multi-subplot composite tracking in generator.py.
"""
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import random

from chart import (
    _generate_bar_chart,
    _generate_line_chart,
    _generate_area_chart,
    _generate_scatter_chart,
    _generate_boxplot_chart,
    _generate_histogram,
    _generate_heatmap_chart,
    add_treatment_key_xaxis,
    DUAL_AXIS_PROBABILITY,
    TREATMENT_KEY_PROBABILITY,
)
from synth.semantics.catalog import METRIC_BY_LABEL, COMPARATIVE_PAIRS_CATALOG, AxisRole, ScaleType


def test_dual_axis_domain_re_resolution_and_sampling():
    """Verify that dual Y-axis bar charts re-resolve business domain to scientific subdomains

    before multivariate table generation and write back effective domain and series info (T116, T122).
    """
    fig, ax = plt.subplots()
    theme_config = {
        "palette": "viridis",
        "semantic_domain": "business",
        "scientific_subdomain_weights": {"biomedical": 0.70, "engineering": 0.30},
        "use_synthetic_data_engine": False,
    }
    style_config = {
        "orientation": "vertical",
        "style": "grouped",
        "pattern": "none",
        "force_dual_axis": True,
    }

    data_artists, other_artists, bar_info, orientation, error_tops, axis_rel, scale_axis_info, _ = _generate_bar_chart(
        ax, "nature", theme_config, style_config
    )

    # 1. Assert domain was re-resolved from 'business' to scientific subdomain
    assert theme_config.get("semantic_domain") in ("biomedical", "engineering")
    assert theme_config.get("effective_is_scientific") is True

    # 2. Assert series info was written back
    assert theme_config.get("series_count") == 2
    assert len(theme_config.get("series_names", [])) == 2

    # 3. Assert dual-axis structure
    assert scale_axis_info.get("secondary_scale_axis") == "y2"

    # 4. Check labels and concept stem / group collision avoidance
    y1_label = ax.get_ylabel()
    # Find secondary axis from other_artists or figure
    ax2 = [a for a in fig.axes if a != ax][0]
    y2_label = ax2.get_ylabel()

    assert y1_label != "" and y2_label != ""
    assert y1_label != y2_label
    m1 = METRIC_BY_LABEL.get(y1_label)
    m2 = METRIC_BY_LABEL.get(y2_label)
    if m1 and m2:
        assert m1.concept_stem != m2.concept_stem
        if m1.concept_group and m2.concept_group:
            assert m1.concept_group != m2.concept_group

    plt.close(fig)


def test_standard_bar_immediate_sampling_and_writebacks():
    """Verify standard bar samples immediately after orientation resolution

    and writes back effective domain and series info (T117, T122).
    """
    fig, ax = plt.subplots()
    theme_config = {
        "palette": "viridis",
        "semantic_domain": "business",
        "use_synthetic_data_engine": False,
    }
    style_config = {
        "orientation": "vertical",
        "style": "side_by_side",
        "pattern": "none",
        "is_scientific": False,
        "force_dual_axis": False,
        "force_treatment_key": False,
    }

    data_artists, other_artists, bar_info, orientation, error_tops, axis_rel, scale_axis_info, _ = _generate_bar_chart(
        ax, "corporate", theme_config, style_config
    )

    assert theme_config.get("semantic_domain") == "business"
    assert theme_config.get("series_count") == 2
    assert theme_config.get("series_names") == ["Series 1", "Series 2"]

    y_label = ax.get_ylabel()
    x_label = ax.get_xlabel()
    assert y_label != ""
    assert x_label != ""
    m_y = METRIC_BY_LABEL.get(y_label)
    m_x = METRIC_BY_LABEL.get(x_label)
    if m_y:
        assert "business" in m_y.domain_tags
        assert m_y.role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)
    if m_x:
        assert "business" in m_x.domain_tags
        assert m_x.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)

    plt.close(fig)


def test_treatment_key_business_activation():
    """Verify treatment key works with domain_category and activates business pairs (T122)."""
    fig, ax = plt.subplots()
    bar_info_list = [
        {"center": 0.5},
        {"center": 1.5},
        {"center": 2.5},
        {"center": 3.5},
    ]

    artists = add_treatment_key_xaxis(ax, bar_info_list, domain_category="business")
    assert len(artists) >= 2
    label1 = artists[0].get_text()
    label2 = artists[1].get_text()

    # Verify that the pair drawn is a valid business comparative pair
    business_pairs = [
        (p.control, p.treatment)
        for p in COMPARATIVE_PAIRS_CATALOG
        if p.domain_category == "business"
    ]
    assert (label1, label2) in business_pairs

    plt.close(fig)


def test_line_and_area_chart_semantic_migration():
    """Verify line and area charts draw independent X and write back domain (T118)."""
    # 1. Line Chart
    fig, ax = plt.subplots()
    theme_config = {"palette": "viridis", "semantic_domain": "biomedical"}
    _generate_line_chart(ax, "nature", theme_config, is_scientific=True)

    assert theme_config.get("semantic_domain") == "biomedical"
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    assert x_label != "" and y_label != ""
    mx = METRIC_BY_LABEL.get(x_label)
    my = METRIC_BY_LABEL.get(y_label)
    if mx:
        assert mx.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)
        assert mx.scale_type in (ScaleType.CONTINUOUS, ScaleType.TEMPORAL, ScaleType.DISCRETE_COUNT)
    if my:
        assert my.role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)
    plt.close(fig)

    # 2. Area Chart
    fig, ax = plt.subplots()
    theme_config = {"palette": "viridis", "semantic_domain": "engineering"}
    _generate_area_chart(ax, "science", theme_config, is_scientific=True)

    assert theme_config.get("semantic_domain") == "engineering"
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    assert x_label != "" and y_label != ""
    mx = METRIC_BY_LABEL.get(x_label)
    if mx:
        assert mx.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)
    plt.close(fig)


def test_scatter_chart_semantic_migration():
    """Verify scatter plot draws continuous metrics on both X and Y (T119)."""
    fig, ax = plt.subplots()
    theme_config = {"palette": "viridis", "semantic_domain": "biomedical"}
    _generate_scatter_chart(ax, "nature", theme_config, is_scientific=True)

    assert theme_config.get("semantic_domain") == "biomedical"
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    assert x_label != "" and y_label != ""
    mx = METRIC_BY_LABEL.get(x_label)
    my = METRIC_BY_LABEL.get(y_label)
    if mx and my:
        assert mx.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT, ScaleType.TEMPORAL)
        assert my.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT, ScaleType.TEMPORAL)
    plt.close(fig)


def test_boxplot_chart_semantic_migration():
    """Verify box plots support orientation and write back domain (T120)."""
    fig, ax = plt.subplots()
    theme_config = {"palette": "viridis", "semantic_domain": "business"}
    _generate_boxplot_chart(ax, "corporate", theme_config, is_scientific=False)

    assert theme_config.get("semantic_domain") == "business"
    assert ax.get_ylabel() != "" or ax.get_xlabel() != ""
    plt.close(fig)


def test_histogram_chart_semantic_migration():
    """Verify histogram draws continuous metric on X and writes back domain (T121)."""
    fig, ax = plt.subplots()
    theme_config = {"palette": "viridis", "semantic_domain": "engineering"}
    _generate_histogram(ax, "technical", theme_config, is_scientific=True)

    assert theme_config.get("semantic_domain") == "engineering"
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    assert x_label != ""
    assert y_label != ""
    plt.close(fig)


def test_multi_subplot_composite_tracking(tmp_path):
    """Verify generator.py creates subplots array and additive composite fields (T123)."""
    import copy
    import json
    import os
    from custom_config import OCR_TRAINING_CONFIG
    from generator import generate_single_chart

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = 1
    cfg["engine"] = "matplotlib"
    cfg["debug_mode"] = False
    cfg["scenario_weights"] = {"single": 0.0, "multi": 1.0}
    cfg["realism_effects"] = {}
    cfg["export_legacy_json"] = True

    run_dir = str(tmp_path / "composite_test")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
    base_name = "chart_00000"

    det_path = os.path.join(lbl_dir, f"{base_name}_detailed.json")
    assert os.path.isfile(det_path)
    with open(det_path, "r", encoding="utf-8") as f:
        det = json.load(f)

    # Multi-subplot assertions
    assert "subplots" in det
    assert len(det["subplots"]) > 1
    assert det["is_composite"] is True
    assert "composite_chart_types" in det
    assert len(det["composite_chart_types"]) == len(det["subplots"])
    assert "composite_domains" in det
    assert len(det["composite_domains"]) == len(det["subplots"])

    # Root scalar reflection of primary subplot
    assert det["chart_type"] == det["subplots"][0]["chart_type"]
    assert det["chart_type"] not in ("composite", "mixed")
    assert det["semantic_domain"] == det["subplots"][0]["semantic_domain"]
    assert det["semantic_domain"] not in ("composite", "mixed")

    # Meta JSON chart_types reflects all subplots
    meta_path = os.path.join(lbl_dir, f"{base_name}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["chart_types"] == det["composite_chart_types"]


def test_config_driven_probabilities():
    """Verify that probabilities across all chart types can be driven from config."""
    # 1. Bar chart: dual_axis_probability (0.0 disables, 1.0 enables)
    fig, ax = plt.subplots()
    theme_cfg = {"palette": "viridis", "semantic_domain": "biomedical"}
    style_cfg = {"orientation": "vertical", "style": "grouped", "pattern": "none", "dual_axis_probability": 0.0}
    _, _, _, _, _, _, scale_info, _ = _generate_bar_chart(ax, "default", theme_cfg, style_cfg)
    assert scale_info.get("secondary_scale_axis") is None
    plt.close(fig)

    fig, ax = plt.subplots()
    theme_cfg = {"palette": "viridis", "semantic_domain": "biomedical"}
    style_cfg = {"orientation": "vertical", "style": "grouped", "pattern": "none", "dual_axis_probability": 1.0}
    _, _, _, _, _, _, scale_info, _ = _generate_bar_chart(ax, "default", theme_cfg, style_cfg)
    assert scale_info.get("secondary_scale_axis") == "y2"
    plt.close(fig)

    # 2. Bar chart: error_bar_probability (0.0 disables)
    fig, ax = plt.subplots()
    style_cfg = {
        "orientation": "vertical", "style": "standard", "pattern": "none",
        "error_bar_probability": 0.0, "data_label_probability": 0.0,
        "is_scientific": False
    }
    _, other_artists, _, _, _, _, _, _ = _generate_bar_chart(ax, "default", {}, style_cfg)
    # With 0.0 error bar and data label prob and not scientific, no error bars or data labels added
    assert len(other_artists) == 0
    plt.close(fig)

    # 3. Line chart: marker_probability (0.0 disables, 1.0 enables)
    fig, ax = plt.subplots()
    theme_cfg = {"line_chart_config": {"marker_probability": 0.0, "legend_probability": 0.0}}
    data_artists, _, _, _, _, _, _, _ = _generate_line_chart(ax, "default", theme_cfg, is_scientific=False)
    for line in data_artists:
        assert line.get_marker() in (None, '', 'None')
    plt.close(fig)

    fig, ax = plt.subplots()
    theme_cfg = {"line_chart_config": {"marker_probability": 1.0, "legend_probability": 0.0}}
    data_artists, _, _, _, _, _, _, _ = _generate_line_chart(ax, "default", theme_cfg, is_scientific=False)
    for line in data_artists:
        assert line.get_marker() not in (None, '', 'None')
    plt.close(fig)

    # 4. Line chart: legend_probability (0.0 disables)
    fig, ax = plt.subplots()
    theme_cfg = {"line_chart_config": {"legend_probability": 0.0}}
    _, other_artists, _, _, _, _, _, _ = _generate_line_chart(ax, "default", theme_cfg, is_scientific=False)
    assert len(other_artists) == 0
    plt.close(fig)

    # 5. Boxplot chart: horizontal_probability (1.0 forces horizontal, 0.0 forces vertical)
    fig, ax = plt.subplots()
    theme_cfg = {"box_plot_config": {"horizontal_probability": 1.0}}
    _, _, _, orientation, _, _, _, _ = _generate_boxplot_chart(ax, "default", theme_cfg, is_scientific=False)
    assert orientation == 'horizontal'
    plt.close(fig)

    fig, ax = plt.subplots()
    theme_cfg = {"box_plot_config": {"horizontal_probability": 0.0}}
    _, _, _, orientation, _, _, _, _ = _generate_boxplot_chart(ax, "default", theme_cfg, is_scientific=False)
    assert orientation == 'vertical'
    plt.close(fig)

    # 6. Area chart: legend_probability (0.0 disables)
    fig, ax = plt.subplots()
    theme_cfg = {"area_chart_config": {"legend_probability": 0.0}}
    _, other_artists, _, _, _, _, _, _ = _generate_area_chart(ax, "default", theme_cfg, is_scientific=False)
    # In area chart, other_artists contains white boundary lines + optional legend
    # With legend_probability: 0.0, no Legend instances are in other_artists
    assert not any(isinstance(a, matplotlib.legend.Legend) for a in other_artists)
    plt.close(fig)

    # 7. Heatmap chart: annotate_cells_probability (0.0 disables cell text annotations)
    fig, ax = plt.subplots()
    theme_cfg = {"heatmap_config": {"annotate_cells_probability": 0.0}}
    data_art, other_art, _, _, _, _, _, meta = _generate_heatmap_chart(ax, "default", theme_cfg, is_scientific=False)
    # Check that no matplotlib.text.Text data labels were added to the axes for cells
    cell_texts = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Text) and child.get_gid() != 'title']
    # Default without annotations only has tick labels, no cell value texts in other_art
    assert len([a for a in other_art if isinstance(a, matplotlib.text.Text)]) == 0
    plt.close(fig)

