"""
tests/test_p5_baseline_defects.py

Failing reproducers for baseline defects documented in spec 005 (T216):
- N-01: multi_chart_detection raises UnboundLocalError: clsmap_obj on line/area/pie
- N-02: v4.0 detailed.json produces class_name == 'unknown' for line/area
- N-03: scenario=multi 2x1 composite emits 0 annotations for bottom half
- N-04: error_bar class is never produced (ErrorbarContainer lacks get_window_extent)
- N-06: _detailed.json lacks filter_stats
- N-14: line/area charts emit bar class IDs in labels/*.txt
- N-15: apply_legend_variation with ncol > 1 leaves legend extent unchanged
- N-20: twin-axis chart produces phantom annotations for invisible secondary x-axis
"""
import copy
import glob
import json
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from config_defaults import OCR_TRAINING_CONFIG
import generator as G
import chart


def _make_cfg(tmp_path, chart_types=None, dataset_format=None, **kwargs):
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg.update(num_images=1, debug_mode=False, use_parallel=False, **kwargs)
    for v in cfg.get("realism_effects", {}).values():
        if isinstance(v, dict):
            v["p"] = 0.0
    if chart_types:
        for k in cfg["chart_types"]:
            cfg["chart_types"][k]["weight"] = chart_types.get(k, 0)
    if dataset_format:
        cfg["dataset_format"] = dataset_format
    run_dir = str(tmp_path)
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    return cfg, img_dir, lbl_dir, run_dir


# -----------------------------------------------------------------------------
# N-01: multi_chart_detection UnboundLocalError: clsmap_obj
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("chart_type", [
    "bar", "line", "area", "pie", "scatter", "box", "histogram", "heatmap"
])
def test_n01_multi_chart_detection_primary_types(tmp_path, chart_type):
    """N-01: multi_chart_detection must generate 3 images per primary type without crashing."""
    cfg, img_dir, lbl_dir, run_dir = _make_cfg(
        tmp_path,
        chart_types={chart_type: 100},
        dataset_format="multi_chart_detection"
    )
    for i in range(3):
        random.seed(42 + i)
        np.random.seed(42 + i)
        G.generate_single_chart(i, cfg, img_dir, lbl_dir, run_dir)
        plt.close("all")


# -----------------------------------------------------------------------------
# N-02: line/area 24-image run has no class_name == 'unknown'
# -----------------------------------------------------------------------------
def test_n02_line_area_no_unknown_classes(tmp_path):
    """N-02: 24-image line/area run must have 0 annotations with class_name == 'unknown'."""
    cfg, img_dir, lbl_dir, run_dir = _make_cfg(
        tmp_path,
        chart_types={"line": 50, "area": 50}
    )
    for i in range(24):
        random.seed(100 + i)
        np.random.seed(100 + i)
        G.generate_single_chart(i, cfg, img_dir, lbl_dir, run_dir)
        plt.close("all")

    det_files = sorted(glob.glob(os.path.join(lbl_dir, "*_detailed.json")))
    assert len(det_files) == 24
    unknown_count = 0
    total_count = 0
    for p in det_files:
        with open(p, "r", encoding="utf-8") as f:
            det = json.load(f)
        for ann in det.get("annotations", []):
            total_count += 1
            if ann.get("class_name") == "unknown":
                unknown_count += 1

    assert unknown_count == 0, f"Found {unknown_count}/{total_count} annotations with class_name == 'unknown'"


# -----------------------------------------------------------------------------
# N-03: forced 2x1 composite has annotations in both halves
# -----------------------------------------------------------------------------
def test_n03_multi_axes_composite_both_halves(tmp_path):
    """N-03: forced 2x1 composite must produce annotations in both top and bottom halves."""
    cfg, img_dir, lbl_dir, run_dir = _make_cfg(
        tmp_path,
        chart_types={"bar": 100}
    )
    cfg["scenario_weights"] = {"single": 0, "multi": 100}

    orig_choice = random.choice
    def forced_choice(seq):
        s = list(seq)
        if (1, 2) in s and (2, 1) in s:
            return (2, 1)
        return orig_choice(seq)

    random.choice = forced_choice
    try:
        random.seed(7)
        np.random.seed(7)
        G.generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
    finally:
        random.choice = orig_choice
        plt.close("all")

    det_files = glob.glob(os.path.join(lbl_dir, "*_detailed.json"))
    assert len(det_files) == 1
    with open(det_files[0], "r", encoding="utf-8") as f:
        det = json.load(f)

    h = det["image"]["height"]
    top = sum(1 for a in det["annotations"] if (a["xyxy"][1] + a["xyxy"][3]) / 2 < h / 2)
    bot = sum(1 for a in det["annotations"] if (a["xyxy"][1] + a["xyxy"][3]) / 2 >= h / 2)

    assert bot > 0, f"Bottom half has {bot} annotations (top half: {top}); expected > 0"


# -----------------------------------------------------------------------------
# N-04: bar charts with error bars produce error_bar annotations
# -----------------------------------------------------------------------------
def test_n04_error_bar_count_greater_than_zero(tmp_path):
    """N-04: bar charts with forced error bars must produce error_bar annotations."""
    cfg, img_dir, lbl_dir, run_dir = _make_cfg(
        tmp_path,
        chart_types={"bar": 100}
    )
    # Force error bars via style_config and is_scientific=True
    cfg["chart_types"]["bar"]["style_config"] = {
        "error_bar_probability": 1.0,
        "is_scientific": True
    }

    # Generate 5 charts
    total_error_bars = 0
    for i in range(5):
        random.seed(200 + i)
        np.random.seed(200 + i)
        G.generate_single_chart(i, cfg, img_dir, lbl_dir, run_dir)
        plt.close("all")

    det_files = sorted(glob.glob(os.path.join(lbl_dir, "*_detailed.json")))
    for p in det_files:
        with open(p, "r", encoding="utf-8") as f:
            det = json.load(f)
        total_error_bars += sum(1 for a in det["annotations"] if a.get("class_name") == "error_bar")

    assert total_error_bars > 0, f"Expected error_bar count > 0, got {total_error_bars}"


# -----------------------------------------------------------------------------
# N-06: _detailed.json contains filter_stats
# -----------------------------------------------------------------------------
def test_n06_detailed_json_contains_filter_stats(tmp_path):
    """N-06: _detailed.json must contain a filter_stats dictionary."""
    cfg, img_dir, lbl_dir, run_dir = _make_cfg(tmp_path, chart_types={"bar": 100})
    random.seed(42)
    np.random.seed(42)
    G.generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
    plt.close("all")

    det_files = glob.glob(os.path.join(lbl_dir, "*_detailed.json"))
    assert len(det_files) == 1
    with open(det_files[0], "r", encoding="utf-8") as f:
        det = json.load(f)

    assert "filter_stats" in det, "Missing required key 'filter_stats' in _detailed.json"
    assert isinstance(det["filter_stats"], dict), "'filter_stats' must be a dict"


# -----------------------------------------------------------------------------
# N-14: line/area charts emit chart-specific class IDs in labels/*.txt
# -----------------------------------------------------------------------------
def test_n14_line_area_labels_use_proper_class_ids(tmp_path):
    """N-14: labels/*.txt for line charts must emit line class IDs (e.g. line_segment=1), not bar IDs."""
    cfg, img_dir, lbl_dir, run_dir = _make_cfg(tmp_path, chart_types={"line": 100})
    random.seed(42)
    np.random.seed(42)
    G.generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
    plt.close("all")

    txt_files = glob.glob(os.path.join(lbl_dir, "chart_*.txt"))
    assert len(txt_files) >= 1
    # Read YOLO labels from standard labels/chart_00000.txt
    std_label = os.path.join(lbl_dir, "chart_00000.txt")
    with open(std_label, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    class_ids = [int(parts[0]) for parts in lines]
    # CLASS_MAP_LINE_OBJ: 1: line_segment, 6: axis_labels
    # CLASS_MAP_BAR: 0: bar, 8: axis_labels
    # In buggy state: class 8 (bar axis_labels) appears, class 1 (line_segment) never appears
    assert 8 not in class_ids, f"labels/*.txt contains Bar axis_labels class ID 8 instead of Line ID 6: {class_ids}"
    assert 1 in class_ids, f"labels/*.txt missing line_segment class ID 1: {class_ids}"


# -----------------------------------------------------------------------------
# N-15: apply_legend_variation with ncol > 1 alters legend extent
# -----------------------------------------------------------------------------
def test_n15_legend_ncol_alters_extent():
    """N-15: apply_legend_variation with ncol > 1 must alter legend width/height extent."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    for i in range(8):
        ax.plot([0, 1], [i, i], label=f"series_{i}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 1 column baseline
    lg1 = ax.legend(loc="upper right")
    fig.canvas.draw()
    w1 = lg1.get_window_extent(renderer).width

    # Using apply_legend_variation with forced ncol=2
    # In buggy code: sets legend._ncol = 2 which does NOT change layout extent
    orig_choice = np.random.choice
    def forced_choice(a, *args, **kwargs):
        if isinstance(a, list) and a == [1, 2]:
            return 2
        return orig_choice(a, *args, **kwargs)

    np.random.choice = forced_choice
    try:
        lg2 = chart.apply_legend_variation(ax, num_items=8)
        fig.canvas.draw()
        w2 = lg2.get_window_extent(renderer).width
    finally:
        np.random.choice = orig_choice
        plt.close("all")

    # A 2-column 8-item legend must be significantly wider than a 1-column 8-item legend
    assert abs(w2 - w1) > 20, f"Legend extent unchanged after ncol=2: w1={w1:.1f}, w2={w2:.1f}"


# -----------------------------------------------------------------------------
# N-20: twin-axis chart produces zero ghost annotations for secondary x-axis
# -----------------------------------------------------------------------------
def test_n20_twin_axis_zero_ghost_annotations():
    """N-20: twin-axis chart (ax.twinx()) must generate zero ghost annotations for secondary x-axis."""
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    ax1.plot([0, 1, 2], [10, 20, 30], label="primary")
    ax2 = ax1.twinx()
    ax2.plot([0, 1, 2], [100, 200, 300], color="red", label="secondary")
    assert not ax2.xaxis.get_visible()

    fig.delaxes(ax1)
    chart_info_map = {ax2: {"chart_type_str": "line"}}
    cls_map = G.CHART_CLASS_MAPS.get("line_obj", G.CHART_CLASS_MAPS["bar"])
    anns = G.get_granular_annotations(fig, chart_info_map, cls_map)

    # In buggy state, ax2 emits x-axis tick labels even though ax2.xaxis is invisible
    # Since ax2 has no x-axis ticks visible, any x-axis tick annotations are phantom
    x_texts = {t.get_text() for t in ax2.get_xticklabels() if t.get_text()}
    phantom_boxes = [a for a in anns if str(a.get("class_id")) == "6" and a.get("text") in x_texts]
    plt.close("all")
    assert len(phantom_boxes) == 0, f"Twin-axis generated {len(phantom_boxes)} phantom secondary x-axis annotations"


def test_strict_mode_unresolved_class_raises():
    """T218: _serialize in strict mode must raise ValueError on unresolved class."""
    import detailed
    ann = {"class_id": 9999, "bbox": [10, 10, 50, 50]}
    # In strict mode, should raise ValueError
    with pytest.raises(ValueError, match="Unresolved class"):
        detailed._serialize(ann, "test_stream", cls_map={0: "chart"}, kind="mat", w=100, h=100, strict=True)

    # In lenient mode, should warn instead of raise
    with pytest.warns(UserWarning, match="Unresolved class"):
        res = detailed._serialize(ann, "test_stream", cls_map={0: "chart"}, kind="mat", w=100, h=100, strict=False)
        assert res is not None
        assert res["class_name"] == "unknown"
