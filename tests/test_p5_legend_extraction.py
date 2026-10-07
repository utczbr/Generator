"""
tests/test_p5_legend_extraction.py

Acceptance tests for Phase 1 Task T234b:
Granular Legend Extraction & Multi-Series Legend Fix (FR-074b, FR-114, OC-9).
"""
import copy
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pytest

from chart import extract_legend_elements, _generate_bar_chart
from config_defaults import OCR_TRAINING_CONFIG
from generator import generate_single_chart


def test_extract_legend_elements_no_degenerate_boxes():
    """Verify decompose legend produces boxes >= 2px for all handle types including markerless Line2D."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    line1 = ax.plot([0, 1], [0, 1], label="Line Series", linewidth=1.5)[0]  # marker-less line
    line2 = ax.plot([0, 1], [1, 0], 'o-', label="Point Series", markersize=6)[0]
    patch1 = mpatches.Patch(color="red", label="Patch Series")

    legend = ax.legend(handles=[line1, line2, patch1], title="Test Legend")
    fig.canvas.draw()

    elements = extract_legend_elements(legend, ax, img_h=400, img_w=600)
    plt.close(fig)

    assert len(elements) == 7  # 1 title + 3 labels + 3 markers
    class_names = [e["class_name"] for e in elements]
    assert class_names.count("legend_title") == 1
    assert class_names.count("legend_label") == 3
    assert class_names.count("legend_marker") == 3

    for el in elements:
        bbox = el["bbox"]
        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0
        assert width >= 2.0, f"Degenerate width {width} in {el}"
        assert height >= 2.0, f"Degenerate height {height} in {el}"
        assert el["role"] in ("legend_title", "legend_label", "legend_marker")


def test_legend_spawn_rate_multi_series():
    """Verify multi-series charts spawn legends >= 60% of the time under domain_gap_v1."""
    clustered_count = 0
    total_clustered = 30
    for i in range(total_clustered):
        fig, ax = plt.subplots(figsize=(6, 4))
        fig._generator_config = {"profile": "domain_gap_v1", "seed": 2000 + i}
        fig._generator_image_idx = i
        theme_cfg = {"profile": "domain_gap_v1"}

        _generate_bar_chart(
            ax,
            theme_name="default",
            theme_config=theme_cfg,
            style_config={
                "style": "side_by_side",
                "pattern": "solid",
                "force_dual_axis": False,
                "is_scientific": False,
            },
        )
        if ax.get_legend() and ax.get_legend().get_visible():
            clustered_count += 1
        plt.close(fig)

    rate = clustered_count / total_clustered
    assert rate >= 0.60, f"Multi-series bar legend spawn rate {rate:.2f} < 0.60"


def test_detailed_json_and_yolo_export_consistency(tmp_path):
    """
    Test end-to-end multi-series chart generation:
    - Detailed JSON has granular legend_elements stream.
    - YOLO txt preserves only the single monolithic legend box (OC-9).
    """
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["profile"] = "domain_gap_v1"
    cfg["output_dir"] = str(tmp_path)
    cfg["seed"] = 42
    for ct in cfg.get("chart_types", {}):
        cfg["chart_types"][ct]["enabled"] = (ct == "bar")
    cfg["bar_chart_config"]["styles"] = {
        "side_by_side": {"weight": 1.0, "enabled": True},
        "stacked": {"weight": 0.0, "enabled": False},
        "simple": {"weight": 0.0, "enabled": False},
    }
    cfg["bar_chart_config"]["force_dual_axis"] = False
    cfg["scenario_weights"] = {"single": 1.0, "multi": 0.0}
    cfg["export_formats"] = ["yolo", "yolo_obb"]
    cfg["debug_mode"] = False
    cfg["export_detailed_json"] = True

    # Generate images until we get one with a legend
    found_legend = False
    for i in range(25):
        generate_single_chart(i, cfg, str(images_dir), str(labels_dir), str(tmp_path))

        base_name = f"chart_{i:05d}"
        det_file = labels_dir / f"{base_name}_detailed.json"
        txt_file = labels_dir / f"{base_name}.txt"

        if det_file.exists():
            with open(det_file, "r") as f:
                data = json.load(f)

            anns = data.get("annotations", [])
            leg_els = [a for a in anns if a.get("stream") == "legend_elements"]
            monolithic = [a for a in anns if a.get("stream") == "annotations" and a.get("class_name") == "legend"]

            if monolithic and leg_els:
                found_legend = True
                assert len(leg_els) >= 2, "Expected decomposed legend elements in detailed JSON"
                for el in leg_els:
                    assert el["class_name"] in ("legend_title", "legend_label", "legend_marker")
                    xyxy = el["xyxy"]
                    assert xyxy[2] - xyxy[0] >= 2.0
                    assert xyxy[3] - xyxy[1] >= 2.0

                # Check YOLO txt contains monolithic legend and NOT individual decomposed ones
                if txt_file.exists():
                    with open(txt_file, "r") as f:
                        lines = [l.strip().split() for l in f if l.strip()]
                    bar_cls = cfg.get("CLASS_MAP_BAR", {})
                    legend_cls_id = next((int(k) for k, v in bar_cls.items() if v == "legend"), 5)
                    yolo_legends = [l for l in lines if int(l[0]) == legend_cls_id]
                    assert len(yolo_legends) == 1, f"Expected 1 monolithic legend in YOLO txt, got {len(yolo_legends)}"
                break

    assert found_legend, "Did not encounter a chart with legend in 25 attempts"
