"""
Unit tests for Phase 5: Topological Keypoints, YOLOv8-OBB Exporter, and Dual Modal/Amodal Masks.

Verifies:
a) labels_obb/ produces valid 8-coordinate normalized lines for rotated texts and bars, passing clockwise convexity assertions.
b) Standard labels/ AABB files remain 100% backwards-compatible and identical in schema.
c) Line charts generate valid topological keypoints with correct categorical tags (peak, valley, inflection, endpoint) and an edge adjacency list.
d) Partially occluded areas expose distinct amodal_polygon and modal_polygon structures where modal_area < amodal_area.
e) Area fill polygons clipped by axis spines are non-degenerate, valid Shapely polygons.
"""
import copy
import json
import math
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pytest
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box

from custom_config import OCR_TRAINING_CONFIG
from generator import (
    generate_single_chart,
    save_annotations_yolo_obb,
    save_annotations_yolo,
    extract_area_segmentation_annotations,
    create_unified_annotation,
    canonicalize_and_validate_obb,
    clip_polygon_to_viewport,
    CHART_CLASS_MAPS,
)
from chart import _generate_line_chart, _generate_area_chart


def test_yolov8_obb_exporter_format_and_convexity(tmp_path):
    """
    Verify labels_obb/ produces valid 8-coordinate normalized lines for rotated texts and bars,
    passing clockwise winding and convexity assertions.
    """
    img_w, img_h = 800, 600
    output_obb_path = str(tmp_path / "test_labels_obb.txt")

    # 1. Test direct export on synthetic mixed annotations (rotated text OBB + standard bar AABB)
    # 45-degree rotated box centered at (400, 300) with w=100, h=40
    cx, cy, bw, bh = 400.0, 300.0, 100.0, 40.0
    angle_rad = math.radians(45.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dx, dy = bw / 2.0, bh / 2.0
    local_corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    rotated_corners = [
        (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
        for x, y in local_corners
    ]

    test_annotations = [
        # Rotated text with canonicalized OBB
        {
            "class_id": 1,
            "bbox": (330.0, 230.0, 470.0, 370.0),
            "obb": rotated_corners,
        },
        # Standard axis-aligned bar element
        {
            "class_id": 0,
            "bbox": (100.0, 200.0, 180.0, 450.0),
        },
    ]

    save_annotations_yolo_obb(test_annotations, img_w, img_h, output_obb_path)
    assert os.path.exists(output_obb_path)

    with open(output_obb_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    assert len(lines) == 2

    for line in lines:
        tokens = line.split()
        assert len(tokens) == 9, f"Expected 9 tokens (class_id + 8 coords), got {len(tokens)}: {line}"
        class_id = int(tokens[0])
        coords = [float(v) for v in tokens[1:]]

        # Verify all coordinates normalized in [0, 1]
        for c in coords:
            assert 0.0 <= c <= 1.0, f"Coordinate {c} out of normalized range [0, 1]"

        # Reconstruct polygon points in pixel space
        px_pts = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, 8, 2)]
        assert len(px_pts) == 4

        # Verify clockwise winding and convexity via canonicalize_and_validate_obb
        canonical = canonicalize_and_validate_obb(px_pts, y_down=True)
        assert canonical is not None, "OBB points failed convexity or canonical clockwise validation"

        # Verify consecutive edge cross products are strictly positive (clockwise in y-down space)
        for i in range(4):
            e1 = np.array(px_pts[(i + 1) % 4]) - np.array(px_pts[i])
            e2 = np.array(px_pts[(i + 2) % 4]) - np.array(px_pts[(i + 1) % 4])
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            assert cross > 1e-4, f"Edge cross product {cross} not positive (non-convex or counter-clockwise)"

        # Verify top-left start: vertex 0 has minimum x+y among the 4 corners
        x_plus_y = [p[0] + p[1] for p in px_pts]
        min_xy = min(x_plus_y)
        assert abs(x_plus_y[0] - min_xy) < 1e-4, f"Vertex 0 {px_pts[0]} is not top-left (x+y={x_plus_y[0]}, min={min_xy})"


def test_standard_labels_aabb_backwards_compatibility(tmp_path):
    """
    Verify standard labels/ AABB files remain 100% backwards-compatible and identical in schema
    even when additive export_obb is enabled.
    """
    target_dir = str(tmp_path / "chart_run")
    images_dir = os.path.join(target_dir, "images")
    labels_dir = os.path.join(target_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["export_obb"] = True
    cfg["num_images"] = 1
    cfg["scenario_weights"] = {"single": 1.0, "multi": 0.0}
    cfg["chart_types"] = {k: {"weight": 1.0 if k == "bar" else 0.0, "enabled": (k == "bar")} for k in cfg["chart_types"]}
    cfg["realism_effects"] = {}

    np.random.seed(42)
    generate_single_chart(0, cfg, images_dir, labels_dir, target_dir)

    # 1. Standard AABB labels/ must exist and conform to 5-token YOLO format
    std_label_path = os.path.join(labels_dir, "chart_00000.txt")
    assert os.path.exists(std_label_path)

    with open(std_label_path, "r") as f:
        std_lines = [l.strip() for l in f if l.strip()]

    assert len(std_lines) > 0
    for line in std_lines:
        tokens = line.split()
        assert len(tokens) == 5, f"Expected 5 tokens (class_id cx cy w h), got: {tokens}"
        class_id = int(tokens[0])
        cx, cy, w, h = [float(v) for v in tokens[1:]]
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0

    # 2. Additive labels_obb/ directory must exist and contain 9-token lines
    obb_dir = os.path.join(target_dir, "labels_obb")
    assert os.path.isdir(obb_dir)
    obb_label_path = os.path.join(obb_dir, "chart_00000.txt")
    assert os.path.exists(obb_label_path)

    with open(obb_label_path, "r") as f:
        obb_lines = [l.strip() for l in f if l.strip()]

    assert len(obb_lines) == len(std_lines)
    for line in obb_lines:
        tokens = line.split()
        assert len(tokens) == 9, f"Expected 9 tokens for YOLOv8-OBB, got: {tokens}"


def test_line_chart_topological_keypoints_and_adjacency():
    """
    Verify line charts generate valid topological keypoints with correct categorical tags
    (peak, valley, inflection, endpoint) and a directed edge adjacency list.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    # Synthetic series designed with known peak, valley, and inflection
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = np.array([10.0, 45.0, 30.0, 15.0, 5.0, 20.0, 40.0, 35.0])
    line, = ax.plot(x, y, label="Test Series")

    chart_info_map = {
        ax: {
            "chart_type_str": "line",
            "data_artists": [line],
            "keypoint_info": [
                {
                    "series_idx": 0,
                    "start": (0.0, 10.0, 0),
                    "end": (7.0, 35.0, 7),
                    "peaks": [(1.0, 45.0, 1), (6.0, 40.0, 6)],
                    "valleys": [(4.0, 5.0, 4)],
                    "inflections": [(2.0, 30.0, 2)],
                    "all_points": [(float(x[i]), float(y[i]), int(i)) for i in range(len(x))],
                }
            ],
        }
    }

    cls_map = CHART_CLASS_MAPS["line_obj"]
    unified_json = create_unified_annotation(fig, chart_info_map, cls_map, 700, 500, annotations=[])
    plt.close(fig)

    keypoint_info = unified_json.get("keypoint_info", [])
    assert len(keypoint_info) == 1
    series_kpi = keypoint_info[0]

    # Legacy points schema preserved
    assert "points" in series_kpi
    assert len(series_kpi["points"]) == 8

    # Topological keypoints present
    assert "keypoints" in series_kpi
    kpts = series_kpi["keypoints"]
    assert len(kpts) == 8

    tags = [kp["tag"] for kp in kpts]
    # Endpoint assertions
    assert tags[0] == "endpoint"
    assert tags[-1] == "endpoint"

    # Peak assertions
    assert tags[1] == "peak"
    assert tags[6] == "peak"

    # Valley assertion
    assert tags[4] == "valley"

    # Inflection assertion
    assert tags[2] == "inflection"

    # Edge adjacency list assertions
    assert "edges" in series_kpi
    edges = series_kpi["edges"]
    assert len(edges) == 7  # 8 vertices -> 7 directed edges
    for idx, edge in enumerate(edges):
        assert edge == [idx, idx + 1], f"Edge {idx} mismatch: expected [{idx}, {idx+1}], got {edge}"

    assert "adjacency_list" in series_kpi
    assert series_kpi["adjacency_list"] == edges


def test_amodal_and_modal_polygon_separation():
    """
    Verify partially occluded areas expose distinct amodal_polygon and modal_polygon
    structures where modal_area < amodal_area.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    x = np.arange(6)

    # Series 0 (underneath): spans from baseline 0 up to y=50
    y0 = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    # Series 1 (drawn on top): covers middle section from baseline 0 up to y=40
    y1 = np.array([0.0, 40.0, 40.0, 40.0, 0.0, 0.0])

    ax.fill_between(x, 0, y0, alpha=0.5, label="Series 1")
    ax.fill_between(x, 0, y1, alpha=0.5, label="Series 2")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 60)

    chart_info_map = {
        ax: {
            "chart_type_str": "area",
            "keypoint_info": [
                {
                    "series_idx": 0,
                    "fill_top": [(float(x[i]), float(y0[i])) for i in range(len(x))],
                    "fill_bottom": [(float(x[i]), 0.0) for i in range(len(x))],
                    "stacking_mode": "overlapping",
                },
                {
                    "series_idx": 1,
                    "fill_top": [(float(x[i]), float(y1[i])) for i in range(len(x))],
                    "fill_bottom": [(float(x[i]), 0.0) for i in range(len(x))],
                    "stacking_mode": "overlapping",
                },
            ],
        }
    }

    fig.canvas.draw()
    seg_anns = extract_area_segmentation_annotations(fig, chart_info_map, 800, 600)
    plt.close(fig)

    assert len(seg_anns) == 2
    underlying_ann = seg_anns[0]
    top_ann = seg_anns[1]

    # Verify both amodal and modal structures are clearly exposed
    assert "amodal_polygon" in underlying_ann
    assert "modal_polygon" in underlying_ann
    assert "amodal_area" in underlying_ann
    assert "modal_area" in underlying_ann
    assert "segmentation" in underlying_ann

    amodal_area = underlying_ann["amodal_area"]
    modal_area = underlying_ann["modal_area"]

    # Series 0 is partially covered by Series 1: modal_area must be strictly less than amodal_area
    assert modal_area < amodal_area, f"Expected modal_area ({modal_area}) < amodal_area ({amodal_area})"
    assert underlying_ann["modal_polygon"] != underlying_ann["amodal_polygon"]

    # Topmost series (Series 1) has nothing occluding it from above
    assert top_ann["modal_area"] == pytest.approx(top_ann["amodal_area"], rel=1e-3)


def test_area_polygon_viewport_clipping_shapely_validity():
    """
    Verify area fill polygons clipped by axis spines are non-degenerate, valid Shapely polygons
    without self-intersecting loops or edge foldbacks.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(0.0, 30.0)

    # Series points deliberately extend beyond axis x limits [1.0, 4.0] and y limits [0.0, 30.0]
    x = np.array([-2.0, 0.0, 2.0, 3.0, 5.0, 7.0])
    y_top = np.array([45.0, 50.0, 25.0, 28.0, 60.0, 40.0])  # Exceeds y_max=30.0

    chart_info_map = {
        ax: {
            "chart_type_str": "area",
            "keypoint_info": [
                {
                    "series_idx": 0,
                    "fill_top": [(float(x[i]), float(y_top[i])) for i in range(len(x))],
                    "fill_bottom": [(float(x[i]), 0.0) for i in range(len(x))],
                    "stacking_mode": "single",
                }
            ],
        }
    }

    fig.canvas.draw()
    seg_anns = extract_area_segmentation_annotations(fig, chart_info_map, 600, 500)
    ax_bbox = ax.bbox
    ax_x0, ax_x1 = float(ax_bbox.x0), float(ax_bbox.x1)
    ax_y0 = float(500 - ax_bbox.y1)
    ax_y1 = float(500 - ax_bbox.y0)
    plt.close(fig)

    assert len(seg_anns) == 1
    ann = seg_anns[0]
    poly_pts = ann["polygon"]
    assert len(poly_pts) >= 3

    # Build Shapely polygon and assert validity and non-degeneracy
    shapely_poly = ShapelyPolygon(poly_pts)
    assert shapely_poly.is_valid, "Clipped polygon must be a valid Shapely polygon"
    assert not shapely_poly.is_empty, "Clipped polygon must not be empty"
    assert shapely_poly.area > 10.0, "Clipped polygon must have non-trivial area"
    assert shapely_poly.is_simple, "Clipped polygon must be simple (no self-intersections or bowtie loops)"

    # Verify all coordinates lie strictly within axis viewport bounding box
    for px, py in poly_pts:
        assert ax_x0 - 1e-4 <= px <= ax_x1 + 1e-4, f"Vertex x={px} outside ax_bbox [{ax_x0}, {ax_x1}]"
        assert ax_y0 - 1e-4 <= py <= ax_y1 + 1e-4, f"Vertex y={py} outside ax_bbox [{ax_y0}, {ax_y1}]"
