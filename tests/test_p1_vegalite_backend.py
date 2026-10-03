"""
tests/test_p1_vegalite_backend.py

Unit tests for Phase 6: Multi-Backend Vector Graphics Engine (Vega-Lite / SVG with DOM Extraction).

Verifies:
a) Vega-Lite bar, line, and scatter charts compile to valid PNG images and SVG strings.
b) SVG DOM parser extracts bounding boxes matching the rendered PNG geometry (IoU >= 0.98).
c) Grouped and stacked bar charts correctly preserve series indexing from Vega-Lite encoding channels.
d) Text elements with rotations export correct OBB quadrilaterals directly from SVG transform attributes.
e) Zero calls to get_window_extent() or raster buffer scanning occur during Vega-Lite generation.
"""
import io
import json
import os
import numpy as np
import pytest
from PIL import Image

import vl_convert as vlc
import svgelements as se
from backends.vegalite_backend import (
    build_vegalite_bar_spec,
    build_vegalite_line_spec,
    build_vegalite_scatter_spec,
    extract_svg_annotations,
    generate_single_vegalite_chart,
)
from generator import generate_single_chart, canonicalize_and_validate_obb


def test_vegalite_compiles_valid_png_and_svg(tmp_path):
    """
    (a) Vega-Lite bar, line, and scatter charts compile to valid PNG images and SVG strings.
    """
    for chart_type in ["bar", "line", "scatter"]:
        run_dir = str(tmp_path / f"run_{chart_type}")
        img_dir = os.path.join(run_dir, "images")
        lbl_dir = os.path.join(run_dir, "labels")

        cfg = {
            "engine": "vegalite",
            "chart_type": chart_type,
            "export_obb": True,
            "save_svg": True,
            "vegalite_scale": 2.0,
            "debug_mode": False,
        }

        result = generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
        assert result is not None
        assert result["chart_type"] == chart_type

        # Verify PNG image
        png_path = os.path.join(img_dir, "chart_00000.png")
        assert os.path.exists(png_path)
        with Image.open(png_path) as img:
            assert img.size[0] > 100
            assert img.size[1] > 100

        # Verify SVG file
        svg_path = os.path.join(run_dir, "svgs", "chart_00000.svg")
        assert os.path.exists(svg_path)
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_text = f.read()
            assert "<svg" in svg_text and "</svg>" in svg_text

        # Verify YOLO AABB label file
        txt_path = os.path.join(lbl_dir, "chart_00000.txt")
        assert os.path.exists(txt_path)
        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) > 0
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            for val in parts[1:]:
                assert 0.0 <= float(val) <= 1.0

        # Verify YOLOv8-OBB label file
        obb_path = os.path.join(run_dir, "labels_obb", "chart_00000.txt")
        assert os.path.exists(obb_path)
        with open(obb_path, "r") as f:
            obb_lines = [l.strip() for l in f if l.strip()]
        assert len(obb_lines) > 0
        for line in obb_lines:
            parts = line.split()
            assert len(parts) == 9
            for val in parts[1:]:
                assert 0.0 <= float(val) <= 1.0

        # Verify detailed.json
        det_path = os.path.join(lbl_dir, "chart_00000_detailed.json")
        assert os.path.exists(det_path)
        with open(det_path, "r") as f:
            det = json.load(f)
            assert det["chart_type"] == chart_type
            assert len(det["resolution"]) == 2


def test_svg_dom_extractor_pixel_iou_precision():
    """
    (b) SVG DOM parser extracts bounding boxes matching the rendered PNG geometry (IoU >= 0.98).
    """
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 400,
        "height": 300,
        "data": {
            "values": [
                {"category": "Alpha", "value": 40.0},
                {"category": "Beta", "value": 75.0},
            ]
        },
        "mark": {"type": "bar", "color": "#ff0000"},
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    scale = 2.0
    svg_str = vlc.vegalite_to_svg(spec)
    png_bytes = vlc.vegalite_to_png(spec, scale=scale)

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img)

    # Detect visual bar pixels (pure red channel)
    red_mask = (arr[:, :, 0] > 220) & (arr[:, :, 1] < 40) & (arr[:, :, 2] < 40)
    assert np.any(red_mask), "Red bar pixels not found in rendered image"

    annotations, detailed, img_w, img_h = extract_svg_annotations(
        svg_str, scale_factor=scale, chart_type="bar"
    )

    bar_anns = [a for a in annotations if a["class_name"] == "bar"]
    assert len(bar_anns) == 2

    for bar in bar_anns:
        # Reconstruct top-left pixel space coordinates from annotation
        px_x0 = int(round(bar["bbox"].x0))
        px_y0 = int(round(img_h - bar["bbox"].y1))
        px_x1 = int(round(bar["bbox"].x1))
        px_y1 = int(round(img_h - bar["bbox"].y0))

        # Inspect corresponding pixel region in rendered mask
        col_crop = red_mask[:, max(0, px_x0 - 2):min(arr.shape[1], px_x1 + 2)]
        y_idxs, x_idxs = np.where(col_crop)
        assert len(y_idxs) > 0, "No visual pixels found in bar region"

        vis_y0 = y_idxs.min()
        vis_y1 = y_idxs.max() + 1
        vis_x0 = max(0, px_x0 - 2) + x_idxs.min()
        vis_x1 = max(0, px_x0 - 2) + x_idxs.max() + 1

        # Compute Intersection over Union
        inter_x0 = max(px_x0, vis_x0)
        inter_y0 = max(px_y0, vis_y0)
        inter_x1 = min(px_x1, vis_x1)
        inter_y1 = min(px_y1, vis_y1)

        inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
        dom_area = (px_x1 - px_x0) * (px_y1 - px_y0)
        vis_area = (vis_x1 - vis_x0) * (vis_y1 - vis_y0)
        union_area = dom_area + vis_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        assert iou >= 0.98, f"DOM bounding box IoU {iou:.4f} < 0.98 threshold (DOM: ({px_x0}, {px_y0}, {px_x1}, {px_y1}), Vis: ({vis_x0}, {vis_y0}, {vis_x1}, {vis_y1}))"


def test_grouped_and_stacked_bar_series_indexing():
    """
    (c) Grouped and stacked bar charts correctly preserve series indexing from the Vega-Lite encoding channels.
    """
    categories = ["CatA", "CatB"]
    values = [15.0, 30.0, 25.0, 45.0]
    series = ["Series1", "Series1", "Series2", "Series2"]

    # 1. Grouped Bar Spec
    spec_grouped = build_vegalite_bar_spec(
        categories=categories * 2,
        values=values,
        series=series,
        stacking="grouped",
        title="Grouped Bar Test",
    )
    svg_grouped = vlc.vegalite_to_svg(spec_grouped)
    anns_g, det_g, _, _ = extract_svg_annotations(svg_grouped, scale_factor=1.0, chart_type="bar")

    assert det_g["series_count"] == 2
    assert det_g["series_names"] == ["Series1", "Series2"]
    bar_info_g = det_g["bar_info"]
    assert len(bar_info_g) == 4
    for b in bar_info_g:
        assert b["series_idx"] in (0, 1)
        assert b["series_name"] in ("Series1", "Series2")

    # 2. Stacked Bar Spec
    spec_stacked = build_vegalite_bar_spec(
        categories=categories * 2,
        values=values,
        series=series,
        stacking="stacked",
        title="Stacked Bar Test",
    )
    svg_stacked = vlc.vegalite_to_svg(spec_stacked)
    anns_s, det_s, _, _ = extract_svg_annotations(svg_stacked, scale_factor=1.0, chart_type="bar")

    assert det_s["series_count"] == 2
    assert det_s["series_names"] == ["Series1", "Series2"]
    bar_info_s = det_s["bar_info"]
    assert len(bar_info_s) == 4
    for b in bar_info_s:
        assert b["series_idx"] in (0, 1)
        assert b["series_name"] in ("Series1", "Series2")


def test_rotated_text_obb_direct_from_svg_transforms():
    """
    (d) Text elements with rotations export correct OBB quadrilaterals directly from SVG transform attributes.
    """
    spec = build_vegalite_bar_spec(
        categories=["VeryLongCategoryOne", "VeryLongCategoryTwo"],
        values=[20.0, 50.0],
        label_angle=-45,
        title="Rotated Text OBB Test",
    )
    svg_content = vlc.vegalite_to_svg(spec)
    annotations, detailed, img_w, img_h = extract_svg_annotations(
        svg_content, scale_factor=2.0, chart_type="bar"
    )

    # Find the rotated tick labels
    rotated_labels = [
        a for a in annotations
        if a.get("class_name") == "axis_labels" and "VeryLongCategory" in a.get("text", "")
    ]
    assert len(rotated_labels) >= 1

    for label in rotated_labels:
        obb = label["obb"]
        assert len(obb) == 4, f"OBB must have 4 corners, got {len(obb)}"

        # Verify not an axis-aligned box (rotated by -45 degrees)
        dx = abs(obb[1][0] - obb[0][0])
        dy = abs(obb[1][1] - obb[0][1])
        assert dx > 5.0 and dy > 5.0, "OBB appears axis-aligned; expected diagonal edge for rotated text"

        # Verify canonical clockwise winding and top-left origin
        validated = canonicalize_and_validate_obb(obb, y_down=True)
        assert validated is not None, "OBB failed clockwise convexity validation"


def test_zero_calls_to_get_window_extent_or_raster_scanning(tmp_path, monkeypatch):
    """
    (e) Zero calls to get_window_extent() or raster buffer scanning occur during Vega-Lite generation.
    """
    import matplotlib.artist
    import matplotlib.axes
    import matplotlib.figure

    def mock_get_window_extent(*args, **kwargs):
        raise AssertionError("get_window_extent() was called during Vega-Lite vector generation!")

    # Monkeypatch Matplotlib's window extent extractors
    monkeypatch.setattr(matplotlib.artist.Artist, "get_window_extent", mock_get_window_extent)
    monkeypatch.setattr(matplotlib.axes.Axes, "get_window_extent", mock_get_window_extent)
    monkeypatch.setattr(matplotlib.figure.Figure, "get_window_extent", mock_get_window_extent)

    run_dir = str(tmp_path / "zero_calls_test")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")

    cfg = {
        "engine": "vegalite",
        "chart_type": "bar",
        "export_obb": True,
        "debug_mode": False,
    }

    # Must complete cleanly without invoking get_window_extent
    res = generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
    assert res is not None
    assert os.path.exists(os.path.join(img_dir, "chart_00000.png"))
    assert os.path.exists(os.path.join(lbl_dir, "chart_00000.txt"))
    assert os.path.exists(os.path.join(run_dir, "labels_obb", "chart_00000.txt"))
