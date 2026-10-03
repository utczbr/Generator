import json
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from unittest import mock

from generator import (
    canonicalize_and_validate_obb,
    get_text_obb_and_bbox,
    filter_overlapping_annotations,
    json_default_fallback,
    stroke_to_polygon_ribbon,
    extract_line_segmentation_annotations,
    generate_single_chart,
)


def test_legend_occlusion_culling():
    """
    Test FR-024 / Claim #5:
    Data markers covered by an opaque legend box are flagged as occluded (visibility = 0)
    or culled from primary detection lists.
    """
    cls_map = {1: "legend", 2: "bar", 3: "data_point"}
    
    # Legend box from (100, 100) to (200, 200)
    anns = [
        {"class_id": 1, "class_name": "legend", "bbox": [100.0, 100.0, 200.0, 200.0]},
        # Mark 1: centroid (150, 150) strictly inside legend, 100% coverage
        {"class_id": 3, "class_name": "data_point", "bbox": [140.0, 140.0, 160.0, 160.0]},
        # Mark 2: bar of width 40, height 100 (area 4000) from x=90 to 130, y=100 to 200
        # Overlap with legend [100, 100, 200, 200] is [100, 100, 130, 200] -> area 3000 / 4000 = 75% >= 70%
        {"class_id": 2, "class_name": "bar", "bbox": [90.0, 100.0, 130.0, 200.0]},
        # Mark 3: unoccluded bar at [300, 300, 350, 400]
        {"class_id": 2, "class_name": "bar", "bbox": [300.0, 300.0, 350.0, 400.0]},
    ]
    
    # Pass 1: cull_occluded=False -> retains entries but sets visibility=0 and occluded=True
    filtered_no_cull = filter_overlapping_annotations(anns, cls_map=cls_map, cull_occluded=False)
    occluded_marks = [a for a in filtered_no_cull if a.get("visibility") == 0]
    assert len(occluded_marks) == 2, f"Expected 2 occluded marks, got {len(occluded_marks)}"
    for m in occluded_marks:
        assert m.get("occluded") is True
        
    # Mark 3 must be visible
    unoccluded = [a for a in filtered_no_cull if a.get("class_name") == "bar" and a.get("visibility") != 0]
    assert len(unoccluded) == 1
    assert unoccluded[0]["bbox"] == [300.0, 300.0, 350.0, 400.0]

    # Pass 2: cull_occluded=True -> completely removes occluded marks from the output list
    filtered_culled = filter_overlapping_annotations(anns, cls_map=cls_map, cull_occluded=True)
    culled_classes = [a.get("class_name") for a in filtered_culled]
    assert "data_point" not in culled_classes
    bars = [a for a in filtered_culled if a.get("class_name") == "bar"]
    assert len(bars) == 1
    assert bars[0]["bbox"] == [300.0, 300.0, 350.0, 400.0]


def test_spine_viewport_clipping():
    """
    Test FR-025 / Claim #6:
    Stroke ribbons and markers touching y=0 or x=0 do not export vertices outside ax.bbox.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    img_w, img_h = fig.canvas.get_width_height()

    ax_bbox = ax.bbox
    ax_x0 = float(ax_bbox.x0)
    ax_x1 = float(ax_bbox.x1)
    ax_y0 = float(img_h - ax_bbox.y1)
    ax_y1 = float(img_h - ax_bbox.y0)

    # 1. Stroke ribbon polygon touching spines
    # Points at extremes: (0, 0), (5, 10), (10, 0) in data coords
    pts = [(0, 0), (5, 10), (10, 0)]
    px_pts = []
    for x, y in pts:
        px, py = ax.transData.transform_point((x, y))
        px_pts.append((px, img_h - py))

    # Without clipping, stroke ribbon extends outward by linewidth/2
    ribbon_unclipped = stroke_to_polygon_ribbon(px_pts, linewidth_px=20.0)
    has_out_of_bounds = any(
        px < ax_x0 - 1e-4 or px > ax_x1 + 1e-4 or py < ax_y0 - 1e-4 or py > ax_y1 + 1e-4
        for px, py in ribbon_unclipped
    )
    assert has_out_of_bounds, "Unclipped ribbon should extend outside axes spine"

    # With clip_bbox passed
    ribbon_clipped = stroke_to_polygon_ribbon(
        px_pts, linewidth_px=20.0, clip_bbox=(ax_x0, ax_y0, ax_x1, ax_y1)
    )
    assert len(ribbon_clipped) >= 3
    for px, py in ribbon_clipped:
        assert ax_x0 - 1e-5 <= px <= ax_x1 + 1e-5, f"Vertex x={px} outside ax_bbox [{ax_x0}, {ax_x1}]"
        assert ax_y0 - 1e-5 <= py <= ax_y1 + 1e-5, f"Vertex y={py} outside ax_bbox [{ax_y0}, {ax_y1}]"

    # 2. Extract line segmentation annotations with marker clipping
    chart_info_map = {
        ax: {
            "chart_type_str": "line",
            "keypoint_info": [
                {
                    "series_idx": 0,
                    "plotted_points": [(0, 0), (10, 10)],
                    "linewidth": 4.0,
                    "marker": "o",
                    "markersize": 15.0,
                }
            ],
        }
    }
    seg_anns, marker_anns = extract_line_segmentation_annotations(fig, chart_info_map, img_w, img_h)
    assert len(seg_anns) == 1
    for px, py in seg_anns[0]["polygon"]:
        assert ax_x0 - 1e-5 <= px <= ax_x1 + 1e-5
        assert ax_y0 - 1e-5 <= py <= ax_y1 + 1e-5

    assert len(marker_anns) == 2
    for m in marker_anns:
        mx0, my0, mx1, my1 = m["bbox"]
        assert mx0 >= float(ax_bbox.x0) - 1e-5
        assert my0 >= float(ax_bbox.y0) - 1e-5
        assert mx1 <= float(ax_bbox.x1) + 1e-5
        assert my1 <= float(ax_bbox.y1) + 1e-5

    plt.close(fig)


def test_rotated_text_obb_tighter_than_aabb():
    """
    Test FR-019 / FR-020:
    A 45-degree rotated text label produces an OBB whose area is substantially
    tighter than the default Matplotlib AABB.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    text_artist = ax.text(0.5, 0.5, "Long Rotated Axis Label String", rotation=45, ha="center", va="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    orig_bbox, obb = get_text_obb_and_bbox(text_artist, renderer, img_h=400)
    assert obb is not None
    assert len(obb) == 4

    aabb_area = orig_bbox.width * orig_bbox.height
    
    # Compute polygon area of OBB via shoelace formula
    pts = np.array(obb, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    obb_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # For 45 deg rotation, the inflated bounding box area is typically 2.5x to 6x larger
    # than the tight OBB area (ratio < 0.40)
    area_ratio = obb_area / aabb_area
    assert area_ratio < 0.45, f"OBB area {obb_area} should be substantially tighter than AABB area {aabb_area} (ratio: {area_ratio:.2f})"

    # Convexity and winding order check
    validated_obb = canonicalize_and_validate_obb(obb, y_down=True)
    assert validated_obb is not None
    assert len(validated_obb) == 4

    plt.close(fig)


def test_json_serialization_numpy_types():
    """
    Test FR-020 / FR-027:
    JSON export correctly serializes NumPy int64/float64/bool_ values and Matplotlib bboxes.
    """
    # 1. json_default_fallback scalar conversions
    assert json_default_fallback(np.bool_(True)) is True
    assert isinstance(json_default_fallback(np.bool_(True)), bool)
    assert not isinstance(json_default_fallback(np.bool_(True)), str)

    assert json_default_fallback(np.int64(12345)) == 12345
    assert isinstance(json_default_fallback(np.int64(12345)), int)

    assert json_default_fallback(np.float64(3.14159)) == 3.14159
    assert isinstance(json_default_fallback(np.float64(3.14159)), float)

    arr = np.array([1, 2, 3], dtype=np.int32)
    assert json_default_fallback(arr) == [1, 2, 3]

    bbox = transforms.Bbox.from_extents(10.0, 20.0, 30.0, 40.0)
    assert json_default_fallback(bbox) == [10.0, 20.0, 30.0, 40.0]

    # 2. json.dumps fallback
    payload = {
        "is_valid": np.bool_(False),
        "count": np.int64(99),
        "score": np.float32(0.875),
        "matrix": np.zeros((2, 2), dtype=np.float64),
        "bbox": bbox,
    }
    dumped = json.dumps(payload, default=json_default_fallback)
    loaded = json.loads(dumped)
    assert loaded["is_valid"] is False
    assert loaded["count"] == 99
    assert loaded["score"] == 0.875
    assert loaded["matrix"] == [[0.0, 0.0], [0.0, 0.0]]
    assert loaded["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_worker_process_memory_hygiene():
    """
    Test FR-020 / Worker Process Memory Hygiene:
    Workers explicitly invoke plt.close(fig) upon task completion.
    """
    from generator import GENERATION_CONFIG
    cfg = GENERATION_CONFIG.copy()
    cfg['num_images'] = 1
    cfg['debug_mode'] = False
    cfg['chart_types'] = {'bar': {'enabled': True, 'weight': 1.0}}

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        lbl_dir = os.path.join(tmpdir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        with mock.patch("matplotlib.pyplot.close", wraps=plt.close) as mock_close:
            generate_single_chart(0, cfg, img_dir, lbl_dir, tmpdir)
            assert mock_close.called, "plt.close was not called"
            # Ensure at least one call was with a matplotlib Figure instance
            figure_calls = [
                call.args[0] for call in mock_close.call_args_list
                if call.args and hasattr(call.args[0], "canvas")
            ]
            assert len(figure_calls) > 0, "plt.close(fig) was not explicitly called with a Figure instance"

        # Verify no open figures remain in Gcf
        assert len(plt.get_fignums()) == 0, f"Figures leaked in Gcf registry: {plt.get_fignums()}"
