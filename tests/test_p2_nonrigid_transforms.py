"""Unit tests verifying Phase 8 Non-Rigid Document Distortions & Equivariant Mesh Warping."""
import math
import os
import shutil
import tempfile
import numpy as np
import pytest
from PIL import Image, ImageDraw
from matplotlib import transforms

import effects
from effects import apply_page_curl
import generator
from generator import (
    BoundingBox,
    apply_realism_effects,
    canonicalize_and_validate_obb,
    generate_single_chart,
)


def test_page_curl_zero_nans_and_smooth_pixel_interpolation():
    """
    Verify apply_page_curl produces continuous, NaN-free, Inf-free pixel grids
    without corrupt borders or channel distortions for both curl axes.
    """
    w, h = 400, 300
    # Create test image with distinct RGB gradient and pattern
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)
    img_arr[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)
    img_arr[:, :, 1] = np.linspace(30, 220, h, dtype=np.uint8)[:, None]
    img_arr[:, :, 2] = 180
    test_img = Image.fromarray(img_arr)

    # 1. Test curl along Y axis (displaces Y based on X)
    warped_y, f_map_y = apply_page_curl(
        test_img,
        curl_axis="y",
        amplitude=15.0,
        wavelength=200.0,
        phase=0.0,
        border_value=(255, 255, 255),
        return_forward_map=True,
    )
    assert isinstance(warped_y, Image.Image)
    assert warped_y.size == (w, h)
    arr_y = np.array(warped_y)
    assert not np.isnan(arr_y).any(), "NaNs detected in Y-axis page curl remap"
    assert not np.isinf(arr_y).any(), "Infs detected in Y-axis page curl remap"
    assert arr_y.shape == (h, w, 3)

    # Verify forward map callable
    pt = np.array([100.0, 150.0])
    warped_pt = f_map_y(pt)
    assert warped_pt.shape == (2,)
    # At x=100 with wavelength=200, phase=0: sin(2*pi*100/200) = sin(pi) ~ 0
    assert abs(warped_pt[0] - 100.0) < 1e-5
    assert abs(warped_pt[1] - 150.0) < 1e-4

    # At x=50: sin(2*pi*50/200) = sin(pi/2) = 1.0 => dy = +15.0
    pt_peak = np.array([50.0, 150.0])
    warped_peak = f_map_y(pt_peak)
    assert abs(warped_peak[0] - 50.0) < 1e-5
    assert abs(warped_peak[1] - 165.0) < 1e-4

    # 2. Test curl along X axis (displaces X based on Y)
    warped_x, f_map_x = apply_page_curl(
        test_img,
        curl_axis="x",
        amplitude=12.0,
        wavelength=150.0,
        phase=0.0,
        border_value=(240, 240, 240),
        return_forward_map=True,
    )
    assert isinstance(warped_x, Image.Image)
    assert warped_x.size == (w, h)
    arr_x = np.array(warped_x)
    assert not np.isnan(arr_x).any(), "NaNs detected in X-axis page curl remap"
    assert not np.isinf(arr_x).any(), "Infs detected in X-axis page curl remap"

    # Test return_forward_map=False returns Image directly
    img_only = apply_page_curl(test_img, return_forward_map=False)
    assert isinstance(img_only, Image.Image)


def test_dense_perimeter_sampling_enclosing_envelope():
    """
    Verify dense perimeter sampling along bounding box edges strictly encloses
    non-linear curl wave peaks where naive 4-corner sampling clips.
    """
    w, h = 600, 400
    img = Image.new("RGB", (w, h), color=(255, 255, 255))

    # Box positioned where corners land at nodes (sin=0) but center hits antinode (sin=1.0)
    # Wavelength = 200, so sin(2*pi*x/200):
    # x=0 -> sin=0; x=50 -> sin=1.0; x=100 -> sin=0; x=200 -> sin=0
    # Let box x range from 0 to 100, y range from 150 to 250 in display coords (matplotlib origin bottom-left)
    # At x=0 and x=100, sin is 0. If only corners are sampled, dy is 0 at both x=0 and x=100.
    # At intermediate point x=50, sin(pi/2) = 1.0, dy is +amplitude!
    amplitude = 25.0
    wavelength = 200.0

    init_bbox = transforms.Bbox.from_extents(0.0, 150.0, 100.0, 250.0)
    box_ann = {"class_id": 0, "bbox": init_bbox}

    effects_cfg = {
        "page_curl": {
            "p": 1.0,
            "params": {
                "curl_axis": "y",
                "amplitude": amplitude,
                "wavelength": wavelength,
                "phase": 0.0,
            },
        }
    }

    out_img, out_anns = apply_realism_effects(img, [box_ann], effects_cfg)
    warped_bbox = out_anns[0]["bbox"]

    # In Matplotlib coords:
    # y_img = h - y_m
    # y_img_warped = y_img + dy => y_m_warped = h - (y_img + dy) = y_m - dy
    # dy is +amplitude at x=50, so y_m_warped decreases by 25 at x=50 (minimum dips to 150 - 25 = 125)
    # A naive 4-corner bounding box would only see corners at x=0, 100 where dy=0, resulting in y0=150!
    # With dense perimeter sampling, y0 must extend down to ~125.0!
    assert warped_bbox.y0 <= 126.0, (
        f"Dense perimeter sampling failed to capture sine wave trough! "
        f"Expected y0 <= 126.0, got {warped_bbox.y0:.2f}"
    )


def test_keypoint_and_polygon_equivariance():
    """
    Verify keypoints and polygon contours deform equivariantly with the
    displacement field within 1.0 px tolerance.
    """
    w, h = 800, 600
    img = Image.new("RGB", (w, h), color=(245, 245, 245))

    amplitude = 20.0
    wavelength = 300.0
    phase = 0.5

    # Known keypoint coordinates in normalized [0, 1] range
    kpts = [
        [0.2, 0.4, 2],
        [0.5, 0.5, 2],
        [0.8, 0.6, 2],
    ]
    kpt_ann = {
        "class_id": 1,
        "keypoints": [list(kp) for kp in kpts],
        "bbox": (0.5, 0.5, 0.6, 0.2),
    }

    # Instance polygon contour in pixel coordinates
    poly = [(150.0, 200.0), (450.0, 200.0), (450.0, 350.0), (150.0, 350.0)]
    poly_ann = {
        "class_id": 2,
        "polygon": list(poly),
        "amodal_polygon": list(poly),
    }

    effects_cfg = {
        "page_curl": {
            "p": 1.0,
            "params": {
                "curl_axis": "y",
                "amplitude": amplitude,
                "wavelength": wavelength,
                "phase": phase,
            },
        }
    }

    out_img, _ = apply_realism_effects(
        img, [], effects_cfg, extra_annotation_sets=[[kpt_ann], [poly_ann]]
    )

    # 1. Verify keypoint displacement
    for orig_kp, warped_kp in zip(kpts, kpt_ann["keypoints"]):
        x_px = orig_kp[0] * w
        y_px = orig_kp[1] * h
        expected_dy = amplitude * math.sin(2.0 * math.pi * x_px / wavelength + phase)
        expected_y_norm = (y_px + expected_dy) / h

        # Transformed keypoint X should remain unchanged for Y-curl
        assert abs(warped_kp[0] - orig_kp[0]) < 1e-4
        # Transformed keypoint Y should match expected displacement within 1.0 pixel (in norm units)
        pixel_diff = abs(warped_kp[1] - expected_y_norm) * h
        assert pixel_diff <= 1.0, (
            f"Keypoint displacement error {pixel_diff:.3f} px exceeds 1.0 px tolerance"
        )

    # 2. Verify polygon vertices deformed and amodal polygon updated
    assert len(poly_ann["polygon"]) >= len(poly)
    for px, py in poly_ann["polygon"]:
        assert 0.0 <= px <= float(w)
        assert 0.0 <= py <= float(h)

    assert "amodal_polygon" in poly_ann
    assert len(poly_ann["amodal_polygon"]) >= len(poly)


def test_obb_non_rigid_curvature_envelope():
    """
    Verify OBB transforms under non-rigid curl into a valid, convex, 4-vertex
    oriented bounding box with clockwise winding enclosing the curved boundary.
    """
    w, h = 600, 400
    img = Image.new("RGB", (w, h), color=(255, 255, 255))

    # OBB rectangle in image space (top-left origin, y down)
    raw_obb = [
        (100.0, 150.0),
        (300.0, 150.0),
        (300.0, 250.0),
        (100.0, 250.0),
    ]
    obb_ann = {"class_id": 3, "obb": list(raw_obb)}

    effects_cfg = {
        "page_curl": {
            "p": 1.0,
            "params": {
                "curl_axis": "y",
                "amplitude": 18.0,
                "wavelength": 250.0,
                "phase": 0.0,
            },
        }
    }

    out_img, out_anns = apply_realism_effects(img, [obb_ann], effects_cfg)
    warped_obb = out_anns[0]["obb"]

    assert len(warped_obb) == 4, f"OBB must have 4 vertices, got {len(warped_obb)}"

    # Validate convexity and clockwise winding order
    validated = canonicalize_and_validate_obb(warped_obb, y_down=True)
    assert validated is not None, "Transformed OBB failed convexity or winding validation"

    # Verify OBB is not degenerate (has non-zero area)
    pts = np.asarray(warped_obb)
    dx = pts[1] - pts[0]
    dy = pts[3] - pts[0]
    approx_area = abs(dx[0] * dy[1] - dx[1] * dy[0])
    assert approx_area > 100.0, f"Transformed OBB area {approx_area:.1f} is degenerate"


def test_generator_end_to_end_page_curl():
    """
    Verify generate_single_chart runs end-to-end with page_curl enabled,
    writing valid image and annotation files with zero regressions.
    """
    tmp_dir = tempfile.mkdtemp(prefix="test_phase8_curl_")
    try:
        import copy
        from custom_config import OCR_TRAINING_CONFIG

        cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
        cfg["num_images"] = 1
        cfg["export_obb"] = True
        cfg["export_labels_obb"] = True
        cfg["scenario_weights"] = {"single": 1.0, "multi": 0.0}
        cfg["chart_types"] = {
            k: {"weight": 1.0 if k == "line" else 0.0, "enabled": (k == "line")}
            for k in cfg["chart_types"]
        }
        cfg["realism_effects"] = {
            "page_curl": {
                "p": 1.0,
                "params": {
                    "curl_axis": "y",
                    "amplitude_ratio": 0.03,
                    "wavelength_ratio": 1.0,
                },
            }
        }

        images_dir = os.path.join(tmp_dir, "images")
        labels_dir = os.path.join(tmp_dir, "labels")
        labels_obb_dir = os.path.join(tmp_dir, "labels_obb")

        # Generate 1 line chart with page curl
        generate_single_chart(0, cfg, images_dir=images_dir, labels_dir=labels_dir, output_dir=tmp_dir)

        # Verify image exists and is readable
        img_path = os.path.join(images_dir, "chart_00000.png")
        assert os.path.exists(img_path), "Generated chart image does not exist"
        img = Image.open(img_path)
        assert img.size[0] > 0 and img.size[1] > 0

        # Verify YOLO standard labels exist and are well-formed
        label_path = os.path.join(labels_dir, "chart_00000.txt")
        assert os.path.exists(label_path), "Generated YOLO label file does not exist"
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) > 0, "YOLO labels should not be empty"
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < bw <= 1.0
            assert 0.0 < bh <= 1.0

        # Verify OBB labels exist and are well-formed
        obb_path = os.path.join(labels_obb_dir, "chart_00000.txt")
        assert os.path.exists(obb_path), "Generated OBB label file does not exist"
        with open(obb_path, "r") as f:
            obb_lines = [l.strip() for l in f if l.strip()]
        assert len(obb_lines) > 0, "OBB labels should not be empty"
        for line in obb_lines:
            parts = line.split()
            assert len(parts) == 9, f"OBB label line should have 9 elements: {line}"
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            for coord in coords:
                assert 0.0 <= coord <= 1.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
