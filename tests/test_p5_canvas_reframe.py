"""
tests/test_p5_canvas_reframe.py

Acceptance tests for Phase 1 Tasks T254 & T255: canvas_reframe effect,
content guard, feature RNG isolation, and config validation (FR-109..FR-113).
"""
import copy
import numpy as np
from PIL import Image
import pytest

from generator import (
    BoundingBox,
    apply_realism_effects,
    _compute_content_guard,
)
from effects import apply_canvas_reframe_effect
from synth.semantics.admin.configspec import validate, Issue
from config_loader import load_config


def test_canvas_reframe_synthetic_exactness():
    """Verify synthetic boxes, polygons, OBBs, and keypoints transform exactly (<= 1e-6)."""
    W, H = 200, 200
    img = Image.new("RGB", (W, H), color=(255, 255, 255))

    # In Matplotlib coordinates: y from bottom.
    # [40, 40, 160, 160] in Matplotlib corresponds to [40, 40, 160, 160] in image coordinates when H=200.
    raw_bbox = BoundingBox(40.0, 40.0, 160.0, 160.0)
    raw_poly = [(40.0, 40.0), (160.0, 40.0), (160.0, 160.0), (40.0, 160.0)]
    raw_obb = [(40.0, 40.0), (160.0, 40.0), (160.0, 160.0), (40.0, 160.0)]
    raw_kpts = [[100.0, 100.0, 2]]

    anns = [
        {
            "class_id": 1,
            "class_name": "data_label",
            "bbox": raw_bbox,
            "polygon": list(raw_poly),
            "obb": list(raw_obb),
        },
        {
            "class_id": 2,
            "class_name": "series_keypoints",
            "keypoints": copy.deepcopy(raw_kpts),
        },
    ]

    # Test Case 1: Uniform pad of 10 px
    # Content box [40, 40, 160, 160] padded by 10 px -> [30, 30, 170, 170]
    # new_w = 140, new_h = 140, dx = -30, dy = 170 - 200 = -30
    effects_cfg = {
        "canvas_reframe": {
            "p": 1.0,
            "params": {
                "p_tight": 1.0,
                "tight_px": (10, 10),
                "margin_frac_range": (0.0, 0.0),
                "content_bbox": (40.0, 40.0, 160.0, 160.0),
                "use_guard": True,
            },
        }
    }

    padded_img, padded_anns = apply_realism_effects(
        img, copy.deepcopy(anns), effects_cfg, seed=123, image_idx=0
    )
    assert padded_img.size == (140, 140)

    ann0 = padded_anns[0]
    b = ann0["bbox"]
    assert abs(b.x0 - 10.0) <= 1e-6
    assert abs(b.y0 - 10.0) <= 1e-6
    assert abs(b.x1 - 130.0) <= 1e-6
    assert abs(b.y1 - 130.0) <= 1e-6

    poly = ann0["polygon"]
    for (ox, oy), (px, py) in zip(raw_poly, poly):
        assert abs(px - (ox - 30.0)) <= 1e-6
        assert abs(py - (oy - 30.0)) <= 1e-6

    obb = ann0["obb"]
    for (ox, oy), (bx, by) in zip(raw_obb, obb):
        assert abs(bx - (ox - 30.0)) <= 1e-6
        assert abs(by - (oy - 30.0)) <= 1e-6

    kpts = padded_anns[1]["keypoints"]
    assert abs(kpts[0][0] - 70.0) <= 1e-6
    assert abs(kpts[0][1] - 70.0) <= 1e-6

    # Test Case 2: Zero-slack crop
    # Content box [40, 40, 160, 160] with 0 margin -> [40, 40, 160, 160]
    # new_w = 120, new_h = 120, dx = -40, dy = 160 - 200 = -40
    effects_cfg_crop = {
        "canvas_reframe": {
            "p": 1.0,
            "params": {
                "p_tight": 1.0,
                "tight_px": (0, 0),
                "margin_frac_range": (0.0, 0.0),
                "content_bbox": (40.0, 40.0, 160.0, 160.0),
                "use_guard": True,
            },
        }
    }
    cropped_img, cropped_anns = apply_realism_effects(
        img, copy.deepcopy(anns), effects_cfg_crop, seed=123, image_idx=0
    )
    assert cropped_img.size == (120, 120)
    cb = cropped_anns[0]["bbox"]
    assert abs(cb.x0 - 0.0) <= 1e-6
    assert abs(cb.y0 - 0.0) <= 1e-6
    assert abs(cb.x1 - 120.0) <= 1e-6
    assert abs(cb.y1 - 120.0) <= 1e-6


def test_canvas_reframe_content_guard_negative_control():
    """
    Verify content guard protects annotations outside ink bounds,
    and use_guard=False allows cropping them off-canvas.
    """
    W, H = 200, 200
    # Create white canvas with a small black ink dot only in the center [90, 90, 110, 110]
    arr = np.full((H, W, 3), 255, dtype=np.uint8)
    arr[90:110, 90:110] = 0
    img = Image.fromarray(arr)

    # Annotation extends well beyond the ink dot to [20, 20, 180, 180]
    raw_bbox = BoundingBox(20.0, 20.0, 180.0, 180.0)
    anns = [{"class_id": 1, "class_name": "data_label", "bbox": raw_bbox}]

    # With use_guard=True: guard encompasses [20, 20, 180, 180]
    effects_guard = {
        "canvas_reframe": {
            "p": 1.0,
            "params": {
                "p_tight": 1.0,
                "tight_px": (0, 0),
                "margin_frac_range": (0.0, 0.0),
                "use_guard": True,
            },
        }
    }
    guarded_img, guarded_anns = apply_realism_effects(
        img, copy.deepcopy(anns), effects_guard, seed=42, image_idx=0
    )
    # The guarded canvas must not crop out the annotation [20, 20, 180, 180]
    assert guarded_img.size[0] >= 160
    assert guarded_img.size[1] >= 160
    gb = guarded_anns[0]["bbox"]
    assert gb.x0 >= 0.0 and gb.y0 >= 0.0
    assert gb.x1 <= guarded_img.size[0] and gb.y1 <= guarded_img.size[1]

    # With use_guard=False: crops only to ink [90, 90, 110, 110]
    effects_no_guard = {
        "canvas_reframe": {
            "p": 1.0,
            "params": {
                "p_tight": 1.0,
                "tight_px": (0, 0),
                "margin_frac_range": (0.0, 0.0),
                "use_guard": False,
            },
        }
    }
    cropped_img, cropped_anns = apply_realism_effects(
        img, copy.deepcopy(anns), effects_no_guard, seed=42, image_idx=0
    )
    # Cropped to ink size ~ 20x20
    assert cropped_img.size[0] <= 30
    assert cropped_img.size[1] <= 30
    # Original annotation [20, 20, 180, 180] is now shifted and off-canvas
    cb = cropped_anns[0]["bbox"]
    assert cb.x0 < 0.0 or cb.x1 > cropped_img.size[0]


def test_canvas_reframe_configspec_validation():
    """Verify FR-112 ordering and parameter limits in config validation."""
    # 1. Valid profile domain_gap_v1
    cfg = load_config("profiles/domain_gap_v1.json")
    issues = [i for i in validate(cfg) if "canvas_reframe" in i.path]
    assert len(issues) == 0, f"Unexpected validation issues: {issues}"

    # 2. Out-of-order: canvas_reframe before geometric effect
    bad_order_cfg = copy.deepcopy(cfg)
    bad_effects = {
        "canvas_reframe": {"p": 0.8},
        "scan_rotation": {"p": 0.1},
    }
    bad_order_cfg["realism_effects"] = bad_effects
    issues = validate(bad_order_cfg)
    order_errors = [i for i in issues if i.level == "error" and "canvas_reframe" in i.path and "must follow geometric effect" in i.msg]
    assert len(order_errors) == 1

    # 3. Out-of-order: canvas_reframe after resize
    bad_resize_cfg = copy.deepcopy(cfg)
    bad_resize_effects = {
        "resize": {"p": 0.5},
        "canvas_reframe": {"p": 0.8},
    }
    bad_resize_cfg["realism_effects"] = bad_resize_effects
    issues = validate(bad_resize_cfg)
    resize_errors = [i for i in issues if i.level == "error" and "canvas_reframe" in i.path and "must precede 'resize'" in i.msg]
    assert len(resize_errors) == 1

    # 4. Out-of-bounds tight_px > 12
    bad_tpx_cfg = copy.deepcopy(cfg)
    bad_tpx_cfg["realism_effects"]["canvas_reframe"]["params"]["tight_px"] = [0, 16]
    issues = validate(bad_tpx_cfg)
    tpx_errors = [i for i in issues if i.level == "error" and "tight_px" in i.path]
    assert len(tpx_errors) == 1

    # 5. Out-of-bounds margin_frac_range > 0.25
    bad_mfr_cfg = copy.deepcopy(cfg)
    bad_mfr_cfg["realism_effects"]["canvas_reframe"]["params"]["margin_frac_range"] = [0.0, 0.40]
    issues = validate(bad_mfr_cfg)
    mfr_errors = [i for i in issues if i.level == "error" and "margin_frac_range" in i.path]
    assert len(mfr_errors) == 1


def _ink_ratio(img: Image.Image, box) -> float:
    W, H = img.size
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    arr = np.asarray(img.convert("L")).astype(int)
    perimeter = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(perimeter)
    crop = arr[y0:y1, x0:x1]
    return float((np.abs(crop - bg) > 25).mean())


@pytest.mark.slow
def test_canvas_reframe_composed_guard_24_images(tmp_path):
    """
    Over >= 24 images with scan_rotation + perspective + page_curl + canvas_reframe (tight_px=[0,0]),
    verify off-canvas OBBs do not exceed control and empty text boxes <= 1% (FR-110, FR-113).
    """
    import glob
    import json
    import random
    from config_defaults import OCR_TRAINING_CONFIG
    from generator import generate_single_chart

    N = 24
    seed = 42

    def run_suite(sub_name, reframe_on):
        run_dir = tmp_path / sub_name
        images_dir = run_dir / "images"
        labels_dir = run_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
        cfg["num_images"] = N
        cfg["seed"] = seed
        cfg["output_dir"] = str(run_dir)
        cfg["debug_mode"] = False
        cfg["use_parallel"] = False

        for k in cfg.get("realism_effects", {}):
            cfg["realism_effects"][k]["p"] = 0.0

        cfg["realism_effects"]["scan_rotation"] = {"p": 1.0, "params": {"angle_range": [-1.5, 1.5]}}
        cfg["realism_effects"]["perspective"] = {"p": 1.0, "params": {"magnitude": [0.02, 0.06]}}
        cfg["realism_effects"]["page_curl"] = {"p": 1.0, "params": {"amplitude_ratio": [0.01, 0.02]}}

        if reframe_on:
            cfg["realism_effects"]["canvas_reframe"] = {
                "p": 1.0,
                "params": {
                    "p_tight": 1.0,
                    "tight_px": [0, 0],
                    "margin_frac_range": [0.0, 0.0],
                    "use_guard": True,
                },
            }

        for i in range(N):
            random.seed(seed + i)
            np.random.seed(seed + i)
            generate_single_chart(i, cfg, str(images_dir), str(labels_dir), str(run_dir))

        det_files = sorted(glob.glob(str(labels_dir / "*_detailed.json")))
        img_files = sorted(glob.glob(str(images_dir / "*.png")))

        off_canvas_obbs = 0
        total_obbs = 0
        total_text = 0
        empty_text = 0

        for df, ip in zip(det_files, img_files):
            with open(df, "r", encoding="utf-8") as f:
                det = json.load(f)
            img = Image.open(ip)
            w, h = img.size

            for a in det.get("annotations", []):
                obb = a.get("obb")
                if obb:
                    total_obbs += 1
                    is_off = any(
                        pt[0] < -1.0 or pt[0] > (w + 1.0) or pt[1] < -1.0 or pt[1] > (h + 1.0)
                        for pt in obb
                    )
                    if is_off:
                        off_canvas_obbs += 1

                if a.get("class_name") in {"title", "x_axis_label", "y_axis_label", "tick_label", "legend"}:
                    box = a.get("xyxy")
                    if box:
                        total_text += 1
                        if _ink_ratio(img, box) < 0.05:
                            empty_text += 1

        empty_pct = (empty_text / max(1, total_text)) * 100.0
        return off_canvas_obbs, total_obbs, empty_pct

    ctrl_off, ctrl_tot, ctrl_empty = run_suite("control", reframe_on=False)
    reframe_off, reframe_tot, reframe_empty = run_suite("reframe", reframe_on=True)

    # FR-113: 0 off-canvas OBBs added relative to control
    assert reframe_off <= ctrl_off, (
        f"canvas_reframe added off-canvas OBBs: reframe={reframe_off}/{reframe_tot} vs control={ctrl_off}/{ctrl_tot}"
    )
    # <= 1% empty text boxes
    assert reframe_empty <= 1.0, f"Empty text boxes {reframe_empty:.2f}% exceeds 1% threshold!"

