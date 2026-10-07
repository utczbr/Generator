"""
tests/test_p5_resize_effect.py

Acceptance tests for Phase 1 Task T232: resize effect (FR-085).
Verifies:
1. Exact mathematical scaling of synthetic bounding boxes, OBBs, polygons, and keypoints.
2. Correct transform composition order when composed with scan_rotation and perspective.
3. Ink preservation across 24 images at scales 0.35–2.0 (<= 1% empty text boxes for s >= 0.5).
"""
import copy
import glob
import json
import math
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest
import matplotlib.transforms as transforms

from config_defaults import OCR_TRAINING_CONFIG
from generator import (
    BoundingBox,
    EFFECT_REGISTRY,
    apply_realism_effects,
    generate_single_chart,
)
from effects import apply_resize_effect


TEXT_CLASSES = {
    "title", "x_axis_label", "y_axis_label", "tick_label", "legend",
    "axis_label", "axis_title", "legend_title", "legend_label", "data_label",
}


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


def test_resize_synthetic_scaling():
    """Verify synthetic boxes, polygons, OBBs, and keypoints scale exactly."""
    W, H = 200, 200
    img = Image.new("RGB", (W, H), color=(255, 255, 255))

    # Matplotlib coordinates: y from bottom. (20, 30, 80, 120) -> PIL y in [200-120, 200-30] = [80, 170]
    raw_bbox = BoundingBox(20.0, 30.0, 80.0, 120.0)
    raw_poly = [(20.0, 80.0), (80.0, 80.0), (80.0, 170.0), (20.0, 170.0)]
    raw_obb = [(20.0, 80.0), (80.0, 80.0), (80.0, 170.0), (20.0, 170.0)]
    raw_kpts = [[0.25, 0.50, 2], [50.0, 125.0, 2]]

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

    # Scale 2.0x
    effects_cfg = {
        "resize": {
            "p": 1.0,
            "params": {"scale": 2.0},
        }
    }
    scaled_img, scaled_anns = apply_realism_effects(img, copy.deepcopy(anns), effects_cfg)
    assert scaled_img.size == (400, 400)

    ann_box = scaled_anns[0]
    b = ann_box["bbox"]
    assert pytest.approx(b.x0, rel=1e-5) == 40.0
    assert pytest.approx(b.y0, rel=1e-5) == 60.0
    assert pytest.approx(b.x1, rel=1e-5) == 160.0
    assert pytest.approx(b.y1, rel=1e-5) == 240.0

    poly = ann_box["polygon"]
    assert len(poly) == 4
    for orig_pt, pt in zip(raw_poly, poly):
        assert pytest.approx(pt[0], rel=1e-5) == orig_pt[0] * 2.0
        assert pytest.approx(pt[1], rel=1e-5) == orig_pt[1] * 2.0

    obb = ann_box["obb"]
    assert len(obb) == 4
    for orig_pt, pt in zip(raw_obb, obb):
        assert pytest.approx(pt[0], rel=1e-5) == orig_pt[0] * 2.0
        assert pytest.approx(pt[1], rel=1e-5) == orig_pt[1] * 2.0

    ann_pose = scaled_anns[1]
    kpts = ann_pose["keypoints"]
    # Normalized keypoint remains invariant [0.25, 0.50]
    assert pytest.approx(kpts[0][0], rel=1e-5) == 0.25
    assert pytest.approx(kpts[0][1], rel=1e-5) == 0.50
    # Pixel keypoint scales 2.0x
    assert pytest.approx(kpts[1][0], rel=1e-5) == 100.0
    assert pytest.approx(kpts[1][1], rel=1e-5) == 250.0

    # Scale 0.5x
    effects_cfg_down = {
        "resize": {
            "p": 1.0,
            "params": {"scale": 0.5},
        }
    }
    down_img, down_anns = apply_realism_effects(img, copy.deepcopy(anns), effects_cfg_down)
    assert down_img.size == (100, 100)
    ann_down = down_anns[0]
    b_down = ann_down["bbox"]
    assert pytest.approx(b_down.x0, rel=1e-5) == 10.0
    assert pytest.approx(b_down.y0, rel=1e-5) == 15.0
    assert pytest.approx(b_down.x1, rel=1e-5) == 40.0
    assert pytest.approx(b_down.y1, rel=1e-5) == 60.0


def test_resize_composition_order():
    """Verify composition order with scan_rotation and perspective."""
    W, H = 200, 200
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(0, 0, 0))

    anns = [{
        "class_id": 1,
        "class_name": "data_label",
        "bbox": BoundingBox(50.0, 50.0, 150.0, 150.0),
        "polygon": [(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)],
        "obb": [(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)],
    }]

    effects_cfg = {
        "scan_rotation": {"p": 1.0, "params": {"angle_range": [2.0, 2.0]}},
        "perspective": {"p": 1.0, "params": {"magnitude": 0.05}},
        "resize": {"p": 1.0, "params": {"scale": 1.5}},
    }

    out_img, out_anns = apply_realism_effects(img, copy.deepcopy(anns), effects_cfg)
    assert out_img.size == (300, 300)
    res_ann = out_anns[0]
    b = res_ann["bbox"]
    # Final bbox should expand roughly by 1.5x with rotation/perspective dilation
    assert b.x1 > b.x0 > 0
    assert b.y1 > b.y0 > 0
    assert pytest.approx((b.x1 - b.x0), rel=0.25) == 100.0 * 1.5
    assert pytest.approx((b.y1 - b.y0), rel=0.25) == 100.0 * 1.5


def test_resize_ink_preservation_24_images(tmp_path):
    """Generate 24 images at scales 0.35-2.0; assert <= 1% empty text for s >= 0.5."""
    N = 24
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    scales = [0.35, 0.45, 0.50, 0.60, 0.75, 0.85, 1.0, 1.25, 1.5, 1.75, 2.0, 0.50] * 2
    scales = scales[:N]

    total_text_ge05 = 0
    empty_text_ge05 = 0

    for i in range(N):
        scale = scales[i]
        cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
        cfg["num_images"] = 1
        cfg["seed"] = 100 + i
        cfg["output_dir"] = str(tmp_path)
        cfg["debug_mode"] = False
        cfg["use_parallel"] = False

        # Disable all effects except resize
        for k in cfg.get("realism_effects", {}):
            cfg["realism_effects"][k]["p"] = 0.0

        cfg["realism_effects"]["resize"] = {
            "p": 1.0,
            "params": {"scale": scale},
        }

        random.seed(100 + i)
        np.random.seed(100 + i)
        generate_single_chart(i, cfg, str(images_dir), str(labels_dir), str(tmp_path))

        img_path = images_dir / f"chart_{i:05d}.png"
        det_path = labels_dir / f"chart_{i:05d}_detailed.json"
        assert img_path.exists(), f"Missing image {img_path}"
        assert det_path.exists(), f"Missing detailed JSON {det_path}"

        img = Image.open(img_path)
        with open(det_path, "r", encoding="utf-8") as f:
            det = json.load(f)

        anns = det.get("annotations", [])
        text_anns = [a for a in anns if a.get("class_name") in TEXT_CLASSES]

        if scale >= 0.5:
            for a in text_anns:
                box = a.get("xyxy")
                if box:
                    total_text_ge05 += 1
                    ink = _ink_ratio(img, box)
                    if ink < 0.05:
                        empty_text_ge05 += 1

    empty_pct = (empty_text_ge05 / max(1, total_text_ge05)) * 100.0
    assert total_text_ge05 > 0, "No text annotations collected for s >= 0.5!"
    assert empty_pct <= 1.0, (
        f"Empty text rate {empty_pct:.2f}% ({empty_text_ge05}/{total_text_ge05}) "
        f"exceeded 1.0% threshold for s >= 0.5!"
    )
