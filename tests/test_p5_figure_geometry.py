"""
tests/test_p5_figure_geometry.py

Acceptance tests for Phase 1 Tasks T231 and T231b (FR-086, SC-005):
- figure.size_mode: fixed|randomized|publication
- aspect set {4:3, 16:9, 1:1, 3:4, 5:4, 16:10, 3:2, 7:5}
- width 5.5–9.5 in
- typography scale clamp(sqrt(W·H/35), 0.75, 1.25)
- publication themes sizing and native DPI
- SC-005 holds over 48 images: 0 tight_layout warnings, text collisions <= baseline + 0.10, <= 1% empty text, >= 8 distinct aspect ratios.
"""
import copy
import glob
import itertools
import json
import math
import os
import random
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pytest

from config_defaults import OCR_TRAINING_CONFIG
from generator import generate_single_chart
from themes import PUBLICATION_THEMES


TEXT_CLASSES = {
    "title", "x_axis_label", "y_axis_label", "tick_label", "legend",
    "axis_label", "axis_title", "legend_title", "legend_label", "data_label",
}


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _box_inter(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


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


def test_typography_scale_formula():
    """Verify clamp(sqrt(W·H/35), 0.75, 1.25) clamp behavior."""
    # Baseline 7x5 -> 35 sq in -> 1.0
    s_base = min(max(math.sqrt((7 * 5) / 35.0), 0.75), 1.25)
    assert abs(s_base - 1.0) < 1e-6

    # Very small 3.2 x 2.4 -> 7.68 sq in -> sqrt(7.68/35) = 0.468 -> clamped to 0.75
    s_small = min(max(math.sqrt((3.2 * 2.4) / 35.0), 0.75), 1.25)
    assert s_small == 0.75

    # Large 10 x 8 -> 80 sq in -> sqrt(80/35) = 1.511 -> clamped to 1.25
    s_large = min(max(math.sqrt((10 * 8) / 35.0), 0.75), 1.25)
    assert s_large == 1.25


def test_figure_size_mode_publication(tmp_path):
    """Verify figure.size_mode == 'publication' sets target dimensions and native DPI."""
    run_dir = tmp_path / "publication"
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["output_dir"] = str(run_dir)
    cfg["figure"] = {"size_mode": "publication"}
    cfg["scenario_weights"] = {"single": 1.0, "multi": 0.0}
    for v in cfg.get("realism_effects", {}).values():
        v["p"] = 0.0

    # Test each of the 9 publication themes
    for i, theme_name in enumerate(PUBLICATION_THEMES.keys()):
        cfg_theme = copy.deepcopy(cfg)
        cfg_theme["theme"] = theme_name
        random.seed(100 + i)
        np.random.seed(100 + i)

        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            generate_single_chart(i, cfg_theme, str(images_dir), str(labels_dir), str(run_dir))
            tl_warns = [w for w in warn_list if "tight_layout" in str(w.message)]
            assert len(tl_warns) == 0, f"Theme {theme_name} triggered tight_layout warning!"

        pub_spec = PUBLICATION_THEMES[theme_name]
        expected_w_in, expected_h_in = pub_spec["figure_size"]
        expected_dpi = pub_spec["dpi"]
        expected_w_px = int(round(expected_w_in * expected_dpi))
        expected_h_px = int(round(expected_h_in * expected_dpi))

        img_path = images_dir / f"chart_{i:05d}.png"
        assert img_path.exists()
        img = Image.open(img_path)
        # Check DPI / pixel dimensions match expected within 2 px rounding
        assert abs(img.width - expected_w_px) <= 2, (
            f"Theme {theme_name} width {img.width} != expected {expected_w_px}"
        )
        assert abs(img.height - expected_h_px) <= 2, (
            f"Theme {theme_name} height {img.height} != expected {expected_h_px}"
        )


@pytest.mark.slow
def test_figure_size_mode_randomized_sc005(tmp_path):
    """
    Verify SC-005 over 48 images with figure.size_mode: randomized:
    - 0 tight_layout warnings
    - collisions/img <= baseline (0.00) + 0.10
    - <= 1% empty text boxes (< 2% ink)
    - >= 8 distinct aspect ratios appear
    """
    N = 48
    run_dir = tmp_path / "randomized"
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["output_dir"] = str(run_dir)
    cfg["figure"] = {
        "size_mode": "randomized",
        "aspect_ratios": [[4, 3], [16, 9], [1, 1], [3, 4], [5, 4], [16, 10], [3, 2], [7, 5]],
        "width_range": [5.5, 9.5],
        "dpi_set": [96, 120, 150],
    }
    cfg["scenario_weights"] = {"single": 1.0, "multi": 0.0}
    for v in cfg.get("realism_effects", {}).values():
        v["p"] = 0.0

    tl_warnings = 0
    aspect_ratios = set()
    total_text = 0
    empty_text = 0
    total_collisions = 0

    for i in range(N):
        random.seed(42 + i)
        np.random.seed(42 + i)
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            generate_single_chart(i, cfg, str(images_dir), str(labels_dir), str(run_dir))
            for w in warn_list:
                if "tight_layout" in str(w.message):
                    tl_warnings += 1

        img_path = images_dir / f"chart_{i:05d}.png"
        img = Image.open(img_path)
        ar = round(img.width / img.height, 2)
        aspect_ratios.add(ar)

        det_path = labels_dir / f"chart_{i:05d}_detailed.json"
        with open(det_path, "r", encoding="utf-8") as f:
            det = json.load(f)

        text_anns = [a for a in det.get("annotations", []) if a.get("class_name") in TEXT_CLASSES]
        total_text += len(text_anns)

        for a, b in itertools.combinations(text_anns, 2):
            b_a = a.get("xyxy")
            b_b = b.get("xyxy")
            if b_a and b_b:
                m = min(_box_area(b_a), _box_area(b_b))
                if m > 0 and _box_inter(b_a, b_b) / m > 0.3:
                    total_collisions += 1

        for a in text_anns:
            box = a.get("xyxy")
            if box:
                r = _ink_ratio(img, box)
                if r < 0.02:
                    empty_text += 1

    coll_per_img = total_collisions / N
    empty_pct = (empty_text / max(1, total_text)) * 100.0

    assert tl_warnings == 0, f"Encountered {tl_warnings} tight_layout warnings!"
    assert coll_per_img <= 0.10, f"Collisions per image {coll_per_img:.2f} exceeded threshold 0.10!"
    assert empty_pct <= 1.0, f"Empty text boxes {empty_pct:.2f}% exceeded threshold 1.0%!"
    assert len(aspect_ratios) >= 8, (
        f"Only {len(aspect_ratios)} distinct aspect ratios appeared ({sorted(aspect_ratios)}), "
        f"expected at least 8!"
    )
