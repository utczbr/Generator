"""
tests/test_p5_effects_integrity.py

Acceptance tests for Phase 1 Task T229: Effects-Integrity Gate (FR-093).
Marked as @pytest.mark.slow.

For each effect enabled in profile domain_gap_v1:
Forced p=1.0 over >= 16 images:
- >= 98% annotations preserved relative to clean baseline.
- <= 1% text boxes with < 5% ink.

Plus:
- Composed case (all domain_gap_v1 effects enabled with their profile parameters): passes gate.
- Negative control: perspective magnitude 0.5 fails the gate (research §4: -19% preserved, 4.6% empty text).
"""
import copy
import glob
import json
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from config_defaults import OCR_TRAINING_CONFIG
from generator import generate_single_chart


TEXT_CLASSES = {
    "title", "x_axis_label", "y_axis_label", "tick_label", "legend",
    "axis_label", "axis_title", "legend_title", "legend_label",
}


def _ink_ratio(img: Image.Image, box) -> float:
    W, H = img.size
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    arr = np.asarray(img.convert("L")).astype(int)
    # Estimate background from canvas perimeter
    perimeter = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(perimeter)
    # Ink is pixels differing from background by > 25 intensity levels
    crop = arr[y0:y1, x0:x1]
    return float((np.abs(crop - bg) > 25).mean())


def _run_condition(tmp_path: Path, n_images: int, effects_dict: dict, seed: int = 42):
    run_dir = tmp_path / "run"
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = n_images
    cfg["seed"] = seed
    cfg["output_dir"] = str(run_dir)
    cfg["debug_mode"] = False
    cfg["use_parallel"] = False

    # Disable all effects first
    for k in cfg.get("realism_effects", {}):
        cfg["realism_effects"][k]["p"] = 0.0

    # Overlay specified effects
    for k, v in effects_dict.items():
        if k not in cfg["realism_effects"]:
            cfg["realism_effects"][k] = {}
        cfg["realism_effects"][k].update(v)

    for i in range(n_images):
        random.seed(seed + i)
        np.random.seed(seed + i)
        generate_single_chart(i, cfg, str(images_dir), str(labels_dir), str(run_dir))

    det_files = sorted(glob.glob(str(labels_dir / "*_detailed.json")))
    img_files = sorted(glob.glob(str(images_dir / "*.png")))

    total_ann = 0
    total_text = 0
    empty_text = 0

    for df, img_path in zip(det_files, img_files):
        with open(df, "r", encoding="utf-8") as f:
            det = json.load(f)
        img = Image.open(img_path)
        anns = det.get("annotations", [])
        total_ann += len(anns)
        for a in anns:
            if a.get("class_name") in TEXT_CLASSES:
                box = a.get("xyxy")
                if box:
                    total_text += 1
                    r = _ink_ratio(img, box)
                    if r < 0.05:
                        empty_text += 1

    empty_pct = (empty_text / max(1, total_text)) * 100.0
    return total_ann, total_text, empty_pct


@pytest.mark.slow
def test_p5_effects_integrity_gate(tmp_path):
    """
    Run FR-093 effects integrity gate:
    1. Baseline clean run (N=16).
    2. Single-effect forced p=1 runs for all effects in domain_gap_v1.
    3. Composed domain_gap_v1 run.
    4. Negative control: perspective 0.5 must fail gate.
    """
    N = 16
    base_dir = tmp_path / "baseline"
    base_ann, base_text, base_empty = _run_condition(base_dir, N, {})
    assert base_ann > 0, "Baseline produced 0 annotations!"

    # Load domain_gap_v1 profile
    prof_path = Path("profiles/domain_gap_v1.json")
    with open(prof_path, "r", encoding="utf-8") as f:
        prof_data = json.load(f)
    prof_effects = prof_data.get("realism_effects", {})

    # Test each enabled effect individually with forced p=1.0
    for eff_name, eff_cfg in prof_effects.items():
        eff_dir = tmp_path / f"eff_{eff_name}"
        forced_eff = {
            eff_name: {
                "p": 1.0,
                "params": copy.deepcopy(eff_cfg.get("params", {})),
            }
        }
        eff_ann, eff_text, eff_empty = _run_condition(eff_dir, N, forced_eff)

        preservation_rate = (eff_ann / base_ann) * 100.0
        assert preservation_rate >= 98.0, (
            f"Effect '{eff_name}' preserved only {preservation_rate:.1f}% annotations "
            f"({eff_ann}/{base_ann}), below 98% gate threshold!"
        )
        assert eff_empty <= 1.0, (
            f"Effect '{eff_name}' had {eff_empty:.2f}% empty text boxes (<5% ink), "
            f"above 1% gate threshold!"
        )

    # Test composed case (all domain_gap_v1 effects with profile parameters)
    composed_dir = tmp_path / "composed"
    composed_ann, composed_text, composed_empty = _run_condition(composed_dir, N, prof_effects)
    composed_preservation = (composed_ann / base_ann) * 100.0
    assert composed_preservation >= 98.0, (
        f"Composed profile preserved only {composed_preservation:.1f}% annotations, below 98%!"
    )
    assert composed_empty <= 1.0, (
        f"Composed profile had {composed_empty:.2f}% empty text boxes, above 1%!"
    )

    # Negative control: perspective 0.5 must fail the gate
    neg_dir = tmp_path / "neg_perspective_05"
    neg_eff = {
        "perspective": {
            "p": 1.0,
            "params": {"magnitude": 0.50},
        }
    }
    neg_ann, neg_text, neg_empty = _run_condition(neg_dir, N, neg_eff)
    neg_preservation = (neg_ann / base_ann) * 100.0
    gate_failed = (neg_preservation < 98.0) or (neg_empty > 1.0)
    assert gate_failed, (
        f"Negative control (perspective 0.5) unexpectedly passed the gate! "
        f"Preserved: {neg_preservation:.1f}%, Empty: {neg_empty:.2f}%"
    )
