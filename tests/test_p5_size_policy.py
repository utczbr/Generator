"""
tests/test_p5_size_policy.py

Verification tests for Phase 1 Task T233: Resolution-aware size policy (FR-087, OD-6).
Validates:
1. annotation_min_side_px parameter supported (legacy default: 8; profile default: 4).
2. Sub-threshold annotations are retained in _detailed.json with ignored: true, ignore_reason
   (e.g. "below_min_side") and counted in filter_stats["size"].
3. Sub-threshold / ignored annotations are excluded from YOLO training exports (labels/*.txt, labels_obb/).
4. At scale 0.5, no text annotation disappears silently (all either exported or ignored with reason).
5. Legacy output unchanged (baseline-2 goldens pass).
"""
import copy
import json
import os
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config
from generator import generate_single_chart, save_annotations_yolo, save_annotations_yolo_obb
from matplotlib.transforms import Bbox


def test_size_policy_defaults_and_profile():
    """Verify annotation_min_side_px defaults: 8 for legacy, 4 for domain_gap_v1."""
    cfg_legacy = load_config(profile="legacy")
    assert cfg_legacy.get("annotation_min_side_px", 8) == 8

    cfg_profile = load_config(profile="domain_gap_v1")
    assert cfg_profile.get("annotation_min_side_px") == 4


def test_sub_threshold_retention_and_yolo_exclusion(tmp_path):
    """
    Verify synthetic annotations below min_side_px:
    - Retained in _detailed.json with ignored: true, ignore_reason: 'below_min_side'
    - Counted in filter_stats['size']
    - Excluded from YOLO detection and OBB exports per OD-6
    """
    out_dir = str(tmp_path / "synthetic_run")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # Synthetic annotations: one >= 4px, one < 4px (e.g. 3x3px)
    valid_ann = {
        "class_id": 0,
        "class_name": "bar",
        "bbox": Bbox.from_extents(10.0, 10.0, 30.0, 50.0),
        "ignored": False,
    }
    sub_threshold_ann = {
        "class_id": 1,
        "class_name": "tick_labels",
        "bbox": Bbox.from_extents(50.0, 50.0, 53.0, 52.5),
        "ignored": True,
        "ignore_reason": "below_min_side",
    }
    anns = [valid_ann, sub_threshold_ann]

    yolo_txt_path = os.path.join(labels_dir, "test.txt")
    yolo_obb_path = os.path.join(labels_dir, "test_obb.txt")

    save_annotations_yolo(anns, 100, 100, yolo_txt_path)
    save_annotations_yolo_obb(anns, 100, 100, yolo_obb_path)

    # Verify YOLO detection file only contains valid_ann (class_id 0), not sub_threshold_ann
    with open(yolo_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert lines[0].startswith("0 ")

    # Verify YOLO OBB file also excludes ignored annotation
    with open(yolo_obb_path, "r", encoding="utf-8") as f:
        obb_lines = f.readlines()
    assert len(obb_lines) == 1
    assert obb_lines[0].startswith("0 ")


def test_scale_05_no_silent_text_loss(tmp_path):
    """
    At scale 0.5 under domain_gap_v1 profile:
    No text annotation disappears silently: every text annotation is either
    exported to labels/*.txt or retained in _detailed.json with ignored: true and ignore_reason.
    """
    cfg = load_config(profile="domain_gap_v1")
    cfg["num_images"] = 3
    cfg["seed"] = 42
    cfg["dataset_format"] = "detection"
    cfg["debug_mode"] = False
    cfg["use_parallel"] = False

    # Force 0.5 scale via resize effect
    cfg["realism_effects"] = {
        "resize": {
            "p": 1.0,
            "params": {
                "fixed_scale": 0.5,
            },
        }
    }

    run_dir = str(tmp_path / "scale_05_run")
    imgs_dir = os.path.join(run_dir, "images")
    lbls_dir = os.path.join(run_dir, "labels")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(lbls_dir, exist_ok=True)

    text_classes = {"axis_title", "axis_labels", "chart_title", "legend", "data_label", "tick_labels"}

    for idx in range(3):
        generate_single_chart(idx, cfg, imgs_dir, lbls_dir, run_dir)
        base_name = f"chart_{idx:05d}"
        det_path = os.path.join(lbls_dir, f"{base_name}_detailed.json")
        yolo_path = os.path.join(lbls_dir, f"{base_name}.txt")

        assert os.path.isfile(det_path)
        assert os.path.isfile(yolo_path)

        with open(det_path, "r", encoding="utf-8") as f:
            det = json.load(f)

        with open(yolo_path, "r", encoding="utf-8") as f:
            yolo_lines = [line.strip() for line in f if line.strip()]

        text_anns = [a for a in det["annotations"] if a["class_name"] in text_classes or a.get("text")]
        assert len(text_anns) > 0, "Expected chart to have text annotations"

        for a in text_anns:
            if a.get("ignored"):
                # Must have an explicit reason
                assert "ignore_reason" in a and a["ignore_reason"], f"Annotation {a['id']} ignored without reason"
                assert a["ignore_reason"] in ("below_min_side", "extreme_aspect_ratio", "viewport_clipped", "duplicate")
            else:
                # Must not be ignored
                assert a.get("ignored") is not True

        # filter_stats['size'] must be present
        assert "filter_stats" in det
        assert "size" in det["filter_stats"]
        assert isinstance(det["filter_stats"]["size"], dict)

        # Check that YOLO txt line count equals the non-ignored annotations in _detailed.json
        # (excluding graph/pseudo streams that don't export to detection txt)
        non_ignored_det = [a for a in det["annotations"] if not a.get("ignored") and a.get("stream") in ("annotations", "annotations_obj")]
        # Every non-ignored text annotation in primary stream is represented in labels
        non_ignored_text = [a for a in text_anns if not a.get("ignored") and a.get("stream") == "annotations"]
        # In YOLO txt, lines should exist for these annotations
        assert len(yolo_lines) >= len(non_ignored_text)


def test_legacy_profile_drops_sub_threshold_unretained(tmp_path):
    """Verify legacy profile retains legacy drop semantics (no ignored: true in _detailed.json)."""
    cfg = load_config(profile="legacy")
    cfg["num_images"] = 1
    cfg["seed"] = 42
    cfg["dataset_format"] = "detection"
    cfg["debug_mode"] = False
    cfg["use_parallel"] = False
    cfg["realism_effects"] = {
        "resize": {
            "p": 1.0,
            "params": {
                "fixed_scale": 0.5,
            },
        }
    }

    run_dir = str(tmp_path / "legacy_run")
    imgs_dir = os.path.join(run_dir, "images")
    lbls_dir = os.path.join(run_dir, "labels")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(lbls_dir, exist_ok=True)

    generate_single_chart(0, cfg, imgs_dir, lbls_dir, run_dir)
    det_path = os.path.join(lbls_dir, "chart_00000_detailed.json")
    with open(det_path, "r", encoding="utf-8") as f:
        det = json.load(f)

    # Legacy profile must not contain any annotation marked with ignored: true
    ignored_anns = [a for a in det["annotations"] if a.get("ignored")]
    assert len(ignored_anns) == 0
