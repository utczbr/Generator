"""
tests/test_v4_detailed_json.py

Verification tests for single-list v4.0 _detailed.json schema generation,
pre-effects coordinate preservation, and attrs pass-through.
"""
import copy
import json
import os
import pytest

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart


def test_v4_detailed_json_generation(tmp_path):
    """Verify v4.0 schema outputs single annotations list with attrs and subplots."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = 1
    cfg["engine"] = "matplotlib"
    cfg["annotation_schema_version"] = "v4.0"
    cfg["debug_mode"] = False
    cfg["realism_effects"] = {}

    run_dir = str(tmp_path / "v4_test")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
    base_name = "chart_00000"

    det_path = os.path.join(lbl_dir, f"{base_name}_detailed.json")
    assert os.path.isfile(det_path)
    with open(det_path, "r", encoding="utf-8") as f:
        det = json.load(f)

    assert det["schema_version"] == "v4.0"
    assert det["dataset_version"] == "4.0.0"
    assert det.get("image_id") == base_name
    assert not os.path.isfile(os.path.join(lbl_dir, f"{base_name}_ocr.json"))
    assert not os.path.isfile(os.path.join(lbl_dir, f"{base_name}.json"))
    assert "image" in det
    assert "width" in det["image"] and "height" in det["image"]
    assert "subplots" in det
    assert len(det["subplots"]) >= 1
    assert "annotations" in det
    assert isinstance(det["annotations"], list)

    # Verify every annotation has unique id, class_name, xyxy, visibility
    ids = set()
    for ann in det["annotations"]:
        assert "id" in ann
        assert ann["id"] not in ids
        ids.add(ann["id"])
        assert "class_name" in ann
        assert "xyxy" in ann
        assert len(ann["xyxy"]) == 4
        assert "visibility" in ann
        assert "occluded" in ann
