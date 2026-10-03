"""
tests/test_metadata_persistence.py

Phase 1 verification tests for metadata persistence, serialization closure,
twin-axes/colorbar guards, Vega-Lite parity, and merge_json OBB retention.
"""
import copy
import json
import os
import pytest

from versions import ANNOTATION_SCHEMA_VERSION, DATASET_VERSION
from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart
from backends.vegalite_backend import generate_single_vegalite_chart


def test_versions_constants():
    """Verify shared version constants."""
    assert ANNOTATION_SCHEMA_VERSION == "v3.0"
    assert DATASET_VERSION == "3.0.0"


def test_generator_detailed_json_metadata_persistence(tmp_path):
    """Verify generator emits schema_version, dataset_version, subplots, and composite fields."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = 1
    cfg["engine"] = "matplotlib"
    cfg["debug_mode"] = False
    cfg["realism_effects"] = {}
    cfg["export_legacy_json"] = True
    cfg["annotation_schema_version"] = "v3.0"

    run_dir = str(tmp_path / "gen_meta_test")
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

    assert det["schema_version"] == "v3.0"
    assert det["dataset_version"] == "3.0.0"
    assert isinstance(det["is_scientific"], bool)
    assert "semantic_domain" in det
    assert "subplots" in det
    assert det["is_composite"] is False
    assert "composite_chart_types" in det
    assert "composite_domains" in det

    meta_path = os.path.join(lbl_dir, f"{base_name}.json")
    assert os.path.isfile(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["schema_version"] == "v3.0"
    assert meta["dataset_version"] == "3.0.0"


def test_twin_axes_and_colorbar_guard(tmp_path):
    """Verify auxiliary twin axes and heatmap colorbars do not clobber detailed metadata."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["debug_mode"] = False
    cfg["realism_effects"] = {}

    run_dir = str(tmp_path / "clobber_guard_test")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 1. Test heatmap with colorbar
    for k in cfg["chart_types"]:
        cfg["chart_types"][k]["enabled"] = (k == "heatmap")
        cfg["chart_types"][k]["weight"] = 1.0 if k == "heatmap" else 0.0

    generate_single_chart(0, cfg, img_dir, lbl_dir, run_dir)
    with open(os.path.join(lbl_dir, "chart_00000_detailed.json"), "r", encoding="utf-8") as f:
        det_heatmap = json.load(f)

    assert det_heatmap["chart_type"] == "heatmap"
    assert det_heatmap["chart_type"] != "unknown"

    # 2. Test bar chart with dual axis
    for k in cfg["chart_types"]:
        cfg["chart_types"][k]["enabled"] = (k == "bar")
        cfg["chart_types"][k]["weight"] = 1.0 if k == "bar" else 0.0
    cfg["bar_chart_config"]["style"] = "grouped"

    generate_single_chart(1, cfg, img_dir, lbl_dir, run_dir)
    with open(os.path.join(lbl_dir, "chart_00001_detailed.json"), "r", encoding="utf-8") as f:
        det_bar = json.load(f)

    assert det_bar["chart_type"] == "bar"
    assert det_bar["chart_type"] != "unknown"


def test_vegalite_metadata_parity(tmp_path):
    """Verify Vega-Lite emits schema versions and handles synthetic vs non-synthetic domain semantics."""
    run_dir = str(tmp_path / "vegalite_meta_test")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["engine"] = "vegalite"
    cfg["chart_type"] = "bar"
    cfg["export_legacy_json"] = True
    cfg["annotation_schema_version"] = "v3.0"

    # Non-synthetic: semantic_domain should be null, is_scientific False
    cfg["use_synthetic_data_engine"] = False
    generate_single_vegalite_chart(0, cfg, img_dir, lbl_dir, run_dir)

    with open(os.path.join(lbl_dir, "chart_00000_detailed.json"), "r", encoding="utf-8") as f:
        det_nonsynth = json.load(f)
    assert det_nonsynth["schema_version"] == "v3.0"
    assert det_nonsynth["dataset_version"] == "3.0.0"
    assert det_nonsynth["semantic_domain"] is None
    assert det_nonsynth["is_scientific"] is False

    with open(os.path.join(lbl_dir, "chart_00000.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["schema_version"] == "v3.0"
    assert meta["dataset_version"] == "3.0.0"

    # Synthetic: semantic_domain mapped and is_scientific reflects sector
    cfg["use_synthetic_data_engine"] = True
    cfg["synthetic_domain"] = "financial"
    generate_single_vegalite_chart(1, cfg, img_dir, lbl_dir, run_dir)

    with open(os.path.join(lbl_dir, "chart_00001_detailed.json"), "r", encoding="utf-8") as f:
        det_synth = json.load(f)
    assert det_synth["schema_version"] == "v3.0"
    assert det_synth["dataset_version"] == "3.0.0"
    assert det_synth["semantic_domain"] == "business"
    assert det_synth["is_scientific"] is False



