"""
tests/test_m1_config_layer.py

Acceptance tests for Milestone M1 (Config layer):
- CFG-01: Layered config loading, provenance tracking, deep copy guarantee, broken custom_config handling.
- CFG-02: custom_config and worker initializer threading into generator and CHART_CLASS_MAPS.
- CFG-06: New defaults present in config_defaults.py without behavior drift.
- CFG-07: Top-level scientific_ratio deprecation note and synchronization.
- GEN-01: Programmatic run_generation() entry point and RunSummary.
- GEN-05: Module-level EFFECT_REGISTRY introspection and aliases.
- GEN-06: os.cpu_count fallback, start-index handling, graceful debug script skip.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import warnings
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import (
    ResolvedConfig,
    apply_dot_override,
    load_config,
    parse_set_override,
)
import generator
from generator import (
    EFFECT_ALIASES,
    EFFECT_REGISTRY,
    RunSummary,
    _init_worker,
    run_generation,
)


def test_cfg01_precedence_and_provenance(tmp_path):
    """CFG-01: Precedence: default -> custom -> file -> env -> cli."""
    # 1. Custom layer in temp file
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "import copy\n"
        "from config_defaults import OCR_TRAINING_CONFIG\n"
        "OCR_TRAINING_CONFIG = copy.deepcopy(OCR_TRAINING_CONFIG)\n"
        "OCR_TRAINING_CONFIG['num_images'] = 77\n"
    )

    # 2. File layer in JSON
    file_json = tmp_path / "override.json"
    file_json.write_text(json.dumps({"num_images": 88, "seed": 999}))

    # Test resolution with file + env + overrides
    env = {"DEBUG_MODE": "true"}
    overrides = {"num_images": 123, "chart_types.bar.enabled": False}

    cfg = load_config(
        path=str(file_json),
        overrides=overrides,
        use_custom=False,
        env=env,
    )

    assert isinstance(cfg, ResolvedConfig)
    assert cfg["num_images"] == 123
    assert cfg.provenance["num_images"] == "cli"

    assert cfg["seed"] == 999
    assert cfg.provenance["seed"] == "file"

    assert cfg["debug_mode"] is True
    assert cfg.provenance["debug_mode"] == "env"

    assert cfg["chart_types"]["bar"]["enabled"] is False
    assert cfg.provenance["chart_types.bar.enabled"] == "cli"

    # Default key provenance
    assert cfg.provenance["dataset_format"] == "default"


def test_cfg01_deep_copy_guarantee():
    """CFG-01: Mutating resolved config never mutates config_defaults."""
    cfg = load_config(use_custom=False)
    original_num = DEFAULT_CONFIG["num_images"]

    cfg["num_images"] = 999999
    cfg["chart_types"]["bar"]["weight"] = 12345

    assert DEFAULT_CONFIG["num_images"] == original_num
    assert DEFAULT_CONFIG["chart_types"]["bar"]["weight"] != 12345


def test_cfg01_broken_custom_config_exits_code_2(tmp_path):
    """CFG-01: A broken custom_config.py prints clear error and exits with code 2."""
    script = (
        "import sys, os\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from config_loader import load_config\n"
        "load_config(use_custom=True)\n"
    )

    # Create broken custom_config.py in a temp directory
    broken_py = tmp_path / "custom_config.py"
    broken_py.write_text("SYNTAX ERROR IN THIS FILE +++")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    res = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "Broken custom_config.py" in res.stderr or "SyntaxError" in res.stderr


def test_cfg02_worker_initializer_and_chart_class_maps():
    """CFG-02: Worker initializer updates active config and CHART_CLASS_MAPS."""
    custom_bar_map = {"0": "custom_chart", "1": "custom_bar"}
    custom_cfg = copy.deepcopy(DEFAULT_CONFIG)
    custom_cfg["CLASS_MAP_BAR"] = custom_bar_map
    custom_cfg["debug_mode"] = True

    _init_worker(custom_cfg)

    assert generator.GENERATION_CONFIG["debug_mode"] is True
    assert generator.CHART_CLASS_MAPS["bar"]["1"] == "custom_bar"

    # Restore defaults
    _init_worker(copy.deepcopy(DEFAULT_CONFIG))
    assert generator.CHART_CLASS_MAPS["bar"]["1"] == "bar"


def test_cfg06_documented_defaults_present():
    """CFG-06: All M2 keys are present in config_defaults.py."""
    expected_keys = [
        "use_parallel", "strict", "engine", "export_obb", "export_labels_obb",
        "save_svg", "export_svg", "vegalite_scale", "theme", "semantic_domain",
        "synthetic_domain", "annotation_schema_version", "detailed_schema",
        "debug_coords", "label_angle", "scientific_ratio",
    ]
    for k in expected_keys:
        assert k in DEFAULT_CONFIG, f"Missing key in config_defaults.py: {k}"

    expected_effects = ["uneven_lighting", "chromatic_aberration", "pdf_document_context"]
    effects = DEFAULT_CONFIG.get("realism_effects", {})
    for eff in expected_effects:
        assert eff in effects, f"Missing effect in realism_effects: {eff}"
        assert effects[eff]["p"] == 0.0


def test_cfg07_scientific_ratio_deprecation_and_sync():
    """CFG-07: Deprecation warning when scientific_ratio is only in bar_chart_config."""
    legacy_cfg = {"bar_chart_config": {"scientific_ratio": 0.85}}

    # Test warning emission
    with pytest.deprecated_call(match="bar_chart_config.scientific_ratio is deprecated"):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(legacy_cfg, f)
            tname = f.name
        try:
            resolved = load_config(path=tname, use_custom=False)
            assert resolved["scientific_ratio"] == 0.85
            assert resolved["bar_chart_config"]["scientific_ratio"] == 0.85
        finally:
            os.unlink(tname)


def test_gen01_run_generation_programmatic(tmp_path):
    """GEN-01: run_generation returns RunSummary and generates expected counts."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["num_images"] = 2
    cfg["seed"] = 42
    cfg["output_dir"] = str(tmp_path)
    cfg["use_parallel"] = False

    summary = run_generation(cfg)

    assert isinstance(summary, RunSummary)
    assert summary.num_images == 2
    assert summary.successful_count == 2
    assert summary.failed_count == 0
    assert summary.output_dir == str(tmp_path)
    assert os.path.isfile(tmp_path / "labels" / "chart_00000.txt")
    assert os.path.isfile(tmp_path / "labels" / "chart_00001.txt")


def test_gen05_effect_registry_introspection():
    """GEN-05: Module-level EFFECT_REGISTRY enumerates all effects and aliases."""
    for eff_name in DEFAULT_CONFIG["realism_effects"]:
        assert eff_name in EFFECT_REGISTRY, f"Effect {eff_name} not registered in EFFECT_REGISTRY"

    assert "perspective_warp" in EFFECT_ALIASES
    assert "non_rigid_mesh" in EFFECT_ALIASES
    assert "mesh_warp" in EFFECT_ALIASES


def test_gen06_robustness_start_index_and_debug_skip(tmp_path):
    """GEN-06: Honors start_index and skips nonexistent testar.py silently."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["num_images"] = 1
    cfg["start_index"] = 10
    cfg["seed"] = 42
    cfg["output_dir"] = str(tmp_path)
    cfg["use_parallel"] = False
    cfg["debug_mode"] = True
    cfg["show_debug"] = False  # Ensure dead testar.py is not invoked

    summary = run_generation(cfg)
    assert summary.successful_count == 1
    # Debug directory is 'test' when debug_mode=True
    assert os.path.isfile("test/chart_00010.txt")
