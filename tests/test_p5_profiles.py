"""
tests/test_p5_profiles.py

Verification tests for Phase 1 Task T225: Profile Loader (FR-083, FR-084).
Validates:
1. Profile resolution precedence and provenance tagging (default -> profile -> custom -> file -> env -> cli).
2. Unknown profile raises clear error and exits with code 2 on CLI.
3. --validate-only --profile domain_gap_v1 prints per-key provenance.
4. Legacy profile omits profile metadata in _detailed.json and run_manifest.json (preserving byte identity).
5. Non-legacy profile (domain_gap_v1) records profile name and config hash in _detailed.json and run_manifest.json.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

from config_defaults import OCR_TRAINING_CONFIG
from config_loader import (
    ResolvedConfig,
    get_available_profiles,
    load_config,
    load_profile_overlay,
)
from generator import generate_single_chart, run_generation


def test_available_profiles_contains_legacy_and_domain_gap_v1():
    """Verify built-in profiles are discoverable."""
    profiles = get_available_profiles()
    assert "legacy" in profiles
    assert "domain_gap_v1" in profiles


def test_unknown_profile_raises_error():
    """Verify unknown profile raises ValueError with clear available list."""
    with pytest.raises(ValueError, match="Unknown profile 'non_existent'"):
        load_config(profile="non_existent", use_custom=False)


def test_profile_precedence_and_provenance(tmp_path):
    """Verify profile overlay precedence: default -> profile -> custom -> file -> env -> cli."""
    # Create a custom profile overlay
    pdir = tmp_path / "profiles"
    pdir.mkdir()
    (pdir / "legacy.json").write_text("{}", encoding="utf-8")
    (pdir / "test_prof.json").write_text(
        json.dumps({
            "num_images": 50,
            "seed": 100,
            "scientific_ratio": 0.8,
            "profile_custom_key": "from_profile",
        }),
        encoding="utf-8",
    )

    # 1. Load with profile only
    cfg = load_config(profile="test_prof", profiles_dir=pdir, use_custom=False)
    assert cfg["profile"] == "test_prof"
    assert cfg.provenance["profile"] == "cli"
    assert cfg["num_images"] == 50
    assert cfg.provenance["num_images"] == "profile"
    assert cfg["profile_custom_key"] == "from_profile"
    assert cfg.provenance["profile_custom_key"] == "profile"
    assert cfg.provenance["dataset_format"] == "default"

    # 2. Override profile setting via file
    file_cfg_path = tmp_path / "file_override.json"
    file_cfg_path.write_text(json.dumps({"num_images": 75}), encoding="utf-8")

    # 3. Override via CLI
    overrides = {"num_images": 99}

    cfg2 = load_config(
        path=file_cfg_path,
        overrides=overrides,
        profile="test_prof",
        profiles_dir=pdir,
        use_custom=False,
    )
    assert cfg2["num_images"] == 99
    assert cfg2.provenance["num_images"] == "cli"
    assert cfg2["seed"] == 100
    assert cfg2.provenance["seed"] == "profile"


def test_cli_validate_only_prints_provenance():
    """Verify generator.py --validate-only --profile domain_gap_v1 prints per-key provenance."""
    cmd = [
        sys.executable,
        "generator.py",
        "--validate-only",
        "--profile",
        "domain_gap_v1",
        "--no-custom",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Configuration valid." in res.stdout
    assert "profile: cli" in res.stdout
    assert "dataset_format: default" in res.stdout


def test_cli_unknown_profile_exits_code_2():
    """Verify generator.py --profile non_existent exits 2 with clear error message."""
    cmd = [
        sys.executable,
        "generator.py",
        "--validate-only",
        "--profile",
        "non_existent_profile",
        "--no-custom",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "[CONFIG ERROR] Unknown profile 'non_existent_profile'" in res.stderr


def test_detailed_json_and_manifest_profile_metadata(tmp_path):
    """
    Verify FR-084:
    - Legacy profile omits profile from _detailed.json and run_manifest.json.
    - Non-legacy profile records profile name and config hash in _detailed.json and run_manifest.json.
    """
    # 1. Legacy run
    legacy_dir = tmp_path / "legacy_run"
    legacy_images = legacy_dir / "images"
    legacy_labels = legacy_dir / "labels"
    legacy_images.mkdir(parents=True)
    legacy_labels.mkdir(parents=True)

    cfg_legacy = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg_legacy["profile"] = "legacy"
    cfg_legacy["num_images"] = 1
    cfg_legacy["output_dir"] = str(legacy_dir)
    cfg_legacy["debug_mode"] = False
    cfg_legacy["realism_effects"] = {}

    generate_single_chart(0, cfg_legacy, str(legacy_images), str(legacy_labels), str(legacy_dir))
    summary_legacy = run_generation(cfg_legacy)

    det_legacy_path = legacy_labels / "chart_00000_detailed.json"
    assert det_legacy_path.exists()
    det_legacy = json.loads(det_legacy_path.read_text(encoding="utf-8"))
    assert "profile" not in det_legacy
    assert "config_hash" not in det_legacy

    manifest_legacy_path = legacy_dir / "run_manifest.json"
    assert manifest_legacy_path.exists()
    manifest_legacy = json.loads(manifest_legacy_path.read_text(encoding="utf-8"))
    assert "profile" not in manifest_legacy

    # 2. domain_gap_v1 run
    profile_dir = tmp_path / "profile_run"
    profile_images = profile_dir / "images"
    profile_labels = profile_dir / "labels"
    profile_images.mkdir(parents=True)
    profile_labels.mkdir(parents=True)

    cfg_profile = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg_profile["profile"] = "domain_gap_v1"
    cfg_profile["num_images"] = 1
    cfg_profile["output_dir"] = str(profile_dir)
    cfg_profile["debug_mode"] = False
    cfg_profile["realism_effects"] = {}

    generate_single_chart(0, cfg_profile, str(profile_images), str(profile_labels), str(profile_dir))
    summary_profile = run_generation(cfg_profile)

    det_profile_path = profile_labels / "chart_00000_detailed.json"
    assert det_profile_path.exists()
    det_profile = json.loads(det_profile_path.read_text(encoding="utf-8"))
    assert det_profile.get("profile") == "domain_gap_v1"
    assert "config_hash" in det_profile
    assert len(det_profile["config_hash"]) == 64

    manifest_profile_path = profile_dir / "run_manifest.json"
    assert manifest_profile_path.exists()
    manifest_profile = json.loads(manifest_profile_path.read_text(encoding="utf-8"))
    assert manifest_profile.get("profile") == "domain_gap_v1"
    assert manifest_profile.get("config_hash") == det_profile["config_hash"]
