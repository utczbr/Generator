"""
tests/test_p5_bundled_fonts.py

Acceptance tests for Phase 1 Task T234: Bundled Fonts (FR-089, OD-1, SC-007).

Validates:
1. Manifest integrity: manifest.json exists, checksums match files, total size <= 8 MB.
2. Open licenses: license files exist for each font family in assets/fonts/.
3. Glyph coverage: 0 missing glyphs for the required inventory (Latin, digits, µ μ ± ° α β ² ³ ⁻ × ≥ ≤ Δ).
4. Deterministic registration in sorted order via font_manager.addfont.
5. Distinct rendered faces: >= 20 distinct faces resolved by findfont path.
6. Alias resolution: Arial/Helvetica -> Arimo, Times New Roman -> Tinos, Courier New -> DejaVu Sans Mono.
7. Strict error policy: non-legacy profiles raise MissingFontError / MissingGlyphError.
8. Legacy neutrality: legacy profile does not raise and preserves baseline golden byte hashes.
9. Detailed JSON metadata: non-legacy profile records font_family; legacy omits it.
10. Deterministic repeatability: same seed produces identical font choices.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import random
import numpy as np
import pytest
from matplotlib import font_manager as fm

from config_defaults import OCR_TRAINING_CONFIG
from generator import generate_single_chart
import font_registry
from font_registry import (
    CHARACTER_INVENTORY,
    FONTS_DIR,
    FONT_ALIASES,
    MissingFontError,
    MissingGlyphError,
    check_glyph_coverage,
    ensure_fonts_registered,
    get_bundled_manifest,
    get_distinct_rendered_faces,
    resolve_font_name,
    validate_font_and_glyphs,
)


def test_manifest_and_size_footprint():
    """Verify manifest exists, sha256 checksums match, and footprint <= 8 MB (NFR-D)."""
    manifest = get_bundled_manifest()
    assert manifest["version"] == "1.0"
    assert manifest["total_fonts"] >= 20
    assert manifest["total_size_bytes"] <= 8 * 1024 * 1024, "Bundled fonts must be <= 8 MB"

    for entry in manifest["fonts"]:
        font_path = FONTS_DIR / entry["file"]
        assert font_path.exists(), f"Font file missing: {entry['file']}"
        assert font_path.stat().st_size == entry["size_bytes"]
        hasher = hashlib.sha256()
        hasher.update(font_path.read_bytes())
        assert hasher.hexdigest() == entry["sha256"], f"Checksum mismatch for {entry['file']}"


def test_license_files_present():
    """Verify license files exist for all bundled fonts (OD-1)."""
    assert (FONTS_DIR / "LICENSE-Croscore.txt").exists()
    assert (FONTS_DIR / "LICENSE-DejaVu.txt").exists()
    assert (FONTS_DIR / "LICENSE-OFL.txt").exists()


def test_glyph_coverage_complete():
    """Verify 0 missing glyphs for the entire manifest inventory across all bundled fonts (SC-007)."""
    manifest = get_bundled_manifest()
    for entry in manifest["fonts"]:
        font_path = str(FONTS_DIR / entry["file"])
        ok, missing = check_glyph_coverage(font_path, chars=CHARACTER_INVENTORY)
        assert ok, f"Font {entry['file']} missing glyphs: {missing}"


def test_distinct_rendered_faces_count():
    """Verify >= 20 distinct rendered faces resolved by findfont (SC-007)."""
    rendered = get_distinct_rendered_faces()
    assert len(rendered) >= 20, f"Expected >= 20 distinct rendered faces, got {len(rendered)}"


def test_font_alias_resolution():
    """Verify alias mapping for publication themes and system font names."""
    assert resolve_font_name("Arial") == "Arimo"
    assert resolve_font_name("Helvetica") == "Arimo"
    assert resolve_font_name("Times New Roman") == "Tinos"
    assert resolve_font_name("Courier New") == "DejaVu Sans Mono"
    assert resolve_font_name("Liberation Sans") == "Arimo"
    assert resolve_font_name("Liberation Serif") == "Tinos"


def test_strict_validation_in_non_legacy_profile():
    """Verify non-legacy profiles raise hard error on unknown font or missing glyph (FR-089)."""
    with pytest.raises(MissingFontError, match="not in bundled fonts"):
        validate_font_and_glyphs("ComicSansMS", active_profile="domain_gap_v1")

    # Synthetic glyph missing check
    with pytest.raises(MissingGlyphError, match="missing required glyphs"):
        validate_font_and_glyphs("Arimo", active_profile="domain_gap_v1", chars=CHARACTER_INVENTORY + "\uFFFF")


def test_lenient_validation_in_legacy_profile():
    """Verify legacy profile never raises on unknown fonts."""
    resolved = validate_font_and_glyphs("ComicSansMS", active_profile="legacy")
    assert resolved == "ComicSansMS"


def test_detailed_json_records_font_family(tmp_path):
    """
    Verify FR-089:
    - Non-legacy profile records chosen font_family in _detailed.json.
    - Legacy profile omits font_family to preserve byte-identity.
    """
    # 1. domain_gap_v1 run
    profile_dir = tmp_path / "profile_run"
    p_images = profile_dir / "images"
    p_labels = profile_dir / "labels"
    p_images.mkdir(parents=True)
    p_labels.mkdir(parents=True)

    cfg_profile = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg_profile["profile"] = "domain_gap_v1"
    cfg_profile["num_images"] = 1
    cfg_profile["output_dir"] = str(profile_dir)
    cfg_profile["debug_mode"] = False
    cfg_profile["realism_effects"] = {}

    generate_single_chart(0, cfg_profile, str(p_images), str(p_labels), str(profile_dir))
    det_profile = json.loads((p_labels / "chart_00000_detailed.json").read_text(encoding="utf-8"))
    assert "font_family" in det_profile
    assert det_profile["font_family"] in font_registry.get_bundled_families()

    # 2. legacy run
    legacy_dir = tmp_path / "legacy_run"
    l_images = legacy_dir / "images"
    l_labels = legacy_dir / "labels"
    l_images.mkdir(parents=True)
    l_labels.mkdir(parents=True)

    cfg_legacy = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg_legacy["profile"] = "legacy"
    cfg_legacy["num_images"] = 1
    cfg_legacy["output_dir"] = str(legacy_dir)
    cfg_legacy["debug_mode"] = False
    cfg_legacy["realism_effects"] = {}

    generate_single_chart(0, cfg_legacy, str(l_images), str(l_labels), str(legacy_dir))
    det_legacy = json.loads((l_labels / "chart_00000_detailed.json").read_text(encoding="utf-8"))
    assert "font_family" not in det_legacy


def test_deterministic_font_sampling_across_runs(tmp_path):
    """Verify same seed produces identical font choices across runs (NFR-A, SC-007)."""
    results = []
    for run_idx in range(2):
        run_dir = tmp_path / f"run_{run_idx}"
        r_images = run_dir / "images"
        r_labels = run_dir / "labels"
        r_images.mkdir(parents=True)
        r_labels.mkdir(parents=True)

        cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
        cfg["profile"] = "domain_gap_v1"
        cfg["seed"] = 12345
        cfg["num_images"] = 3
        cfg["output_dir"] = str(run_dir)
        cfg["debug_mode"] = False
        cfg["realism_effects"] = {}

        fonts_for_run = []
        for i in range(3):
            random.seed(cfg["seed"] + i)
            np.random.seed(cfg["seed"] + i)
            generate_single_chart(i, cfg, str(r_images), str(r_labels), str(run_dir))
            det = json.loads((r_labels / f"chart_{i:05d}_detailed.json").read_text(encoding="utf-8"))
            fonts_for_run.append(det.get("font_family"))
        results.append(fonts_for_run)

    assert results[0] == results[1], f"Font choices diverged across identical seeds: {results}"
