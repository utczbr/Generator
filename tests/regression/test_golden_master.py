"""Regression tests verifying Golden Master baseline determinism.

Executes Tier 1 (Platform-Agnostic JSON invariance) and
Tier 2 (Environment-Guarded headless Linux raster diff) across all 8 golden fixtures.
"""
import copy
import json
import os
import shutil
import pytest

from tests.regression.generate_golden import (
    FIXTURES_DIR,
    FIXTURE_SPECS,
    generate_fixture,
)
from tests.regression.harness import (
    verify_tier1_annotations,
    verify_tier2_raster,
)


@pytest.mark.parametrize("spec", FIXTURE_SPECS, ids=[s["name"] for s in FIXTURE_SPECS])
def test_golden_master_fixture_regression(spec, tmp_path):
    """Test each golden fixture against a fresh run with identical seed and config."""
    base_filename = f"chart_{spec['fixture_id']:05d}"
    target_dir = str(tmp_path)

    # 1. Run generation into isolated temporary directory
    generate_fixture(spec, target_dir)

    # 2. Tier 1 Verification (Platform-Agnostic)
    diffs = verify_tier1_annotations(
        golden_labels_dir=os.path.join(FIXTURES_DIR, "labels"),
        test_labels_dir=os.path.join(target_dir, "labels"),
        base_filename=base_filename,
    )
    if diffs:
        report = "\n  - ".join(str(d) for d in diffs)
        pytest.fail(f"Tier 1 regression failed for {spec['name']} ({len(diffs)} diffs):\n  - {report}")

    # 3. Tier 2 Verification (Environment-Guarded)
    passed, msg, metrics = verify_tier2_raster(
        golden_images_dir=os.path.join(FIXTURES_DIR, "images"),
        test_images_dir=os.path.join(target_dir, "images"),
        base_filename=base_filename,
    )
    if not passed:
        pytest.fail(f"Tier 2 raster check failed for {spec['name']}: {msg}\nMetrics: {metrics}")


def test_harness_mutation_reporting(tmp_path):
    """Verify harness assertion mode pinpoints exact field mutations (T006 readiness)."""
    base_filename = "chart_00000"
    golden_labels = os.path.join(FIXTURES_DIR, "labels")
    test_labels = tmp_path / "labels"
    test_labels.mkdir(parents=True, exist_ok=True)

    # Copy files
    for suffix in ["_detailed.json", "_ocr.json", ".json", ".txt"]:
        src = os.path.join(golden_labels, f"{base_filename}{suffix}")
        if os.path.exists(src):
            shutil.copy(src, test_labels / f"{base_filename}{suffix}")

    # Intentionally mutate one specific field in detailed.json
    det_path = test_labels / f"{base_filename}_detailed.json"
    with open(det_path, "r") as f:
        data = json.load(f)

    original_class = data["raw_annotations"][0]["class_name"]
    data["raw_annotations"][0]["class_name"] = "deliberate_mutation_test"

    with open(det_path, "w") as f:
        json.dump(data, f)

    # Run Tier 1 verification
    diffs = verify_tier1_annotations(
        golden_labels_dir=golden_labels,
        test_labels_dir=str(test_labels),
        base_filename=base_filename,
    )

    # Assert exactly that field was identified
    assert len(diffs) == 1, f"Expected 1 diff, got {len(diffs)}: {[str(d) for d in diffs]}"
    assert diffs[0].field == "raw_annotations[0].class_name"
    assert diffs[0].expected == original_class
    assert diffs[0].actual == "deliberate_mutation_test"
