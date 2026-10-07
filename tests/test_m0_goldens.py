"""
tests/test_m0_goldens.py

Milestone M0 Safety Net characterization tests.
Validates:
1. SemanticRegistry catalogs match the baseline golden snapshot
   (SEM-03 golden criteria: LEGACY_METRICS_CATALOG, CANONICAL_INDEPENDENT_METRICS,
   HISTOGRAM_Y_LABELS, LEGACY_METRIC_BY_LABEL, legacy pairs identical in content and order).
2. Deterministic 20-image seeded generation hash against baseline snapshot.
   Supports --update-goldens flag (or UPDATE_GOLDENS=1 env var) to generate baseline-2 goldens.
"""
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import pytest

from config_defaults import OCR_TRAINING_CONFIG
from generator import generate_single_chart
from synth.semantics.loader import registry
from scripts.env_fingerprint import get_env_fingerprint

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
REGISTRY_BASELINE_PATH = GOLDEN_DIR / "registry_catalogs_baseline.json"
SEEDED_BASELINE_PATH = GOLDEN_DIR / "seeded_generation_20_baseline.json"


def _should_update(pytestconfig=None) -> bool:
    if pytestconfig is not None:
        try:
            if pytestconfig.getoption("--update-goldens", default=False):
                return True
        except Exception:
            pass
    return bool(os.environ.get("UPDATE_GOLDENS"))


def _metric_to_dict(m):
    return {
        "label": m.label,
        "scale_type": m.scale_type.value,
        "role": m.role.value,
        "concept_stem": m.concept_stem,
        "unit": m.unit,
        "concept_group": m.concept_group,
        "domain_tags": list(m.domain_tags),
        "is_scientific": m.is_scientific,
    }


def _pair_to_dict(p):
    return {
        "control": p.control,
        "treatment": p.treatment,
        "domain_category": p.domain_category,
        "context_tags": list(p.context_tags),
        "is_scientific": p.is_scientific,
    }


def test_registry_catalogs_golden_snapshot(pytestconfig=None):
    """Verify registry catalogs match the baseline snapshot."""
    curr_canon = [_metric_to_dict(m) for m in registry.canonical_metrics]
    curr_legacy = [_metric_to_dict(m) for m in registry.legacy_metrics]
    curr_legacy_by_label = {k: _metric_to_dict(v) for k, v in registry.legacy_metric_by_label.items()}
    curr_legacy_pairs = {k: [_pair_to_dict(p) for p in v] for k, v in registry.legacy_pairs_by_domain.items()}

    if _should_update(pytestconfig) or not REGISTRY_BASELINE_PATH.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        baseline_data = {
            "version": "baseline-2",
            "env_fingerprint": get_env_fingerprint(),
            "summary": {
                "total_metrics": len(registry.all_metrics),
                "total_titles": len(registry.all_titles),
                "total_pairs": len(registry.all_pairs),
                "legacy_metrics_count": len(registry.legacy_metrics),
                "canonical_metrics_count": len(registry.canonical_metrics),
                "histogram_y_labels_count": len(registry.histogram_y_labels),
            },
            "canonical_independent_metrics": curr_canon,
            "histogram_y_labels": list(registry.histogram_y_labels),
            "legacy_metrics_catalog": curr_legacy,
            "legacy_metric_by_label": curr_legacy_by_label,
            "legacy_pairs_by_domain": curr_legacy_pairs,
        }
        with open(REGISTRY_BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)

    assert REGISTRY_BASELINE_PATH.exists(), f"Missing {REGISTRY_BASELINE_PATH}"
    with open(REGISTRY_BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    # 1. Summary counts: assert aggregate counts as lower bounds per T223 (>= 495, >= 192, >= 96)
    assert len(registry.all_metrics) >= baseline["summary"]["total_metrics"]
    assert len(registry.all_metrics) >= 495
    assert len(registry.all_titles) >= baseline["summary"]["total_titles"]
    assert len(registry.all_titles) >= 192
    assert len(registry.all_pairs) >= baseline["summary"]["total_pairs"]
    assert len(registry.all_pairs) >= 96
    assert len(registry.legacy_metrics) == baseline["summary"]["legacy_metrics_count"] == 264
    assert len(registry.canonical_metrics) == baseline["summary"]["canonical_metrics_count"] == 40
    assert len(registry.histogram_y_labels) == baseline["summary"]["histogram_y_labels_count"] == 51

    # 2. Canonical independent metrics (order + content)
    assert curr_canon == baseline["canonical_independent_metrics"]

    # 3. Histogram Y labels (order + content)
    assert list(registry.histogram_y_labels) == baseline["histogram_y_labels"]

    # 4. Legacy metrics catalog (order + content)
    assert curr_legacy == baseline["legacy_metrics_catalog"]

    # 5. Legacy metric by label
    assert curr_legacy_by_label == baseline["legacy_metric_by_label"]

    # 6. Legacy pairs by domain
    assert curr_legacy_pairs == baseline["legacy_pairs_by_domain"]


def test_seeded_generation_20_golden_hash(tmp_path, pytestconfig=None):
    """Verify deterministic 20-image generation produces identical labels & _detailed.json hashes."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = 20
    cfg["seed"] = 42
    cfg["output_dir"] = str(tmp_path)
    images_dir = os.path.join(str(tmp_path), "images")
    labels_dir = os.path.join(str(tmp_path), "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    base_seed = cfg["seed"]
    for i in range(20):
        random.seed(base_seed + i)
        np.random.seed(base_seed + i)
        generate_single_chart(i, cfg, images_dir, labels_dir, str(tmp_path))

    generated_hashes = {}
    for fname in sorted(os.listdir(labels_dir)):
        if fname.endswith(".txt") or fname.endswith("_detailed.json"):
            path = os.path.join(labels_dir, fname)
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            generated_hashes[fname] = h

    combined = hashlib.sha256(
        ";".join(f"{k}={v}" for k, v in sorted(generated_hashes.items())).encode("utf-8")
    ).hexdigest()

    if _should_update(pytestconfig) or not SEEDED_BASELINE_PATH.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        baseline_data = {
            "version": "baseline-2",
            "seed": cfg["seed"],
            "env_fingerprint": get_env_fingerprint(),
            "combined_sha256": combined,
            "file_hashes": generated_hashes,
        }
        with open(SEEDED_BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)

    assert SEEDED_BASELINE_PATH.exists(), f"Missing {SEEDED_BASELINE_PATH}"
    with open(SEEDED_BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    curr_fp = get_env_fingerprint()
    base_fp = baseline.get("env_fingerprint", {})
    fp_match = (
        curr_fp.get("matplotlib", {}).get("version") == base_fp.get("matplotlib", {}).get("version")
        and curr_fp.get("matplotlib", {}).get("freetype_version") == base_fp.get("matplotlib", {}).get("freetype_version")
    )
    is_ci = bool(os.environ.get("CI"))

    if is_ci or fp_match:
        assert generated_hashes == baseline["file_hashes"]
        assert combined == baseline["combined_sha256"]
    else:
        assert len(generated_hashes) == len(baseline["file_hashes"])
        for fname in generated_hashes:
            assert fname in baseline["file_hashes"]


if __name__ == "__main__":
    import tempfile
    if "--update-goldens" in sys.argv:
        os.environ["UPDATE_GOLDENS"] = "1"
    print("Testing registry catalogs snapshot...")
    test_registry_catalogs_golden_snapshot()
    print("Testing 20-image seeded generation...")
    with tempfile.TemporaryDirectory() as td:
        test_seeded_generation_20_golden_hash(Path(td))
    print("All golden tests passed successfully!")
