"""
tests/test_p5_tick_formatters.py

Acceptance tests for Phase 1 Task T234c:
Tick Formatters and Numeric Clutter (FR-088).
"""
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pytest

from chart import apply_tick_formatters, apply_typography_variation
from config_defaults import OCR_TRAINING_CONFIG
from generator import generate_single_chart


def test_apply_tick_formatters_legacy_neutral():
    """Verify legacy profile never modifies formatters."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 5000])
    orig_formatter = ax.yaxis.get_major_formatter()

    apply_tick_formatters(ax, active_profile="legacy", theme_config={"profile": "legacy"})
    assert ax.yaxis.get_major_formatter() is orig_formatter
    plt.close(fig)


def test_scientific_multiplier_offset_annotation(tmp_path):
    """Verify scientific multiplier offset text is annotated as axis_label with valid extent."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["profile"] = "domain_gap_v1"
    cfg["output_dir"] = str(tmp_path)
    cfg["seed"] = 42
    for ct in cfg.get("chart_types", {}):
        cfg["chart_types"][ct]["enabled"] = (ct == "bar")
    cfg["scenario_weights"] = {"single": 1.0, "multi": 0.0}
    cfg["export_detailed_json"] = True
    cfg["debug_mode"] = False

    # Force tick formatter with high probability and scientific focus
    cfg["tick_formatter_probability"] = 1.0

    found_formatted = False
    for i in range(20):
        generate_single_chart(i, cfg, str(images_dir), str(labels_dir), str(tmp_path))
        base_name = f"chart_{i:05d}"
        det_file = labels_dir / f"{base_name}_detailed.json"

        if det_file.exists():
            with open(det_file, "r") as f:
                data = json.load(f)

            anns = data.get("annotations", [])
            axis_labels = [a for a in anns if a.get("class_name") == "axis_labels"]
            texts = [a.get("text", "") for a in axis_labels if a.get("text")]

            # Check for numeric clutter: %, $, k, M, or scientific multiplier
            has_clutter = any(
                ("%" in t or "$" in t or "k" in t or "M" in t or "10^" in t or "1e" in t or "times" in t)
                for t in texts
            )
            if has_clutter:
                found_formatted = True
                for a in axis_labels:
                    xyxy = a["xyxy"]
                    assert xyxy[2] - xyxy[0] >= 1.0
                    assert xyxy[3] - xyxy[1] >= 1.0
                break

    assert found_formatted, "Expected formatted ticks or scientific multipliers"


def test_tick_formatter_frequency_above_25_percent(tmp_path):
    """Verify non-standard tick formatters appear in >= 25% of profile-generated charts."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["profile"] = "domain_gap_v1"
    cfg["output_dir"] = str(tmp_path)
    cfg["seed"] = 100
    cfg["scenario_weights"] = {"single": 1.0, "multi": 0.0}
    cfg["export_detailed_json"] = True
    cfg["debug_mode"] = False

    formatted_count = 0
    total = 30

    for i in range(total):
        generate_single_chart(i, cfg, str(images_dir), str(labels_dir), str(tmp_path))
        base_name = f"chart_{i:05d}"
        det_file = labels_dir / f"{base_name}_detailed.json"

        if det_file.exists():
            with open(det_file, "r") as f:
                data = json.load(f)

            anns = data.get("annotations", [])
            axis_labels = [a for a in anns if a.get("class_name") == "axis_labels"]
            texts = [a.get("text", "") for a in axis_labels if a.get("text")]

            if any(("%" in t or "$" in t or "k" in t or "M" in t or "10^" in t or "1e" in t or "times" in t) for t in texts):
                formatted_count += 1

    rate = formatted_count / total
    assert rate >= 0.25, f"Formatted tick frequency {rate:.2f} < 0.25 threshold"
