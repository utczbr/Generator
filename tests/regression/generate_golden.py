"""Generate golden master baseline fixtures for regression testing.

Generates 8 representative fixtures:
- Fixtures 0-4: Clean renderings (bar, line, scatter, pie, area) with effects disabled.
- Fixtures 5-7: Augmented renderings with deterministic effects enabled.
"""
import copy
import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart
from tests.regression.font_config import configure_deterministic_fonts

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "golden_fixtures")

FIXTURE_SPECS = [
    # 1–5: Clean renderings (p = 0)
    {
        "fixture_id": 0,
        "name": "00_bar_clean",
        "chart_type": "bar",
        "seed": 1001,
        "effects": {},
    },
    {
        "fixture_id": 1,
        "name": "01_line_clean",
        "chart_type": "line",
        "seed": 1002,
        "effects": {},
    },
    {
        "fixture_id": 2,
        "name": "02_scatter_clean",
        "chart_type": "scatter",
        "seed": 1003,
        "effects": {},
    },
    {
        "fixture_id": 3,
        "name": "03_pie_clean",
        "chart_type": "pie",
        "seed": 1004,
        "effects": {},
    },
    {
        "fixture_id": 4,
        "name": "04_area_clean",
        "chart_type": "area",
        "seed": 1005,
        "effects": {},
    },
    # 6–8: Augmented renderings (exercising effects under fixed seeds)
    {
        "fixture_id": 5,
        "name": "05_bar_augmented",
        "chart_type": "bar",
        "seed": 2001,
        "effects": {
            "blur": {"p": 1.0, "params": {"radius_range": [0.3, 0.4]}},
            "noise": {"p": 1.0, "params": {"sigma_range": [2, 3]}},
        },
    },
    {
        "fixture_id": 6,
        "name": "06_line_augmented",
        "chart_type": "line",
        "seed": 2002,
        "effects": {
            "color_variation": {"p": 1.0, "params": {"shift_range": [0.98, 1.02]}},
            "printing_artifacts": {"p": 1.0, "params": {"texture_alpha": [0.08, 0.08], "blur_radius": [0.3, 0.3]}},
        },
    },
    {
        "fixture_id": 7,
        "name": "07_area_augmented",
        "chart_type": "area",
        "seed": 2003,
        "effects": {
            "low_res": {"p": 1.0, "params": {"scale_range": [0.3, 0.35]}},
            "jpeg_compression": {"p": 1.0, "params": {"quality_range": [70, 75]}},
        },
    },
]


def build_fixture_config(spec: dict) -> dict:
    """Build isolated config for a specific fixture run."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["debug_mode"] = False
    cfg["num_images"] = 1
    cfg["dataset_format"] = "detection"
    cfg["scenario_weights"] = {"single": 100, "multi": 0}

    # Pin single target chart type
    for c_type in cfg["chart_types"]:
        if c_type == spec["chart_type"]:
            cfg["chart_types"][c_type]["enabled"] = True
            cfg["chart_types"][c_type]["weight"] = 100
        else:
            cfg["chart_types"][c_type]["enabled"] = False
            cfg["chart_types"][c_type]["weight"] = 0

    # Configure effects
    cfg["realism_effects"] = spec["effects"]
    cfg["annotation_schema_version"] = "v3.0"
    return cfg


def generate_fixture(spec: dict, target_dir: str):
    """Generate a single golden fixture with cleanup guardrails."""
    configure_deterministic_fonts()
    cfg = build_fixture_config(spec)

    images_dir = os.path.join(target_dir, "images")
    labels_dir = os.path.join(target_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    seed = spec["seed"]
    random.seed(seed)
    np.random.seed(seed)

    fixture_id = spec["fixture_id"]
    try:
        generate_single_chart(
            i=fixture_id,
            cfg=cfg,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=target_dir,
        )
    finally:
        plt.close("all")


def generate_all_golden_fixtures(output_dir: str = FIXTURES_DIR):
    """Generate all 8 golden master baseline fixtures."""
    print(f"Generating 8 Golden Master fixtures under {output_dir}...")
    for spec in FIXTURE_SPECS:
        print(f"  -> Fixture {spec['fixture_id']}: {spec['name']} (Type: {spec['chart_type']}, Seed: {spec['seed']})")
        generate_fixture(spec, output_dir)
    print("✓ All golden master fixtures generated successfully.")


if __name__ == "__main__":
    generate_all_golden_fixtures()
