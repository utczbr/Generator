#!/usr/bin/env python3
"""
scripts/verify_e2e_500_charts.py

T214: End-to-end multi-chart generation and schema v3.0 validation suite.
Generates 500 images across all 8 chart types with multi-chart mixes,
dual-axis configurations, and treatment keys.
Validates 100% compliance across both _detailed.json and _unified.json metadata.
"""
import copy
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart_task
from merge_json import merge_json_files
from config_defaults import ANNOTATION_SCHEMA_VERSION, DATASET_VERSION


VALID_DOMAINS = {"biomedical", "engineering", "business", "demographic", "common"}
CHART_TYPES = ["bar", "line", "scatter", "box", "area", "histogram", "pie", "heatmap"]


def run_e2e_validation(num_images: int = 500, output_dir: str = "/tmp/v3_e2e_500") -> bool:
    print(f"================================================================================")
    print(f"  T214: E2E GENERATION & SCHEMA v3.0 VALIDATION ({num_images} CHARTS)")
    print(f"================================================================================")

    # 1. Clean output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 2. Build configuration for balanced representation across all 8 chart types
    base_cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    base_cfg["realism_effects"] = {}  # Fast rendering without lossy filters for validation
    base_cfg["debug_mode"] = False
    base_cfg["merge_json_files"] = False  # Merge manually after detailed.json audit

    tasks = []
    for i in range(num_images):
        cfg = copy.deepcopy(base_cfg)
        cfg["num_images"] = num_images

        # Evenly cycle through all 8 chart types
        target_ctype = CHART_TYPES[i % len(CHART_TYPES)]
        for c in cfg["chart_types"]:
            cfg["chart_types"][c]["enabled"] = (c == target_ctype)
            cfg["chart_types"][c]["weight"] = 100 if c == target_ctype else 0

        # Alternate domains and enable multi-chart mixes / dual-axis / treatment keys
        domain = ["biomedical", "engineering", "business", "demographic"][i % 4]
        is_sci = domain in ("biomedical", "engineering")

        # Multi-chart mix on 25% of images
        is_multi = (i % 4 == 1)
        cfg["scenario_weights"] = {"single": 0 if is_multi else 100, "multi": 100 if is_multi else 0}

        # Force dual-axis on 20% of images
        force_dual = (i % 5 == 0) and target_ctype in ("bar", "line")
        # Force treatment key on bar charts
        force_tk = (target_ctype == "bar" and (i % 2 == 0))

        cfg["theme_config"] = {
            "semantic_domain": domain,
            "force_dual_axis": force_dual,
            "force_treatment_key": force_tk,
        }
        cfg["style_config"] = {
            "force_dual_axis": force_dual,
            "force_treatment_key": force_tk,
            "is_scientific": is_sci,
        }

        tasks.append((i, cfg, images_dir, labels_dir, output_dir))

    # 3. Parallel generation across process pool
    num_cores = min(os.cpu_count() or 4, 16)
    print(f"Generating {num_images} charts using {num_cores} parallel workers...")
    t0 = time.perf_counter()
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_cores, mp_context=mp_context) as executor:
        results = list(executor.map(generate_single_chart_task, tasks))
    gen_elapsed = time.perf_counter() - t0
    print(f"Generation completed in {gen_elapsed:.2f}s ({gen_elapsed / num_images:.3f}s per chart).")

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    if failed:
        print(f"[FAIL] {len(failed)} charts failed to generate:")
        for r in failed[:5]:
            print(f"  - Image {r[0]}: {r[2]}")
        return False
    print(f"✓ All {len(successful)}/{num_images} charts generated successfully.")

    # 4. Audit detailed.json files BEFORE merge
    print("\nAuditing detailed.json metadata compliance...")
    detailed_cache = {}
    detailed_violations = []

    for i in range(num_images):
        base_name = f"chart_{i:05d}"
        det_path = os.path.join(labels_dir, f"{base_name}_detailed.json")
        img_path = os.path.join(images_dir, f"{base_name}.png")

        if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
            detailed_violations.append(f"{base_name}: Missing or empty PNG image")
            continue
        if not os.path.isfile(det_path):
            detailed_violations.append(f"{base_name}: Missing detailed.json")
            continue

        with open(det_path, "r", encoding="utf-8") as f:
            det = json.load(f)

        # Check versions
        if det.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            detailed_violations.append(
                f"{base_name}: schema_version mismatch ({det.get('schema_version')} != {ANNOTATION_SCHEMA_VERSION})"
            )
        if det.get("dataset_version") != DATASET_VERSION:
            detailed_violations.append(
                f"{base_name}: dataset_version mismatch ({det.get('dataset_version')} != {DATASET_VERSION})"
            )

        # Check semantic domain
        sem_dom = det.get("semantic_domain")
        if sem_dom not in VALID_DOMAINS:
            detailed_violations.append(f"{base_name}: Invalid semantic_domain '{sem_dom}'")

        # Check subplots
        subplots = det.get("subplots", [])
        if not isinstance(subplots, list) or len(subplots) == 0:
            detailed_violations.append(f"{base_name}: subplots list missing or empty")
        for s_idx, sp in enumerate(subplots):
            sp_dom = sp.get("semantic_domain")
            if sp_dom and sp_dom not in VALID_DOMAINS:
                detailed_violations.append(f"{base_name}: subplot[{s_idx}] invalid domain '{sp_dom}'")

        # Check placeholder / unmapped labels
        for ann in det.get("raw_annotations", []):
            txt = ann.get("text")
            if txt is not None and isinstance(txt, str):
                txt_lower = txt.strip().lower()
                if txt_lower in ("placeholder", "unknown", "none"):
                    detailed_violations.append(f"{base_name}: Placeholder label detected '{txt}'")

        detailed_cache[base_name] = det

    if detailed_violations:
        print(f"[FAIL] {len(detailed_violations)} detailed.json violations found:")
        for v in detailed_violations[:10]:
            print(f"  - {v}")
        return False
    print(f"✓ All {num_images} detailed.json files passed 100% compliance audit.")

    # 5. Merge JSON files to produce unified.json
    print("\nMerging metadata into unified.json format...")
    t_merge = time.perf_counter()
    merge_failures = []
    for i in range(num_images):
        base_name = f"chart_{i:05d}"
        try:
            merge_json_files(base_name, labels_dir=labels_dir)
        except Exception as e:
            merge_failures.append(f"{base_name}: {e}")

    if merge_failures:
        print(f"[FAIL] {len(merge_failures)} merge failures:")
        for mf in merge_failures[:5]:
            print(f"  - {mf}")
        return False
    print(f"✓ Merged {num_images} charts in {time.perf_counter() - t_merge:.2f}s.")

    # 6. Audit unified.json files
    print("\nAuditing unified.json metadata compliance...")
    unified_violations = []
    domain_counts = {}
    chart_counts = {}

    for i in range(num_images):
        base_name = f"chart_{i:05d}"
        uni_path = os.path.join(labels_dir, f"{base_name}_unified.json")
        if not os.path.isfile(uni_path):
            unified_violations.append(f"{base_name}: Missing unified.json")
            continue

        with open(uni_path, "r", encoding="utf-8") as f:
            uni = json.load(f)

        if uni.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            unified_violations.append(
                f"{base_name}: unified schema_version mismatch ({uni.get('schema_version')})"
            )
        if uni.get("dataset_version") != DATASET_VERSION:
            unified_violations.append(
                f"{base_name}: unified dataset_version mismatch ({uni.get('dataset_version')})"
            )

        sem_dom = uni.get("semantic_domain")
        if sem_dom not in VALID_DOMAINS:
            unified_violations.append(f"{base_name}: unified invalid semantic_domain '{sem_dom}'")
        domain_counts[sem_dom] = domain_counts.get(sem_dom, 0) + 1

        ctype = uni.get("chart_type") or "composite"
        chart_counts[ctype] = chart_counts.get(ctype, 0) + 1

        stats = uni.get("statistics")
        if not stats or not isinstance(stats, dict):
            unified_violations.append(f"{base_name}: Missing statistics in unified.json")

    if unified_violations:
        print(f"[FAIL] {len(unified_violations)} unified.json violations found:")
        for v in unified_violations[:10]:
            print(f"  - {v}")
        return False

    print(f"✓ All {num_images} unified.json files passed 100% compliance audit.")

    # 7. Print summary distribution
    print("\n================================================================================")
    print("  E2E VALIDATION SUMMARY")
    print("================================================================================")
    print(f"Total Charts Tested     : {num_images}")
    print(f"Schema Version Emitted  : {ANNOTATION_SCHEMA_VERSION}")
    print(f"Dataset Version Emitted : {DATASET_VERSION}")
    print("\nChart Type Distribution:")
    for ct, cnt in sorted(chart_counts.items(), key=lambda x: str(x[0])):
        print(f"  - {str(ct):15s}: {cnt:4d} charts")
    print("\nSemantic Domain Distribution:")
    for dom, cnt in sorted(domain_counts.items()):
        print(f"  - {dom:15s}: {cnt:4d} charts")
    print("-" * 80)
    print("ALL 500 CHARTS VALIDATED WITH 100% METADATA COMPLIANCE.")
    print("================================================================================")

    # Cleanup temporary verification directory to save disk space
    shutil.rmtree(output_dir, ignore_errors=True)
    return True


if __name__ == "__main__":
    success = run_e2e_validation(num_images=500)
    sys.exit(0 if success else 1)
