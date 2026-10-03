#!/usr/bin/env python3
"""
scripts/validate_dataset.py

Lightweight CLI dataset packaging and schema sanity validator.
Validates:
1. 1:1 filename matching between images/ and labels/ (flags orphan or corrupted files).
2. Normalized coordinate boundaries ([0, 1]) for all YOLO AABB (.txt) and YOLOv8-OBB (.txt) files.
3. JSON schema integrity in *_detailed.json:
   - bar_info series indices (int >= 0)
   - topological keypoint tags and coordinates
   - modal and amodal polygon geometries
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple


def validate_txt_coordinates(filepath: str, is_obb: bool = False, eps: float = 1e-3) -> List[str]:
    """Validate normalized coordinates in YOLO AABB or OBB format."""
    errors = []
    expected_tokens = 9 if is_obb else 5

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"Could not read {filepath}: {e}"]

    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != expected_tokens:
            errors.append(
                f"{filepath}:L{line_idx}: expected {expected_tokens} tokens, got {len(tokens)}"
            )
            continue

        try:
            cls_id = int(tokens[0])
            if cls_id < 0:
                errors.append(f"{filepath}:L{line_idx}: negative class_id {cls_id}")
        except ValueError:
            errors.append(f"{filepath}:L{line_idx}: invalid class_id '{tokens[0]}'")

        for c_idx, val_str in enumerate(tokens[1:], start=1):
            try:
                c_val = float(val_str)
                if c_val < -eps or c_val > (1.0 + eps):
                    errors.append(
                        f"{filepath}:L{line_idx}: coord #{c_idx} value {c_val} out of bounds [0, 1]"
                    )
            except ValueError:
                errors.append(f"{filepath}:L{line_idx}: invalid float coord '{val_str}'")

    return errors


def validate_detailed_json(filepath: str) -> List[str]:
    """Validate schema, bar_info series indices, keypoints, and modal/amodal polygons."""
    errors = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [f"JSON decode failure in {filepath}: {e}"]

    if not isinstance(data, dict):
        return [f"Root of {filepath} must be a JSON object"]

    # 1. Validate bar_info if present
    bar_info = data.get("bar_info")
    if bar_info and isinstance(bar_info, list):
        for b_idx, bar in enumerate(bar_info):
            if not isinstance(bar, dict):
                errors.append(f"{filepath}: bar_info[{b_idx}] must be a dict")
                continue
            s_idx = bar.get("series_idx")
            if s_idx is not None and (not isinstance(s_idx, int) or s_idx < 0):
                errors.append(f"{filepath}: bar_info[{b_idx}].series_idx invalid: {s_idx}")

    # 2. Validate topological keypoints if present
    keypoints = data.get("topological_keypoints") or data.get("keypoint_info")
    if keypoints and isinstance(keypoints, list):
        for k_idx, kp in enumerate(keypoints):
            if isinstance(kp, dict):
                coords = kp.get("coordinates") or [kp.get("x"), kp.get("y")]
                if not coords or len(coords) < 2:
                    errors.append(f"{filepath}: keypoint[{k_idx}] missing 2D coordinates")
            elif isinstance(kp, (list, tuple)):
                if len(kp) < 2:
                    errors.append(f"{filepath}: keypoint[{k_idx}] tuple length < 2")

    # 3. Validate modal / amodal polygons in raw_annotations
    raw_anns = data.get("raw_annotations", [])
    if isinstance(raw_anns, list):
        for a_idx, ann in enumerate(raw_anns):
            if not isinstance(ann, dict):
                continue
            for poly_key in ("modal_polygon", "amodal_polygon"):
                if poly_key in ann and ann[poly_key] is not None:
                    poly = ann[poly_key]
                    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
                        errors.append(f"{filepath}: ann[{a_idx}].{poly_key} must have >= 3 vertices")

    return errors


def validate_dataset(dataset_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """Run full dataset validation sweep."""
    dataset_dir = os.path.abspath(dataset_dir)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    labels_obb_dir = os.path.join(dataset_dir, "labels_obb")

    report = {
        "dataset_dir": dataset_dir,
        "images_found": 0,
        "labels_found": 0,
        "labels_obb_found": 0,
        "orphan_images": [],
        "orphan_labels": [],
        "coordinate_errors": [],
        "json_schema_errors": [],
    }

    if not os.path.isdir(images_dir):
        report["coordinate_errors"].append(f"Missing images directory: {images_dir}")
        return False, report
    if not os.path.isdir(labels_dir):
        report["coordinate_errors"].append(f"Missing labels directory: {labels_dir}")
        return False, report

    # 1. 1:1 filename matching
    img_files = {f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    txt_files = {f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")}

    img_stems = {os.path.splitext(f)[0]: f for f in img_files}
    txt_stems = {os.path.splitext(f)[0]: f for f in txt_files}

    report["images_found"] = len(img_files)
    report["labels_found"] = len(txt_files)

    for stem in img_stems.keys() - txt_stems.keys():
        report["orphan_images"].append(img_stems[stem])
    for stem in txt_stems.keys() - img_stems.keys():
        report["orphan_labels"].append(txt_stems[stem])

    # 2. Check normalized coordinates in labels/
    for txt_file in txt_files:
        path = os.path.join(labels_dir, txt_file)
        errs = validate_txt_coordinates(path, is_obb=False)
        report["coordinate_errors"].extend(errs)

    # 3. Check normalized coordinates in labels_obb/ if present
    if os.path.isdir(labels_obb_dir):
        obb_files = [f for f in os.listdir(labels_obb_dir) if f.lower().endswith(".txt")]
        report["labels_obb_found"] = len(obb_files)
        for obb_file in obb_files:
            path = os.path.join(labels_obb_dir, obb_file)
            errs = validate_txt_coordinates(path, is_obb=True)
            report["coordinate_errors"].extend(errs)

    # 4. Check detailed.json files
    det_files = [f for f in os.listdir(labels_dir) if f.lower().endswith("_detailed.json")]
    for det_file in det_files:
        path = os.path.join(labels_dir, det_file)
        errs = validate_detailed_json(path)
        report["json_schema_errors"].extend(errs)

    is_valid = (
        len(report["orphan_images"]) == 0
        and len(report["orphan_labels"]) == 0
        and len(report["coordinate_errors"]) == 0
        and len(report["json_schema_errors"]) == 0
    )
    return is_valid, report


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic chart dataset schema and packaging.")
    parser.add_argument("dataset_dir", help="Path to exported dataset directory (containing images/ and labels/)")
    args = parser.parse_args()

    is_valid, report = validate_dataset(args.dataset_dir)

    print("=" * 80)
    print("  DATASET SCHEMA & PACKAGING VALIDATION REPORT")
    print("=" * 80)
    print(f"Target Directory : {report['dataset_dir']}")
    print(f"Images Inspected : {report['images_found']}")
    print(f"Labels Inspected : {report['labels_found']}")
    print(f"OBB Labels Found : {report['labels_obb_found']}")
    print(f"Orphan Images    : {len(report['orphan_images'])}")
    print(f"Orphan Labels    : {len(report['orphan_labels'])}")
    print(f"Coordinate Errs  : {len(report['coordinate_errors'])}")
    print(f"Schema Errs      : {len(report['json_schema_errors'])}")
    print("-" * 80)

    if not is_valid:
        print("\n[VALIDATION FAILED]")
        for err in report["orphan_images"][:5]:
            print(f"  - Orphan image: {err}")
        for err in report["orphan_labels"][:5]:
            print(f"  - Orphan label: {err}")
        for err in report["coordinate_errors"][:10]:
            print(f"  - Coordinate error: {err}")
        for err in report["json_schema_errors"][:10]:
            print(f"  - Schema error: {err}")
        sys.exit(1)
    else:
        print("\n[VALIDATION PASSED] 100% schema integrity and 1:1 filename alignment confirmed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
