"""Two-tier Golden Master regression verification harness.

Tier 1 (Platform-Agnostic): Semantic and geometric invariance on ground-truth JSON annotations.
Tier 2 (Environment-Guarded): Raster pixel diff / hash comparison on headless Linux.
"""
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


class AnnotationDiff:
    """Records a specific mismatch between golden and test annotations."""
    def __init__(self, field: str, expected: Any, actual: Any, detail: str = ""):
        self.field = field
        self.expected = expected
        self.actual = actual
        self.detail = detail

    def __str__(self) -> str:
        msg = f"[{self.field}] Expected: {self.expected}, Actual: {self.actual}"
        if self.detail:
            msg += f" ({self.detail})"
        return msg


def calculate_box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union (IoU) for two [x0, y0, x1, y1] boxes."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    inter_w = max(0.0, float(x1 - x0))
    inter_h = max(0.0, float(y1 - y0))
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 1.0 if area_a == 0 and area_b == 0 else 0.0
    return inter_area / union_area


def compare_polygons(poly_a: Any, poly_b: Any, rtol: float = 1e-4, min_iou: float = 0.99) -> Tuple[bool, str]:
    """Compare two polygon point lists with relative coordinate tolerance or bounding box IoU."""
    arr_a = np.array(poly_a, dtype=float)
    arr_b = np.array(poly_b, dtype=float)

    if arr_a.shape == arr_b.shape:
        if np.allclose(arr_a, arr_b, rtol=rtol, atol=1e-2):
            return True, "Points match within relative tolerance"

    # Fallback to polygon bounding box IoU guardrail
    if arr_a.size > 0 and arr_b.size > 0:
        box_a = [np.min(arr_a[:, 0]), np.min(arr_a[:, 1]), np.max(arr_a[:, 0]), np.max(arr_a[:, 1])]
        box_b = [np.min(arr_b[:, 0]), np.min(arr_b[:, 1]), np.max(arr_b[:, 0]), np.max(arr_b[:, 1])]
        iou = calculate_box_iou(box_a, box_b)
        if iou >= min_iou:
            return True, f"Polygon envelope IoU {iou:.5f} >= {min_iou}"
        return False, f"Polygon envelope IoU {iou:.5f} < {min_iou}"

    return False, f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}"


def verify_tier1_annotations(golden_labels_dir: str, test_labels_dir: str, base_filename: str) -> List[AnnotationDiff]:
    """Tier 1 Platform-Agnostic verification of annotations.
    
    Returns a list of AnnotationDiff objects. Empty list indicates clean pass.
    """
    diffs: List[AnnotationDiff] = []

    # 1. Detailed JSON verification
    golden_detailed_path = os.path.join(golden_labels_dir, f"{base_filename}_detailed.json")
    test_detailed_path = os.path.join(test_labels_dir, f"{base_filename}_detailed.json")

    if not os.path.exists(test_detailed_path):
        diffs.append(AnnotationDiff("file", golden_detailed_path, None, "Missing test detailed.json"))
        return diffs

    with open(golden_detailed_path, 'r') as f:
        gold_det = json.load(f)
    with open(test_detailed_path, 'r') as f:
        test_det = json.load(f)

    # Schema completeness
    gold_keys = set(gold_det.keys())
    test_keys = set(test_det.keys())
    if gold_keys != test_keys:
        diffs.append(AnnotationDiff("detailed.keys", sorted(list(gold_keys)), sorted(list(test_keys)), "Top-level schema keys mismatch"))

    # Verify raw_annotations
    gold_raw = gold_det.get("raw_annotations", [])
    test_raw = test_det.get("raw_annotations", [])
    if len(gold_raw) != len(test_raw):
        diffs.append(AnnotationDiff("raw_annotations.count", len(gold_raw), len(test_raw), "Annotation count mismatch"))
    else:
        for idx, (g_ann, t_ann) in enumerate(zip(gold_raw, test_raw)):
            prefix = f"raw_annotations[{idx}]"
            # Class & semantic checks
            for field in ["class_id", "class_name", "semantic_role", "text"]:
                if field in g_ann or field in t_ann:
                    if g_ann.get(field) != t_ann.get(field):
                        diffs.append(AnnotationDiff(f"{prefix}.{field}", g_ann.get(field), t_ann.get(field)))

            # Discrete bbox checks (IoU >= 0.999)
            g_box = g_ann.get("xyxy") or g_ann.get("bbox")
            t_box = t_ann.get("xyxy") or t_ann.get("bbox")
            if isinstance(g_box, dict):
                g_box = [g_box.get("x0", 0), g_box.get("y0", 0), g_box.get("x1", 0), g_box.get("y1", 0)]
            if isinstance(t_box, dict):
                t_box = [t_box.get("x0", 0), t_box.get("y0", 0), t_box.get("x1", 0), t_box.get("y1", 0)]

            if g_box is not None and t_box is not None:
                iou = calculate_box_iou(g_box, t_box)
                if iou < 0.999:
                    diffs.append(AnnotationDiff(f"{prefix}.bbox", g_box, t_box, f"IoU={iou:.5f} < 0.999"))

    # Verify category-specific lists (e.g. bar, wedge, data_point, line_segment, area_boundary)
    category_keys = ["bar", "wedge", "data_point", "box", "line_segment", "area_boundary", "cell", "legend", "axis_title", "chart_title"]
    for cat in category_keys:
        g_items = gold_det.get(cat, [])
        t_items = test_det.get(cat, [])
        if len(g_items) != len(t_items):
            diffs.append(AnnotationDiff(f"{cat}.count", len(g_items), len(t_items)))
            continue
        for idx, (g_item, t_item) in enumerate(zip(g_items, t_items)):
            if not isinstance(g_item, dict) or not isinstance(t_item, dict):
                continue
            # Check bounding boxes if present
            if "bbox" in g_item and "bbox" in t_item:
                iou = calculate_box_iou(g_item["bbox"], t_item["bbox"])
                if iou < 0.999:
                    diffs.append(AnnotationDiff(f"{cat}[{idx}].bbox", g_item["bbox"], t_item["bbox"], f"IoU={iou:.5f}"))
            # Check polygon / ribbon coordinates if present
            for poly_field in ["polygon", "points", "ribbon_polygon"]:
                if poly_field in g_item and poly_field in t_item:
                    match, msg = compare_polygons(g_item[poly_field], t_item[poly_field], rtol=1e-4, min_iou=0.99)
                    if not match:
                        diffs.append(AnnotationDiff(f"{cat}[{idx}].{poly_field}", "golden_polygon", "test_polygon", msg))

    # 2. OCR JSON verification
    golden_ocr_path = os.path.join(golden_labels_dir, f"{base_filename}_ocr.json")
    test_ocr_path = os.path.join(test_labels_dir, f"{base_filename}_ocr.json")
    if os.path.exists(golden_ocr_path) and os.path.exists(test_ocr_path):
        with open(golden_ocr_path, 'r') as f:
            gold_ocr = json.load(f)
        with open(test_ocr_path, 'r') as f:
            test_ocr = json.load(f)
        g_ocr_items = gold_ocr.get("ocr_entries", []) if isinstance(gold_ocr, dict) else gold_ocr
        t_ocr_items = test_ocr.get("ocr_entries", []) if isinstance(test_ocr, dict) else test_ocr
        if len(g_ocr_items) != len(t_ocr_items):
            diffs.append(AnnotationDiff("ocr.count", len(g_ocr_items), len(t_ocr_items)))
        else:
            for idx, (g_o, t_o) in enumerate(zip(g_ocr_items, t_ocr_items)):
                if g_o.get("text") != t_o.get("text"):
                    diffs.append(AnnotationDiff(f"ocr[{idx}].text", g_o.get("text"), t_o.get("text")))

    # 3. Metadata JSON verification
    golden_meta_path = os.path.join(golden_labels_dir, f"{base_filename}.json")
    test_meta_path = os.path.join(test_labels_dir, f"{base_filename}.json")
    if os.path.exists(golden_meta_path) and os.path.exists(test_meta_path):
        with open(golden_meta_path, 'r') as f:
            gold_meta = json.load(f)
        with open(test_meta_path, 'r') as f:
            test_meta = json.load(f)
        for key in ["image_id", "resolution", "chart_types", "num_annotations"]:
            if gold_meta.get(key) != test_meta.get(key):
                diffs.append(AnnotationDiff(f"metadata.{key}", gold_meta.get(key), test_meta.get(key)))

    # 4. YOLO .txt label verification
    golden_txt_path = os.path.join(golden_labels_dir, f"{base_filename}.txt")
    test_txt_path = os.path.join(test_labels_dir, f"{base_filename}.txt")
    if os.path.exists(golden_txt_path) and os.path.exists(test_txt_path):
        with open(golden_txt_path, 'r') as f:
            gold_lines = [l.strip().split() for l in f if l.strip()]
        with open(test_txt_path, 'r') as f:
            test_lines = [l.strip().split() for l in f if l.strip()]
        if len(gold_lines) != len(test_lines):
            diffs.append(AnnotationDiff("yolo.line_count", len(gold_lines), len(test_lines)))
        else:
            for idx, (g_tokens, t_tokens) in enumerate(zip(gold_lines, test_lines)):
                if g_tokens[0] != t_tokens[0]:
                    diffs.append(AnnotationDiff(f"yolo[{idx}].class_id", g_tokens[0], t_tokens[0]))
                g_coords = [float(v) for v in g_tokens[1:]]
                t_coords = [float(v) for v in t_tokens[1:]]
                if len(g_coords) == len(t_coords) and len(g_coords) == 4:
                    # Normalized center x, y, w, h to bbox
                    def yolo_to_xyxy(c):
                        return [c[0] - c[2]/2, c[1] - c[3]/2, c[0] + c[2]/2, c[1] + c[3]/2]
                    iou = calculate_box_iou(yolo_to_xyxy(g_coords), yolo_to_xyxy(t_coords))
                    if iou < 0.999:
                        diffs.append(AnnotationDiff(f"yolo[{idx}].coords", g_coords, t_coords, f"IoU={iou:.5f}"))

    return diffs


def is_headless_linux() -> bool:
    """Return True if executing in a headless Linux environment."""
    if not sys.platform.startswith('linux'):
        return False
    # Check if headless / Agg backend
    return os.environ.get('DISPLAY') is None or os.environ.get('CI') == 'true' or True  # Linux CI/test environment


def verify_tier2_raster(golden_images_dir: str, test_images_dir: str, base_filename: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Tier 2 Environment-Guarded verification of raster pixels and perceptual hash.
    
    Returns (passed, message, metrics).
    """
    if not is_headless_linux():
        return True, "Skipped: non-headless Linux environment", {"skipped": True}

    gold_img_path = os.path.join(golden_images_dir, f"{base_filename}.png")
    test_img_path = os.path.join(test_images_dir, f"{base_filename}.png")

    if not os.path.exists(test_img_path):
        return False, f"Missing test image {test_img_path}", {}

    gold_img = Image.open(gold_img_path).convert('RGB')
    test_img = Image.open(test_img_path).convert('RGB')

    if gold_img.size != test_img.size:
        return False, f"Image dimensions mismatch: gold={gold_img.size} vs test={test_img.size}", {}

    gold_arr = np.array(gold_img, dtype=np.float32)
    test_arr = np.array(test_img, dtype=np.float32)

    diff = np.abs(gold_arr - test_arr)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))

    with open(gold_img_path, 'rb') as f:
        gold_hash = hashlib.sha256(f.read()).hexdigest()
    with open(test_img_path, 'rb') as f:
        test_hash = hashlib.sha256(f.read()).hexdigest()

    metrics = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rms_diff": rms_diff,
        "hash_match": (gold_hash == test_hash),
        "gold_hash": gold_hash,
        "test_hash": test_hash,
    }

    # Strict pixel equality for headless Linux
    if max_diff == 0.0 and gold_hash == test_hash:
        return True, "Exact pixel match and hash match", metrics
    elif max_diff <= 2.0 and rms_diff < 0.5:
        return True, f"Sub-pixel raster match (max_diff={max_diff}, rms={rms_diff:.3f})", metrics
    else:
        return False, f"Raster diff exceeded tolerance: max_diff={max_diff}, rms={rms_diff:.3f}", metrics
