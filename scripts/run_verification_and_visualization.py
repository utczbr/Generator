"""
Comprehensive Dataset Generation, Verification, and Annotation Visualizer
==========================================================================
1. Generates individual charts of each type across Matplotlib and Vega-Lite vector backends.
2. Generates multi-panel composite charts.
3. Generates modern v2 annotations: YOLOv8/v26-OBB, topological keypoints, modal/amodal polygons.
4. Verifies file system integrity and label validity (Detection, OBB, Segmentation, Pose, Topologies).
5. Generates visual overlay renderings for each annotation modality (AABB, OBB, Segmentation, Pose, Keypoints).
"""

import os
import sys
import time
import copy
import json
import random
import argparse
import traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import custom_config
    OCR_TRAINING_CONFIG = custom_config.OCR_TRAINING_CONFIG
except (ImportError, AttributeError):
    import config_defaults
    OCR_TRAINING_CONFIG = config_defaults.OCR_TRAINING_CONFIG

import generator
from generator import ensure_dir

OUTPUT_BASE = "dataset_verification_run"
VIS_DIR = os.path.join(OUTPUT_BASE, "visualizations")

CHART_TYPES = ['line', 'area', 'pie', 'bar', 'scatter', 'box', 'histogram', 'heatmap']

# Color palette for visualization (RGB for PIL / BGR for OpenCV)
COLORS = [
    (255, 56, 56),    # Bright Red
    (56, 156, 255),   # Sky Blue
    (56, 255, 56),    # Bright Green
    (255, 214, 56),   # Gold / Yellow
    (255, 56, 255),   # Magenta
    (56, 255, 255),   # Cyan
    (255, 112, 56),   # Orange
    (140, 56, 255),   # Purple
    (56, 255, 140),   # Mint
    (255, 56, 140),   # Pink
    (180, 180, 180),  # Grey
    (255, 165, 0),    # Amber
    (0, 200, 200),    # Teal
    (128, 0, 128),    # Violet
    (0, 128, 0)       # Dark Green
]

def get_color(idx):
    return COLORS[idx % len(COLORS)]

# =========================================================================
# STEP 1: GENERATION ENGINE
# =========================================================================

def run_generation(num_per_type=20, num_multi=20):
    print("=" * 80)
    print("STEP 1: GENERATING SYNTHETIC CHARTS (v2 PIPELINE)")
    print(f"Target: {num_per_type} images for each of {len(CHART_TYPES)} chart types + {num_multi} multi-panel charts")
    print(f"Total charts to generate: {len(CHART_TYPES) * num_per_type + num_multi}")
    print("=" * 80)

    images_dir = os.path.join(OUTPUT_BASE, "images")
    labels_dir = os.path.join(OUTPUT_BASE, "labels")
    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    ensure_dir(VIS_DIR)

    base_cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    base_cfg['output_dir'] = OUTPUT_BASE
    base_cfg['debug_mode'] = False
    base_cfg['export_obb'] = True
    base_cfg['export_labels_obb'] = True
    base_cfg['annotation_schema_version'] = 'v2'

    tasks = []
    global_img_idx = 0

    # 1. Prepare single chart types (balanced between Matplotlib and Vega-Lite vector backends)
    for ctype in CHART_TYPES:
        for local_i in range(num_per_type):
            cfg = copy.deepcopy(base_cfg)
            cfg['num_images'] = num_per_type
            cfg['scenario_weights'] = {'single': 100, 'multi': 0}
            for k in cfg['chart_types']:
                cfg['chart_types'][k]['enabled'] = (k == ctype)
                cfg['chart_types'][k]['weight'] = 100 if k == ctype else 0
            cfg['chart_type'] = ctype

            # Exercise modern vector backend & copula tabular synthesis on supported chart types
            if ctype in ('bar', 'line', 'scatter') and local_i % 2 == 1:
                cfg['engine'] = 'vegalite'
                cfg['use_synthetic_data_engine'] = True

            tasks.append((global_img_idx, cfg, images_dir, labels_dir, OUTPUT_BASE))
            global_img_idx += 1

    # 2. Prepare multi-panel charts
    for local_i in range(num_multi):
        multi_cfg = copy.deepcopy(base_cfg)
        multi_cfg['num_images'] = num_multi
        multi_cfg['scenario_weights'] = {'single': 0, 'multi': 100}
        for k in multi_cfg['chart_types']:
            multi_cfg['chart_types'][k]['enabled'] = True
            multi_cfg['chart_types'][k]['weight'] = 10

        tasks.append((global_img_idx, multi_cfg, images_dir, labels_dir, OUTPUT_BASE))
        global_img_idx += 1

    import concurrent.futures
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"Executing {len(tasks)} generation tasks in parallel across {max_workers} worker processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generator.generate_single_chart_task, task) for task in tasks]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if not res[1]:
                print(f"[ERROR] Failed to generate image {res[0]}: {res[2]}")

    print(f"\n✓ Generation complete! Total generated: {global_img_idx} images.")


# =========================================================================
# STEP 2: VERIFICATION & AUDITING ENGINE
# =========================================================================

def verify_dataset():
    print("\n" + "=" * 80)
    print("STEP 2: VERIFYING CREATED FILES, FOLDERS, AND LABEL CONSTRAINTS")
    print("=" * 80)

    # 1. Directory audit
    expected_dirs = [
        "images",
        "labels",
        "labels_obb",
        "line_obj_labels",
        "line_seg_labels",
        "line_marker_labels",
        "area_obj_labels",
        "area_seg_labels",
        "pie_obj_labels",
        "pie_pose_labels",
        "bar_obj_labels",
        "scatter_obj_labels",
        "box_elements_labels",
        "box_global_labels",
        "histogram_obj_labels",
        "heatmap_macro_labels",
        "heatmap_text_labels",
        "heatmap_lattice_images",
        "heatmap_lattice_labels",
        "heatmap_colorbar_images",
        "heatmap_colorbar_labels",
    ]

    print("\n[Directory Inventory & File Counts]")
    dir_counts = {}
    for d in expected_dirs:
        full_path = os.path.join(OUTPUT_BASE, d)
        if os.path.exists(full_path):
            files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
            dir_counts[d] = len(files)
            print(f"  ✓ {d:26s}: {len(files):4d} files")
        else:
            dir_counts[d] = 0
            print(f"  ✗ {d:26s}: MISSING DIRECTORY")

    # 2. Parse and validate labels and JSON metadata
    print("\n[Label Integrity & Semantic Validation]")

    validation_issues = []
    stats = defaultdict(int)

    for root, _, files in os.walk(OUTPUT_BASE):
        if "visualizations" in root:
            continue
        rel_dir = os.path.relpath(root, OUTPUT_BASE)
        for fname in files:
            fpath = os.path.join(root, fname)

            # Audit detailed JSON metadata
            if fname.endswith("_detailed.json"):
                stats['detailed_json_files'] += 1
                try:
                    with open(fpath, 'r', encoding='utf-8') as jf:
                        jdata = json.load(jf)
                    # Verify bar_info series indices
                    bar_info = jdata.get("bar_info", [])
                    if isinstance(bar_info, list):
                        for b_idx, bar in enumerate(bar_info):
                            if isinstance(bar, dict) and bar.get("series_idx") is not None:
                                s_idx = bar["series_idx"]
                                if not isinstance(s_idx, int) or s_idx < 0:
                                    validation_issues.append(f"{rel_dir}/{fname}: bar_info[{b_idx}].series_idx invalid: {s_idx}")
                    # Verify topological keypoint structures
                    kpt_info = jdata.get("keypoint_info", [])
                    if isinstance(kpt_info, list) and kpt_info:
                        for s_dict in kpt_info:
                            if isinstance(s_dict, dict):
                                kpts = s_dict.get("keypoints", [])
                                stats['topological_keypoints'] += len(kpts)
                except Exception as e:
                    validation_issues.append(f"{rel_dir}/{fname}: JSON decode error: {e}")
                continue

            if not fname.endswith(".txt"):
                continue

            stats['total_label_files'] += 1

            with open(fpath, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                if rel_dir in ['labels', 'labels_obb', 'line_obj_labels', 'bar_obj_labels']:
                    validation_issues.append(f"{rel_dir}/{fname}: Empty label file")
                continue

            for line_idx, line in enumerate(lines):
                tokens = line.split()
                if not tokens:
                    continue
                stats['total_instances'] += 1

                # Class ID check
                try:
                    cls_id = int(tokens[0])
                except ValueError:
                    validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Invalid class ID '{tokens[0]}'")
                    continue

                # Coordinate floats check
                try:
                    coords = [float(t) for t in tokens[1:]]
                except ValueError:
                    validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Non-numeric coordinates")
                    continue

                # Check for NaN / Inf
                if any(np.isnan(c) or np.isinf(c) for c in coords):
                    validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: NaN/Inf coordinate detected")
                    continue

                # A. Instance Segmentation Polygon: line_seg_labels, area_seg_labels
                if "seg_labels" in rel_dir:
                    stats['segmentation_masks'] += 1
                    if len(coords) < 6:
                        validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Polygon has fewer than 3 vertices ({len(coords)} tokens)")
                    elif len(coords) % 2 != 0:
                        validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Polygon coords count is odd ({len(coords)})")
                    else:
                        out_of_bounds = [c for c in coords if c < -0.01 or c > 1.01]
                        if out_of_bounds:
                            validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Polygon coords out of [0, 1]: {out_of_bounds[:3]}")

                # B. Pose Keypoints: pie_pose_labels
                elif "pose_labels" in rel_dir:
                    stats['pose_annotations'] += 1
                    if len(coords) != 19:
                        validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Expected 19 tokens for 5-kpt pose, got {len(coords)}")
                    else:
                        cx, cy, w, h = coords[:4]
                        if w <= 0 or h <= 0:
                            validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Pose bbox has zero/negative dimension ({w}, {h})")
                        kpt_tokens = coords[4:]
                        for k_idx in range(5):
                            kx = kpt_tokens[k_idx*3]
                            ky = kpt_tokens[k_idx*3 + 1]
                            kv = int(kpt_tokens[k_idx*3 + 2])
                            if kv not in (0, 1, 2):
                                validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Invalid visibility flag {kv}")
                            if kv > 0 and (kx < -0.01 or kx > 1.01 or ky < -0.01 or ky > 1.01):
                                validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Visible keypoint out of bounds ({kx}, {ky})")

                # C. Oriented Bounding Boxes: labels_obb or *_obb.txt (8 coordinates)
                elif "labels_obb" in rel_dir or fname.endswith("_obb.txt"):
                    stats['obb_boxes'] += 1
                    if len(coords) != 8:
                        validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Expected 8 OBB coordinates, got {len(coords)}")
                    else:
                        out_of_bounds = [c for c in coords if c < -0.01 or c > 1.01]
                        if out_of_bounds:
                            validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: OBB coords out of [0, 1]: {out_of_bounds[:3]}")
                        pts = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                        area = 0.5 * abs(sum(pts[i][0]*pts[(i+1)%4][1] - pts[(i+1)%4][0]*pts[i][1] for i in range(4)))
                        if area <= 1e-7:
                            validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Degenerate OBB area ({area})")

                # D. Axis-Aligned Bounding Box Detection: all standard detection dirs
                else:
                    stats['detection_boxes'] += 1
                    if len(coords) != 4:
                        validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Expected 4 bbox tokens, got {len(coords)}")
                    else:
                        cx, cy, w, h = coords
                        if w <= 0 or h <= 0:
                            validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Detection box zero/negative size ({w}, {h})")
                        if cx < -0.01 or cx > 1.01 or cy < -0.01 or cy > 1.01:
                            validation_issues.append(f"{rel_dir}/{fname} L{line_idx+1}: Bbox center out of bounds ({cx}, {cy})")

    print(f"\n  • Total label files audited:     {stats['total_label_files']}")
    print(f"  • Total detailed JSON files:     {stats['detailed_json_files']}")
    print(f"  • Total annotated instances:     {stats['total_instances']}")
    print(f"    - Axis-Aligned Bounding Boxes:  {stats['detection_boxes']}")
    print(f"    - Oriented Bounding Boxes (OBB):{stats['obb_boxes']}")
    print(f"    - Instance Segmentation Masks:  {stats['segmentation_masks']}")
    print(f"    - 5-Point Slice Pose Instances: {stats['pose_annotations']}")
    print(f"    - Topological Keypoint Nodes:   {stats['topological_keypoints']}")

    if validation_issues:
        print(f"\n[WARNING] Found {len(validation_issues)} validation issues:")
        for issue in validation_issues[:15]:
            print(f"  - {issue}")
        if len(validation_issues) > 15:
            print(f"  ... and {len(validation_issues) - 15} more.")
    else:
        print("\n✓ ALL LABEL FILES AND ANNOTATIONS PASSED VALIDATION PERFECTLY (0 errors)!")

    return validation_issues


# =========================================================================
# STEP 3: ANNOTATION VISUALIZATION OVERLAY ENGINE
# =========================================================================

def draw_detection_labels(img_path, label_path, class_map, output_vis_path, title_text=""):
    """Renders high-quality bounding box annotations with colored labels."""
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        return False

    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h

        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        x1 = cx + w / 2.0
        y1 = cy + h / 2.0

        color = get_color(cls_id)
        cls_name = class_map.get(str(cls_id), class_map.get(cls_id, f"cls_{cls_id}"))

        # Semi-transparent box fill
        fill_color = (color[0], color[1], color[2], 40)
        draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=color, width=2)

        # Label tag pill
        label_text = f"{cls_name} ({cls_id})"
        text_bbox = draw.textbbox((x0, max(0, y0 - 16)), label_text)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x0, max(0, y0 - 16)), label_text, fill=(255, 255, 255))

    if title_text:
        header_bbox = draw.textbbox((10, 10), title_text)
        draw.rectangle([header_bbox[0]-4, header_bbox[1]-4, header_bbox[2]+4, header_bbox[3]+4], fill=(0, 0, 0, 200))
        draw.text((10, 10), title_text, fill=(255, 255, 255))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    ensure_dir(os.path.dirname(output_vis_path))
    result.save(output_vis_path)
    return True


def draw_obb_labels(img_path, label_path, class_map, output_vis_path, title_text=""):
    """Renders 4-corner Oriented Bounding Box (YOLOv8/v26-OBB) annotations with angled polygons."""
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        return False

    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 9:
            continue
        cls_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        pts = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, 8, 2)]

        color = get_color(cls_id)
        cls_name = class_map.get(str(cls_id), class_map.get(cls_id, f"cls_{cls_id}"))

        # Semi-transparent polygon fill
        fill_color = (color[0], color[1], color[2], 50)
        draw.polygon(pts, fill=fill_color, outline=color, width=2)

        # Draw small corner dots
        for px, py in pts:
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255), outline=color)

        # Draw label tag near the first vertex
        label_text = f"{cls_name} ({cls_id}) [OBB]"
        tx, ty = pts[0]
        text_bbox = draw.textbbox((tx, max(0, ty - 16)), label_text)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((tx, max(0, ty - 16)), label_text, fill=(255, 255, 255))

    if title_text:
        header_bbox = draw.textbbox((10, 10), title_text)
        draw.rectangle([header_bbox[0] - 4, header_bbox[1] - 4, header_bbox[2] + 4, header_bbox[3] + 4], fill=(0, 0, 0, 200))
        draw.text((10, 10), title_text, fill=(255, 255, 255))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    ensure_dir(os.path.dirname(output_vis_path))
    result.save(output_vis_path)
    return True


def draw_topological_keypoints(img_path, json_path, output_vis_path, title_text=""):
    """Renders topological series keypoints and directed edges from detailed JSON metadata."""
    if not os.path.exists(img_path) or not os.path.exists(json_path):
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    kpt_info = data.get("keypoint_info", [])
    if not kpt_info:
        return False

    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    role_colors = {
        "endpoint": (0, 220, 255),    # Cyan
        "peak": (255, 50, 50),        # Red
        "valley": (50, 120, 255),     # Blue
        "inflection": (255, 215, 0),  # Gold
        "vertex": (50, 220, 50),      # Lime Green
    }

    drawn_any = False
    for s_idx, series_dict in enumerate(kpt_info):
        if not isinstance(series_dict, dict):
            continue
        kpts = series_dict.get("keypoints", [])
        if not kpts:
            continue

        series_color = get_color(s_idx)
        pt_coords = []
        for kp in kpts:
            if "pixel_x" in kp and "pixel_y" in kp:
                kx = float(kp["pixel_x"])
                ky = float(kp["pixel_y"])
            else:
                kx = float(kp.get("x", 0))
                ky = float(kp.get("y", 0))
                # Convert normalized coordinates if required
                if 0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0 and img_w > 1 and img_h > 1:
                    kx *= img_w
                    ky *= img_h
            pt_coords.append((kx, ky))

        # Draw series connective skeleton lines
        for j in range(len(pt_coords) - 1):
            draw.line([pt_coords[j], pt_coords[j + 1]], fill=(series_color[0], series_color[1], series_color[2], 180), width=2)

        # Draw topological keypoint markers
        for j, kp in enumerate(kpts):
            kx, ky = pt_coords[j]
            tag = kp.get("tag") or kp.get("role") or "vertex"
            color = role_colors.get(tag, (200, 200, 200))
            r = 5
            draw.ellipse([kx - r - 1, ky - r - 1, kx + r + 1, ky + r + 1], fill=(255, 255, 255))
            draw.ellipse([kx - r, ky - r, kx + r, ky + r], fill=color, outline=(0, 0, 0), width=1)
            draw.text((kx + 6, ky - 6), tag[:4], fill=color)
            drawn_any = True

    if not drawn_any:
        return False

    header = title_text or "Topological Series Keypoints & Adjacency"
    header_bbox = draw.textbbox((10, 10), header)
    draw.rectangle([header_bbox[0] - 4, header_bbox[1] - 4, header_bbox[2] + 4, header_bbox[3] + 4], fill=(0, 0, 0, 220))
    draw.text((10, 10), header, fill=(255, 255, 255))

    # Top-right role legend
    leg_x = max(10, img_w - 320)
    leg_y = 10
    draw.rectangle([leg_x - 5, leg_y - 4, img_w - 10, leg_y + 24], fill=(0, 0, 0, 220))
    cur_x = leg_x
    for role_name, r_col in role_colors.items():
        draw.ellipse([cur_x, leg_y + 4, cur_x + 8, leg_y + 12], fill=r_col, outline=(255, 255, 255))
        draw.text((cur_x + 10, leg_y + 2), role_name[:4], fill=(255, 255, 255))
        cur_x += 58

    result = Image.alpha_composite(img, overlay).convert("RGB")
    ensure_dir(os.path.dirname(output_vis_path))
    result.save(output_vis_path)
    return True


def draw_segmentation_labels(img_path, label_path, class_map, output_vis_path, title_text=""):
    """Renders instance segmentation masks with colored translucent polygon fills and crisp boundaries."""
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        return False

    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    mask_overlay = img.copy()

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    for series_idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]

        pts = []
        for j in range(0, len(coords), 2):
            px = int(coords[j] * w)
            py = int(coords[j+1] * h)
            pts.append([px, py])

        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        color_rgb = get_color(series_idx)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.fillPoly(mask_overlay, [pts], color_bgr)
        cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

    cv2.addWeighted(mask_overlay, 0.45, img, 0.55, 0, img)

    if title_text:
        cv2.putText(img, title_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, title_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    ensure_dir(os.path.dirname(output_vis_path))
    cv2.imwrite(output_vis_path, img)
    return True


def draw_pose_labels(img_path, label_path, output_vis_path, title_text=""):
    """Renders 5-point slice pose keypoint skeletons and connection lines for pie wedges."""
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        return False

    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    kpt_names = ["Center", "Start", "Inter1", "Inter2", "End"]
    kpt_colors = [
        (255, 50, 50),     # Center: Bright Red
        (50, 220, 50),     # Start: Green
        (255, 215, 0),     # Inter1: Gold
        (0, 220, 255),     # Inter2: Cyan
        (255, 0, 255)      # End: Magenta
    ]

    for wedge_idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 20:
            continue

        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h

        wedge_color = get_color(wedge_idx)
        draw.rectangle([cx - w/2, cy - h/2, cx + w/2, cy + h/2], outline=(wedge_color[0], wedge_color[1], wedge_color[2], 140), width=1)

        kpt_tokens = [float(p) for p in parts[5:]]
        points = []
        for k in range(5):
            kx = kpt_tokens[k*3] * img_w
            ky = kpt_tokens[k*3 + 1] * img_h
            kv = int(kpt_tokens[k*3 + 2])
            points.append((kx, ky, kv))

        valid_points = [p for p in points if p[2] > 0]
        if len(valid_points) >= 5:
            draw.line([(points[0][0], points[0][1]), (points[1][0], points[1][1])], fill=(wedge_color[0], wedge_color[1], wedge_color[2], 220), width=2)
            draw.line([(points[1][0], points[1][1]), (points[2][0], points[2][1])], fill=(wedge_color[0], wedge_color[1], wedge_color[2], 220), width=2)
            draw.line([(points[2][0], points[2][1]), (points[3][0], points[3][1])], fill=(wedge_color[0], wedge_color[1], wedge_color[2], 220), width=2)
            draw.line([(points[3][0], points[3][1]), (points[4][0], points[4][1])], fill=(wedge_color[0], wedge_color[1], wedge_color[2], 220), width=2)
            draw.line([(points[4][0], points[4][1]), (points[0][0], points[0][1])], fill=(wedge_color[0], wedge_color[1], wedge_color[2], 220), width=2)

        for k, (kx, ky, kv) in enumerate(points):
            if kv > 0:
                kc = kpt_colors[k]
                r = 5
                draw.ellipse([kx-r-1, ky-r-1, kx+r+1, ky+r+1], fill=(255, 255, 255))
                draw.ellipse([kx-r, ky-r, kx+r, ky+r], fill=kc, outline=(0, 0, 0), width=1)

    header_text = title_text or "Pie 5-Point Wedge Pose Keypoints"
    header_bbox = draw.textbbox((10, 10), header_text)
    draw.rectangle([header_bbox[0]-4, header_bbox[1]-4, header_bbox[2]+4, header_bbox[3]+4], fill=(0, 0, 0, 220))
    draw.text((10, 10), header_text, fill=(255, 255, 255))

    leg_x = max(10, img_w - 280)
    leg_y = 10
    draw.rectangle([leg_x - 5, leg_y - 4, img_w - 10, leg_y + 24], fill=(0, 0, 0, 220))
    cur_x = leg_x
    for k_name, k_col in zip(kpt_names, kpt_colors):
        draw.ellipse([cur_x, leg_y + 4, cur_x + 8, leg_y + 12], fill=k_col, outline=(255, 255, 255))
        draw.text((cur_x + 12, leg_y + 2), k_name[:4], fill=(255, 255, 255))
        cur_x += 52

    result = Image.alpha_composite(img, overlay).convert("RGB")
    ensure_dir(os.path.dirname(output_vis_path))
    result.save(output_vis_path)
    return True


def render_all_visualizations(max_per_folder=10):
    print("\n" + "=" * 80)
    print("STEP 3: RENDERING ANNOTATION OVERLAY VISUALIZATIONS")
    print("=" * 80)

    vis_counts = defaultdict(int)

    # 1. Detection Visualizations (AABB)
    detection_folders = [
        ("bar_obj_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_BAR', {}), "Bar Detection"),
        ("scatter_obj_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_SCATTER', {}), "Scatter Detection"),
        ("histogram_obj_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_HISTOGRAM', {}), "Histogram Detection"),
        ("box_elements_labels", {"0": "box", "1": "range_indicator", "2": "median_line", "3": "outlier", "4": "significance_marker"}, "Box Elements Specialist"),
        ("box_global_labels", {"0": "chart", "1": "axis_title", "2": "legend", "3": "chart_title", "4": "axis_labels"}, "Box Global Layout"),
        ("line_obj_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_LINE_OBJ', {}), "Line Objects Detection"),
        ("line_marker_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_LINE_MARKERS', {}), "Line Markers Detection"),
        ("area_obj_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_AREA_OBJ', {}), "Area Objects Detection"),
        ("pie_obj_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_PIE_OBJ', {}), "Pie Objects Detection"),
        ("heatmap_macro_labels", {"0": "chart", "1": "color_bar_region", "2": "legend"}, "Heatmap Macro Router"),
        ("heatmap_text_labels", {"0": "axis_labels", "1": "axis_title", "2": "chart_title"}, "Heatmap Text Line Specialist"),
    ]

    for folder_name, cmap, title_prefix in detection_folders:
        dir_path = os.path.join(OUTPUT_BASE, folder_name)
        if not os.path.exists(dir_path):
            continue
        txt_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".txt")])
        for fname in txt_files[:max_per_folder]:
            base_name = os.path.splitext(fname)[0]
            img_path = os.path.join(OUTPUT_BASE, "images", f"{base_name}.png")
            lbl_path = os.path.join(dir_path, fname)
            out_vis = os.path.join(VIS_DIR, "detection", f"{folder_name}_{base_name}.png")
            if draw_detection_labels(img_path, lbl_path, cmap, out_vis, f"{title_prefix} - {base_name}"):
                vis_counts['detection'] += 1

    # 2. Oriented Bounding Box Visualizations (OBB)
    obb_dir = os.path.join(OUTPUT_BASE, "labels_obb")
    if os.path.exists(obb_dir):
        obb_cmap = {
            "0": "chart", "1": "data_mark", "2": "axis_title", "3": "significance_marker",
            "4": "error_bar", "5": "legend", "6": "chart_title", "7": "data_label", "8": "axis_labels"
        }
        obb_files = sorted([f for f in os.listdir(obb_dir) if f.endswith(".txt")])
        for fname in obb_files[:max_per_folder]:
            base_name = os.path.splitext(fname)[0]
            img_path = os.path.join(OUTPUT_BASE, "images", f"{base_name}.png")
            lbl_path = os.path.join(obb_dir, fname)
            out_vis = os.path.join(VIS_DIR, "obb", f"obb_{base_name}.png")
            if draw_obb_labels(img_path, lbl_path, obb_cmap, out_vis, f"YOLOv8/v26-OBB Rotated Envelopes - {base_name}"):
                vis_counts['obb'] += 1

    # 3. Cropped Heatmap Specialists (Lattice & Colorbar)
    cropped_domains = [
        ("heatmap_lattice_images", "heatmap_lattice_labels", {"0": "cell", "1": "data_label"}, "Heatmap Lattice Specialist Crop"),
        ("heatmap_colorbar_images", "heatmap_colorbar_labels", {"0": "color_bar", "1": "color_bar_label", "2": "color_bar_title"}, "Heatmap Colorbar Specialist Crop")
    ]
    for img_folder, lbl_folder, cmap, title_prefix in cropped_domains:
        img_dir = os.path.join(OUTPUT_BASE, img_folder)
        lbl_dir = os.path.join(OUTPUT_BASE, lbl_folder)
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        for fname in img_files[:max_per_folder]:
            base_name = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            out_vis = os.path.join(VIS_DIR, "detection", f"{img_folder}_{base_name}.png")
            if draw_detection_labels(img_path, lbl_path, cmap, out_vis, f"{title_prefix} - {base_name}"):
                vis_counts['detection'] += 1

    # 4. Instance Segmentation Visualizations (Line & Area)
    seg_domains = [
        ("line_seg_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_LINE_SEG', {}), "Line Instance Segmentation (Ribbon Polygons)"),
        ("area_seg_labels", OCR_TRAINING_CONFIG.get('CLASS_MAP_AREA_SEG', {}), "Area Instance Segmentation (Stacked Fill Polygons)")
    ]
    for folder_name, cmap, title_prefix in seg_domains:
        dir_path = os.path.join(OUTPUT_BASE, folder_name)
        if not os.path.exists(dir_path):
            continue
        txt_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".txt")])
        for fname in txt_files[:max_per_folder]:
            base_name = os.path.splitext(fname)[0]
            img_path = os.path.join(OUTPUT_BASE, "images", f"{base_name}.png")
            lbl_path = os.path.join(dir_path, fname)
            out_vis = os.path.join(VIS_DIR, "segmentation", f"{folder_name}_{base_name}.png")
            if draw_segmentation_labels(img_path, lbl_path, cmap, out_vis, f"{title_prefix} - {base_name}"):
                vis_counts['segmentation'] += 1

    # 5. Pose Keypoints Visualizations (Pie Wedges)
    pie_pose_dir = os.path.join(OUTPUT_BASE, "pie_pose_labels")
    if os.path.exists(pie_pose_dir):
        txt_files = sorted([f for f in os.listdir(pie_pose_dir) if f.endswith(".txt")])
        for fname in txt_files[:max_per_folder]:
            base_name = os.path.splitext(fname)[0]
            img_path = os.path.join(OUTPUT_BASE, "images", f"{base_name}.png")
            lbl_path = os.path.join(pie_pose_dir, fname)
            out_vis = os.path.join(VIS_DIR, "pose", f"pie_pose_{base_name}.png")
            if draw_pose_labels(img_path, lbl_path, out_vis, f"Pie 5-Point Wedge Pose - {base_name}"):
                vis_counts['pose'] += 1

    # 6. Topological Series Keypoint Visualizations (Detailed JSON)
    labels_dir = os.path.join(OUTPUT_BASE, "labels")
    if os.path.exists(labels_dir):
        json_files = sorted([f for f in os.listdir(labels_dir) if f.endswith("_detailed.json")])
        for fname in json_files[:max_per_folder]:
            base_name = fname.replace("_detailed.json", "")
            img_path = os.path.join(OUTPUT_BASE, "images", f"{base_name}.png")
            json_path = os.path.join(labels_dir, fname)
            out_vis = os.path.join(VIS_DIR, "topologies", f"topology_{base_name}.png")
            if draw_topological_keypoints(img_path, json_path, out_vis, f"Series Topology Graph - {base_name}"):
                vis_counts['topologies'] += 1

    # 7. Multi-panel Chart Detection Visualizations
    multi_chart_cls_map = {
        "0": "chart", "1": "bar/line/wedge", "2": "axis_title", "3": "significance_marker",
        "4": "error_bar/legend", "5": "legend/title", "6": "chart_title", "7": "data_label", "8": "axis_labels"
    }
    if os.path.exists(labels_dir):
        multi_candidates = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt") and not f.endswith("_obb.txt")])
        for fname in multi_candidates[-max_per_folder:]:
            base_name = os.path.splitext(fname)[0]
            img_path = os.path.join(OUTPUT_BASE, "images", f"{base_name}.png")
            lbl_path = os.path.join(labels_dir, fname)
            out_vis = os.path.join(VIS_DIR, "multipanel", f"multipanel_{base_name}.png")
            if draw_detection_labels(img_path, lbl_path, multi_chart_cls_map, out_vis, f"Multi-Panel Layout - {base_name}"):
                vis_counts['multipanel'] += 1

    print(f"\n✓ Generated Visualizations in '{VIS_DIR}':")
    print(f"  • Detection overlays (AABB):  {vis_counts['detection']} images")
    print(f"  • Oriented Bounding Boxes (OBB): {vis_counts['obb']} images")
    print(f"  • Segmentation overlays:      {vis_counts['segmentation']} images")
    print(f"  • Pose skeleton overlays:     {vis_counts['pose']} images")
    print(f"  • Topological graph overlays: {vis_counts['topologies']} images")
    print(f"  • Multi-panel overlays:       {vis_counts['multipanel']} images")
    print(f"  • Total preview images:       {sum(vis_counts.values())} images")


def main():
    parser = argparse.ArgumentParser(description="Dataset Generation, Verification, and Annotation Visualizer")
    parser.add_argument("--num-per-type", type=int, default=20, help="Number of single charts per chart type (default: 20)")
    parser.add_argument("--num-multi", type=int, default=20, help="Number of multi-panel charts (default: 20)")
    parser.add_argument("--num-vis", type=int, default=10, help="Number of annotated visualizations per folder/modality (default: 10)")
    parser.add_argument("--quick", action="store_true", help="Quick run (3 per type + 2 multi-panel)")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation and verify/visualize existing output")
    args = parser.parse_args()

    num_per_type = 3 if args.quick else args.num_per_type
    num_multi = 2 if args.quick else args.num_multi

    start_time = time.time()
    import shutil

    if not args.skip_gen and os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)

    # 1. Run Generation
    if not args.skip_gen:
        run_generation(num_per_type=num_per_type, num_multi=num_multi)

    # 2. Verify Dataset
    validation_issues = verify_dataset()

    # 3. Render Visualizations
    render_all_visualizations(max_per_folder=args.num_vis)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"SUCCESS: Pipeline completed in {elapsed:.2f} seconds.")
    print(f"Dataset root:        {os.path.abspath(OUTPUT_BASE)}")
    print(f"Visualizations root: {os.path.abspath(VIS_DIR)}")
    print("=" * 80)

    if validation_issues:
        sys.exit(1)


if __name__ == '__main__':
    main()
