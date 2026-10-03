import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')  # Thread-safe, headless backend
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib import patches, rcParams, transforms, colormaps
from matplotlib.colors import ListedColormap
from matplotlib.collections import PolyCollection, PathCollection, QuadMesh

import os
import io
import sys
import time
import math
import json
import random
import warnings
import traceback
import subprocess
import hashlib
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont
try:
    from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box, MultiPolygon as ShapelyMultiPolygon
    from shapely.validation import make_valid as shapely_make_valid
except ImportError:
    ShapelyPolygon = None
    shapely_box = None
    ShapelyMultiPolygon = None
    shapely_make_valid = None

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

from versions import (
    ANNOTATION_SCHEMA_VERSION, DATASET_VERSION,
    ANNOTATION_SCHEMA_VERSION_V4, DATASET_VERSION_V4,
)
from synth.semantics import sample_chart_title
from themes import THEMES
from effects import (
    apply_jpeg_compression_effect, apply_noise_effect, apply_blur_effect,
    apply_motion_blur_effect, apply_low_res_effect, apply_pixelation_effect,
    apply_posterize_effect, apply_color_variation_effect, apply_ui_chrome_effect,
    apply_watermark_effect, apply_vignette_effect, apply_scanner_streaks_effect,
    apply_clipping_effect, apply_printing_artifacts_effect, apply_mouse_cursor_effect,
    apply_text_degradation_effect, apply_grid_occlusion_effect, apply_scan_rotation_effect,
    apply_grayscale_effect, apply_perspective_warp_effect,
    apply_uneven_lighting_effect, apply_chromatic_aberration_effect,
    apply_pdf_document_context_effect, apply_page_curl
)
from chart import (
    _generate_bar_chart, _generate_line_chart, _generate_scatter_chart,
    _generate_boxplot_chart, _generate_heatmap_chart, _generate_pie_chart,
    _generate_area_chart, _generate_histogram, add_data_labels, apply_chart_theme
)
warnings.filterwarnings("ignore", category=UserWarning)

BoundingBox = namedtuple('BoundingBox', ['x0', 'y0', 'x1', 'y1'])

def validate_coordinates(coords, context="unknown"):
    """
    Validate coordinate lists for debugging and consistency.
    """
    if not coords:
        return True

    try:
        for i, coord in enumerate(coords):
            if len(coord) < 2:
                continue
            x, y = coord[0], coord[1]
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
        return True
    except Exception as e:
        return False

# ===================================================================================
# ==                               CONFIGURATION                                   ==
# ===================================================================================
from config_loader import load_config, parse_set_override
from config_defaults import OCR_TRAINING_CONFIG as GENERATION_CONFIG

def _build_chart_class_maps(cfg):
    return {
        'bar': cfg.get('CLASS_MAP_BAR', {}),
        'scatter': cfg.get('CLASS_MAP_SCATTER', {}),
        'box': cfg.get('CLASS_MAP_BOX', {}),
        'histogram': cfg.get('CLASS_MAP_HISTOGRAM', {}),
        'heatmap': cfg.get('CLASS_MAP_HEATMAP', {}),
        'area_obj': cfg.get('CLASS_MAP_AREA_OBJ', {}),
        'area_seg': cfg.get('CLASS_MAP_AREA_SEG', {}),
        'pie': cfg.get('CLASS_MAP_PIE_OBJ', {}),
        'pie_pose': cfg.get('CLASS_MAP_PIE_POSE', {}),
        'line_obj': cfg.get('CLASS_MAP_LINE_OBJ', {}),
        'line_seg': cfg.get('CLASS_MAP_LINE_SEG', {}),
        'line_markers': cfg.get('CLASS_MAP_LINE_MARKERS', {})
    }

CHART_CLASS_MAPS = _build_chart_class_maps(GENERATION_CONFIG)

def set_active_config(cfg):
    global GENERATION_CONFIG
    GENERATION_CONFIG = cfg
    CHART_CLASS_MAPS.clear()
    CHART_CLASS_MAPS.update(_build_chart_class_maps(cfg))

def _init_worker(cfg):
    """Initializer for worker processes to thread resolved cfg and update class maps."""
    set_active_config(cfg)

# ===================================================================================
# == UTILITY FUNCTIONS
# ===================================================================================

def is_float(text):
    try:
        float(text)
        return True
    except (ValueError, TypeError):
        return False

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def bbox_to_yolo_norm(x0, y0, x1, y1, img_w, img_h):
    if img_w == 0 or img_h == 0:
        return 0, 0, 0, 0
    dw = 1. / img_w
    dh = 1. / img_h
    x = (x0 + x1) / 2.0
    y = (y0 + y1) / 2.0
    w = x1 - x0
    h = y1 - y0
    return x * dw, y * dh, w * dw, h * dh

def bbox_to_xyxy(bbox, img_h):
    """Convert matplotlib bbox to [x0, y0, x1, y1] xyxy format"""
    if hasattr(bbox, 'extents'):
        x0, y0, x1, y1 = bbox.extents
    else:
        x0, y0, x1, y1 = bbox
    return [int(x0), int(img_h - y1), int(x1), int(img_h - y0)]

def bbox_to_xyxy_absolute(bbox, img_h):
    """Convert matplotlib bbox to [x0, y0, x1, y1] absolute xyxy format"""
    if hasattr(bbox, 'extents'):
        x0, y0, x1, y1 = bbox.extents
    else:
        x0, y0, x1, y1 = bbox
    # Convert from matplotlib coordinates to image coordinates
    abs_y0 = int(img_h - y1)
    abs_y1 = int(img_h - y0)
    return [int(x0), abs_y0, int(x1), abs_y1]

def ensure_min_bbox_thickness(bbox, min_size=4.0):
    """
    Guarantee a bounding box has at least `min_size` pixels of extent along
    BOTH the X and Y axes by applying bidirectional symmetric padding.

    Thin 1D elements (vertical/horizontal median lines, whiskers without caps,
    single-line error bars, connector lines) can otherwise collapse to 0-1px
    bounding boxes, which vanish or degrade feature gradients during CNN/ViT
    downsampling. Because the deficit is computed independently per axis, this
    works correctly regardless of chart orientation (e.g. a horizontal
    boxplot's vertical median line gets padded in X, not just Y).

    Args:
        bbox: A matplotlib Bbox-like object (has `.extents`) or an
              (x0, y0, x1, y1) tuple/list.
        min_size: Minimum guaranteed width and height, in pixels.

    Returns:
        A matplotlib Bbox with width/height >= min_size (original bbox
        returned unchanged, as-is, if it already meets the minimum).
    """
    if bbox is None:
        return bbox

    if hasattr(bbox, 'extents'):
        x0, y0, x1, y1 = bbox.extents
    else:
        x0, y0, x1, y1 = bbox

    width = x1 - x0
    height = y1 - y0

    pad_x = max(0.0, (min_size - width) / 2.0)
    pad_y = max(0.0, (min_size - height) / 2.0)

    if pad_x == 0.0 and pad_y == 0.0:
        return bbox

    return transforms.Bbox.from_extents(
        x0 - pad_x, y0 - pad_y,
        x1 + pad_x, y1 + pad_y
    )


def canonicalize_and_validate_obb(pts, y_down=True):
    """
    Ensure all 4-vertex OBBs maintain a canonical clockwise winding order
    and pass a convexity check.
    If non-convex or bowtie, reorders points via a pure-NumPy 4-point convex hull.
    Returns list of 4 (x, y) tuples if valid and convex, or None if degenerate.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) != 4:
        return None

    def is_convex_cw(p):
        for i in range(4):
            e1 = p[(i + 1) % 4] - p[i]
            e2 = p[(i + 2) % 4] - p[(i + 1) % 4]
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            if not y_down:
                cross = -cross
            if cross <= 1e-6:
                return False
        return True

    if is_convex_cw(pts):
        return [(float(x), float(y)) for x, y in pts]

    pts_rev = pts[::-1]
    if is_convex_cw(pts_rev):
        return [(float(x), float(y)) for x, y in pts_rev]

    # 4-point convex hull to repair self-intersecting bowtie
    start = np.lexsort((pts[:, 1], pts[:, 0]))[0]
    hull = []
    p = start
    while True:
        hull.append(p)
        q = (p + 1) % len(pts)
        for i in range(len(pts)):
            v1 = pts[q] - pts[p]
            v2 = pts[i] - pts[p]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross > 1e-7:
                q = i
        p = q
        if p == start:
            break

    if len(hull) != 4:
        return None

    repaired = pts[hull]
    if is_convex_cw(repaired):
        return [(float(x), float(y)) for x, y in repaired]
    elif is_convex_cw(repaired[::-1]):
        return [(float(x), float(y)) for x, y in repaired[::-1]]
    return None


def get_text_obb_and_bbox(text_artist, renderer, img_h=None):
    """
    Calculate tight unrotated text dimensions from font metrics, and analytically
    compute the 4-corner oriented bounding box (OBB) using text rotation angle and
    anchor alignment (ha, va), eliminating square AABB inflation.

    Returns:
        bbox: Matplotlib Bbox or BoundingBox in display space
        obb: 4-vertex list [(x1, y1), ...] in clockwise winding order
    """
    orig_bbox = text_artist.get_window_extent(renderer)
    try:
        rot = text_artist.get_rotation()

        if abs(rot) < 1e-3:
            x0, y0, x1, y1 = orig_bbox.x0, orig_bbox.y0, orig_bbox.x1, orig_bbox.y1
            if img_h is not None:
                obb = [
                    (float(x0), float(img_h - y1)),
                    (float(x1), float(img_h - y1)),
                    (float(x1), float(img_h - y0)),
                    (float(x0), float(img_h - y0))
                ]
            else:
                obb = [
                    (float(x0), float(y0)),
                    (float(x1), float(y0)),
                    (float(x1), float(y1)),
                    (float(x0), float(y1))
                ]
            return orig_bbox, obb

        text = text_artist.get_text()
        clean_text, ismath = (
            text_artist._preprocess_math(text)
            if hasattr(text_artist, '_preprocess_math')
            else (text, False)
        )
        fontprop = text_artist.get_fontproperties()
        w, h, d = renderer.get_text_width_height_descent(clean_text, fontprop, ismath=ismath)

        if w <= 0 or h <= 0:
            return orig_bbox, None

        corners_horiz = np.array([(0.0, -h), (0.0, 0.0), (w, 0.0), (w, -h)], dtype=np.float64)
        M = transforms.Affine2D().rotate_deg(float(rot))
        corners_rot = M.transform(corners_horiz)

        c_xmin, c_xmax = corners_rot[:, 0].min(), corners_rot[:, 0].max()
        c_ymin, c_ymax = corners_rot[:, 1].min(), corners_rot[:, 1].max()

        halign = text_artist.get_ha()
        valign = text_artist.get_va()

        if halign == 'center':
            offsetx = (c_xmin + c_xmax) / 2.0
        elif halign == 'right':
            offsetx = c_xmax
        else:
            offsetx = c_xmin

        if valign == 'center':
            offsety = (c_ymin + c_ymax) / 2.0
        elif valign == 'top':
            offsety = c_ymax
        elif valign == 'baseline':
            offsety = c_ymin + d
        else:
            offsety = c_ymin

        pos_coords = (
            text_artist.get_unitless_position()
            if hasattr(text_artist, 'get_unitless_position')
            else text_artist.get_position()
        )
        pos = text_artist.get_transform().transform(pos_coords)
        final_corners = corners_rot - [offsetx, offsety] + pos

        if img_h is not None:
            raw_obb = [(float(p[0]), float(img_h - p[1])) for p in final_corners]
            obb = canonicalize_and_validate_obb(raw_obb, y_down=True)
        else:
            raw_obb = [(float(p[0]), float(p[1])) for p in final_corners]
            obb = canonicalize_and_validate_obb(raw_obb, y_down=False)

        return orig_bbox, obb
    except Exception:
        return orig_bbox, None


def has_non_background_pixels(label_artist, fig, ax, bgcolor=None, threshold=1):
    """
    Analytically check if label artist is visible, non-empty, and has a valid bounding box.
    Eliminates full-figure canvas redraws and buffer pixel scanning.
    """
    try:
        if hasattr(label_artist, 'get_visible') and not label_artist.get_visible():
            return False
        if hasattr(label_artist, 'get_text') and not label_artist.get_text().strip():
            return False
        renderer = fig.canvas.get_renderer()
        if renderer is None:
            return True
        bbox = label_artist.get_window_extent(renderer)
        if bbox.width < 1 or bbox.height < 1 or bbox.x1 <= bbox.x0 or bbox.y1 <= bbox.y0:
            return False
        if bbox.x1 <= 0 or bbox.y1 <= 0 or bbox.x0 >= fig.bbox.width or bbox.y0 >= fig.bbox.height:
            return False
        return True
    except Exception:
        return False




def get_granular_annotations(fig, chart_info_map, cls_map):
    """
    Extract detailed bounding box annotations for all visible chart components
    (bars, lines, text, scatter points, wedges, heatmap cells, legends, etc.).
    """
    reverse_map = {v: k for k, v in cls_map.items()}

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        print("WARNING: Could not obtain renderer, annotations will be empty")
        return []

    annotations = []
    fig_bbox = fig.get_window_extent(renderer)
    img_h = fig_bbox.height
    seen_annotations = set()

    def add_unique_annotation(class_id, bbox, text=None, obb=None, artist=None, attrs=None):
        # ``ax_idx`` is the enclosing loop variable (index into fig.axes); it is resolved at call
        # time. ``artist`` / ``attrs`` are per-element detail hooks consumed (and stripped) by
        # detailed.prepare_detail_annotations. Neither is read by the legacy writers.
        if bbox is None:
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Skipping annotation - bbox is None")
            return False

        try:
            if hasattr(bbox, 'width') and hasattr(bbox, 'height'):
                width, height = bbox.width, bbox.height
                x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            elif hasattr(bbox, 'extents'):
                x0, y0, x1, y1 = bbox.extents
                width, height = x1 - x0, y1 - y0
            else:
                x0, y0, x1, y1 = bbox
                width, height = x1 - x0, y1 - y0

        except (AttributeError, ValueError, TypeError) as e:
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Bbox extraction failed: {e}")
            return False

        if width <= 0.5 or height <= 0.5:
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Bbox too small - w:{width:.2f}, h:{height:.2f}")
            return False

        key = (class_id, round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))
        if key not in seen_annotations:
            entry = {'class_id': class_id, 'bbox': bbox, 'ax_index': ax_idx}
            if text:
                entry['text'] = text
            if obb is not None:
                entry['obb'] = obb
            if artist is not None:
                entry['_artist'] = artist
            if attrs:
                entry['attrs'] = attrs
            annotations.append(entry)
            seen_annotations.add(key)
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Added annotation - class:{class_id}, bbox:[{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}]")
            return True
        else:
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Duplicate annotation filtered - class:{class_id}")
            return False

    for ax_idx, ax in enumerate(fig.axes):
        if not ax.get_visible():
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Skipping invisible axis")
            continue

        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str')

        if GENERATION_CONFIG.get('debug_mode', False):
            print(f"DEBUG: AX[{ax_idx}]: Processing chart_type={chart_type}")

        # Chart Title
        if 'chart_title' in reverse_map:
            title = ax.title
            if title and title.get_visible() and title.get_text().strip():
                try:
                    title_bbox, title_obb = get_text_obb_and_bbox(title, renderer, img_h=img_h)
                    if add_unique_annotation(reverse_map['chart_title'], title_bbox, text=title.get_text().strip(), obb=title_obb):
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Added chart title annotation")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Chart title bbox failed: {e}")

        # Legend
        if 'legend' in reverse_map:
            legend = ax.get_legend()
            if legend and legend.get_visible():
                valid_texts = [t.get_text().strip() for t in legend.get_texts()
                               if t.get_visible() and t.get_text().strip()]
                if valid_texts:
                    try:
                        if has_non_background_pixels(legend, fig, ax, ax.get_facecolor(), threshold=5):
                            legend_bbox = legend.get_window_extent(renderer)
                            if add_unique_annotation(reverse_map['legend'], legend_bbox):
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Added legend annotation")
                        else:
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Legend empty (no pixels), skipping")
                    except Exception as e:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Legend bbox failed: {e}")

        # Axis Titles
        if 'axis_title' in reverse_map:
            # X-axis title
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                try:
                    xlabel_bbox, xlabel_obb = get_text_obb_and_bbox(ax.xaxis.label, renderer, img_h=img_h)
                    if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip(), obb=xlabel_obb):
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Added x-axis title annotation")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: X-axis title bbox failed: {e}")

            # Y-axis title
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                try:
                    ylabel_bbox, ylabel_obb = get_text_obb_and_bbox(ax.yaxis.label, renderer, img_h=img_h)
                    if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip(), obb=ylabel_obb):
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Added y-axis title annotation")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Y-axis title bbox failed: {e}")

        # Axis Tick Labels
        if 'axis_labels' in reverse_map:
            scale_axis_info = chart_info.get('scale_axis_info', {})
            primary_scale_axis = scale_axis_info.get('primary_scale_axis', 'y')
            bg_color = ax.get_facecolor()

            x_min, x_max = sorted(ax.get_xlim())
            y_min, y_max = sorted(ax.get_ylim())

            # X-axis labels
            x_labels_added = 0
            for label in ax.get_xticklabels():
                if label.get_visible() and label.get_text().strip():
                    if x_min <= label.get_position()[0] <= x_max:
                        if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                            try:
                                label_bbox, label_obb = get_text_obb_and_bbox(label, renderer, img_h=img_h)
                                if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip(), obb=label_obb):
                                    x_labels_added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: X-label bbox failed: {e}")

            # Y-axis labels
            y_labels_added = 0
            for label in ax.get_yticklabels():
                if label.get_visible() and label.get_text().strip():
                    if y_min <= label.get_position()[1] <= y_max:
                        if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                            try:
                                label_bbox, label_obb = get_text_obb_and_bbox(label, renderer, img_h=img_h)
                                if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip(), obb=label_obb):
                                    y_labels_added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Y-label bbox failed: {e}")

            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Added {x_labels_added} x-labels, {y_labels_added} y-labels")

        # General chart / plot-area bounding box
        if 'chart' in reverse_map:
            try:
                chart_bbox = ax.get_window_extent(renderer)
                if add_unique_annotation(reverse_map['chart'], chart_bbox):
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added general chart layout box")
            except Exception as e:
                if GENERATION_CONFIG.get('debug_mode', False):
                    print(f"DEBUG: AX[{ax_idx}]: General chart layout box failed: {e}")

        # Bar Chart Elements
        if chart_type == 'bar' and 'bar' in reverse_map:
            data_artists = chart_info.get('data_artists', [])
            bars_added = 0

            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Processing {len(data_artists)} bar data artists")

            for artist_idx, artist in enumerate(data_artists):
                if artist is None:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx} is None")
                    continue

                try:
                    is_visible = artist.get_visible()
                except:
                    is_visible = True

                if not is_visible:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx} not visible")
                    continue

                is_rectangle = (isinstance(artist, patches.Rectangle) or
                              str(type(artist).__name__) == 'Rectangle' or
                              hasattr(artist, 'get_x') and hasattr(artist, 'get_y') and
                              hasattr(artist, 'get_width') and hasattr(artist, 'get_height'))

                if is_rectangle:
                    try:
                        artist_bbox = artist.get_window_extent(renderer)
                        if artist_bbox and add_unique_annotation(reverse_map['bar'], artist_bbox, artist=artist):
                            bars_added += 1
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Added bar annotation #{artist_idx}")
                    except Exception as e:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Bar bbox failed for artist {artist_idx}: {e}")
                else:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx} type: {type(artist).__name__} - not Rectangle")

            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Successfully added {bars_added} bar annotations")

        elif chart_type == 'histogram':
            debug = GENERATION_CONFIG.get('debug_mode', False)
            if debug:
                print(f"DEBUG [AX{ax_idx}] HISTOGRAM axis processing override")

            if "bar" in reverse_map:
                dataartists = chart_info.get("data_artists", [])
                barsadded = 0
                if debug:
                    print(f"DEBUG: AX{ax_idx} Processing {len(dataartists)} histogram bar patches")

                for artistidx, artist in enumerate(dataartists):
                    if artist is None:
                        continue
                    try:
                        isvisible = artist.get_visible()
                    except:
                        isvisible = True

                    if not isvisible:
                        continue

                    # Histogram patches are always Rectangle objects
                    if isinstance(artist, patches.Rectangle):
                        try:
                            artistbbox = artist.get_window_extent(renderer)
                            if artistbbox and add_unique_annotation(reverse_map["bar"], artistbbox, artist=artist):
                                barsadded += 1
                                if debug:
                                    print(f"DEBUG: AX{ax_idx} Added histogram bar {artistidx}")
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX{ax_idx} Histogram bar bbox failed: {e}")

                if debug:
                    print(f"DEBUG: AX{ax_idx} Total histogram bars annotated: {barsadded}")

            # Override the general axis processing for histograms
            if 'axis_title' in reverse_map:
                titles_added = 0
                # Axis TITLES (xlabel/ylabel - these should be axis_title)
                if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                    try:
                        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip()):
                            titles_added += 1
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Added histogram X-axis TITLE")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG [AX{ax_idx}] Histogram X-title error: {e}")

                # X-axis title
                if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                    try:
                        ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip()):
                            titles_added += 1
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Added histogram Y-axis TITLE")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG [AX{ax_idx}] Histogram Y-title error: {e}")

                if debug:
                    print(f"DEBUG [AX{ax_idx}] HISTOGRAM axis titles: {titles_added}")
                # Y-axis title

            if 'axis_labels' in reverse_map:
                labels_added = 0
                bg_color = ax.get_facecolor()
                x_min, x_max = sorted(ax.get_xlim())
                y_min, y_max = sorted(ax.get_ylim())

                # X-axis tick labels
                for label in ax.get_xticklabels():
                    if label.get_visible() and label.get_text().strip():
                        if x_min <= label.get_position()[0] <= x_max:
                            if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                                try:
                                    label_bbox = label.get_window_extent(renderer)
                                    if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                        labels_added += 1
                                except Exception as e:
                                    if debug:
                                        print(f"DEBUG [AX{ax_idx}] Histogram X-label error: {e}")

                # Y-axis tick labels
                for label in ax.get_yticklabels():
                    if label.get_visible() and label.get_text().strip():
                        if y_min <= label.get_position()[1] <= y_max:
                            if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                                try:
                                    label_bbox = label.get_window_extent(renderer)
                                    if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                        labels_added += 1
                                except Exception as e:
                                    if debug:
                                        print(f"DEBUG [AX{ax_idx}] Histogram Y-label error: {e}")

                if debug:
                    print(f"DEBUG [AX{ax_idx}] HISTOGRAM axis labels: {labels_added}")
                # Y-axis tick labels

            if 'data_label' in reverse_map:
                data_labels_added = 0
                other_artists = chart_info.get('other_artists', [])

                if debug:
                    print(f"DEBUG [AX{ax_idx}] Processing {len(other_artists)} other_artists for data labels")

                # Data labels are stored in other_artists (text annotations)
                for artist_idx, artist in enumerate(other_artists):
                    # Check if artist is a Text object
                    if hasattr(artist, 'get_text') and hasattr(artist, 'get_window_extent'):
                        try:
                            # Verify it's visible and has content
                            if artist.get_visible() and artist.get_text().strip():
                                label_bbox, label_obb = get_text_obb_and_bbox(artist, renderer, img_h=img_h)

                                if add_unique_annotation(reverse_map['data_label'], label_bbox, text=artist.get_text().strip(), obb=label_obb):
                                    data_labels_added += 1
                                    if debug:
                                        print(f"DEBUG [AX{ax_idx}] Added data label: '{artist.get_text()}'")
                        except Exception as e:
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Data label bbox failed for artist {artist_idx}: {e}")

                if debug:
                    print(f"DEBUG [AX{ax_idx}] HISTOGRAM data labels: {data_labels_added}")

        # Enhanced Box plot processing with fallback
        elif chart_type == 'box':
            scale_axis_info_box = chart_info.get('scale_axis_info', {})
            # Look up the Matplotlib boxplot artists dictionary (contains 'boxes', 'whiskers', 'caps', 'medians', 'fliers')
            bp_artists = (
                chart_info.get('boxplot_artists')
                or scale_axis_info_box.get('boxplot_raw')
                or (chart_info.get('boxplot_dict', {}).get('boxplot_raw') if isinstance(chart_info.get('boxplot_dict'), dict) else None)
            )
            if not bp_artists and isinstance(chart_info.get('boxplot_dict'), dict) and 'boxes' in chart_info['boxplot_dict']:
                bp_artists = chart_info['boxplot_dict']

            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Boxplot dict: {bp_artists is not None} | Keys: {list(bp_artists.keys()) if bp_artists else 'None'}")
                print(f"DEBUG: AX[{ax_idx}]: Boxes: {len(bp_artists.get('boxes', [])) if bp_artists else 0}")

            boxes_processed = False
            if bp_artists and bp_artists.get('boxes'):
                if GENERATION_CONFIG.get('debug_mode', False):
                    print(f"DEBUG: AX[{ax_idx}]: Processing boxplot with {len(bp_artists['boxes'])} boxes")

                # Boxes
                if 'box' in reverse_map:
                    added = 0
                    for group_idx, box_artist in enumerate(bp_artists['boxes']):
                        if box_artist and box_artist.get_visible():
                            try:
                                bbox = box_artist.get_window_extent(renderer)
                                if bbox.width > 0.5 and bbox.height > 0.5 and add_unique_annotation(reverse_map['box'], bbox, attrs={'group_index': group_idx}):
                                    added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Box bbox error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} box annotations")

                # Medians
                if 'median_line' in reverse_map:
                    added = 0
                    for group_idx, median in enumerate(bp_artists.get('medians', [])):
                        if median and median.get_visible():
                            try:
                                orig_bbox = median.get_window_extent(renderer)
                                # Guarantee minimum pixel thickness in BOTH directions
                                # so the median line stays detectable regardless of
                                # box orientation (vertical boxplot -> thin height;
                                # horizontal boxplot -> thin width).
                                padded = ensure_min_bbox_thickness(orig_bbox, min_size=4.0)

                                if padded.width > 0.5 and padded.height > 0.5:
                                    if add_unique_annotation(reverse_map['median_line'], padded, attrs={'group_index': group_idx}):
                                        added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Median error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} median annotations")

                # Range indicators (Whiskers and Caps)
                # The annotation encompasses the full visual range indicator:
                # the whisker lines and the caps at each end.  We union the
                # raw artist bboxes (no per-element padding) so the natural
                # cap width is preserved in the result.
                if 'range_indicator' in reverse_map:
                    added = 0
                    boxes_list = bp_artists.get('boxes', [])
                    num_boxes = len(boxes_list)
                    whiskers = bp_artists.get('whiskers', [])
                    caps = bp_artists.get('caps', [])
                    for i in range(num_boxes):
                        try:
                            artists = []
                            idxs = [2*i, 2*i + 1]
                            for idx in idxs:
                                if len(whiskers) > idx and whiskers[idx]:
                                    artists.append(whiskers[idx])
                                if len(caps) > idx and caps[idx]:
                                    artists.append(caps[idx])

                            bboxes = []
                            for art in artists:
                                if art and art.get_visible():
                                    try:
                                        bbox = art.get_window_extent(renderer)
                                        if bbox:
                                            bboxes.append(bbox)
                                    except Exception as e:
                                        if GENERATION_CONFIG.get('debug_mode', False):
                                            print(f"DEBUG: AX[{ax_idx}]: Error processing range indicator artist: {e}")

                            if bboxes:
                                union_bbox = transforms.Bbox.union(bboxes)

                                # Guarantee minimum pixel thickness in both
                                # directions (e.g. whiskers with no caps can
                                # otherwise collapse to near-zero width/height).
                                union_bbox = ensure_min_bbox_thickness(union_bbox, min_size=2.0)
                                if union_bbox.width > 0.5 and add_unique_annotation(reverse_map['range_indicator'], union_bbox, attrs={'group_index': i}):
                                    added += 1
                        except Exception as e:
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Range {i} error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} range annotations")

                # Outliers (Fliers)
                if 'outlier' in reverse_map:
                    added = 0
                    for group_idx, flier in enumerate(bp_artists.get('fliers', [])):
                        if flier and flier.get_visible():
                            try:
                                xdata, ydata = flier.get_xdata(), flier.get_ydata()
                                for x, y in zip(xdata, ydata):
                                    px, py = ax.transData.transform_point((x, y))
                                    size = 3
                                    bbox = transforms.Bbox.from_extents(
                                        px - size, py - size, px + size, py + size
                                    )
                                    if bbox.width > 0.5 and add_unique_annotation(
                                            reverse_map['outlier'], bbox,
                                            attrs={'group_index': group_idx, 'data': [float(x), float(y)]}):
                                        added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Outlier error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} outlier annotations")

                boxes_processed = True

            # FALLBACK: If boxplot_dict failed, use data_artists
            if not boxes_processed and 'box' in reverse_map:
                added = 0
                for artist in chart_info.get('data_artists', []):
                    if isinstance(artist, patches.Rectangle) and artist.get_visible():
                        try:
                            bbox = artist.get_window_extent(renderer)
                            if bbox.width > 0.5 and add_unique_annotation(reverse_map['box'], bbox):
                                added += 1
                        except Exception as e:
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Fallback box error: {e}")
                if GENERATION_CONFIG.get('debug_mode', False):
                    print(f"DEBUG: AX[{ax_idx}]: Fallback added {added} box annotations from data_artists")

        # Scatter Chart Points
        elif chart_type == 'scatter' and 'data_point' in reverse_map:
            data_artists = chart_info.get('data_artists', [])
            points_added = 0
            debug = GENERATION_CONFIG.get('debug_mode', False)

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: SCATTER processing {len(data_artists)} artists")

            for artist_idx, artist in enumerate(data_artists):
                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx}: {type(artist).__name__}")

                if not isinstance(artist, PathCollection):
                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Not PathCollection, skipping")
                    continue

                try:
                    offsets = artist.get_offsets()
                    sizes = artist.get_sizes()

                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Offsets: {offsets.shape}, Sizes: {sizes}")

                    if len(offsets) == 0:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: No offsets")
                        continue

                    is_uniform_size = (sizes.size == 1)
                    # Convert marker points^2 area to display pixel radius
                    # 1 pt = fig.dpi / 72.0 pixels
                    # For marker of area s in pt^2, side length in pt is sqrt(s), radius in pt is sqrt(s)/2.0
                    points_to_pixels = fig.dpi / 72.0

                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: DPI={fig.dpi}, conversion={points_to_pixels}")
                        print(f"DEBUG: AX[{ax_idx}]: Will process {len(offsets)} points")

                    for i, (x_data, y_data) in enumerate(offsets):
                        px, py = ax.transData.transform_point((x_data, y_data))
                        s = float(sizes[0] if is_uniform_size else sizes[i])
                        # Radius in pixels matching visual marker extent
                        radius = max(3.0, (np.sqrt(s) / 2.0) * points_to_pixels)

                        cb_x0 = max(float(ax.bbox.x0), px - radius)
                        cb_y0 = max(float(ax.bbox.y0), py - radius)
                        cb_x1 = min(float(ax.bbox.x1), px + radius)
                        cb_y1 = min(float(ax.bbox.y1), py + radius)

                        if cb_x1 > cb_x0 and cb_y1 > cb_y0:
                            bbox = transforms.Bbox.from_extents(cb_x0, cb_y0, cb_x1, cb_y1)
                        else:
                            if debug and i < 3:
                                print(f"DEBUG: Point {i}: OUTSIDE AXES")
                            continue

                        if debug and i < 3:
                            print(f"DEBUG: Point {i}: data=({x_data:.2f},{y_data:.2f}) → "
                                  f"display=({px:.1f},{py:.1f}), size={s:.1f}, radius={radius:.2f}")

                        if bbox.width > 0.5 and bbox.height > 0.5:
                            if add_unique_annotation(reverse_map['data_point'], bbox, attrs={'data': [round(float(x_data), 6), round(float(y_data), 6)]}):
                                points_added += 1
                                if debug and i < 3:
                                    print(f"DEBUG: Point {i}: ADDED")
                        else:
                            if debug and i < 3:
                                print(f"DEBUG: Point {i}: TOO SMALL")
                except Exception as e:
                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Scatter error: {e}")
                        import traceback
                        traceback.print_exc()

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: SCATTER TOTAL: {points_added} points added")

        # Pie Chart Wedges
        elif chart_type == 'pie' and 'wedge' in reverse_map:
            data_artists = chart_info.get('data_artists', [])
            wedges_added = 0
            debug = GENERATION_CONFIG.get('debug_mode', False)

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: PIE processing {len(data_artists)} artists")

            for artist_idx, artist in enumerate(data_artists):
                if isinstance(artist, patches.Wedge) and artist.get_visible():
                    try:
                        bbox = artist.get_window_extent(renderer)
                        if bbox.width > 0.5 and bbox.height > 0.5:
                            if add_unique_annotation(reverse_map['wedge'], bbox, artist=artist, attrs={'wedge_index': artist_idx}):
                                wedges_added += 1
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: Added wedge {artist_idx}")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Wedge bbox failed: {e}")

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: PIE TOTAL: {wedges_added} wedges added")

        # Line Chart Segments
        elif chart_type == 'line' and 'line_segment' in reverse_map:
            data_artists = chart_info.get('data_artists', [])
            lines_added = 0
            debug = GENERATION_CONFIG.get('debug_mode', False)

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: LINE processing {len(data_artists)} artists")

            for artist_idx, artist in enumerate(data_artists):
                if isinstance(artist, matplotlib.lines.Line2D) and artist.get_visible():
                    try:
                        bbox = artist.get_window_extent(renderer)
                        if bbox.width > 0.5:
                            padded = ensure_min_bbox_thickness(bbox, min_size=4.0)
                            if add_unique_annotation(reverse_map['line_segment'], padded):
                                lines_added += 1
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: Added line_segment {artist_idx}")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Line bbox failed: {e}")

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: LINE TOTAL: {lines_added} line_segments added")

        # Heatmap Cells and Colorbar
        elif chart_type == 'heatmap':
            cells_added = 0
            colorbar_added = 0
            debug = GENERATION_CONFIG.get('debug_mode', False)

            # Heatmap Axis Titles
            if 'axis_title' in reverse_map:
                titles_added = 0
                if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                    try:
                        xlabel_bbox, xlabel_obb = get_text_obb_and_bbox(ax.xaxis.label, renderer, img_h=img_h)
                        if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip(), obb=xlabel_obb):
                            titles_added += 1
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Heatmap X-title error: {e}")

                if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                    try:
                        ylabel_bbox, ylabel_obb = get_text_obb_and_bbox(ax.yaxis.label, renderer, img_h=img_h)
                        if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip(), obb=ylabel_obb):
                            titles_added += 1
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Heatmap Y-title error: {e}")

                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP axis titles: {titles_added}")

            # Heatmap Axis Labels
            if 'axis_labels' in reverse_map:
                labels_added = 0
                bg_color = ax.get_facecolor()
                x_min, x_max = sorted(ax.get_xlim())
                y_min, y_max = sorted(ax.get_ylim())

                # X-axis tick labels
                for label in ax.get_xticklabels():
                    if label.get_visible() and label.get_text().strip():
                        if x_min <= label.get_position()[0] <= x_max:
                            if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                                try:
                                    label_bbox, label_obb = get_text_obb_and_bbox(label, renderer, img_h=img_h)
                                    if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip(), obb=label_obb):
                                        labels_added += 1
                                except Exception as e:
                                    if debug:
                                        print(f"DEBUG: AX[{ax_idx}]: Heatmap X-label error: {e}")

                # Y-axis tick labels
                for label in ax.get_yticklabels():
                    if label.get_visible() and label.get_text().strip():
                        if y_min <= label.get_position()[1] <= y_max:
                            if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                                try:
                                    label_bbox, label_obb = get_text_obb_and_bbox(label, renderer, img_h=img_h)
                                    if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip(), obb=label_obb):
                                        labels_added += 1
                                except Exception as e:
                                    if debug:
                                        print(f"DEBUG: AX[{ax_idx}]: Heatmap Y-label error: {e}")

                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP axis labels: {labels_added}")

            # Heatmap Cells
            if 'cell' in reverse_map:
                data_artists = chart_info.get('data_artists', [])
                for artist_idx, artist in enumerate(data_artists):
                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Heatmap artist {artist_idx}: {type(artist).__name__}")

                    if isinstance(artist, QuadMesh):
                        try:
                            coords = artist.get_coordinates()

                            if coords is not None and coords.ndim == 3:
                                rows, cols = coords.shape[0] - 1, coords.shape[1] - 1

                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: QuadMesh grid: {rows}x{cols} cells")

                                for i in range(rows):
                                    for j in range(cols):
                                        cell_corners = np.array([
                                            coords[i, j],      # bottom-left
                                            coords[i+1, j],    # bottom-right
                                            coords[i+1, j+1],  # top-right
                                            coords[i, j+1]     # top-left
                                        ])

                                        display_coords = ax.transData.transform(cell_corners)

                                        x0, y0 = display_coords.min(axis=0)
                                        x1, y1 = display_coords.max(axis=0)

                                        bbox = transforms.Bbox.from_extents(x0, y0, x1, y1)

                                        if bbox.width >= 1.0 and bbox.height >= 1.0:
                                            if add_unique_annotation(reverse_map['cell'], bbox):
                                                cells_added += 1
                                        elif debug and i < 3 and j < 3:
                                            print(f"DEBUG: Cell ({i},{j}) TOO SMALL: {bbox.width:.1f}x{bbox.height:.1f}")

                        except AttributeError:
                            try:
                                extent = artist.get_extent()
                                data_array = artist.get_array()
                                if data_array is not None:
                                    if data_array.ndim == 2:
                                        rows, cols = data_array.shape
                                    else:
                                        rows, cols = data_array.shape[0], data_array.shape[1]

                                    if debug:
                                        print(f"DEBUG: AX[{ax_idx}]: AxesImage grid: {rows}x{cols} cells")

                                    x0_data, x1_data, y0_data, y1_data = extent
                                    cell_width = (x1_data - x0_data) / cols
                                    cell_height = (y1_data - y0_data) / rows

                                    for i in range(rows):
                                        for j in range(cols):
                                            cell_x0 = x0_data + j * cell_width
                                            cell_x1 = cell_x0 + cell_width
                                            cell_y0 = y0_data + i * cell_height
                                            cell_y1 = cell_y0 + cell_height

                                            pt0 = ax.transData.transform_point((cell_x0, cell_y0))
                                            pt1 = ax.transData.transform_point((cell_x1, cell_y1))

                                            bbox = transforms.Bbox.from_extents(
                                                pt0[0], pt0[1], pt1[0], pt1[1]
                                            )

                                            if bbox.width >= 1.0 and bbox.height >= 1.0:
                                                if add_unique_annotation(reverse_map['cell'], bbox):
                                                    cells_added += 1
                            except Exception as e:
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: Heatmap cell fallback error: {e}")

                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Heatmap cell error: {e}")

                    elif hasattr(artist, 'get_extent') and hasattr(artist, 'get_array'):
                        try:
                            extent = artist.get_extent()
                            data_array = artist.get_array()

                            if data_array is not None:
                                if data_array.ndim == 2:
                                    rows, cols = data_array.shape
                                else:
                                    rows, cols = data_array.shape[0], data_array.shape[1]

                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: AxesImage: {rows}x{cols} cells")

                                x0_data, x1_data, y0_data, y1_data = extent
                                cell_width = (x1_data - x0_data) / cols
                                cell_height = (y1_data - y0_data) / rows

                                for i in range(rows):
                                    for j in range(cols):
                                        cell_x0 = x0_data + j * cell_width
                                        cell_x1 = cell_x0 + cell_width
                                        cell_y0 = y0_data + i * cell_height
                                        cell_y1 = cell_y0 + cell_height

                                        pt0 = ax.transData.transform_point((cell_x0, cell_y0))
                                        pt1 = ax.transData.transform_point((cell_x1, cell_y1))

                                        bbox = transforms.Bbox.from_extents(pt0[0], pt0[1], pt1[0], pt1[1])

                                        if bbox.width >= 1.0 and bbox.height >= 1.0:
                                            if add_unique_annotation(reverse_map['cell'], bbox):
                                                cells_added += 1

                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: AxesImage error: {e}")

                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP cells: {cells_added}")

            # Heatmap Data Labels
            if 'data_label' in reverse_map:
                data_labels_added = 0
                other_artists = chart_info.get('other_artists', [])

                for artist in other_artists:
                    if isinstance(artist, matplotlib.text.Text):
                        if artist.get_visible() and artist.get_text().strip():
                            try:
                                label_bbox, label_obb = get_text_obb_and_bbox(artist, renderer, img_h=img_h)
                                if add_unique_annotation(reverse_map['data_label'], label_bbox, text=artist.get_text().strip(), obb=label_obb):
                                    data_labels_added += 1
                            except Exception as e:
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: Data label error: {e}")

                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP data labels: {data_labels_added}")

            # Heatmap Colorbar Components
            if 'color_bar' in reverse_map:
                for ax_candidate in fig.axes:
                    if ax_candidate == ax:
                        continue

                    try:
                        ax_bbox = ax_candidate.get_window_extent(renderer)
                        if ax_bbox.height <= 0 or ax_bbox.width <= 0:
                            continue

                        aspect_ratio = ax_bbox.width / ax_bbox.height

                        is_vertical_colorbar = aspect_ratio < 0.5 and ax_bbox.height > 40
                        is_horizontal_colorbar = aspect_ratio > 2.0 and ax_bbox.width > 40

                        if is_vertical_colorbar or is_horizontal_colorbar:
                            if add_unique_annotation(reverse_map['color_bar'], ax_bbox):
                                colorbar_added += 1
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: COLORBAR ADDED ({'vertical' if is_vertical_colorbar else 'horizontal'})")

                                # Extract Color Bar Title
                                if 'color_bar_title' in reverse_map:
                                    if ax_candidate.yaxis.label.get_visible() and ax_candidate.yaxis.label.get_text().strip():
                                        title_bbox = ax_candidate.yaxis.label.get_window_extent(renderer)
                                        add_unique_annotation(reverse_map['color_bar_title'], title_bbox, text=ax_candidate.yaxis.label.get_text().strip())
                                    elif ax_candidate.xaxis.label.get_visible() and ax_candidate.xaxis.label.get_text().strip():
                                        title_bbox = ax_candidate.xaxis.label.get_window_extent(renderer)
                                        add_unique_annotation(reverse_map['color_bar_title'], title_bbox, text=ax_candidate.xaxis.label.get_text().strip())

                                # Extract Color Bar Tick Labels
                                if 'color_bar_label' in reverse_map:
                                    bg_color = ax_candidate.get_facecolor()
                                    for label in ax_candidate.get_yticklabels() + ax_candidate.get_xticklabels():
                                        if label.get_visible() and label.get_text().strip():
                                            if has_non_background_pixels(label, fig, ax_candidate, bg_color, threshold=5):
                                                label_bbox = label.get_window_extent(renderer)
                                                add_unique_annotation(reverse_map['color_bar_label'], label_bbox, text=label.get_text().strip())
                                break
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Colorbar detection error: {e}")

                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP colorbar: {colorbar_added}")
        # Line chart keypoints
        elif chart_type == 'line' and chart_info.get('keypoint_info'):
            for series_kpts in chart_info['keypoint_info']:
                series_idx = series_kpts['series_idx']

                # Start/end points
                for pt_type, pt_data in [('line_start', series_kpts['start']), ('line_end', series_kpts['end'])]:
                    if pt_type in reverse_map:
                        x, y, idx = pt_data
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px-4, py-4, px+4, py+4)
                        add_unique_annotation(reverse_map[pt_type], bbox)

                # Inflection points
                if 'inflection_point' in reverse_map:
                    for x, y, idx in series_kpts['inflections']:
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px-3, py-3, px+3, py+3)
                        add_unique_annotation(reverse_map['inflection_point'], bbox)

        # Area chart keypoints (similar structure to line)
        elif chart_type == 'area' and chart_info.get('keypoint_info'):
            for series_kpts in chart_info['keypoint_info']:
                if 'area_start' in reverse_map:
                    x, y, idx = series_kpts['start']
                    px, py = ax.transData.transform_point((x, y))
                    bbox = transforms.Bbox.from_extents(px-4, py-4, px+4, py+4)
                    add_unique_annotation(reverse_map['area_start'], bbox)

                if 'inflection_point' in reverse_map:
                    for x, y, idx in series_kpts['inflections']:
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px-3, py-3, px+3, py+3)
                        add_unique_annotation(reverse_map['inflection_point'], bbox)

        # Pie chart geometric keypoints
        elif chart_type == 'pie' and chart_info.get('pie_geometry'):
            pie_geo = chart_info['pie_geometry']

            # Center point
            if 'center_point' in reverse_map and 'center_point' in pie_geo:
                cx, cy = pie_geo['center_point']
                px, py = ax.transData.transform_point((cx, cy))
                bbox = transforms.Bbox.from_extents(px-5, py-5, px+5, py+5)
                add_unique_annotation(reverse_map['center_point'], bbox)

            # Arc boundaries for each wedge
            if 'arc_boundary' in reverse_map:
                for wedge_geo in pie_geo['wedges']:
                    for arc_pt in ['arc_start', 'arc_end', 'arc_mid']:
                        ax_pt, ay_pt = wedge_geo[arc_pt]
                        px, py = ax.transData.transform_point((ax_pt, ay_pt))
                        bbox = transforms.Bbox.from_extents(px-3, py-3, px+3, py+3)
                        add_unique_annotation(reverse_map['arc_boundary'], bbox)

            # Wedge centers
            if 'wedge_center' in reverse_map:
                for wedge_geo in pie_geo['wedges']:
                    wx, wy = wedge_geo['wedge_label_point']
                    px, py = ax.transData.transform_point((wx, wy))
                    bbox = transforms.Bbox.from_extents(px-4, py-4, px+4, py+4)
                    add_unique_annotation(reverse_map['wedge_center'], bbox)

        # Process other artists for error bars, text annotations, etc.
        other_artists = chart_info.get('other_artists', [])
        if GENERATION_CONFIG.get('debug_mode', False):
            print(f"DEBUG: AX[{ax_idx}]: Processing {len(other_artists)} other artists")

        for artist_idx, artist in enumerate(other_artists):
            if artist is None:
                continue

            try:
                is_visible = artist.get_visible()
            except:
                is_visible = True

            if not is_visible:
                continue

            # Pie connector lines
            if chart_type == 'pie' and 'connector_line' in reverse_map and isinstance(artist, matplotlib.lines.Line2D):
                try:
                    if artist.get_gid() == 'pie_connector':
                        bbox = artist.get_window_extent(renderer)
                        if bbox.width > 0.5:
                            padded = ensure_min_bbox_thickness(bbox, min_size=4.0)
                            add_unique_annotation(reverse_map['connector_line'], padded)
                            continue
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Connector line failed: {e}")

            # Error bars
            if hasattr(artist, 'lines') and len(getattr(artist, 'lines', [])) >= 3 and 'error_bar' in reverse_map:
                try:
                    # ErrorbarContainer processing
                    plotline, caplines, barlinecols = artist.lines
                    if barlinecols and caplines:
                        artist_bbox = artist.get_window_extent(renderer)
                        artist_bbox = ensure_min_bbox_thickness(artist_bbox, min_size=4.0)
                        add_unique_annotation(reverse_map['error_bar'], artist_bbox)
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Error bar processing failed: {e}")

            # Text annotations
            elif hasattr(artist, 'get_text'):
                try:
                    text_content = artist.get_text().strip()
                    if not text_content:
                        continue

                    # Significance markers
                    if text_content in ['*', '**', '***', 'ns', 'a', 'b', 'c', 'd'] and 'significance_marker' in reverse_map:
                        artist_bbox, artist_obb = get_text_obb_and_bbox(artist, renderer, img_h=img_h)
                        add_unique_annotation(reverse_map['significance_marker'], artist_bbox, text=text_content, obb=artist_obb)
                    elif 'data_label' in reverse_map:
                        artist_bbox, artist_obb = get_text_obb_and_bbox(artist, renderer, img_h=img_h)
                        if artist_bbox.width > 0.5 and artist_bbox.height > 0.5:
                            add_unique_annotation(reverse_map['data_label'], artist_bbox, text=text_content, obb=artist_obb)
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Text processing failed: {e}")

        if GENERATION_CONFIG.get('debug_mode', False) or GENERATION_CONFIG.get('debug_annotations', False):
            print(f"DEBUG: Total annotations generated: {len(annotations)}")
            class_counts = {}
            for ann in annotations:
                class_id = ann['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            print(f"DEBUG: Class distribution: {class_counts}")

        return annotations

def filter_overlapping_annotations(annotations, iou_threshold=0.7, cls_map=None, occlusion_threshold=0.85, cull_occluded=False):
    """
    Remove annotations with high IoU overlap within the same class, and perform
    hierarchical cross-class occlusion detection against legend bounding boxes.

    For any data mark (bar, scatter_point, line_marker, pie_slice, etc.) and legend bounding box:
        Coverage = Area(Mark ∩ Legend) / Area(Mark)
    If Coverage >= occlusion_threshold (0.85) or if the mark centroid lies strictly inside
    the legend box, mark it as occluded ("visibility": 0, "occluded": True).
    If cull_occluded is True, drops it; otherwise retains it with visibility=0 (amodal representation).
    """
    def _extract_coords(bbox):
        if hasattr(bbox, 'x0'):
            return float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return None

    def bbox_iou(bbox1, bbox2):
        c1 = _extract_coords(bbox1)
        c2 = _extract_coords(bbox2)
        if c1 is None or c2 is None:
            return 0.0
        x1 = max(c1[0], c2[0])
        y1 = max(c1[1], c2[1])
        x2 = min(c1[2], c2[2])
        y2 = min(c1[3], c2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (c1[2] - c1[0]) * (c1[3] - c1[1])
        bbox2_area = (c2[2] - c2[0]) * (c2[3] - c2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    # 1. Hierarchical Cross-Class Legend Occlusion Pass
    legend_boxes = []
    for ann in annotations:
        c_id = ann.get('class_id')
        c_name = ann.get('class_name') or (cls_map.get(c_id) if cls_map else None)
        if c_name == 'legend' or ann.get('semantic_role') == 'text_legend':
            coords = _extract_coords(ann.get('bbox'))
            if coords:
                legend_boxes.append(coords)

    occluded_indices = set()
    data_mark_names = {
        'bar', 'data_point', 'scatter_point', 'data_marker', 'line_marker',
        'wedge', 'pie_slice', 'slice_boundary', 'box', 'median_line',
        'cell', 'area_series', 'line_segment'
    }

    if legend_boxes:
        for idx, ann in enumerate(annotations):
            c_id = ann.get('class_id')
            c_name = ann.get('class_name') or (cls_map.get(c_id) if cls_map else None)
            is_data_mark = (
                c_name in data_mark_names or
                ann.get('semantic_role') == 'data_element' or
                (c_name not in {'legend', 'chart_title', 'axis_title', 'axis_labels', 'chart'} if c_name else False)
            )
            if not is_data_mark:
                continue

            coords = _extract_coords(ann.get('bbox'))
            if coords is None:
                continue
            mx0, my0, mx1, my1 = coords
            m_area = (mx1 - mx0) * (my1 - my0)
            if m_area <= 0:
                continue

            cx = (mx0 + mx1) / 2.0
            cy = (my0 + my1) / 2.0

            for lx0, ly0, lx1, ly1 in legend_boxes:
                ix0 = max(mx0, lx0)
                iy0 = max(my0, ly0)
                ix1 = min(mx1, lx1)
                iy1 = min(my1, ly1)
                inter_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                coverage = inter_area / m_area
                centroid_inside = (lx0 <= cx <= lx1) and (ly0 <= cy <= ly1)

                if coverage >= occlusion_threshold or centroid_inside:
                    ann['visibility'] = 0
                    ann['occluded'] = True
                    occluded_indices.add(idx)
                    break

    # 2. Same-class overlap filtering
    by_class = defaultdict(list)
    for idx, ann in enumerate(annotations):
        if cull_occluded and idx in occluded_indices:
            continue
        by_class[ann['class_id']].append(ann)

    filtered = []
    for class_id, class_anns in by_class.items():
        def _area(a):
            c = _extract_coords(a.get('bbox'))
            return (c[2] - c[0]) * (c[3] - c[1]) if c else 0.0
        class_anns.sort(key=_area, reverse=True)

        keep = []
        for ann in class_anns:
            is_duplicate = False
            for kept_ann in keep:
                if bbox_iou(ann['bbox'], kept_ann['bbox']) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(ann)

        filtered.extend(keep)

    return filtered


def extract_pie_pose_annotations(
    fig,
    chart_info_map,
    cls_map_pose: Dict[str, int],
    img_w: int,
    img_h: int
) -> List[Dict]:
    """
    Extract YOLO pose annotations for pie charts (5 keypoints per slice).

    Annotates each slice (wedge) individually:
    - Class: 0 ("slice_boundary")
    - Keypoints: 5 (WedgeCenter, ArcStart, ArcInter1, ArcInter2, ArcEnd)
    """
    keypoint_annotations = []

    for ax in fig.axes:
        if not ax.get_visible():
            continue

        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str', '')

        if chart_type != 'pie':
            continue

        pie_geometry = chart_info.get('pie_geometry', None)
        if not pie_geometry:
            continue

        wedges_info = pie_geometry.get('wedges', [])

        for wedge_geo in wedges_info:
            angle_span = wedge_geo.get('angle_span', 0.0)
            if angle_span < 0.5:
                continue

            kpt1_data = wedge_geo.get('center')
            kpt2_data = wedge_geo.get('arc_start')
            kpt3_data = wedge_geo.get('arc_end')
            kpt4_data = wedge_geo.get('arc_inter_1')
            kpt5_data = wedge_geo.get('arc_inter_2')

            if not all([kpt1_data, kpt2_data, kpt3_data, kpt4_data, kpt5_data]):
                continue

            kpt1_px_data = ax.transData.transform_point(kpt1_data)
            kpt2_px_data = ax.transData.transform_point(kpt2_data)
            kpt3_px_data = ax.transData.transform_point(kpt3_data)
            kpt4_px_data = ax.transData.transform_point(kpt4_data)
            kpt5_px_data = ax.transData.transform_point(kpt5_data)

            all_kpts_px = [
                (kpt1_px_data[0], img_h - kpt1_px_data[1]),  # 0: Center
                (kpt2_px_data[0], img_h - kpt2_px_data[1]),  # 1: ArcStart
                (kpt4_px_data[0], img_h - kpt4_px_data[1]),  # 2: ArcInter1
                (kpt5_px_data[0], img_h - kpt5_px_data[1]),  # 3: ArcInter2
                (kpt3_px_data[0], img_h - kpt3_px_data[1])   # 4: ArcEnd
            ]

            all_x, all_y = zip(*all_kpts_px)
            x0, x1 = min(all_x), max(all_x)
            y0, y1 = min(all_y), max(all_y)

            # Ensure minimum bounding box size of at least 2 pixels
            min_dim_px = 2.0
            if (x1 - x0) < min_dim_px:
                diff = (min_dim_px - (x1 - x0)) / 2.0
                x0 -= diff
                x1 += diff
            if (y1 - y0) < min_dim_px:
                diff = (min_dim_px - (y1 - y0)) / 2.0
                y0 -= diff
                y1 += diff

            cx = max(0.0, min(1.0, (x0 + x1) / 2 / img_w))
            cy = max(0.0, min(1.0, (y0 + y1) / 2 / img_h))
            w = max(min_dim_px / img_w, min(1.0, (x1 - x0) / img_w))
            h = max(min_dim_px / img_h, min(1.0, (y1 - y0) / img_h))

            kp_norm = [
                [
                    max(0.0, min(1.0, x_px / img_w)),
                    max(0.0, min(1.0, y_px / img_h)),
                    2  # Visible
                ]
                for x_px, y_px in all_kpts_px
            ]

            keypoint_annotations.append({
                'class_id': 0,  # slice_boundary
                'bbox': (cx, cy, w, h),
                'keypoints': kp_norm
            })

    return keypoint_annotations


def _select_multi_chart_layout(cfg):
    """Sample grid dimensions (nrows, ncols) for multi_chart_detection layout."""
    mcd_cfg = cfg.get('multi_chart_detection', {})
    weights_dict = mcd_cfg.get('layout_weights', {
        "1x1": 10, "1x2": 25, "2x1": 25, "2x2": 25, "1x3": 7.5, "3x1": 7.5
    })
    min_sub = mcd_cfg.get('min_subplots', 1)
    max_sub = mcd_cfg.get('max_subplots', 4)

    valid_layouts = []
    weights = []
    for k, v in weights_dict.items():
        try:
            r, c = map(int, k.split('x'))
            if min_sub <= r * c <= max_sub:
                valid_layouts.append((r, c))
                weights.append(v)
        except Exception:
            continue

    if not valid_layouts:
        valid_layouts = [(1, 2)]
        weights = [1.0]

    r, c = random.choices(valid_layouts, weights=weights, k=1)[0]
    return r, c


def get_subchart_detection_annotations(fig, chart_info_map, class_map, renderer):
    """Extract tight bounding box annotations for each subplot in a composite figure, including any auxiliary axes (e.g. colorbars, twin axes)."""
    reverse_class_map = {v: int(k) for k, v in class_map.items()}
    annotations = []
    for ax, info in chart_info_map.items():
        chart_type = info.get('chart_type_str', 'unknown')
        class_id = reverse_class_map.get(chart_type, 0)
        try:
            bbox = ax.get_tightbbox(renderer)
        except Exception:
            bbox = ax.get_window_extent(renderer)

        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

        aux_axes = list(info.get('aux_axes', []))
        if not aux_axes:
            for art in list(info.get('other_artists', [])) + list(info.get('data_artists', [])):
                if hasattr(art, 'ax') and isinstance(getattr(art, 'ax'), plt.Axes) and art.ax != ax:
                    if art.ax not in aux_axes:
                        aux_axes.append(art.ax)
                if hasattr(art, 'axes') and isinstance(getattr(art, 'axes'), plt.Axes) and art.axes != ax:
                    if art.axes not in aux_axes:
                        aux_axes.append(art.axes)

        for aux_ax in aux_axes:
            try:
                aux_bbox = aux_ax.get_tightbbox(renderer)
            except Exception:
                aux_bbox = aux_ax.get_window_extent(renderer)
            x0 = min(x0, aux_bbox.x0)
            y0 = min(y0, aux_bbox.y0)
            x1 = max(x1, aux_bbox.x1)
            y1 = max(y1, aux_bbox.y1)

        union_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

        annotations.append({
            'class_id': class_id,
            'bbox': union_bbox,
            'chart_type': chart_type,
            'element_type': 'chart'
        })
    return annotations


EFFECT_REGISTRY = {
    "blur": apply_blur_effect,
    "motion_blur": apply_motion_blur_effect,
    "low_res": apply_low_res_effect,
    "noise": apply_noise_effect,
    "jpeg_compression": apply_jpeg_compression_effect,
    "pixelation": apply_pixelation_effect,
    "posterize": apply_posterize_effect,
    "color_variation": apply_color_variation_effect,
    "ui_chrome": apply_ui_chrome_effect,
    "watermark": apply_watermark_effect,
    "vignette": apply_vignette_effect,
    "scanner_streaks": apply_scanner_streaks_effect,
    "clipping": apply_clipping_effect,
    "printing_artifacts": apply_printing_artifacts_effect,
    "mouse_cursor": apply_mouse_cursor_effect,
    "text_degradation": apply_text_degradation_effect,
    "grid_occlusion": apply_grid_occlusion_effect,
    "scan_rotation": apply_scan_rotation_effect,
    "grayscale": apply_grayscale_effect,
    "perspective": partial(apply_perspective_warp_effect, distortion_factor=0.15),
    "perspective_warp": apply_perspective_warp_effect,
    "page_curl": apply_page_curl,
    "non_rigid_mesh": apply_page_curl,
    "mesh_warp": apply_page_curl,
    "uneven_lighting": apply_uneven_lighting_effect,
    "chromatic_aberration": apply_chromatic_aberration_effect,
    "pdf_document_context": apply_pdf_document_context_effect,
}

EFFECT_ALIASES = {
    "perspective_warp": "perspective",
    "non_rigid_mesh": "page_curl",
    "mesh_warp": "page_curl",
}


def apply_realism_effects(pil_img, annotations, effects_config, extra_annotation_sets=None):
    """Apply realism effects and return modified image and annotations."""
    effect_function_map = EFFECT_REGISTRY

    total_dx, total_dy = 0, 0
    transform_steps = []
    init_w, init_h = pil_img.size

    def _subdivide_polygon(pts, max_segment_len=15.0):
        if len(pts) < 3:
            return pts
        subdivided = []
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.hypot(dx, dy)
            subdivided.append(p1)
            if dist > max_segment_len:
                steps = int(dist // max_segment_len)
                for s in range(1, steps + 1):
                    t = s / (steps + 1)
                    subdivided.append((p1[0] + t * dx, p1[1] + t * dy))
        return subdivided

    def _rotate_point(x, y, angle, step_w, step_h):
        cx = step_w / 2.0
        cy = step_h / 2.0
        rad = np.radians(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        # Convert to standard image coordinates (origin top-left)
        x_img = float(x)
        y_img = float(step_h - y)

        # Rotate around center
        dx = x_img - cx
        dy = y_img - cy
        x_rot = cx + dx * cos_a + dy * sin_a
        y_rot = cy - dx * sin_a + dy * cos_a

        # Convert back to Matplotlib coordinates (origin bottom-left)
        return x_rot, float(step_h - y_rot)

    def _warp_point(x, y, H, step_h):
        x_img = float(x)
        y_img = float(step_h - y)
        denom = (H[2, 0] * x_img) + (H[2, 1] * y_img) + H[2, 2]
        if denom == 0:
            return float(x), float(y)
        x_w = ((H[0, 0] * x_img) + (H[0, 1] * y_img) + H[0, 2]) / denom
        y_w = ((H[1, 0] * x_img) + (H[1, 1] * y_img) + H[1, 2]) / denom
        return float(x_w), float(step_h - y_w)

    def _apply_transform_steps(x, y):
        x_t, y_t = float(x), float(y)
        for step in transform_steps:
            if step[0] == "translate":
                x_t += step[1]
                y_t += step[2]
            elif step[0] == "homography":
                x_t, y_t = _warp_point(x_t, y_t, step[1], step[3])
            elif step[0] == "rotation":
                x_t, y_t = _rotate_point(x_t, y_t, step[1], step[2], step[3])
            elif step[0] == "non_rigid_mesh":
                f_map = step[1]
                step_h = step[3]
                x_img = float(x_t)
                y_img = float(step_h - y_t)
                w_pt = f_map([x_img, y_img])
                x_t = float(w_pt[0])
                y_t = float(step_h - w_pt[1])
        return x_t, y_t

    for effect_name, effect_config in effects_config.items():
        if not isinstance(effect_config, dict):
            continue
        p_val = effect_config.get('p', 1.0) if 'p' in effect_config else 1.0
        if random.random() < p_val:
            func = effect_function_map.get(effect_name)
            if not func:
                continue

            print(f"    - Applying effect: {effect_name}")
            if 'params' in effect_config and isinstance(effect_config['params'], dict):
                params = effect_config['params']
            else:
                params = {k: v for k, v in effect_config.items() if k != 'p'}

            try:
                if effect_name in ['clipping', 'pdf_document_context']:
                    pil_img, dx, dy = func(pil_img, **params)
                    total_dx += dx
                    total_dy += dy
                    transform_steps.append(("translate", dx, dy))
                elif effect_name == 'scan_rotation':
                    cur_w, cur_h = pil_img.size
                    result = func(pil_img, **params)
                    pil_img = result[0]
                    angle = result[1]
                    transform_steps.append(("rotation", angle, cur_w, cur_h))
                elif effect_name in ['perspective', 'perspective_warp']:
                    cur_w, cur_h = pil_img.size
                    p = dict(params)
                    if 'magnitude' in p and 'distortion_factor' not in p:
                        p['distortion_factor'] = p.pop('magnitude')
                    pil_img, H = func(pil_img, return_homography=True, **p)
                    if H is not None:
                        transform_steps.append(("homography", H, cur_w, cur_h))
                elif effect_name in ['page_curl', 'non_rigid_mesh', 'mesh_warp']:
                    cur_w, cur_h = pil_img.size
                    pil_img, forward_map = func(pil_img, return_forward_map=True, **params)
                    if forward_map is not None:
                        transform_steps.append(("non_rigid_mesh", forward_map, cur_w, cur_h))
                else:
                    pil_img = func(pil_img, **params)
            except Exception as e:
                print(f"      [WARNING] Failed to apply effect '{effect_name}': {e}")

    # Apply offset and transforms to annotations across all streams
    if transform_steps:
        final_w, final_h = pil_img.size
        if total_dx != 0 or total_dy != 0:
            print(f"    - Applying total annotation offset: dx={total_dx}, dy={total_dy}")

        target_lists = [annotations]
        if extra_annotation_sets:
            target_lists.extend(extra_annotation_sets)

        has_non_rigid = any(step[0] == "non_rigid_mesh" for step in transform_steps)

        for ann_list in target_lists:
            for ann in ann_list:
                has_kpts = ("keypoints" in ann and ann["keypoints"])

                # 1. Standard bounding box transform
                if 'bbox' in ann and ann['bbox'] is not None and not (has_kpts and isinstance(ann['bbox'], tuple)):
                    bbox = ann['bbox']
                    if hasattr(bbox, 'x0'):
                        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                    else:
                        x0, y0, x1, y1 = bbox
                    if has_non_rigid:
                        num_subdiv = 16
                        pts_to_warp = []
                        for t in np.linspace(0.0, 1.0, num_subdiv, endpoint=False):
                            pts_to_warp.append((x0 + t * (x1 - x0), y0))
                        for t in np.linspace(0.0, 1.0, num_subdiv, endpoint=False):
                            pts_to_warp.append((x1, y0 + t * (y1 - y0)))
                        for t in np.linspace(0.0, 1.0, num_subdiv, endpoint=False):
                            pts_to_warp.append((x1 - t * (x1 - x0), y1))
                        for t in np.linspace(0.0, 1.0, num_subdiv, endpoint=False):
                            pts_to_warp.append((x0, y1 - t * (y1 - y0)))
                    else:
                        pts_to_warp = [
                            (x0, y0),
                            (x1, y0),
                            (x1, y1),
                            (x0, y1)
                        ]
                    warped = [_apply_transform_steps(x, y) for x, y in pts_to_warp]
                    xs = [p[0] for p in warped]
                    ys = [p[1] for p in warped]
                    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
                    if isinstance(bbox, BoundingBox):
                        ann['bbox'] = BoundingBox(min_x, min_y, max_x, max_y)
                    else:
                        ann['bbox'] = transforms.Bbox.from_extents(min_x, min_y, max_x, max_y)

                # 2. Pose keypoints transform
                if has_kpts:
                    keypoints = ann["keypoints"]
                    updated = []
                    for kp in keypoints:
                        x_val, y_val = kp[0], kp[1]
                        vis = kp[2] if len(kp) > 2 else 2
                        is_norm = (0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0)
                        x_img = x_val * init_w if is_norm else x_val
                        y_img = y_val * init_h if is_norm else y_val
                        x_m = x_img
                        y_m = init_h - y_img
                        x_t, y_t = _apply_transform_steps(x_m, y_m)
                        x_final_img = x_t
                        y_final_img = final_h - y_t
                        if is_norm:
                            x_out = max(0.0, min(1.0, x_final_img / final_w))
                            y_out = max(0.0, min(1.0, y_final_img / final_h))
                        else:
                            x_out, y_out = x_final_img, y_final_img
                        updated.append([x_out, y_out, vis])
                    ann["keypoints"] = updated

                    # If bbox is normalized tuple (cx, cy, w, h), update envelope from transformed keypoints
                    if 'bbox' in ann and isinstance(ann['bbox'], tuple) and len(ann['bbox']) == 4:
                        xs = [p[0] for p in updated]
                        ys = [p[1] for p in updated]
                        x0, x1 = min(xs), max(xs)
                        y0, y1 = min(ys), max(ys)
                        min_dim_w = 2.0 / final_w
                        min_dim_h = 2.0 / final_h
                        w = max(min_dim_w, x1 - x0)
                        h = max(min_dim_h, y1 - y0)
                        cx = max(0.0, min(1.0, (x0 + x1) / 2.0))
                        cy = max(0.0, min(1.0, (y0 + y1) / 2.0))
                        ann['bbox'] = (cx, cy, w, h)

                # 3. Instance segmentation polygon contour transform
                if "polygon" in ann and ann["polygon"]:
                    poly_pts = ann["polygon"]
                    if has_non_rigid and len(poly_pts) >= 3:
                        poly_pts = _subdivide_polygon(poly_pts, max_segment_len=15.0)
                    warped_poly = []
                    for px, py in poly_pts:
                        x_m = float(px)
                        y_m = float(init_h - py)
                        x_t, y_t = _apply_transform_steps(x_m, y_m)
                        warped_poly.append((x_t, float(final_h - y_t)))
                    ann["polygon"] = warped_poly

                for p_key in ["amodal_polygon", "modal_polygon"]:
                    if p_key in ann and ann[p_key]:
                        poly_pts = ann[p_key]
                        if has_non_rigid and len(poly_pts) >= 3:
                            poly_pts = _subdivide_polygon(poly_pts, max_segment_len=15.0)
                        warped_p = []
                        for px, py in poly_pts:
                            x_m = float(px)
                            y_m = float(init_h - py)
                            x_t, y_t = _apply_transform_steps(x_m, y_m)
                            warped_p.append((x_t, float(final_h - y_t)))
                        ann[p_key] = warped_p
                        if "segmentation" in ann and isinstance(ann["segmentation"], dict):
                            ann["segmentation"][p_key] = warped_p
                        if ShapelyPolygon is not None and len(warped_p) >= 3:
                            try:
                                sp = ShapelyPolygon(warped_p)
                                if not sp.is_valid:
                                    sp = shapely_make_valid(sp)
                                area_val = float(sp.area)
                                area_key = p_key.replace("_polygon", "_area")
                                ann[area_key] = area_val
                                if "segmentation" in ann and isinstance(ann["segmentation"], dict):
                                    ann["segmentation"][area_key] = area_val
                            except Exception:
                                pass

                # 4. OBB vertices transform
                if "obb" in ann and ann["obb"]:
                    raw_obb = ann["obb"]
                    canonical_obb = None
                    if has_non_rigid and len(raw_obb) == 4 and _HAS_CV2:
                        try:
                            dense_obb_pts = []
                            for i in range(4):
                                p1 = raw_obb[i]
                                p2 = raw_obb[(i + 1) % 4]
                                for t in np.linspace(0.0, 1.0, 16, endpoint=False):
                                    dense_obb_pts.append((p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])))
                            warped_dense = []
                            for px, py in dense_obb_pts:
                                x_m = float(px)
                                y_m = float(init_h - py)
                                x_t, y_t = _apply_transform_steps(x_m, y_m)
                                warped_dense.append((x_t, float(final_h - y_t)))
                            rect = cv2.minAreaRect(np.asarray(warped_dense, dtype=np.float32))
                            box = cv2.boxPoints(rect)
                            canonical_obb = canonicalize_and_validate_obb(box, y_down=True)
                        except Exception:
                            canonical_obb = None

                    if canonical_obb is None:
                        warped_obb = []
                        for px, py in raw_obb:
                            x_m = float(px)
                            y_m = float(init_h - py)
                            x_t, y_t = _apply_transform_steps(x_m, y_m)
                            warped_obb.append((x_t, float(final_h - y_t)))
                        canonical_obb = canonicalize_and_validate_obb(warped_obb, y_down=True)

                    if canonical_obb is not None:
                        ann["obb"] = canonical_obb

    return pil_img, annotations


def clip_bbox_to_viewport(ann_list, img_w, img_h, debug_mode=False):
    """
    Clip bounding box annotations to image viewport [0, img_w] x [0, img_h].
    Discards annotations only if post-clip area is degenerate (< 4 px² or w/h < 2 px).
    Preserves BoundingBox, tuple, and transforms.Bbox types.
    """
    kept = []
    for ann in ann_list:
        bbox = ann.get('bbox')
        if bbox is None:
            continue
        if hasattr(bbox, 'x0'):
            bx0, by0, bx1, by1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        else:
            bx0, by0, bx1, by1 = bbox[:4]
        x0 = max(0.0, min(float(bx0), img_w))
        x1 = max(x0, min(float(bx1), img_w))
        y0 = max(0.0, min(float(by0), img_h))
        y1 = max(y0, min(float(by1), img_h))
        bw = x1 - x0
        bh = y1 - y0
        if (bw * bh) >= 4.0 and bw >= 2.0 and bh >= 2.0:
            if isinstance(bbox, BoundingBox):
                ann['bbox'] = BoundingBox(x0, y0, x1, y1)
            elif isinstance(bbox, (list, tuple)) and not hasattr(bbox, 'x0'):
                ann['bbox'] = (x0, y0, x1, y1)
            else:
                ann['bbox'] = transforms.Bbox.from_extents(x0, y0, x1, y1)
            kept.append(ann)
        else:
            if debug_mode:
                print(f"    - Discarded annotation (degenerate area after clipping): class {ann.get('class_id')}")
    return kept


def clip_polygon_to_viewport(ann_list, img_w, img_h, debug_mode=False):
    """
    Clip polygon contours to viewport [0, img_w] x [0, img_h] using analytical polygon intersection.
    Discards polygons only if post-clip bounding envelope is degenerate (< 4 px² or w/h < 2 px).
    """
    kept = []
    viewport_box = shapely_box(0.0, 0.0, float(img_w), float(img_h)) if shapely_box is not None else None

    for ann in ann_list:
        poly = ann.get('polygon', [])
        if len(poly) < 3:
            continue

        if ShapelyPolygon is not None and viewport_box is not None:
            try:
                sp = ShapelyPolygon(poly)
                if not sp.is_valid:
                    sp = shapely_make_valid(sp)
                clipped = sp.intersection(viewport_box)
                if clipped.is_empty or clipped.area < 1e-4:
                    clamped = []
                else:
                    if clipped.geom_type == 'MultiPolygon':
                        clipped = max(clipped.geoms, key=lambda g: g.area)
                    clamped = [(float(x), float(y)) for x, y in clipped.exterior.coords[:-1]]
            except Exception:
                clamped = [
                    (max(0.0, min(float(px), float(img_w))), max(0.0, min(float(py), float(img_h))))
                    for px, py in poly
                ]
        else:
            clamped = [
                (max(0.0, min(float(px), float(img_w))), max(0.0, min(float(py), float(img_h))))
                for px, py in poly
            ]

        if len(clamped) < 3:
            if debug_mode:
                print(f"    - Discarded polygon annotation (fewer than 3 vertices after clipping): class {ann.get('class_id')}")
            continue

        xs = [p[0] for p in clamped]
        ys = [p[1] for p in clamped]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if (bw * bh) >= 4.0 and bw >= 2.0 and bh >= 2.0:
            ann['polygon'] = clamped

            # Also clip amodal_polygon and modal_polygon analytically if present
            if ShapelyPolygon is not None and viewport_box is not None:
                for extra_key in ['amodal_polygon', 'modal_polygon']:
                    if extra_key in ann and ann[extra_key]:
                        try:
                            sp_extra = ShapelyPolygon(ann[extra_key])
                            if not sp_extra.is_valid:
                                sp_extra = shapely_make_valid(sp_extra)
                            c_extra = sp_extra.intersection(viewport_box)
                            if not c_extra.is_empty and c_extra.area >= 1e-4:
                                if c_extra.geom_type == 'MultiPolygon':
                                    c_extra = max(c_extra.geoms, key=lambda g: g.area)
                                ann[extra_key] = [(float(x), float(y)) for x, y in c_extra.exterior.coords[:-1]]
                                area_key = extra_key.replace('_polygon', '_area')
                                ann[area_key] = float(c_extra.area)
                                if 'segmentation' in ann and isinstance(ann['segmentation'], dict):
                                    ann['segmentation'][extra_key] = ann[extra_key]
                                    ann['segmentation'][area_key] = ann[area_key]
                        except Exception:
                            pass

            kept.append(ann)
        else:
            if debug_mode:
                print(f"    - Discarded polygon annotation (degenerate area after clipping): class {ann.get('class_id')}")
    return kept


def save_annotations_yolo(annotations, img_w, img_h, output_path):
    """Save in proper YOLO format with normalization"""
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            bbox = ann['bbox']

            # Extract bbox coordinates
            if hasattr(bbox, 'extents'):
                x0, y0, x1, y1 = bbox.extents
            else:
                x0, y0, x1, y1 = bbox

            # Convert matplotlib (bottom-left) to image (top-left) coordinates
            img_y0 = img_h - y1  # Top edge
            img_y1 = img_h - y0  # Bottom edge

            # Clamp to bounds
            img_x0 = max(0.0, min(float(x0), img_w))
            img_x1 = max(img_x0, min(float(x1), img_w))
            img_y0 = max(0.0, min(float(img_y0), img_h))
            img_y1 = max(img_y0, min(float(img_y1), img_h))

            # YOLO format: normalized center and dimensions
            x_center = (img_x0 + img_x1) / 2.0 / img_w
            y_center = (img_y0 + img_y1) / 2.0 / img_h
            width = (img_x1 - img_x0) / img_w
            height = (img_y1 - img_y0) / img_h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# In generator.py after save_annotations_yolo()
def save_annotations_pose(
    annotations: List[Dict],
    img_w: int,
    img_h: int,
    output_path: str
):
    """
    Save YOLO pose format annotations to file.

    Format per line:
    class_id center_x center_y width height kpt1_x kpt1_y vis1 kpt2_x kpt2_y vis2 ...
    """
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            cx, cy, w, h = ann['bbox']
            keypoints = ann['keypoints']

            line_parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for x, y, vis in keypoints:
                line_parts.extend([f"{x:.6f}", f"{y:.6f}", str(vis)])

            f.write(" ".join(line_parts) + "\n")


def save_annotations_yolo_obb(
    annotations: List[Dict],
    img_w: int,
    img_h: int,
    output_path: str
):
    """
    Save annotations in standard YOLOv8-OBB format:
    class_id x1 y1 x2 y2 x3 y3 x4 y4
    Normalized to [0, 1] relative to final image width and height,
    ordered in clockwise winding sequence starting from top-left.
    Consumes canonicalized OBB vertices if present, or derives them from bbox.
    """
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann.get('class_id')
            if class_id is None:
                continue

            pts = None
            if "obb" in ann and ann["obb"]:
                pts = ann["obb"]
            elif "bbox" in ann and ann["bbox"] is not None:
                bbox = ann["bbox"]
                if hasattr(bbox, "extents"):
                    x0, y0, x1, y1 = bbox.extents
                elif hasattr(bbox, "x0"):
                    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x0, y0, x1, y1 = bbox[:4]
                else:
                    continue

                # Convert matplotlib (bottom-left) to image (top-left) coordinates
                img_y0 = img_h - y1
                img_y1 = img_h - y0
                img_x0 = max(0.0, min(float(x0), float(img_w)))
                img_x1 = max(img_x0, min(float(x1), float(img_w)))
                img_y0 = max(0.0, min(float(img_y0), float(img_h)))
                img_y1 = max(img_y0, min(float(img_y1), float(img_h)))
                if img_x1 <= img_x0 or img_y1 <= img_y0:
                    continue
                pts = [
                    (img_x0, img_y0),
                    (img_x1, img_y0),
                    (img_x1, img_y1),
                    (img_x0, img_y1),
                ]

            if not pts or len(pts) != 4:
                continue

            canonical = canonicalize_and_validate_obb(pts, y_down=True)
            if canonical is None or len(canonical) != 4:
                continue

            # Start from top-left (min x+y, tie-break min y, min x) while preserving clockwise winding
            start_idx = min(
                range(4),
                key=lambda idx: (
                    canonical[idx][0] + canonical[idx][1],
                    canonical[idx][1],
                    canonical[idx][0],
                ),
            )
            ordered_pts = [canonical[(start_idx + k) % 4] for k in range(4)]

            norm_parts = []
            for x, y in ordered_pts:
                nx = max(0.0, min(1.0, float(x) / img_w))
                ny = max(0.0, min(1.0, float(y) / img_h))
                norm_parts.extend([f"{nx:.6f}", f"{ny:.6f}"])

            f.write(f"{class_id} " + " ".join(norm_parts) + "\n")


# ===================================================================================
# == DUAL-STREAM ANNOTATION: INSTANCE SEGMENTATION & MARKER/EXTREMA DETECTION
# ===================================================================================
# Extracts polygon masks for line/area series and discrete bounding boxes for data
# markers and extrema (peaks, valleys, inflections) directly from chart coordinate spaces.

def stroke_to_polygon_ribbon(
    pixel_points: List[Tuple[float, float]],
    linewidth_px: float,
    clip_bbox: Optional[Tuple[float, float, float, float]] = None
) -> List[Tuple[float, float]]:
    """
    Converts an ordered sequence of pixel coordinates into a closed ribbon polygon by
    offsetting each vertex along its averaged segment normal.
    Optionally clips vertices to a bounding box (x0, y0, x1, y1) such as the axes viewport.
    """
    pts = np.array(pixel_points, dtype=np.float64)
    if len(pts) < 2:
        return []

    tangents = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1] / lengths[:, 0]
    normals[:, 1] = tangents[:, 0] / lengths[:, 0]

    v_normals = np.zeros_like(pts)
    v_normals[0] = normals[0]
    v_normals[-1] = normals[-1]
    if len(pts) > 2:
        v_normals[1:-1] = (normals[:-1] + normals[1:]) / 2.0
    v_normals_len = np.linalg.norm(v_normals, axis=1, keepdims=True) + 1e-8
    v_normals = v_normals / v_normals_len

    half_w = max(1.5, linewidth_px / 2.0)
    top_edge = pts + v_normals * half_w
    bot_edge = pts - v_normals * half_w

    polygon = np.vstack([top_edge, bot_edge[::-1]])
    if clip_bbox is not None:
        cx0, cy0, cx1, cy1 = clip_bbox
        polygon[:, 0] = np.clip(polygon[:, 0], cx0, cx1)
        polygon[:, 1] = np.clip(polygon[:, 1], cy0, cy1)
    return [(float(p[0]), float(p[1])) for p in polygon]


def extract_line_segmentation_annotations(
    fig, chart_info_map, img_w: int, img_h: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extracts instance segmentation polygons and discrete marker/extrema bounding boxes for line charts.

    Returns:
        seg_annotations:    One polygon per line series (class_id 0 = "line_series").
                            'polygon' is in image-space pixel coordinates (top-left origin).
        marker_annotations: One bounding box per rendered marker glyph or extremum
                            (class_ids per CLASS_MAP_LINE_MARKERS) in Matplotlib display coordinates.
    """
    seg_annotations = []
    marker_annotations = []
    dpi_scale = fig.dpi / 72.0  # points -> pixels

    for ax in fig.axes:
        if not ax.get_visible():
            continue
        chart_info = chart_info_map.get(ax, {})
        if chart_info.get('chart_type_str') != 'line':
            continue

        ax_bbox = ax.bbox
        ax_x0 = float(ax_bbox.x0)
        ax_x1 = float(ax_bbox.x1)
        ax_y0 = float(img_h - ax_bbox.y1)
        ax_y1 = float(img_h - ax_bbox.y0)

        for series in chart_info.get('keypoint_info', []):
            pts_data = series.get('plotted_points') or series.get('all_points', [])
            if len(pts_data) < 2:
                continue

            # 1. Ribbon polygon from plotted vertices
            px_pts = []
            for x, y, *_ in pts_data:
                px, py = ax.transData.transform_point((x, y))
                px_pts.append((px, img_h - py))  # Y-flip to image space

            linewidth_pt = series.get('linewidth', 2.0)
            poly_px = stroke_to_polygon_ribbon(
                px_pts, linewidth_px=linewidth_pt * dpi_scale, clip_bbox=(ax_x0, ax_y0, ax_x1, ax_y1)
            )
            if len(poly_px) >= 3:
                clipped_poly = [
                    (max(ax_x0, min(float(px), ax_x1)), max(ax_y0, min(float(py), ax_y1)))
                    for px, py in poly_px
                ]
                seg_annotations.append({
                    'class_id': 0,  # "line_series"
                    'polygon': clipped_poly,
                    'series_idx': series.get('series_idx')
                })

            # 2. Marker glyphs (only when markers are rendered on the series)
            marker = series.get('marker')
            if marker:
                marker_r = max(2.5, (series.get('markersize', 6.0) / 2.0) * dpi_scale)
                for point_idx, (x, y, *idx) in enumerate(pts_data):
                    mx, my = ax.transData.transform_point((x, y))
                    cmx0 = max(float(ax_bbox.x0), mx - marker_r)
                    cmy0 = max(float(ax_bbox.y0), my - marker_r)
                    cmx1 = min(float(ax_bbox.x1), mx + marker_r)
                    cmy1 = min(float(ax_bbox.y1), my + marker_r)
                    if cmx1 > cmx0 and cmy1 > cmy0:
                        marker_annotations.append({
                            'class_id': 0,  # "data_marker"
                            'bbox': (cmx0, cmy0, cmx1, cmy1)
                        })

    return seg_annotations, marker_annotations


def extract_area_segmentation_annotations(
    fig, chart_info_map, img_w: int, img_h: int
) -> List[Dict]:
    """
    Extracts instance segmentation polygon masks for area charts with dual mask representation:
    - amodal_polygon: full un-occluded mathematical geometry filled to the baseline,
      analytically clipped against axes viewport to prevent self-intersecting loops.
    - modal_polygon: strictly visible pixel surface after z-order rendering and cross-class
      occlusion culling (subtracting overlapping layers and foreground legend).
    """
    seg_annotations = []
    renderer = fig.canvas.get_renderer()

    for ax in fig.axes:
        if not ax.get_visible():
            continue
        chart_info = chart_info_map.get(ax, {})
        if chart_info.get('chart_type_str') != 'area':
            continue

        ax_bbox = ax.bbox
        ax_x0 = float(min(ax_bbox.x0, ax_bbox.x1))
        ax_x1 = float(max(ax_bbox.x0, ax_bbox.x1))
        ax_y0 = float(min(img_h - ax_bbox.y1, img_h - ax_bbox.y0))
        ax_y1 = float(max(img_h - ax_bbox.y1, img_h - ax_bbox.y0))
        ax_clip_box = shapely_box(ax_x0, ax_y0, ax_x1, ax_y1) if shapely_box is not None else None

        # Detect legend geometry on this axes for occlusion subtraction
        legend_geom = None
        legend = ax.get_legend()
        if legend and legend.get_visible() and renderer and shapely_box is not None:
            try:
                l_bbox = legend.get_window_extent(renderer)
                lx0 = float(min(l_bbox.x0, l_bbox.x1))
                lx1 = float(max(l_bbox.x0, l_bbox.x1))
                ly0 = float(min(img_h - l_bbox.y1, img_h - l_bbox.y0))
                ly1 = float(max(img_h - l_bbox.y1, img_h - l_bbox.y0))
                if lx1 > lx0 and ly1 > ly0:
                    legend_geom = shapely_box(lx0, ly0, lx1, ly1)
            except Exception:
                legend_geom = None

        series_list = chart_info.get('keypoint_info', [])

        # Pass 1: Build analytical amodal geometries for all series
        amodal_geoms = []
        raw_polys = []
        for series in series_list:
            top = series.get('fill_top', [])
            bottom = series.get('fill_bottom', [])
            if len(top) < 2 or len(bottom) < 2:
                amodal_geoms.append(None)
                raw_polys.append([])
                continue

            poly_data = top + list(reversed(bottom))
            px_pts = []
            for x, y, *_ in poly_data:
                px, py = ax.transData.transform_point((x, y))
                px_pts.append((float(px), float(img_h - py)))

            if len(px_pts) < 3:
                amodal_geoms.append(None)
                raw_polys.append([])
                continue

            raw_polys.append(px_pts)

            if ShapelyPolygon is not None and ax_clip_box is not None:
                try:
                    sp = ShapelyPolygon(px_pts)
                    if not sp.is_valid:
                        sp = shapely_make_valid(sp)
                    clipped = sp.intersection(ax_clip_box)
                    if clipped.is_empty or clipped.area < 1e-4:
                        amodal_geoms.append(None)
                    else:
                        if clipped.geom_type == 'MultiPolygon':
                            clipped = max(clipped.geoms, key=lambda g: g.area)
                        amodal_geoms.append(clipped)
                except Exception:
                    amodal_geoms.append(None)
            else:
                amodal_geoms.append(None)

        # Pass 2: Derive modal polygon by subtracting later series and legend
        for idx, series in enumerate(series_list):
            geom = amodal_geoms[idx]
            raw_pts = raw_polys[idx] if idx < len(raw_polys) else []

            if geom is not None and hasattr(geom, 'exterior'):
                amodal_pts = [(float(x), float(y)) for x, y in geom.exterior.coords[:-1]]
                amodal_area = float(geom.area)

                # Modal geometry starts from amodal geometry
                modal_geom = geom

                # Subtract all layers drawn on top (series_idx > idx)
                for later_idx in range(idx + 1, len(series_list)):
                    later_geom = amodal_geoms[later_idx]
                    if later_geom is not None and not modal_geom.is_empty:
                        try:
                            modal_geom = modal_geom.difference(later_geom)
                        except Exception:
                            pass

                # Subtract legend occlusion
                if legend_geom is not None and not modal_geom.is_empty:
                    try:
                        modal_geom = modal_geom.difference(legend_geom)
                    except Exception:
                        pass

                modal_pts = []
                modal_area = 0.0
                if not modal_geom.is_empty and modal_geom.area >= 1e-4:
                    if modal_geom.geom_type == 'MultiPolygon':
                        primary_m = max(modal_geom.geoms, key=lambda g: g.area)
                        modal_pts = [(float(x), float(y)) for x, y in primary_m.exterior.coords[:-1]]
                        modal_area = float(modal_geom.area)
                    elif hasattr(modal_geom, 'exterior'):
                        modal_pts = [(float(x), float(y)) for x, y in modal_geom.exterior.coords[:-1]]
                        modal_area = float(modal_geom.area)

                if not modal_pts:
                    modal_pts = amodal_pts.copy()
                    modal_area = amodal_area
            else:
                # Fallback to clamped coordinates
                clamped = [
                    (max(ax_x0, min(float(px), ax_x1)), max(ax_y0, min(float(py), ax_y1)))
                    for px, py in raw_pts
                ]
                amodal_pts = clamped
                amodal_area = float(len(clamped))
                modal_pts = clamped
                modal_area = amodal_area

            if len(amodal_pts) < 3:
                continue

            seg_annotations.append({
                'class_id': 0,  # "area_series"
                'polygon': amodal_pts,
                'amodal_polygon': amodal_pts,
                'modal_polygon': modal_pts,
                'amodal_area': amodal_area,
                'modal_area': modal_area,
                'segmentation': {
                    'amodal_polygon': amodal_pts,
                    'modal_polygon': modal_pts,
                    'amodal_area': amodal_area,
                    'modal_area': modal_area,
                },
                'series_idx': series.get('series_idx')
            })

    return seg_annotations


def save_annotations_yolo_seg(annotations: List[Dict], img_w: int, img_h: int, output_path: str):
    """
    Saves annotations in YOLO instance-segmentation label format:
    <class_id> x1 y1 x2 y2 ... xn yn (normalized coordinates in [0, 1]).
    Expects ann['polygon'] in image-space pixel coordinates (top-left origin).

    Out-of-frame geometry is handled by clamping each vertex independently to
    [0.0, 1.0] in normalized space, not by clipping the polygon against the
    canvas (e.g. Sutherland-Hodgman) to derive a new, re-shaped boundary. Per-
    vertex clamping is simpler and cheap, at the cost of a vertex far outside
    the frame being dragged straight to the nearest edge rather than the edge
    being interpolated where the true boundary actually crosses it. That
    trade-off is accepted here; if sub-pixel edge accuracy for heavily
    off-canvas polygons ever matters, that's a real polygon-clip that would
    need to be added deliberately, not a bug in the clamp below.
    """
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            polygon = ann.get('polygon', [])
            if len(polygon) < 3:
                continue

            # Clamping a vertex near/outside the canvas edge to [0,1] can produce
            # consecutive duplicate points (e.g. two clamped-to-0.0 vertices in a
            # row); some polygon loaders are picky about that, so drop repeats.
            norm_pts = []
            for x, y in polygon:
                x_norm = max(0.0, min(1.0, float(x) / img_w))
                y_norm = max(0.0, min(1.0, float(y) / img_h))
                if norm_pts and norm_pts[-1] == (x_norm, y_norm):
                    continue
                norm_pts.append((x_norm, y_norm))

            if len(norm_pts) < 3:
                continue

            poly_parts = [str(class_id)]
            for x_norm, y_norm in norm_pts:
                poly_parts.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

            f.write(" ".join(poly_parts) + "\n")

            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [SEG-FORMAT] class={class_id}, {len(norm_pts)} vertices -> {output_path}")


_V3_DEPRECATION_WARNED = False


def _use_schema_v4(cfg):
    """v4.0 is the default. v3.0 is used only when the config asks for something other than v4.0."""
    requested = {cfg.get('annotation_schema_version'), cfg.get('detailed_schema')} - {None}
    return not requested or ANNOTATION_SCHEMA_VERSION_V4 in requested


def _warn_v3_deprecated():
    """Announce (once per process) that annotation schema v3.0 is deprecated."""
    global _V3_DEPRECATION_WARNED
    if not _V3_DEPRECATION_WARNED:
        _V3_DEPRECATION_WARNED = True
        print("[DEPRECATION] annotation schema v3.0 is deprecated and will be removed; v4.0 is now the default. "
              "Remove 'annotation_schema_version' / 'detailed_schema' from the config to switch.")


def json_default_fallback(obj):
    """Universal JSON serializer fallback handling NumPy types and Matplotlib objects."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'extents'):
        x0, y0, x1, y1 = obj.extents
        return [float(x0), float(y0), float(x1), float(y1)]
    return str(obj)

# ===================================================================================
# == GNN TRAINING DATA & GRAPH TOPOLOGY
# ===================================================================================

def add_graph_topology_metadata(fig, detailed_metadata, img_h):
    """
    Augments metadata with Graph structure: Baselines and Bar-to-Baseline links.
    Used for training Graph Neural Networks (GNN) for chart understanding.

    Adds to detailed_metadata:
    - "baselines": List of baseline objects with y_pixel, x_range, type
    - "bars_with_baseline": List of bars with pixel xyxy and baseline_id

    This is CRITICAL for GNN training - it provides the ground truth
    bar-to-baseline grouping without manual annotation.
    """
    renderer = fig.canvas.get_renderer()

    # Initialize graph sections
    detailed_metadata["baselines"] = []
    detailed_metadata["bars_with_baseline"] = []
    ax_to_baseline = {}

    # 1. Identify Baselines (Axis Spines) and collect axis transforms
    axis_transforms = {}
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue

        try:
            bbox = ax.get_window_extent(renderer)

            # Calculate baseline Y-coordinate (bottom spine)
            # Matplotlib (0,0) is bottom-left, Image (0,0) is top-left
            baseline_y_pixel = img_h - bbox.y0

            baseline_id = f"baseline_{i}"
            ax_to_baseline[ax] = baseline_id
            axis_transforms[i] = ax.transData  # Store transform for bar conversion

            # Detect dual-axis (secondary axes often share same bbox)
            is_secondary = False
            if i > 0:
                try:
                    first_bbox = fig.axes[0].get_window_extent(renderer)
                    is_secondary = (abs(bbox.x0 - first_bbox.x0) < 5 and
                                   abs(bbox.x1 - first_bbox.x1) < 5)
                except:
                    pass

            detailed_metadata["baselines"].append({
                "id": baseline_id,
                "y_pixel": float(baseline_y_pixel),
                "x_range": [float(bbox.x0), float(bbox.x1)],
                "width": float(bbox.width),
                "type": "secondary" if is_secondary else "primary",
                "bbox": [float(bbox.x0), float(img_h - bbox.y1),
                        float(bbox.x1), float(img_h - bbox.y0)],
                "axis_index": i
            })

        except Exception as e:
            if GENERATION_CONFIG.get('debug_mode'):
                print(f"DEBUG: Could not process axis {i} for baseline: {e}")

    # 2. Convert bar_info (data coords) to pixel coords and link to baselines
    bar_info_list = detailed_metadata.get("bar_info", [])

    if bar_info_list and detailed_metadata["baselines"]:
        # Get the primary axis for coordinate transformation
        primary_ax = fig.axes[0] if fig.axes else None

        if primary_ax:
            for bar_idx, bar_info in enumerate(bar_info_list):
                try:
                    # Extract data coordinates from bar_info
                    # bar_info fields:
                    #   - bar_idx: The X position (category index: 0, 1, 2...)
                    #   - width: Bar width in X-axis data units (~0.8)
                    #   - bottom: Y value at bar bottom (usually 0)
                    #   - top: Y value at bar top (the actual data value)
                    #   - center: Y-axis midpoint (NOT X position!)

                    # X position from bar_idx (category index) or x_value (histograms)
                    x_position = bar_info.get("x_value")
                    if x_position is None:
                        x_position = bar_info.get("bar_idx", bar_idx)
                    bar_width = bar_info.get("width", 0.8)

                    # Y values from bottom and top
                    bottom = bar_info.get("bottom", 0)
                    top = bar_info.get("top", bar_info.get("height", 0))

                    # Calculate data-space corners
                    data_x0 = x_position - bar_width / 2
                    data_x1 = x_position + bar_width / 2
                    data_y0 = bottom
                    data_y1 = top

                    # Transform data coords to pixel coords
                    # transData converts (data_x, data_y) -> (pixel_x, pixel_y)
                    pixel_bottom_left = primary_ax.transData.transform((data_x0, data_y0))
                    pixel_top_right = primary_ax.transData.transform((data_x1, data_y1))

                    # Convert matplotlib coords (origin bottom-left) to image coords (origin top-left)
                    px0 = float(pixel_bottom_left[0])
                    py1 = float(img_h - pixel_bottom_left[1])  # Bottom in image coords
                    px1 = float(pixel_top_right[0])
                    py0 = float(img_h - pixel_top_right[1])    # Top in image coords

                    # Ensure proper ordering (y0 < y1)
                    if py0 > py1:
                        py0, py1 = py1, py0

                    # Create pixel xyxy bounding box
                    xyxy = [px0, py0, px1, py1]

                    # Calculate bar center for baseline matching
                    bar_cx = (px0 + px1) / 2
                    bar_cy = (py0 + py1) / 2
                    bar_bottom_y = py1  # Bottom of bar in image coords

                    # Find the best matching baseline
                    best_baseline = None
                    min_dist = float('inf')

                    for baseline in detailed_metadata["baselines"]:
                        # Check X-containment (bar center within axis width)
                        x_min, x_max = baseline['x_range']
                        if x_min <= bar_cx <= x_max:
                            # Distance from bar bottom to baseline
                            dist = abs(bar_bottom_y - baseline['y_pixel'])

                            if dist < min_dist:
                                min_dist = dist
                                best_baseline = baseline['id']

                    # Cut off implausibly distant baselines (e.g. from exaggerated margins or twinx)
                    max_baseline_dist = float(img_h) * 1.5
                    if min_dist > max_baseline_dist:
                        best_baseline = None

                    # Skip bars that fall outside the plot area (invalid coords)
                    # This can happen when axis limits don't match bar count
                    if best_baseline is None:
                        # Bar center is outside all baseline x_ranges - skip it
                        if GENERATION_CONFIG.get('debug_mode'):
                            print(f"DEBUG: Skipping bar {bar_idx} - center {bar_cx:.0f} outside baselines")
                        continue

                    # Add bar with pixel coordinates and baseline link
                    bar_entry = {
                        "xyxy": xyxy,
                        "data_value": float(bar_info.get("value", top)),
                        "series_idx": bar_info.get("series_idx", 0),
                        "bar_index": bar_idx,
                        "baseline_id": best_baseline,
                        "baseline_distance": float(min_dist) if best_baseline else None
                    }
                    detailed_metadata["bars_with_baseline"].append(bar_entry)

                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode'):
                        print(f"DEBUG: Could not convert bar {bar_idx} to pixels: {e}")

    if GENERATION_CONFIG.get('debug_mode'):
        print(f"DEBUG: GNN metadata - {len(detailed_metadata['baselines'])} baselines, {len(detailed_metadata['bars_with_baseline'])} bars linked")

    return detailed_metadata


def extract_true_baseline_location(fig, detailed_metadata, img_h):
    """
    Calculates the exact pixel coordinate of the logical baseline (y=0 or x=0).
    Used to train SOTA Keypoint Detectors (ChartOCR-style).

    This is SUPERIOR to manual annotation because matplotlib knows the
    coordinate to 64-bit float precision. Manually clicking has ~2-5px error.

    Returns:
        List of baseline annotations with exact pixel coordinates
    """
    baseline_annotations = []

    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue

        orientation = detailed_metadata.get("orientation", "vertical")

        try:
            # Project (0, 0) from Data Space to Pixel Space
            # This is the "God View" - exact location of the zero line
            origin_pixel = ax.transData.transform((0, 0))
            origin_x_px = origin_pixel[0]
            origin_y_px = img_h - origin_pixel[1]  # Flip Y for image coords

            # Get axis limits to check if 0 is within range
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if orientation == "vertical":
                # For vertical bars, baseline is horizontal at Y = origin_y_px
                # Check if y=0 is within the axis range
                if ylim[0] <= 0 <= ylim[1] and 0 <= origin_y_px <= img_h:
                    baseline_annotations.append({
                        "axis_index": i,
                        "orientation": "vertical",
                        "baseline_coordinate": float(origin_y_px),
                        "type": "zero_line",
                        "axis_limits": {"y_min": float(ylim[0]), "y_max": float(ylim[1])}
                    })
            else:
                # For horizontal bars, baseline is vertical at X = origin_x_px
                img_w = detailed_metadata.get("resolution", [800, 600])[0]
                if xlim[0] <= 0 <= xlim[1] and 0 <= origin_x_px <= img_w:
                    baseline_annotations.append({
                        "axis_index": i,
                        "orientation": "horizontal",
                        "baseline_coordinate": float(origin_x_px),
                        "type": "zero_line",
                        "axis_limits": {"x_min": float(xlim[0]), "x_max": float(xlim[1])}
                    })

        except Exception as e:
            if GENERATION_CONFIG.get('debug_mode'):
                print(f"DEBUG: Could not project baseline for axis {i}: {e}")

    return baseline_annotations


def create_unified_annotation(fig, chart_info_map, cls_map, img_w, img_h, annotations):
    """
    Build unified metadata with consistent xyxy coordinates, semantic text labels,
    and visual element boxes for synthetic benchmarking.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    # Build class-id -> class-name map (supporting both "1" and 1 keys)
    class_id_to_name = {}
    class_names = set()
    for class_id, class_name in cls_map.items():
        class_names.add(class_name)
        class_id_to_name[str(class_id)] = class_name
        try:
            class_id_to_name[int(class_id)] = class_name
        except (ValueError, TypeError):
            pass

    raw_to_element_key = {
        "bar": "bar",
        "data_point": "data_point",
        "box": "box",
        "median_line": "median_line",
        "range_indicator": "range_indicator",
        "outlier": "outlier",
        "wedge": "wedge",
        "line_segment": "line_segment",
        "area_boundary": "area_boundary",
        "cell": "cell",
    "color_bar": "color_bar",
        "color_bar_label": "color_bar_label",
        "color_bar_title": "color_bar_title",
        "connector_line": "connector_line",
        "legend": "legend",
        "error_bar": "error_bar",
        "significance_marker": "significance_marker",
        "data_label": "data_label",
        "chart_title": "chart_title",
        "axis_title": "axis_title",
    }

    semantic_role_map = {
        "chart": "layout",
        "chart_title": "text_title",
        "axis_title": "text_axis",
        "axis_labels": "text_axis_label",
        "legend": "text_legend",
        "data_label": "text_data_label",
        "bar": "data_element",
        "data_point": "data_element",
        "box": "data_element",
        "wedge": "data_element",
        "line_segment": "data_element",
        "area_boundary": "data_element",
        "cell": "data_element",
        "median_line": "statistical_element",
        "range_indicator": "statistical_element",
        "error_bar": "statistical_element",
        "outlier": "statistical_element",
        "significance_marker": "statistical_element",
        "connector_line": "connector",
        "color_bar": "auxiliary",
        "color_bar_label": "text_axis_label",
        "color_bar_title": "text_title",
    }

    detailed_metadata = {
        "chart_type": None,
        "orientation": None,
        "resolution": [int(img_w), int(img_h)],
        "scale_labels": [],
        "tick_labels": [],
        "chart_title": [],
        "axis_title": [],
        "legend": [],
        "bar": [],
        "data_point": [],
        "error_bar": [],
        "significance_marker": [],
        "data_label": [],
        "box": [],
        "median_line": [],
        "range_indicator": [],
        "outlier": [],
        "wedge": [],
        "line_segment": [],
        "area_boundary": [],
        "cell": [],
        "color_bar": [],
        "color_bar_label": [],
        "color_bar_title": [],
        "connector_line": [],
        "scale_axis_info": {},
        "bar_info": [],
        "keypoint_info": [],
        "boxplot_metadata": {},
        "pie_geometry": {},
        "pie_metadata": {},
        "histogram_metadata": {},
        "series_count": 1,
        "series_names": [],
        "stacking_mode": None,
        "dual_axis_info": {},
        "style": None,
        "pattern": None,
        "is_scientific": False,
    }

    seen = set()

    def _clip_xyxy(xyxy):
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            return None
        x0, y0, x1, y1 = [int(round(v)) for v in xyxy[:4]]
        x0 = max(0, min(x0, int(img_w)))
        y0 = max(0, min(y0, int(img_h)))
        x1 = max(0, min(x1, int(img_w)))
        y1 = max(0, min(y1, int(img_h)))
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def _is_numeric_text(text):
        cleaned = str(text).strip().replace("%", "").replace(",", "")
        return is_float(cleaned)

    def add_xyxy_annotation(element_type, xyxy, text="", conf=1.0, extra=None):
        if element_type not in detailed_metadata:
            return None
        clipped = _clip_xyxy(xyxy)
        if not clipped:
            return None

        dedupe_key = (
            element_type,
            clipped[0], clipped[1], clipped[2], clipped[3],
            str(text).strip()
        )
        if dedupe_key in seen:
            return None
        seen.add(dedupe_key)

        entry = {"xyxy": clipped, "conf": float(conf)}
        if text:
            entry["text"] = str(text).strip()
        if extra:
            entry.update(extra)
        detailed_metadata[element_type].append(entry)
        return entry

    def add_bbox_annotation(element_type, bbox, text="", conf=1.0, extra=None):
        if bbox is None:
            return None
        if hasattr(bbox, "width") and hasattr(bbox, "height"):
            if bbox.width < 1 or bbox.height < 1:
                return None
        return add_xyxy_annotation(element_type, bbox_to_xyxy(bbox, img_h), text=text, conf=conf, extra=extra)

    def _annotation_bbox_to_xyxy(ann_bbox):
        if ann_bbox is None:
            return None

        if isinstance(ann_bbox, dict):
            return _clip_xyxy([
                ann_bbox.get("x0", 0),
                ann_bbox.get("y0", 0),
                ann_bbox.get("x1", 0),
                ann_bbox.get("y1", 0),
            ])

        if isinstance(ann_bbox, (list, tuple)) and len(ann_bbox) >= 4:
            return _clip_xyxy(ann_bbox[:4])

        if hasattr(ann_bbox, "extents") or hasattr(ann_bbox, "x0"):
            return _clip_xyxy(bbox_to_xyxy_absolute(ann_bbox, img_h))

        return None

    # Extract chart metadata from chart_info_map
    subplots = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax not in chart_info_map:
            continue

        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get("chart_type_str", "unknown")
        is_primary = (len(subplots) == 0)

        if is_primary:
            detailed_metadata["chart_type"] = chart_type
            detailed_metadata["orientation"] = chart_info.get("orientation", "vertical")
            detailed_metadata["series_count"] = chart_info.get("series_count", 1)
            detailed_metadata["series_names"] = chart_info.get("series_names", [])
            detailed_metadata["stacking_mode"] = chart_info.get("stacking_mode")
            detailed_metadata["dual_axis_info"] = chart_info.get("dual_axis_info", {})
            detailed_metadata["style"] = chart_info.get("style")
            detailed_metadata["pattern"] = chart_info.get("pattern")
            detailed_metadata["is_scientific"] = chart_info.get("is_scientific", False)
            detailed_metadata["semantic_domain"] = chart_info.get("semantic_domain")

        from chart import extract_scale_axis_info
        scale_axis_info = chart_info.get("scale_axis_info") or extract_scale_axis_info(ax, chart_type)
        if scale_axis_info and is_primary:
            detailed_metadata["scale_axis_info"] = scale_axis_info

        bar_info_list = chart_info.get("bar_info") or chart_info.get("bar_info_list")
        if not bar_info_list:
            from chart import extract_bar_info
            bar_info_list = extract_bar_info(ax, chart_type)
        if bar_info_list and is_primary:
            detailed_metadata["bar_info"] = [
                {
                    "center": float(info.get("center", 0)),
                    "height": float(info.get("height", 0)),
                    "width": float(info.get("width", 0)),
                    "bottom": float(info.get("bottom", 0)),
                    "top": float(info.get("top", 0)),
                    "x_value": info.get("x_value"),
                    "series_idx": info.get("series_idx"),
                    "bar_idx": info.get("bar_idx"),
                    "axis": info.get("axis", "primary"),
                }
                for info in bar_info_list
            ]

        from chart import extract_keypoint_info
        raw_keypoint_info = chart_info.get("keypoint_info")
        keypoint_info = extract_keypoint_info(ax, chart_type)
        if keypoint_info and is_primary:
            enriched_kpi = []
            for kp_idx, kp in enumerate(keypoint_info):
                pts_list = kp.get("points", [])
                legacy_points = [
                    {
                        "x": float(pt.get("x", 0)),
                        "y": float(pt.get("y", 0)),
                        "is_inflection": bool(pt.get("is_inflection", False)),
                    }
                    for pt in pts_list
                ]

                # Map peak, valley, and inflection indices from chart_info metadata if present
                raw_kpi = (
                    raw_keypoint_info[kp_idx]
                    if (raw_keypoint_info and isinstance(raw_keypoint_info, list) and kp_idx < len(raw_keypoint_info))
                    else {}
                )
                peak_indices = {int(idx) for _, _, idx in raw_kpi.get("peaks", [])} if isinstance(raw_kpi, dict) else set()
                valley_indices = {int(idx) for _, _, idx in raw_kpi.get("valleys", [])} if isinstance(raw_kpi, dict) else set()
                inflection_indices = {int(idx) for _, _, idx in raw_kpi.get("inflections", [])} if isinstance(raw_kpi, dict) else set()

                # Fallback: Detect peaks and valleys from y sequence if not pre-computed
                if not peak_indices and not valley_indices and len(legacy_points) >= 3:
                    y_vals = [p["y"] for p in legacy_points]
                    for j in range(1, len(y_vals) - 1):
                        if y_vals[j] > y_vals[j - 1] and y_vals[j] >= y_vals[j + 1]:
                            peak_indices.add(j)
                        elif y_vals[j] < y_vals[j - 1] and y_vals[j] <= y_vals[j + 1]:
                            valley_indices.add(j)

                # Construct topological keypoints K_i = (x_i, y_i, v_i, tau_i)
                topological_kpts = []
                n_pts = len(legacy_points)
                for j, pt in enumerate(legacy_points):
                    if j == 0 or j == n_pts - 1:
                        tag = "endpoint"
                    elif j in peak_indices:
                        tag = "peak"
                    elif j in valley_indices:
                        tag = "valley"
                    elif j in inflection_indices or pt.get("is_inflection"):
                        tag = "inflection"
                    else:
                        tag = "vertex"

                    data_x = float(pt.get("x", 0))
                    data_y = float(pt.get("y", 0))
                    px_x, px_y = data_x, data_y
                    try:
                        if hasattr(ax, "transData"):
                            disp = ax.transData.transform_point((data_x, data_y))
                            px_x = float(disp[0])
                            px_y = float(img_h - disp[1])
                    except Exception:
                        pass

                    topological_kpts.append({
                        "x": px_x,
                        "y": px_y,
                        "pixel_x": px_x,
                        "pixel_y": px_y,
                        "data_x": data_x,
                        "data_y": data_y,
                        "v": 1,
                        "visibility": 1,
                        "tag": tag,
                        "tau": tag,
                        "point_idx": j,
                    })

                edges = [[j, j + 1] for j in range(n_pts - 1)]

                series_dict = {
                    "series_idx": kp.get("series_idx"),
                    "points": legacy_points,
                    "keypoints": topological_kpts,
                    "topological_keypoints": topological_kpts,
                    "edges": edges,
                    "adjacency_list": edges,
                }

                # Attach dual segmentation mask for area series if available
                if chart_type == "area" and isinstance(raw_kpi, dict):
                    if "segmentation" in raw_kpi:
                        series_dict["segmentation"] = raw_kpi["segmentation"]
                        series_dict["amodal_polygon"] = raw_kpi.get("amodal_polygon")
                        series_dict["modal_polygon"] = raw_kpi.get("modal_polygon")

                enriched_kpi.append(series_dict)

            detailed_metadata["keypoint_info"] = enriched_kpi

        boxplot_dict = chart_info.get("boxplot_dict", {})
        if boxplot_dict and chart_type == "box" and is_primary:
            detailed_metadata["boxplot_metadata"] = {
                "num_groups": boxplot_dict.get("num_groups", 0),
                "box_width": float(boxplot_dict.get("box_width", 0)),
                "orientation": boxplot_dict.get("orientation", "vertical"),
                "medians": [
                    {
                        "group_index": m.get("group_index"),
                        "group_label": m.get("group_label"),
                        "median_value": float(m.get("median_value", 0)),
                        "lower_left": m.get("lower_left", {}),
                        "upper_right": m.get("upper_right", {}),
                        "center_x": m.get("center_x"),
                        "center_y": m.get("center_y"),
                        "line_length": float(m.get("line_length", 0)),
                    }
                    for m in boxplot_dict.get("medians", [])
                ],
            }

        from chart import extract_pie_geometry
        pie_geometry = extract_pie_geometry(ax, chart_type)
        if pie_geometry and is_primary:
            detailed_metadata["pie_geometry"] = {
                "center_point": {
                    "x": float(pie_geometry.get("center_point", {}).get("x", 0)),
                    "y": float(pie_geometry.get("center_point", {}).get("y", 0)),
                },
                "radius": float(pie_geometry.get("radius", 0)),
                "wedges": [
                    {
                        "wedge_index": w.get("wedge_index"),
                        "start_angle": float(w.get("start_angle", 0)),
                        "end_angle": float(w.get("end_angle", 0)),
                        "mid_angle": float(w.get("mid_angle", 0)),
                        "percentage": float(w.get("percentage", 0)),
                    }
                    for w in pie_geometry.get("wedges", [])
                ],
            }

        pie_meta = chart_info.get("pie_metadata")
        if pie_meta and is_primary:
            detailed_metadata["pie_metadata"] = pie_meta

        histogram_meta = chart_info.get("histogram_metadata")
        if histogram_meta and chart_type == "histogram" and is_primary:
            detailed_metadata["histogram_metadata"] = histogram_meta

        if is_primary:
            detailed_metadata["series_count"] = chart_info.get("series_count", 1)
            detailed_metadata["series_names"] = chart_info.get("series_names", [])
            detailed_metadata["stacking_mode"] = chart_info.get("stacking_mode")
            detailed_metadata["dual_axis_info"] = chart_info.get("dual_axis_info", {})
            detailed_metadata["style"] = chart_info.get("style")
            detailed_metadata["pattern"] = chart_info.get("pattern")
            detailed_metadata["is_scientific"] = chart_info.get("is_scientific", False)
            detailed_metadata["semantic_domain"] = chart_info.get("semantic_domain")

        subplot_record = {
            "chart_type": chart_type,
            "semantic_domain": chart_info.get("semantic_domain"),
            "is_scientific": chart_info.get("is_scientific", False),
            "orientation": chart_info.get("orientation", "vertical"),
            "series_count": chart_info.get("series_count", 1),
            "series_names": chart_info.get("series_names", []),
            "stacking_mode": chart_info.get("stacking_mode"),
            "style": chart_info.get("style"),
            "pattern": chart_info.get("pattern"),
            "dual_axis_info": chart_info.get("dual_axis_info", {}),
            "scale_axis_info": scale_axis_info or {},
        }
        subplots.append(subplot_record)

    detailed_metadata["subplots"] = subplots
    detailed_metadata["is_composite"] = len(subplots) > 1
    detailed_metadata["composite_chart_types"] = [s["chart_type"] for s in subplots]
    detailed_metadata["composite_domains"] = [s["semantic_domain"] for s in subplots]

    # Semantic text labels from matplotlib artists
    for ax in fig.axes:
        if not ax.get_visible():
            continue

        if "chart_title" in class_names:
            title = ax.title
            if title and title.get_visible() and title.get_text():
                add_bbox_annotation("chart_title", title.get_window_extent(renderer), text=title.get_text().strip(), conf=1.0)

        if "axis_title" in class_names:
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
                add_bbox_annotation(
                    "axis_title",
                    ax.xaxis.label.get_window_extent(renderer),
                    text=ax.xaxis.label.get_text().strip(),
                    conf=1.0,
                    extra={"axis": "x"},
                )
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
                add_bbox_annotation(
                    "axis_title",
                    ax.yaxis.label.get_window_extent(renderer),
                    text=ax.yaxis.label.get_text().strip(),
                    conf=1.0,
                    extra={"axis": "y"},
                )

        if "axis_labels" in class_names:
            scale_axis_info = detailed_metadata.get("scale_axis_info", {})
            primary_scale_axis = scale_axis_info.get("primary_scale_axis", "y")
            secondary_scale_axis = scale_axis_info.get("secondary_scale_axis", None)
            bgcolor = ax.get_facecolor()

            for label in ax.get_xticklabels():
                if label.get_visible() and label.get_text() and has_non_background_pixels(label, fig, ax, bgcolor, threshold=5):
                    txt = label.get_text().strip()
                    if _is_numeric_text(txt) and (primary_scale_axis == "x" or secondary_scale_axis == "x"):
                        add_bbox_annotation(
                            "scale_labels",
                            label.get_window_extent(renderer),
                            text=txt,
                            conf=1.0,
                            extra={"axis": "x", "is_numeric": True},
                        )
                    else:
                        add_bbox_annotation(
                            "tick_labels",
                            label.get_window_extent(renderer),
                            text=txt,
                            conf=1.0,
                            extra={"axis": "x", "is_numeric": _is_numeric_text(txt)},
                        )

            for label in ax.get_yticklabels():
                if label.get_visible() and label.get_text() and has_non_background_pixels(label, fig, ax, bgcolor, threshold=5):
                    txt = label.get_text().strip()
                    if _is_numeric_text(txt) and (primary_scale_axis == "y" or secondary_scale_axis == "y"):
                        add_bbox_annotation(
                            "scale_labels",
                            label.get_window_extent(renderer),
                            text=txt,
                            conf=1.0,
                            extra={"axis": "y", "is_numeric": True},
                        )
                    else:
                        add_bbox_annotation(
                            "tick_labels",
                            label.get_window_extent(renderer),
                            text=txt,
                            conf=1.0,
                            extra={"axis": "y", "is_numeric": _is_numeric_text(txt)},
                        )

    # Normalize raw annotations and use them to populate element boxes
    normalized_raw_annotations = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        raw_bbox = ann.get("xyxy", ann.get("bbox"))
        xyxy = _annotation_bbox_to_xyxy(raw_bbox)
        if not xyxy:
            continue

        class_id = ann.get("class_id")
        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            pass

        class_name = class_id_to_name.get(class_id)
        if class_name is None:
            class_name = class_id_to_name.get(str(class_id), "unknown")

        text = str(ann.get("text", "")).strip()

        raw_entry = {
            "class_id": class_id,
            "class_name": class_name,
            "semantic_role": semantic_role_map.get(class_name, "other"),
            "xyxy": xyxy,
        }
        if text:
            raw_entry["text"] = text
        if "obb" in ann and ann["obb"]:
            raw_entry["obb"] = ann["obb"]
        normalized_raw_annotations.append(raw_entry)

        element_key = raw_to_element_key.get(class_name)
        if element_key:
            extra = {"class_id": class_id}
            if "obb" in ann and ann["obb"]:
                extra["obb"] = ann["obb"]
            if class_name == "axis_title":
                extra["axis"] = ann.get("axis")
            add_xyxy_annotation(element_key, xyxy, text=text, conf=1.0, extra=extra)
        elif class_name == "axis_labels":
            extra = {"is_numeric": _is_numeric_text(text)}
            if "obb" in ann and ann["obb"]:
                extra["obb"] = ann["obb"]
            add_xyxy_annotation(
                "scale_labels" if _is_numeric_text(text) else "tick_labels",
                xyxy,
                text=text,
                conf=1.0,
                extra=extra,
            )

    detailed_metadata["raw_annotations"] = normalized_raw_annotations

    # Backward-compatible key aliases used by older consumers
    alias_map = {
        "chart_title": "charttitle",
        "axis_title": "axistitle",
        "scale_labels": "scalelabels",
        "tick_labels": "ticklabels",
        "data_point": "datapoint",
        "error_bar": "errorbar",
        "significance_marker": "significancemarker",
        "data_label": "datalabel",
        "median_line": "medianline",
        "range_indicator": "rangeindicator",
        "line_segment": "linesegment",
        "area_boundary": "areaboundary",
        "color_bar": "colorbar",
        "connector_line": "connectorline",
    }
    for canonical_key, alias_key in alias_map.items():
        detailed_metadata[alias_key] = detailed_metadata.get(canonical_key, [])

    # =========================================================================
    # GNN TRAINING DATA: Add graph topology (bar-to-baseline links)
    # =========================================================================
    detailed_metadata = add_graph_topology_metadata(fig, detailed_metadata, img_h)

    # =========================================================================
    # KEYPOINT TRAINING DATA: Extract true baseline locations
    # =========================================================================
    detailed_metadata["baseline_keypoints"] = extract_true_baseline_location(
        fig, detailed_metadata, img_h
    )

    return detailed_metadata

class HeatmapQualityValidator:
    """Comprehensive quality checks for generated heatmaps."""

    def __init__(self, config):
        self.config = config
        self.failures = []

    def _normalize_class_id(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def validate_data_structure(self, data):
        """Check for realistic spatial patterns using Moran's I for spatial autocorrelation."""
        try:
            # Calculate spatial autocorrelation using a simplified approach
            # Moran's I measures spatial autocorrelation (values near 0 = random, > 0 = clustered)
            rows, cols = data.shape

            if rows < 2 or cols < 2:
                return True  # Skip validation for very small matrices

            finite_mask = np.isfinite(data)
            if not np.any(finite_mask):
                self.failures.append("Heatmap data contains no finite values")
                return False

            # Moran's I over 4-neighbour adjacency (each undirected pair counted in both directions).
            # NOTE: preserved from the original implementation: normalises by N rather than by the
            # neighbour-pair count W, so it reads ~W/N (~4x) higher than textbook Moran's I.
            z = data - np.nanmean(data)
            cross, n_pairs = 0.0, 0
            for a, b in ((z[1:], z[:-1]), (z[:, 1:], z[:, :-1])):
                ok = np.isfinite(a) & np.isfinite(b)
                cross += 2.0 * np.sum(a[ok] * b[ok])
                n_pairs += 2 * int(ok.sum())

            if n_pairs == 0:
                return True  # No neighbors (single cell)

            variance = np.nanvar(data)

            if variance == 0:
                # All values are the same - not spatially interesting but valid
                return True

            moran_i = cross / (rows * cols * variance)

            # Realistic heatmaps should have positive spatial autocorrelation (Moran's I > 0.1)
            if moran_i < 0.05:
                self.failures.append(f"Data too uniform, Moran's I = {moran_i:.3f} (should be > 0.05)")
                return False

            # Check for extreme outliers (>3 std dev)
            outlier_ratio = np.sum(np.abs(data - np.nanmean(data)) > 3 * np.nanstd(data)) / data.size
            if outlier_ratio > 0.05:
                self.failures.append(f"Too many outliers: {outlier_ratio:.2%}")
                return False

        except Exception as e:
            self.failures.append(f"Error in data structure validation: {e}")
            return False

        return True

    def validate_annotations(self, annotations, data_shape, fig_size_pixels):
        """Comprehensive annotation validation."""
        rows, cols = data_shape
        expected_cells = rows * cols

        # Count cells
        cell_count = sum(1 for ann in annotations if self._normalize_class_id(ann['class_id']) == 1)

        # Allow for some missing cells due to small size filtering, but expect most
        min_coverage = float(self.config.get("min_cell_coverage", 0.90))
        if cell_count < expected_cells * min_coverage:
            self.failures.append(f"Missing too many cells: {cell_count}/{expected_cells}")
            return False

        # Check bbox validity
        for ann in annotations:
            bbox = ann['bbox']
            if not self._is_valid_bbox(bbox, fig_size_pixels):
                self.failures.append(f"Invalid bbox: {bbox}")
                return False

        # Check for duplicate bboxes (allow some tolerance for floating point)
        bboxes = [(self._normalize_class_id(ann['class_id']), ann['bbox']) for ann in annotations]
        unique_bboxes = set()
        for class_id, bbox in bboxes:
            # Round coordinates to avoid floating point precision issues
            rounded_bbox = tuple(round(coord, 1) for coord in [bbox.x0, bbox.y0, bbox.x1, bbox.y1])
            key = (class_id, rounded_bbox)
            if key in unique_bboxes:
                self.failures.append("Duplicate annotations detected")
                return False
            unique_bboxes.add(key)

        return True

    def validate_visual_elements(self, ax, annotations):
        """Check required elements are present."""
        class_ids = set(self._normalize_class_id(ann['class_id']) for ann in annotations)

        required = {0, 1}  # chart, cell - basic elements
        if not required.issubset(class_ids):
            self.failures.append(f"Missing required classes: {required - class_ids}")
            return False

        # Check colorbar presence (should exist in most heatmap cases)
        has_colorbar = 3 in class_ids
        if not has_colorbar and len([a for a in annotations if a['class_id'] == 0]) > 0:  # if there are charts
            # Only warn, don't fail - colorbars are not always present
            pass

        return True

    def _is_valid_bbox(self, bbox, fig_size):
        """Check bbox is within figure bounds and has positive area."""
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        w, h = fig_size

        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return False
        if x1 <= x0 or y1 <= y0:
            return False

        return True

    def generate_report(self):
        """Generate validation report."""
        if not self.failures:
            return "PASS: All quality checks passed"
        else:
            return "FAIL:\n" + "\n".join(f"- {f}" for f in self.failures)


def generate_single_chart(i, cfg, images_dir, labels_dir, output_dir):
    if cfg.get('engine', 'matplotlib') == 'vegalite':
        from backends.vegalite_backend import generate_single_vegalite_chart
        return generate_single_vegalite_chart(i, cfg, images_dir, labels_dir, output_dir)

    iter_start = time.time()
    chart_generators = {
        "bar": _generate_bar_chart,
        "line": _generate_line_chart,
        "scatter": _generate_scatter_chart,
        "box": _generate_boxplot_chart,
        "pie": _generate_pie_chart,
        "area": _generate_area_chart,
        "histogram": _generate_histogram,
        "heatmap": _generate_heatmap_chart,
    }

    output_dpi = random.choice([96, 120, 150])

    if cfg['debug_mode']:
        print(f"DEBUG: Using DPI {output_dpi}")

    if cfg.get('dataset_format') == 'multi_chart_detection':
        scenario = 'multi'
        nrows, ncols = _select_multi_chart_layout(cfg)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), dpi=output_dpi)
        axes = np.array(axes).flatten()
        if cfg['debug_mode']:
            print(f"DEBUG: Created multi_chart_detection chart: {nrows}x{ncols}")
    else:
        # Determine scenario
        scenarios, weights = zip(*cfg['scenario_weights'].items())
        scenario = random.choices(scenarios, weights=weights, k=1)[0]

        if cfg['debug_mode']:
            print(f"DEBUG: Selected scenario: {scenario}")

        if scenario == 'multi':
            nrows, ncols = random.choice([(1,2), (2,1), (2,2), (1,3), (3,1), (2,3), (3,2), (3,3)])
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), dpi=output_dpi)
            axes = np.array(axes).flatten()

            if cfg['debug_mode']:
                print(f"DEBUG: Created multi-axis chart: {nrows}x{ncols}")
        else:
            fig, ax = plt.subplots(figsize=(7, 5), dpi=output_dpi)
            axes = [ax]

            if cfg['debug_mode']:
                print(f"DEBUG: Created single-axis chart: 1x1")

    try:
        chart_info_map = {}

        for ax_idx, ax in enumerate(axes):
            enabled_chart_types = [(k, v['weight']) for k, v in cfg['chart_types'].items() if v['enabled']]
            if not enabled_chart_types:
                raise ValueError("No chart types enabled. Set at least one chart_types[*].enabled = True.")

            chart_types, weights = zip(*enabled_chart_types)
            weights = list(weights)
            if sum(weights) <= 0:
                if cfg.get('debug_mode', False):
                    print("WARNING: Enabled chart types have zero total weight; using uniform weights.")
                weights = [1] * len(chart_types)

            chart_type = random.choices(chart_types, weights=weights, k=1)[0]
            print(f"  - Type: {chart_type} (Scenario: {scenario})")

            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Chart type selected: {chart_type}")
                print(f"DEBUG: AX[{ax_idx}]: Chart types and weights: {list(zip(chart_types, weights))}")

            # Get correct class map for this chart type
            cls_map = CHART_CLASS_MAPS.get(chart_type, CHART_CLASS_MAPS['bar'])

            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Class map: {cls_map}")

            theme_name = random.choice(list(THEMES.keys()))
            theme_config = dict(THEMES[theme_name])
            if cfg.get('use_synthetic_data_engine', False):
                theme_config['use_synthetic_data_engine'] = True
                theme_config['synthetic_domain'] = cfg.get('synthetic_domain', None)
            print(f"  - Theme: {theme_name}")

            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Theme selected: {theme_name}")

            sci_ratio = cfg.get('scientific_ratio', cfg.get('bar_chart_config', {}).get('scientific_ratio', 0.6))
            is_scientific = random.random() < sci_ratio
            subdomain_weights = cfg.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})
            theme_config['scientific_subdomain_weights'] = subdomain_weights
            theme_config['bar_chart_config'] = cfg.get('bar_chart_config', {})
            theme_config['line_chart_config'] = cfg.get('line_chart_config', {})
            theme_config['scatter_chart_config'] = cfg.get('scatter_chart_config', {})
            theme_config['box_plot_config'] = cfg.get('box_plot_config', {})
            theme_config['histogram_config'] = cfg.get('histogram_config', {})
            theme_config['pie_config'] = cfg.get('pie_config', {})
            theme_config['heatmap_config'] = cfg.get('heatmap_config', {})
            theme_config['area_chart_config'] = cfg.get('area_chart_config', {})
            style_config = {}
            cfg_semantic_domain = (
                cfg.get('theme_config', {}).get('semantic_domain')
                or cfg.get('style_config', {}).get('semantic_domain')
                or cfg.get('semantic_domain')
            )
            if cfg_semantic_domain:
                semantic_domain = cfg_semantic_domain
                try:
                    from synth.semantics.loader import registry
                    if semantic_domain in registry.domains:
                        is_scientific = registry.domains[semantic_domain].is_scientific
                except Exception:
                    pass
            elif is_scientific:
                subdomains = list(subdomain_weights.keys())
                weights = list(subdomain_weights.values())
                semantic_domain = random.choices(subdomains, weights=weights, k=1)[0]
            else:
                non_sci_weights = cfg.get('non_scientific_subdomain_weights', {'business': 0.80, 'demographic': 0.20})
                subdomains = list(non_sci_weights.keys())
                weights = list(non_sci_weights.values())
                semantic_domain = random.choices(subdomains, weights=weights, k=1)[0]
            theme_config['semantic_domain'] = semantic_domain
            style_config['semantic_domain'] = semantic_domain
            theme_config['is_scientific'] = is_scientific
            style_config['is_scientific'] = is_scientific
            if cfg.get('use_synthetic_data_engine', False):
                style_config['use_synthetic_data_engine'] = True
                style_config['synthetic_domain'] = cfg.get('synthetic_domain', None)

            generator_func = chart_generators[chart_type]

            # Initialize all possible return variables
            boxplot_dict = {}
            scale_axis_info = {}
            keypoint_data = None
            histogram_metadata = None
            pie_metadata = None

            heatmap_meta = None
            heatmap_data = None

            if chart_type == 'box':
                if cfg['debug_mode']:
                    print(f"DEBUG: AX[{ax_idx}]: Calling box plot generator")
                data_artists, other_artists, bar_info_list, orientation, error_tops, axis_related_artists, scale_axis_info, boxplot_dict = generator_func(ax, theme_name, theme_config, is_scientific, debug_mode=cfg['debug_mode'])
                keypoint_data = None
            else:
                if cfg['debug_mode']:
                    print(f"DEBUG: AX[{ax_idx}]: Calling generator for {chart_type}")
                if chart_type == 'bar':
                    if cfg['debug_mode']:
                        print(f"DEBUG: AX[{ax_idx}]: Setting up bar chart style config")
                    styles, weights = zip(*[(k, v['weight']) for k, v in cfg['bar_chart_config']['styles'].items()])
                    patterns, weights_p = zip(*[(k, v['weight']) for k, v in cfg['bar_chart_config']['patterns'].items()])
                    bar_cfg = cfg.get('bar_chart_config', {})
                    style_config.update(bar_cfg)
                    style_config['style'] = random.choices(styles, weights=weights, k=1)[0]
                    style_config['pattern'] = random.choices(patterns, weights=weights_p, k=1)[0]
                    style_config['is_scientific'] = is_scientific

                if chart_type == 'pie':
                    result = generator_func(
                        ax, theme_name, theme_config, is_scientific,
                        pie_config=cfg.get('pie_config', {}),
                        debug_mode=cfg['debug_mode']
                    )
                else:
                    result = generator_func(
                        ax, theme_name, theme_config,
                        style_config if chart_type == 'bar' else is_scientific,
                        debug_mode=cfg['debug_mode']
                    )

                if len(result) == 8:
                    data_artists, other_artists, bar_info_list, orientation, error_tops, axis_related_artists, scale_axis_info, keypoint_data = result
                else:
                    data_artists, other_artists, bar_info_list, orientation, error_tops, axis_related_artists, scale_axis_info = result
                    keypoint_data = None

                pie_metadata = None
                if chart_type == 'pie' and isinstance(keypoint_data, dict):
                    if keypoint_data.get('geometry') is not None:
                        pie_metadata = keypoint_data.get('metadata', {})
                        keypoint_data = keypoint_data.get('geometry')

                if chart_type == 'histogram':
                    histogram_metadata = keypoint_data
                    keypoint_data = None

                heatmap_meta = None
                heatmap_data = None
                if chart_type == 'heatmap' and isinstance(keypoint_data, dict):
                    heatmap_meta = keypoint_data.get('meta')
                    heatmap_data = keypoint_data.get('data')
                    keypoint_data = None

            other_artists.extend(axis_related_artists)

            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Generated {len(data_artists)} data artists, {len(other_artists)} other artists")
                print(f"DEBUG: AX[{ax_idx}]: Scale axis info: {scale_axis_info}")

            if not ax.get_title():
                eff_sci = theme_config.get('effective_is_scientific', is_scientific)
                eff_dom = theme_config.get('semantic_domain')
                title = sample_chart_title(chart_type=chart_type, is_scientific=eff_sci, domain=eff_dom, rng=random)
                ax.set_title(title, fontsize=14, pad=15)
            aux_axes = []
            for art in list(data_artists) + list(other_artists):
                if hasattr(art, 'ax') and isinstance(getattr(art, 'ax'), plt.Axes) and art.ax != ax:
                    if art.ax not in aux_axes:
                        aux_axes.append(art.ax)
                if hasattr(art, 'axes') and isinstance(getattr(art, 'axes'), plt.Axes) and art.axes != ax:
                    if art.axes not in aux_axes:
                        aux_axes.append(art.axes)

            is_dual_axis = isinstance(scale_axis_info, dict) and scale_axis_info.get('secondary_scale_axis') is not None
            dual_axis_dict = {}
            if is_dual_axis:
                dual_axis_dict = {
                    'enabled': True,
                    'primary_axis': scale_axis_info.get('primary_scale_axis', 'y'),
                    'secondary_axis': scale_axis_info.get('secondary_scale_axis', 'y2')
                }

            eff_sci = theme_config.get('effective_is_scientific', is_scientific)
            eff_dom = theme_config.get('semantic_domain')
            series_count_val = theme_config.get('series_count', len(keypoint_data) if ('keypoint_data' in locals() and keypoint_data is not None) else 1)
            series_names_val = theme_config.get('series_names', [])
            bar_style = style_config.get('style') if (chart_type == 'bar' and 'style_config' in locals() and isinstance(style_config, dict)) else None
            bar_pattern = style_config.get('pattern') if (chart_type == 'bar' and 'style_config' in locals() and isinstance(style_config, dict)) else None
            if chart_type == 'area' and 'keypoint_data' in locals() and keypoint_data and len(keypoint_data) > 0:
                stacking_mode_val = keypoint_data[0].get('stacking_mode')
            elif chart_type == 'bar' and bar_style == 'stacked':
                stacking_mode_val = 'stacked'
            else:
                stacking_mode_val = theme_config.get('stacking_mode') if isinstance(theme_config, dict) else None

            chart_info_map[ax] = {
                'chart_type_str': chart_type,
                'data_artists': data_artists,
                'other_artists': other_artists,
                'axis_related_artists': axis_related_artists,
                'aux_axes': aux_axes,
                'dual_axis_info': dual_axis_dict,
                'boxplot_dict': boxplot_dict,
                'boxplot_artists': scale_axis_info.get('boxplot_raw') if isinstance(scale_axis_info, dict) else None,
                'scale_axis_info': scale_axis_info,
                'keypoint_info': keypoint_data,
                'pie_geometry': keypoint_data if chart_type == 'pie' else None,
                'pie_metadata': pie_metadata,
                'histogram_metadata': histogram_metadata,
                'heatmap_meta': heatmap_meta,
                'heatmap_data': heatmap_data,
                'bar_info': bar_info_list,
                'bar_info_list': bar_info_list,
                'orientation': orientation,
                'is_scientific': eff_sci,
                'semantic_domain': eff_dom,
                'style': bar_style,
                'pattern': bar_pattern,
                'series_count': series_count_val,
                'series_names': series_names_val,
                'stacking_mode': stacking_mode_val,
            }

            # Store keypoint metadata for line, area, and pie charts
            if chart_type in ['line', 'area'] and keypoint_data is not None:
                chart_info_map[ax]['keypoint_info'] = keypoint_data
            elif chart_type == 'pie' and keypoint_data is not None:
                chart_info_map[ax]['pie_geometry'] = keypoint_data

        fig.tight_layout(pad=2.0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=output_dpi)
        buf.seek(0)

        # Determine the primary chart type for annotation extraction
        primary_chart_type = chart_info_map.get(axes[0], {}).get('chart_type_str', 'bar')

        if cfg.get('dataset_format') == 'multi_chart_detection':
            class_map = cfg.get('CLASS_MAP_CLASSIFICATION', GENERATION_CONFIG['CLASS_MAP_CLASSIFICATION'])
            renderer = fig.canvas.get_renderer()
            annotations = get_subchart_detection_annotations(fig, chart_info_map, class_map, renderer)
            cls_map = {}
        else:
            cls_map = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])

            if cfg['debug_mode']:
                print(f"DEBUG: Primary chart type for annotation: {primary_chart_type}")
                print(f"DEBUG: Using class map for annotations: {cls_map}")

            annotations = get_granular_annotations(fig, chart_info_map, cls_map)

        if cfg['debug_mode']:
            print(f"DEBUG: Total annotations detected: {len(annotations)}")
            class_counts = {}
            for ann in annotations:
                class_id = ann['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            print(f"DEBUG: Annotation class distribution: {class_counts}")

        pil_img_for_check = Image.open(buf).convert('RGB')
        img_w, img_h = pil_img_for_check.size

        heatmap_validation_cfg = cfg.get('heatmap_validation', {})
        if primary_chart_type == 'heatmap' and heatmap_validation_cfg.get('enabled', False):
            validator = HeatmapQualityValidator(heatmap_validation_cfg)
            heatmap_ax = axes[0]
            heatmap_data = chart_info_map.get(heatmap_ax, {}).get('heatmap_data')

            if heatmap_data is not None:
                is_valid_data = validator.validate_data_structure(heatmap_data)
                is_valid_ann = validator.validate_annotations(annotations, heatmap_data.shape, (img_w, img_h))
                is_valid_vis = validator.validate_visual_elements(heatmap_ax, annotations)

                if not (is_valid_data and is_valid_ann and is_valid_vis):
                    report = validator.generate_report()
                    print(f"    - Heatmap validation failed: {report}")

                    if heatmap_validation_cfg.get('mode') == 'strict':
                        return

        # Filter low-variance annotations
        filtered_annotations = []
        PIXEL_STD_DEV_THRESHOLD = 10

        # Get class IDs for axis_labels and legend
        axis_labels_class_id = next((k for k, v in cls_map.items() if v == 'axis_labels'), None)
        legend_class_id = next((k for k, v in cls_map.items() if v == 'legend'), None)

        for ann in annotations:
            if (axis_labels_class_id is not None and ann['class_id'] == axis_labels_class_id) or \
               (legend_class_id is not None and ann['class_id'] == legend_class_id):
                bbox = ann['bbox']

                if legend_class_id is not None and ann['class_id'] == legend_class_id:
                    padding = 3
                    x0 = max(bbox.x0 + padding, bbox.x0)
                    y0 = max(bbox.y0 + padding, bbox.y0)
                    x1 = max(bbox.x1 - padding, x0 + 1)
                    y1 = max(bbox.y1 - padding, y0 + 1)
                    ann['bbox'] = BoundingBox(x0, y0, x1, y1)
                else:
                    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

                crop_box = (int(x0), int(img_h - y1), int(x1), int(img_h - y0))

                if crop_box[0] < crop_box[2] and crop_box[1] < crop_box[3]:
                    label_crop = pil_img_for_check.crop(crop_box)
                    l_crop = label_crop.convert('L')
                    std_dev = np.array(l_crop).std()

                    if std_dev > PIXEL_STD_DEV_THRESHOLD:
                        filtered_annotations.append(ann)
            else:
                filtered_annotations.append(ann)

        annotations = filtered_annotations

        # Dual-axis post-processing
        if len(fig.axes) == 2 and any(v == 'axis_labels' for v in cls_map.values()):
            print("    - Processing dual-axis chart annotations.")
            axis_label_class_id = next((k for k, v in cls_map.items() if v == 'axis_labels'), None)
            if axis_label_class_id is not None:
                renderer = fig.canvas.get_renderer()
                main_ax_bbox = axes[0].get_window_extent(renderer)
                xaxis_y_threshold = main_ax_bbox.y0

                if cfg['debug_mode']:
                    print(f"DEBUG: Dual-axis processing - threshold Y: {xaxis_y_threshold}")

                x_axis_labels_to_filter = []
                annotations_to_keep = []

                for ann in annotations:
                    is_class_8 = (ann['class_id'] == axis_label_class_id)
                    is_on_x_axis = (ann['bbox'].y1 < xaxis_y_threshold)

                    if is_class_8 and is_on_x_axis:
                        x_axis_labels_to_filter.append(ann)
                    else:
                        annotations_to_keep.append(ann)

                deduplicated_x_labels = []
                seen_x_positions = []
                tolerance = 5

                for ann in sorted(x_axis_labels_to_filter, key=lambda a: a['bbox'].x0):
                    x_center = (ann['bbox'].x0 + ann['bbox'].x1) / 2
                    is_duplicate = any(abs(x_center - seen_x) < tolerance for seen_x in seen_x_positions)

                    if not is_duplicate:
                        deduplicated_x_labels.append(ann)
                        seen_x_positions.append(x_center)

                annotations = annotations_to_keep + deduplicated_x_labels
                print(f"    - Kept {len(deduplicated_x_labels)} of {len(x_axis_labels_to_filter)} X-axis labels.")

                if cfg['debug_mode']:
                    print(f"DEBUG: After dual-axis processing - annotations: {len(annotations)}")

        primary_chart_type = chart_info_map.get(fig.axes[0], {}).get('chart_type_str', 'unknown')

        # Apply realism effects
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        init_w, init_h = pil_img.size

        if cfg['debug_mode']:
            print(f"DEBUG: Before realism effects - annotations: {len(annotations)}")

        # Pre-extract specialized annotations before realism effects so all streams are transformed together
        annotations_obj = []
        area_seg_anns = []
        keypoint_annotations = []
        line_seg_anns = []
        line_marker_anns = []

        if cfg.get('dataset_format') != 'multi_chart_detection':
            if primary_chart_type == 'area':
                clsmap_obj = GENERATION_CONFIG['CLASS_MAP_AREA_OBJ']
                annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
                area_seg_anns = extract_area_segmentation_annotations(fig, chart_info_map, init_w, init_h)
            elif primary_chart_type == 'pie':
                clsmap_obj = GENERATION_CONFIG['CLASS_MAP_PIE_OBJ']
                annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
                clsmap_pose = GENERATION_CONFIG['CLASS_MAP_PIE_POSE']
                clsmap_pose_reverse = {v: k for k, v in clsmap_pose.items()}
                keypoint_annotations = extract_pie_pose_annotations(
                    fig, chart_info_map, clsmap_pose_reverse, init_w, init_h
                )
            elif primary_chart_type == 'line':
                clsmap_obj = GENERATION_CONFIG['CLASS_MAP_LINE_OBJ']
                annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
                line_seg_anns, line_marker_anns = extract_line_segmentation_annotations(
                    fig, chart_info_map, init_w, init_h
                )
            elif primary_chart_type in ['bar', 'histogram', 'scatter']:
                cls_map_specific = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])
                annotations_obj = get_granular_annotations(fig, chart_info_map, cls_map_specific)

        from detailed import prepare_detail_annotations, build_detailed_json
        cls_map_pre = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])
        obj_map = clsmap_obj if primary_chart_type in ('area', 'pie', 'line') else cls_map_pre
        pose_map = clsmap_pose if primary_chart_type == 'pie' else None
        streams_pre = [
            ("annotations", annotations, cls_map_pre, "mat"),
            ("annotations_obj", annotations_obj, obj_map, "mat"),
            ("area_seg", area_seg_anns, None, "mat"),
            ("pose", keypoint_annotations, pose_map, "pose"),
            ("line_seg", line_seg_anns, None, "mat"),
            ("line_marker", line_marker_anns, None, "mat"),
        ]
        graph_pseudo_anns = prepare_detail_annotations(fig, chart_info_map, streams_pre, init_w, init_h)

        effects_to_apply = dict(cfg.get('realism_effects', {}))
        if cfg.get('dataset_format') == 'multi_chart_detection':
            pdf_noise = cfg.get('multi_chart_detection', {}).get('pdf_context_noise', {})
            if pdf_noise:
                effects_to_apply['pdf_document_context'] = pdf_noise

        extra_sets = [annotations_obj, area_seg_anns, keypoint_annotations, line_seg_anns, line_marker_anns, graph_pseudo_anns]
        pil_img, annotations = apply_realism_effects(
            pil_img, annotations, effects_to_apply, extra_annotation_sets=extra_sets
        )

        if cfg['debug_mode']:
            print(f"DEBUG: After realism effects - annotations: {len(annotations)}")

        # Filter annotations by size and aspect ratio
        MIN_BBOX_SIZE = 8
        MAX_ASPECT_RATIO = 20.0

        valid_annotations = []
        for ann in annotations:
            bbox = ann['bbox']
            width = bbox.x1 - bbox.x0
            height = bbox.y1 - bbox.y0

            ann_chart_type = ann.get('chart_type', primary_chart_type)
            # Exempt scatter, box, and heatmap components from aspect ratio pruning
            if ann_chart_type in ['scatter', 'box', 'heatmap']:
                if width == 0 or height == 0:
                    continue
                valid_annotations.append(ann)
            else:
                if width >= MIN_BBOX_SIZE and height >= MIN_BBOX_SIZE :
                    if width > 0 and height > 0:
                        aspect_ratio = max(width / height, height / width)
                        if aspect_ratio > MAX_ASPECT_RATIO:
                            print(f"    - Discarded annotation (extreme aspect ratio): {aspect_ratio:.1f}")
                            continue
                        valid_annotations.append(ann)
                else:
                    print(f"    - Discarded annotation (too small): class {ann['class_id']}, size {width:.1f}x{height:.1f}")

        annotations = valid_annotations

        # Viewport clipping: clip bounding boxes and polygon boundaries to [0, img_w] x [0, img_h]
        img_w, img_h = pil_img.size

        debug_mode = cfg.get('debug_mode', False)
        annotations = clip_bbox_to_viewport(annotations, img_w, img_h, debug_mode)
        annotations_obj = clip_bbox_to_viewport(annotations_obj, img_w, img_h, debug_mode)
        line_marker_anns = clip_bbox_to_viewport(line_marker_anns, img_w, img_h, debug_mode)
        area_seg_anns = clip_polygon_to_viewport(area_seg_anns, img_w, img_h, debug_mode)
        line_seg_anns = clip_polygon_to_viewport(line_seg_anns, img_w, img_h, debug_mode)

        if cfg['debug_mode']:
            print(f"DEBUG: After size/aspect/bounds filtering - annotations: {len(annotations)}")

        annotations = filter_overlapping_annotations(annotations, iou_threshold=0.7)

        if cfg['debug_mode']:
            print(f"DEBUG: After overlap filtering - annotations: {len(annotations)}")

        # Save files
        base_filename = f"chart_{i:05d}"

        ensure_dir(images_dir)
        ensure_dir(labels_dir)
        pil_img.save(os.path.join(images_dir, f"{base_filename}.png"))
        save_annotations_yolo(annotations, img_w, img_h,
                             os.path.join(labels_dir, f"{base_filename}.txt"))

        # Additive YOLOv8-OBB Label Exporter (Phase 5)
        if cfg.get('export_obb', False) or cfg.get('export_labels_obb', False):
            labels_obb_dir = os.path.join(output_dir, 'labels_obb')
            ensure_dir(labels_obb_dir)
            save_annotations_yolo_obb(
                annotations, img_w, img_h,
                os.path.join(labels_obb_dir, f"{base_filename}.txt")
            )

        if cfg.get('dataset_format') == 'multi_chart_detection':
            iter_time = time.time() - iter_start
            print(f"    ✓ Image {i+1}/{cfg['num_images']} complete in {iter_time:.2f}s | Saved {len(annotations)} annotations")
            return

        # =========================================================================
        # AREA CHARTS: Object Detection & Instance Segmentation Masks
        # =========================================================================
        if primary_chart_type == 'area':
            area_obj_dir = os.path.join(output_dir, 'area_obj_labels')
            ensure_dir(area_obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h,
                                os.path.join(area_obj_dir, f"{base_filename}.txt"))

            area_seg_dir = os.path.join(output_dir, 'area_seg_labels')
            ensure_dir(area_seg_dir)
            save_annotations_yolo_seg(area_seg_anns, img_w, img_h,
                                os.path.join(area_seg_dir, f"{base_filename}.txt"))

            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} area object annotations")
                print(f"DEBUG: Saved {len(area_seg_anns)} area segmentation polygons")

        # =========================================================================
        # PIE CHARTS: Object Detection & 5-Point Wedge Pose Keypoints
        # =========================================================================
        elif primary_chart_type == 'pie':
            pie_obj_dir = os.path.join(output_dir, 'pie_obj_labels')
            ensure_dir(pie_obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h,
                                os.path.join(pie_obj_dir, f"{base_filename}.txt"))

            pie_pose_dir = os.path.join(output_dir, 'pie_pose_labels')
            ensure_dir(pie_pose_dir)
            save_annotations_pose(keypoint_annotations, img_w, img_h,
                                os.path.join(pie_pose_dir, f"{base_filename}.txt"))

            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} pie object annotations")
                print(f"DEBUG: Saved {len(keypoint_annotations)} pie pose annotations")

        # =========================================================================
        # LINE CHARTS: Dual-Stream Instance Segmentation & Marker/Extrema Detection
        # =========================================================================
        elif primary_chart_type == 'line':
            line_obj_dir = os.path.join(output_dir, 'line_obj_labels')
            ensure_dir(line_obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h,
                                os.path.join(line_obj_dir, f"{base_filename}.txt"))

            line_seg_dir = os.path.join(output_dir, 'line_seg_labels')
            ensure_dir(line_seg_dir)
            save_annotations_yolo_seg(line_seg_anns, img_w, img_h,
                                os.path.join(line_seg_dir, f"{base_filename}.txt"))

            line_marker_dir = os.path.join(output_dir, 'line_marker_labels')
            ensure_dir(line_marker_dir)
            save_annotations_yolo(line_marker_anns, img_w, img_h,
                                os.path.join(line_marker_dir, f"{base_filename}.txt"))

            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} line object annotations")
                print(f"DEBUG: Saved {len(line_seg_anns)} line segmentation polygons")
                print(f"DEBUG: Saved {len(line_marker_anns)} line marker/extrema annotations")

        # =========================================================================
        # STANDARD CHARTS: Bar, Histogram, Scatter
        # =========================================================================
        elif primary_chart_type in ['bar', 'histogram', 'scatter']:
            obj_dir = os.path.join(output_dir, f"{primary_chart_type}_obj_labels")
            ensure_dir(obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h,
                                os.path.join(obj_dir, f"{base_filename}.txt"))

            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} {primary_chart_type} object annotations to {obj_dir}")

        # =========================================================================
        # BOX PLOT: Dual-Expert Annotation Routing (Elements vs Global Layout)
        # =========================================================================
        # Dispatches the unified master annotations into two dedicated, separately-
        # indexed annotation sets for training two specialist models:
        #   Set 1 (Elements): box, range_indicator, median_line, outlier,
        #                     significance_marker -> Model B / Structural Specialist
        #   Set 2 (Global):   chart, axis_title, legend, chart_title,
        #                     axis_labels         -> Model A / Global Layout
        elif primary_chart_type == 'box':
            cls_map_specific = CHART_CLASS_MAPS['box']

            elements_expert_map = {
                "box": 0,
                "range_indicator": 1,
                "median_line": 2,
                "outlier": 3,
                "significance_marker": 4
            }
            global_expert_map = {
                "chart": 0,
                "axis_title": 1,
                "legend": 2,
                "chart_title": 3,
                "axis_labels": 4
            }

            anns_elements = []
            anns_global = []

            for ann in annotations:
                orig_class_id = str(ann['class_id'])
                class_name = cls_map_specific.get(orig_class_id)

                if class_name in elements_expert_map:
                    ann_copy = ann.copy()
                    ann_copy['class_id'] = elements_expert_map[class_name]
                    anns_elements.append(ann_copy)

                if class_name in global_expert_map:
                    ann_copy = ann.copy()
                    ann_copy['class_id'] = global_expert_map[class_name]
                    anns_global.append(ann_copy)

            # Set 1: Data & Statistical Elements (box_elements_labels + box_obj_labels)
            elements_dir = os.path.join(output_dir, 'box_elements_labels')
            ensure_dir(elements_dir)
            save_annotations_yolo(anns_elements, img_w, img_h,
                                os.path.join(elements_dir, f"{base_filename}.txt"))

            obj_dir = os.path.join(output_dir, 'box_obj_labels')
            ensure_dir(obj_dir)
            save_annotations_yolo(anns_elements, img_w, img_h,
                                os.path.join(obj_dir, f"{base_filename}.txt"))

            # Set 2: Global & Layout Elements (box_global_labels)
            global_dir = os.path.join(output_dir, 'box_global_labels')
            ensure_dir(global_dir)
            save_annotations_yolo(anns_global, img_w, img_h,
                                os.path.join(global_dir, f"{base_filename}.txt"))

            if cfg.get('debug_mode', False):
                print(f"DEBUG: Saved {len(anns_elements)} box element annotations to {elements_dir} (+ {obj_dir})")
                print(f"DEBUG: Saved {len(anns_global)} box global annotations to {global_dir}")

        # =========================================================================
        # HEATMAP: Cascaded Expert Double-Routing Pipeline with Regional Crops
        # =========================================================================
        # Dispatches heatmap annotations across four specialist domains:
        #   - Expert 1 (Macro Layout Router, Global): chart, color_bar_region, legend
        #   - Expert 2 (Color Bar Specialist, Crop):  color_bar, color_bar_label, color_bar_title
        #   - Expert 3 (Cell Lattice Specialist, Crop): cell, data_label
        #   - Expert 4 (Text Line Parser, Global):    axis_labels, axis_title, chart_title
        elif primary_chart_type == 'heatmap':
            cls_map_specific = CHART_CLASS_MAPS['heatmap']

            # 1. Dynamically calculate the unified "color_bar_region" from post-processed elements
            colorbar_bboxes = []
            chart_bbox_global = None

            for ann in annotations:
                orig_class_id = str(ann['class_id'])
                class_name = cls_map_specific.get(orig_class_id)
                if class_name in ["color_bar", "color_bar_label", "color_bar_title"]:
                    colorbar_bboxes.append(ann['bbox'])
                if class_name == "chart":
                    chart_bbox_global = ann['bbox']

            colorbar_region_bbox_global = None
            # Create a virtual unified region if any color bar component is present
            if colorbar_bboxes:
                from matplotlib.transforms import Bbox
                union_bbox = Bbox.union(colorbar_bboxes)

                # Apply cushioned padding to the layout router box to avoid clipping text labels
                padding_pixels = 12
                padded_bbox = Bbox.from_extents(
                    max(0.0, union_bbox.x0 - padding_pixels),
                    max(0.0, union_bbox.y0 - padding_pixels),
                    min(img_w, union_bbox.x1 + padding_pixels),
                    min(img_h, union_bbox.y1 + padding_pixels)
                )
                colorbar_region_bbox_global = padded_bbox
                annotations.append({
                    'class_id': 'color_bar_region',
                    'bbox': padded_bbox
                })

            # 2. Define isolated re-indexed expert lookup tables
            expert1_map = {"chart": 0, "color_bar_region": 1, "legend": 2}
            expert2_map = {"color_bar": 0, "color_bar_label": 1, "color_bar_title": 2}
            expert3_map = {"cell": 0, "data_label": 1}
            expert4_map = {"axis_labels": 0, "axis_title": 1, "chart_title": 2}

            # Initialize label queues for each specialist domain
            anns_expert1 = []
            anns_expert2_raw = []
            anns_expert3_raw = []
            anns_expert4 = []

            # 3. Sort global annotations into respective expert queues
            for ann in annotations:
                if ann['class_id'] == 'color_bar_region':
                    class_name = 'color_bar_region'
                else:
                    orig_class_id = str(ann['class_id'])
                    class_name = cls_map_specific.get(orig_class_id)

                # Expert 1 (Macro Layout Router) - stays in global coordinate space
                if class_name in expert1_map:
                    ann_copy = ann.copy()
                    ann_copy['class_id'] = expert1_map[class_name]
                    anns_expert1.append(ann_copy)

                # Expert 2 (Color Bar Specialist Domain) - gathered for cropping
                if class_name in expert2_map:
                    ann_copy = ann.copy()
                    ann_copy['class_id'] = expert2_map[class_name]
                    anns_expert2_raw.append(ann_copy)

                # Assign to Expert 3 (Cell Lattice Specialist) - gathered for cropping
                if class_name in expert3_map:
                    ann_copy = ann.copy()
                    ann_copy['class_id'] = expert3_map[class_name]
                    anns_expert3_raw.append(ann_copy)

                # Assign to Expert 4 (General Text Line Parser) - stays in global space
                if class_name in expert4_map:
                    ann_copy = ann.copy()
                    ann_copy['class_id'] = expert4_map[class_name]
                    anns_expert4.append(ann_copy)

            # --- SUB-IMAGE CROP AND RE-PROJECTION NESTED ENGINE ---
            def process_expert_crop(global_bbox, raw_expert_anns, folder_img_suffix, folder_lbl_suffix):
                if global_bbox is None:
                    return

                # Convert global matplotlib coords to global image top-left coordinates
                g_x0 = global_bbox.x0
                g_x1 = global_bbox.x1
                g_y0 = img_h - global_bbox.y1
                g_y1 = img_h - global_bbox.y0

                w = g_x1 - g_x0
                h = g_y1 - g_y0
                if w <= 0 or h <= 0:
                    return

                # Generate random padding between 0% and 10% of width and height
                pad_pct_w = random.uniform(0.0, 0.10)
                pad_pct_h = random.uniform(0.0, 0.10)
                pad_w = w * pad_pct_w
                pad_h = h * pad_pct_h

                # Expand crop boundaries and clamp safely inside viewport dimensions
                crop_x0 = max(0.0, g_x0 - pad_w)
                crop_y0 = max(0.0, g_y0 - pad_h)
                crop_x1 = min(float(img_w), g_x1 + pad_w)
                crop_y1 = min(float(img_h), g_y1 + pad_h)

                sub_w = crop_x1 - crop_x0
                sub_h = crop_y1 - crop_y0
                if sub_w <= 0 or sub_h <= 0:
                    return

                # Perform physical image crop and commit to storage
                cropped_img = pil_img.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
                img_dir = os.path.join(output_dir, folder_img_suffix)
                ensure_dir(img_dir)
                cropped_img.save(os.path.join(img_dir, f"{base_filename}.png"))

                # Re-project bounding boxes into the new sub-image viewport space
                transformed_anns = []
                for ann in raw_expert_anns:
                    bbox = ann['bbox']
                    # Get global top-left coordinates of the target item
                    element_g_x0 = bbox.x0
                    element_g_x1 = bbox.x1
                    element_g_y0 = img_h - bbox.y1
                    element_g_y1 = img_h - bbox.y0

                    # Compute relative offset coords inside the new crop space
                    l_x0 = max(0.0, min(element_g_x0 - crop_x0, sub_w))
                    l_x1 = max(l_x0, min(element_g_x1 - crop_x0, sub_w))
                    l_y0 = max(0.0, min(element_g_y0 - crop_y0, sub_h))
                    l_y1 = max(l_y0, min(element_g_y1 - crop_y0, sub_h))

                    # Convert back to Matplotlib Bbox format relative to sub_h canvas height
                    sub_mat_y0 = sub_h - l_y1
                    sub_mat_y1 = sub_h - l_y0

                    ann_copy = ann.copy()
                    from matplotlib.transforms import Bbox
                    ann_copy['bbox'] = Bbox.from_extents(l_x0, sub_mat_y0, l_x1, sub_mat_y1)
                    transformed_anns.append(ann_copy)

                # Write YOLO text files relative to the sub-image canvas coordinates
                lbl_dir = os.path.join(output_dir, folder_lbl_suffix)
                ensure_dir(lbl_dir)
                save_annotations_yolo(
                    transformed_anns,
                    sub_w,
                    sub_h,
                    os.path.join(lbl_dir, f"{base_filename}.txt")
                )

            # 4. Process and export standalone cropped images + synchronized labels
            process_expert_crop(chart_bbox_global, anns_expert3_raw, "heatmap_lattice_images", "heatmap_lattice_labels")
            process_expert_crop(colorbar_region_bbox_global, anns_expert2_raw, "heatmap_colorbar_images", "heatmap_colorbar_labels")

            # 5. Export standard global-space label configurations
            global_bundles = [
                ("heatmap_macro_labels", anns_expert1),
                ("heatmap_text_labels", anns_expert4)
            ]
            for folder_suffix, expert_annotations in global_bundles:
                expert_dir = os.path.join(output_dir, folder_suffix)
                ensure_dir(expert_dir)
                save_annotations_yolo(
                    expert_annotations,
                    img_w,
                    img_h,
                    os.path.join(expert_dir, f"{base_filename}.txt")
                )

            if cfg.get('debug_mode', False):
                print(f"DEBUG [CASCADED-HEATMAP] Dispatched annotations: Macro={len(anns_expert1)}, ColorBar={len(anns_expert2_raw)}, Lattice={len(anns_expert3_raw)}, Text={len(anns_expert4)}")
        # Determine the appropriate class map for unified JSON based on chart type
        cls_map = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])

        streams_post = [
            ("annotations", annotations, cls_map, "mat"),
            ("annotations_obj", annotations_obj, obj_map, "mat"),
            ("area_seg", area_seg_anns, None, "mat"),
            ("pose", keypoint_annotations, pose_map, "pose"),
            ("line_seg", line_seg_anns, None, "mat"),
            ("line_marker", line_marker_anns, None, "mat"),
            ("graph", graph_pseudo_anns, None, "graph"),
        ]

        if _use_schema_v4(cfg):
            detailed_json = build_detailed_json(
                fig, chart_info_map, streams_post, img_w, img_h,
                schema_version=ANNOTATION_SCHEMA_VERSION_V4, dataset_version=DATASET_VERSION_V4,
                image_id=base_filename,
            )
        else:
            _warn_v3_deprecated()
            # Get comprehensive unified JSON with complete metadata
            unified_json = create_unified_annotation(fig, chart_info_map, cls_map, img_w, img_h, annotations)

            def _pick_list(*keys):
                for key in keys:
                    value = unified_json.get(key)
                    if isinstance(value, list) and value:
                        return value
                for key in keys:
                    value = unified_json.get(key)
                    if isinstance(value, list):
                        return value
                return []

            # 1. Create detailed JSON with element annotations and all metadata
            detailed_json = {
                "chart_type": unified_json.get("chart_type"),
                "orientation": unified_json.get("orientation"),

                # Canonical keys
                "scale_labels": _pick_list("scale_labels", "scalelabels"),
                "tick_labels": _pick_list("tick_labels", "ticklabels"),
                "chart_title": _pick_list("chart_title", "charttitle"),
                "axis_title": _pick_list("axis_title", "axistitle"),
                "legend": _pick_list("legend"),
                "bar": _pick_list("bar"),
                "data_point": _pick_list("data_point", "datapoint"),
                "error_bar": _pick_list("error_bar", "errorbar"),
                "significance_marker": _pick_list("significance_marker", "significancemarker"),
                "data_label": _pick_list("data_label", "datalabel"),
                "box": _pick_list("box"),
                "median_line": _pick_list("median_line", "medianline"),
                "range_indicator": _pick_list("range_indicator", "rangeindicator"),
                "outlier": _pick_list("outlier"),
                "wedge": _pick_list("wedge"),
                "line_segment": _pick_list("line_segment", "linesegment"),
                "area_boundary": _pick_list("area_boundary", "areaboundary"),
                "cell": _pick_list("cell"),
                "color_bar": _pick_list("color_bar", "colorbar"),
                "color_bar_label": _pick_list("color_bar_label"),
                "color_bar_title": _pick_list("color_bar_title"),
                "connector_line": _pick_list("connector_line", "connectorline"),

                # Legacy aliases kept for backward compatibility
                "scalelabels": _pick_list("scale_labels", "scalelabels"),
                "ticklabels": _pick_list("tick_labels", "ticklabels"),
                "charttitle": _pick_list("chart_title", "charttitle"),
                "axistitle": _pick_list("axis_title", "axistitle"),
                "datapoint": _pick_list("data_point", "datapoint"),
                "errorbar": _pick_list("error_bar", "errorbar"),
                "significancemarker": _pick_list("significance_marker", "significancemarker"),
                "datalabel": _pick_list("data_label", "datalabel"),
                "medianline": _pick_list("median_line", "medianline"),
                "rangeindicator": _pick_list("range_indicator", "rangeindicator"),
                "linesegment": _pick_list("line_segment", "linesegment"),
                "areaboundary": _pick_list("area_boundary", "areaboundary"),
                "colorbar": _pick_list("color_bar", "colorbar"),
                "connectorline": _pick_list("connector_line", "connectorline"),

                # Include detailed chart metadata
                "scale_axis_info": unified_json.get("scale_axis_info", {}),
                "bar_info": unified_json.get("bar_info", []),
                "keypoint_info": unified_json.get("keypoint_info", []),
                "boxplot_metadata": unified_json.get("boxplot_metadata", {}),
                "pie_geometry": unified_json.get("pie_geometry", {}),
                "pie_metadata": unified_json.get("pie_metadata", {}),
                "histogram_metadata": unified_json.get("histogram_metadata", {}),
                "series_count": unified_json.get("series_count", 1),
                "series_names": unified_json.get("series_names", []),
                "stacking_mode": unified_json.get("stacking_mode"),
                "dual_axis_info": unified_json.get("dual_axis_info", {}),
                "style": unified_json.get("style"),
                "pattern": unified_json.get("pattern"),
                "is_scientific": unified_json.get("is_scientific", False),
                "semantic_domain": unified_json.get("semantic_domain"),
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "subplots": unified_json.get("subplots", []),
                "is_composite": unified_json.get("is_composite", False),
                "composite_chart_types": unified_json.get("composite_chart_types", []),
                "composite_domains": unified_json.get("composite_domains", []),
                "raw_annotations": unified_json.get("raw_annotations", []),

                # GNN TRAINING FIELDS - bar-to-baseline graph topology
                "baselines": unified_json.get("baselines", []),
                "bars_with_baseline": unified_json.get("bars_with_baseline", []),
                "baseline_keypoints": unified_json.get("baseline_keypoints", [])
            }

        # 2. Create OCR JSON with OCR annotations
        # For now, create an empty structure that will be populated by OCR processing
        ocr_json = {
            "ocr_annotations": [],  # This would normally come from OCR processing
            "effects_applied": []   # This might be added during image processing
        }

        # 3. Create basic metadata JSON
        metadata_json = {
            "image_id": base_filename,
            "resolution": [int(img_w), int(img_h)],
            "chart_types": detailed_json.get("composite_chart_types") or [detailed_json.get("chart_type", "unknown")],
            "themes": {},  # Will be populated based on the chart theme
            "num_annotations": len(detailed_json.get("annotations", annotations)),
            "schema_version": detailed_json.get("schema_version", ANNOTATION_SCHEMA_VERSION),
            "dataset_version": detailed_json.get("dataset_version", DATASET_VERSION),
        }

        # Save detailed.json
        with open(os.path.join(labels_dir, f"{base_filename}_detailed.json"), 'w') as f:
            json.dump(detailed_json, f, indent=2, default=json_default_fallback)

        # Auxiliary files (_ocr.json and .json) are skipped unless explicitly requested
        if cfg.get('export_legacy_json', False):
            with open(os.path.join(labels_dir, f"{base_filename}_ocr.json"), 'w') as f:
                json.dump(ocr_json, f, indent=2, default=json_default_fallback)

            with open(os.path.join(labels_dir, f"{base_filename}.json"), 'w') as f:
                json.dump(metadata_json, f, indent=2, default=json_default_fallback)

        iter_time = time.time() - iter_start
        print(f"    ✓ Image {i+1}/{cfg['num_images']} complete in {iter_time:.2f}s | Saved {len(annotations)} annotations")

    finally:
        plt.close(fig)


def generate_single_chart_task(args):
    i, cfg, images_dir, labels_dir, output_dir = args
    # Re-seed to prevent process replication duplication
    base_seed = cfg.get('seed', 42)
    random.seed(base_seed + i)
    np.random.seed(base_seed + i)

    print(f"--- Generating image {i+1}/{cfg['num_images']} (PID: {os.getpid()}) ---")
    try:
        generate_single_chart(i, cfg, images_dir, labels_dir, output_dir)
        return (i, True, None)
    except Exception as e:
        err_msg = f"Process PID {os.getpid()} failed on image {i}: {e}"
        print(f"[ERROR] {err_msg}")
        traceback.print_exc()
        return (i, False, str(e))
    finally:
        plt.close('all')


def _compute_manifest_fingerprint() -> str:
    from synth.semantics.loader import DOMAINS_DIR
    items = []
    for yf in sorted(DOMAINS_DIR.glob("*.yaml")):
        st = yf.stat()
        items.append(f"{yf.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256(";".join(items).encode("utf-8")).hexdigest()


def _compute_dataset_hash(labels_dir: str) -> str:
    if not os.path.exists(labels_dir):
        return ""
    file_hashes = {}
    for fname in sorted(os.listdir(labels_dir)):
        if fname.endswith(".txt") or fname.endswith("_detailed.json"):
            path = os.path.join(labels_dir, fname)
            with open(path, "rb") as f:
                file_hashes[fname] = hashlib.sha256(f.read()).hexdigest()
    return hashlib.sha256(";".join(f"{k}={v}" for k, v in sorted(file_hashes.items())).encode("utf-8")).hexdigest()


def _get_git_revision() -> Optional[str]:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


@dataclass
class RunSummary:
    num_images: int
    successful_count: int
    failed_count: int
    failed_indices: List[int]
    elapsed_seconds: float
    output_dir: str
    manifest_path: str = ""
    class_counts: Dict[Any, int] = field(default_factory=dict)
    interrupted: bool = False


def run_generation(cfg, progress=None) -> RunSummary:
    """Execute chart dataset generation for the given resolved configuration."""
    set_active_config(cfg)

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    if cfg.get('debug_mode', False):
        print("--- DEBUG MODE ENABLED ---")
        debug_dir = 'test'
        print(f"Output will be saved to '{debug_dir}/'")
        images_dir = debug_dir
        labels_dir = debug_dir
        output_dir = debug_dir
        ensure_dir(debug_dir)
    else:
        output_dir = cfg['output_dir']
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        ensure_dir(images_dir)
        ensure_dir(labels_dir)

    if cfg.get('debug_mode', False):
        print(f"DEBUG: Available chart types: {list(cfg['chart_types'].keys())}")
        print(f"DEBUG: Enabled chart types: {[k for k, v in cfg['chart_types'].items() if v['enabled']]}")
        print(f"DEBUG: Scenario weights: {cfg['scenario_weights']}")
        print(f"DEBUG: Number of images to generate: {cfg['num_images']}")

    start_time = time.time()
    num_images = cfg['num_images']
    start_idx = cfg.get('start_index', 0)
    progress_jsonl_path = cfg.get('progress_jsonl')

    def _on_image_done(res_tuple, task_secs=0.0):
        if progress is not None:
            progress(res_tuple)
        if progress_jsonl_path:
            try:
                with open(progress_jsonl_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps({
                        "event": "image_done",
                        "index": res_tuple[0],
                        "ok": res_tuple[1],
                        "secs": round(task_secs, 4),
                        "error": res_tuple[2]
                    }) + "\n")
            except Exception:
                pass

    results = []
    interrupted = False
    use_parallel = cfg.get('use_parallel', True)
    try:
        if use_parallel and num_images > 1:
            worker_limit = cfg.get('workers') or cfg.get('max_workers')
            if worker_limit is not None:
                num_cores = max(1, min(int(worker_limit), num_images))
            else:
                num_cores = min(os.cpu_count() or 1, num_images, 16)
            print(f"Launching batch engine parallelized across {num_cores} process cores...")
            tasks = [(start_idx + i, cfg, images_dir, labels_dir, output_dir) for i in range(num_images)]
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            mp_context = mp.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=num_cores, mp_context=mp_context, initializer=_init_worker, initargs=(cfg,))
            try:
                task_start_times = {i: time.time() for i in range(num_images)}
                futures = {executor.submit(generate_single_chart_task, task): task[0] for task in tasks}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    t_spent = time.time() - task_start_times.get(idx, start_time)
                    res = fut.result()
                    results.append(res)
                    _on_image_done(res, t_spent)
                results.sort(key=lambda x: x[0])
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        else:
            _init_worker(cfg)
            for i in range(num_images):
                t0 = time.time()
                res = generate_single_chart_task((start_idx + i, cfg, images_dir, labels_dir, output_dir))
                t_spent = time.time() - t0
                results.append(res)
                _on_image_done(res, t_spent)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[WARNING] Generation interrupted by user (Ctrl-C). Preserving partial output...")

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    failed_indices = [r[0] for r in failed]

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.2f}s.")
    print(f"SUMMARY: {len(successful)}/{num_images} images generated successfully ({len(failed)} failed).")
    if failed:
        print(f"[WARNING] Failed image indices: {failed_indices}")

    print("\n=== DATASET STATISTICS ===")
    class_counts = defaultdict(int)
    if cfg.get('dataset_format') == 'multi_chart_detection':
        if os.path.exists(labels_dir):
            for fname in os.listdir(labels_dir):
                if fname.endswith('.txt'):
                    with open(os.path.join(labels_dir, fname), 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                class_counts[class_id] += 1
        cls_map = cfg.get('CLASS_MAP_CLASSIFICATION', GENERATION_CONFIG.get('CLASS_MAP_CLASSIFICATION', {}))
        for class_id_str, class_name in sorted(cls_map.items(), key=lambda x: int(x[0])):
            cid = int(class_id_str)
            print(f"  {class_name:20s}: {class_counts[cid]:5d} instances")
    else:
        for i in range(start_idx, start_idx + num_images):
            label_file = os.path.join(labels_dir, f"chart_{i:05d}.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1

        combined_cls_map = {}
        for chart_type, chart_cls_map in CHART_CLASS_MAPS.items():
            if chart_type != 'pie':
                for id_val, class_name in chart_cls_map.items():
                    combined_cls_map[int(id_val)] = class_name

        for class_id, class_name in sorted(combined_cls_map.items(), key=lambda x: x[1]):
            print(f"  {class_name:20s}: {class_counts[class_id]:5d} instances")

    if cfg.get('dataset_format') == 'multi_chart_detection':
        yaml_path = os.path.join(output_dir, 'data.yaml')
        cls_map = cfg.get('CLASS_MAP_CLASSIFICATION', GENERATION_CONFIG.get('CLASS_MAP_CLASSIFICATION', {}))
        sorted_names = [cls_map[str(k)] for k in sorted(map(int, cls_map.keys()))]
        abs_output_dir = os.path.abspath(output_dir)
        yaml_content = f"path: {abs_output_dir}\ntrain: images\nval: images\nnc: {len(sorted_names)}\nnames:\n"
        for name in sorted_names:
            yaml_content += f"  - {name}\n"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"\nWritten Ultralytics dataset config to {yaml_path}")

    # Write run_manifest.json (GEN-04)
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    serializable_cfg = {}
    for k, v in cfg.items():
        try:
            json.dumps(v)
            serializable_cfg[k] = v
        except Exception:
            serializable_cfg[k] = str(v)

    ephemeral_keys = {"output_dir", "progress_jsonl", "log_file", "show_debug"}
    hash_cfg = {k: v for k, v in serializable_cfg.items() if k not in ephemeral_keys}
    cfg_hash = hashlib.sha256(json.dumps(hash_cfg, sort_keys=True).encode("utf-8")).hexdigest()
    dataset_hash = _compute_dataset_hash(labels_dir)
    manifest_data = {
        "schema_version": cfg.get("annotation_schema_version", "v4.0"),
        "dataset_version": cfg.get("dataset_version", DATASET_VERSION_V4),
        "seed": cfg.get("seed", 42),
        "config_hash": cfg_hash,
        "dataset_content_hash": dataset_hash,
        "manifest_registry_fingerprint": _compute_manifest_fingerprint(),
        "git_revision": _get_git_revision(),
        "wall_time_seconds": round(elapsed, 4),
        "counts": {
            "num_images": num_images,
            "successful": len(successful),
            "failed": len(failed)
        },
        "failed_indices": failed_indices,
        "resolved_config": serializable_cfg,
    }
    if interrupted:
        manifest_data["interrupted"] = True

    try:
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not write run_manifest.json: {e}")

    if cfg.get('debug_mode', False):
        if os.path.isfile("testar.py") and cfg.get('show_debug', False):
            print("\n--- Generation complete. Running visualization script... ---")
            try:
                subprocess.run([sys.executable, "testar.py", debug_dir, "--show"], check=True)
            except Exception as e:
                print(f"\n[ERROR] Visualization script failed: {e}")

    return RunSummary(
        num_images=num_images,
        successful_count=len(successful),
        failed_count=len(failed),
        failed_indices=failed_indices,
        elapsed_seconds=elapsed,
        output_dir=output_dir,
        manifest_path=manifest_path,
        class_counts=dict(class_counts),
        interrupted=interrupted,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic chart training data')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (.py or .json)')
    parser.add_argument('--no-custom', action='store_true', help='Skip loading custom_config.py')
    parser.add_argument('--num', type=int, default=None, help='Number of images to generate')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory')
    parser.add_argument('--mode', '--format', type=str, default=None, choices=['classification', 'detection', 'multi_chart_detection'], help='Dataset format mode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for deterministic generation')
    parser.add_argument('--engine', type=str, default=None, choices=['matplotlib', 'vegalite'], help='Rendering engine')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes for generation')
    parser.add_argument('--strict', action='store_true', help='Exit with non-zero status code if any image generation fails')
    parser.add_argument('--start-index', type=int, default=None, help='Starting index for generated image numbers')
    parser.add_argument('--set', action='append', dest='set_overrides', default=[], help='Override config setting via key.path=value (repeatable)')
    parser.add_argument('--progress-jsonl', type=str, default=None, help='Path to write JSONL progress events')
    parser.add_argument('--log-file', type=str, default=None, help='Path to redirect stdout/stderr logs')
    parser.add_argument('--validate-only', action='store_true', help='Validate resolved configuration and exit')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite non-empty output directory')
    parser.add_argument('--show-debug', action='store_true', help='Launch visualization script testar.py if in debug mode')
    args = parser.parse_args()

    cli_overrides = {}
    if args.num is not None:
        cli_overrides['num_images'] = args.num
    if args.output is not None:
        cli_overrides['output_dir'] = args.output
    if args.mode is not None:
        cli_overrides['dataset_format'] = args.mode
    if args.seed is not None:
        cli_overrides['seed'] = args.seed
    if args.engine is not None:
        cli_overrides['engine'] = args.engine
    if args.workers is not None:
        cli_overrides['workers'] = args.workers
        cli_overrides['max_workers'] = args.workers
    if args.strict:
        cli_overrides['strict'] = True
    if args.start_index is not None:
        cli_overrides['start_index'] = args.start_index
    if args.show_debug:
        cli_overrides['show_debug'] = True
    if args.progress_jsonl:
        cli_overrides['progress_jsonl'] = args.progress_jsonl

    for set_expr in args.set_overrides:
        k, v = parse_set_override(set_expr)
        cli_overrides[k] = v

    cfg = load_config(path=args.config, overrides=cli_overrides, use_custom=not args.no_custom)

    # GEN-02: validate-only check
    if args.validate_only:
        from synth.semantics.admin.configspec import validate
        issues = validate(cfg)
        errors = [i for i in issues if i.level == 'error']
        if errors:
            for err in errors:
                sys.stderr.write(f"[CONFIG ERROR] {err.path}: {err.msg}\n")
            sys.exit(2)
        print("Configuration valid.")
        sys.exit(0)

    # GEN-03: output-dir conflict check
    if not cfg.get('debug_mode', False) and not args.overwrite and args.start_index is None:
        out_dir = cfg.get('output_dir', '')
        lbls = os.path.join(out_dir, 'labels')
        imgs = os.path.join(out_dir, 'images')
        has_existing = False
        for d in (lbls, imgs):
            if os.path.exists(d) and len(os.listdir(d)) > 0:
                has_existing = True
                break
        if has_existing:
            sys.stderr.write(f"[ERROR] Output directory '{out_dir}' already contains generated files. Use --overwrite or --start-index.\n")
            sys.exit(3)

    if args.log_file:
        log_fp = open(args.log_file, "a", encoding="utf-8")
        sys.stdout = log_fp
        sys.stderr = log_fp

    try:
        summary = run_generation(cfg)
    except KeyboardInterrupt:
        sys.exit(130)

    if summary.interrupted:
        sys.exit(130)

    if summary.failed_count > 0 and (args.strict or cfg.get('strict', False)):
        print("\n[ERROR] Strict mode enabled and generation failures occurred. Exiting with status 1.")
        sys.exit(1)


if __name__ == '__main__':
    main()
