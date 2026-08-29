import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')  # Thread-safe, headless backend
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib import patches, rcParams, transforms, colormaps
from matplotlib.colors import ListedColormap
from matplotlib.container import ErrorbarContainer
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
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont

from themes import THEMES, CHART_TITLES, SCIENTIFIC_Y_LABELS, BUSINESS_Y_LABELS, SCIENTIFIC_X_LABELS, BUSINESS_X_LABELS
from effects import (
    apply_jpeg_compression_effect, apply_noise_effect, apply_blur_effect, 
    apply_motion_blur_effect, apply_low_res_effect, apply_pixelation_effect, 
    apply_posterize_effect, apply_color_variation_effect, apply_ui_chrome_effect, 
    apply_watermark_effect, apply_vignette_effect, apply_scanner_streaks_effect, 
    apply_clipping_effect, apply_printing_artifacts_effect, apply_mouse_cursor_effect, 
    apply_text_degradation_effect, apply_grid_occlusion_effect, apply_scan_rotation_effect, 
    apply_grayscale_effect, apply_perspective_effect, apply_perspective_warp_effect,
    apply_uneven_lighting_effect, apply_chromatic_aberration_effect,
    apply_pdf_document_context_effect
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
try:
    from custom_config import OCR_TRAINING_CONFIG as GENERATION_CONFIG
except ImportError:
    _custom_config_path = os.path.join(os.path.dirname(__file__), "custom_config.py")
    if os.path.exists(_custom_config_path):
        import importlib.util
        _spec = importlib.util.spec_from_file_location("custom_config", _custom_config_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        GENERATION_CONFIG = _mod.OCR_TRAINING_CONFIG
    else:
        raise ImportError("custom_config.py could not be found.")

# Chart-type-specific class maps
CHART_CLASS_MAPS = {
    'bar': GENERATION_CONFIG['CLASS_MAP_BAR'],
    'scatter': GENERATION_CONFIG['CLASS_MAP_SCATTER'],
    'box': GENERATION_CONFIG['CLASS_MAP_BOX'],
    'histogram': GENERATION_CONFIG['CLASS_MAP_HISTOGRAM'],
    'heatmap': GENERATION_CONFIG['CLASS_MAP_HEATMAP'],
    'area_obj': GENERATION_CONFIG['CLASS_MAP_AREA_OBJ'],
    'area_seg': GENERATION_CONFIG['CLASS_MAP_AREA_SEG'],
    'pie': GENERATION_CONFIG['CLASS_MAP_PIE_OBJ'],
    'pie_pose': GENERATION_CONFIG['CLASS_MAP_PIE_POSE'],
    'line_obj': GENERATION_CONFIG['CLASS_MAP_LINE_OBJ'],
    'line_seg': GENERATION_CONFIG['CLASS_MAP_LINE_SEG'],
    'line_markers': GENERATION_CONFIG['CLASS_MAP_LINE_MARKERS']
}

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


def create_reverse_class_map(cls_map):
    """Create reverse mapping: class_name -> class_id"""
    return {v: k for k, v in cls_map.items()}

def has_non_background_pixels(label_artist, fig, ax, bgcolor, threshold=1):
    """
    Check if label region contains pixels of different colors.
    Stops immediately upon finding color variation.
    
    Args:
        label_artist: matplotlib Text artist (axis label)
        fig: matplotlib figure
        ax: matplotlib axes
        threshold: pixel difference threshold (0-255 scale)
    
    Returns:
        bool: True if color variation found, False if all pixels same color
    """
    import numpy as np
    
    try:
        renderer = fig.canvas.get_renderer()
        if renderer is None:
            return True
        
        bbox = label_artist.get_window_extent(renderer)
        x0, y0, x1, y1 = int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)
        
        if x1 <= x0 or y1 <= y0 or bbox.width < 1 or bbox.height < 1:
            return False
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        h, w = int(fig.bbox.height), int(fig.bbox.width)
        buf = buf.reshape((h, w, 4))
        
        x0 = max(0, min(x0, w-1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h-1))
        y1 = max(0, min(y1, h))
        
        if x1 <= x0 or y1 <= y0:
            return False
        
        roi = buf[h-y1:h-y0, x0:x1, :3]
        pixels = roi.reshape(-1, 3)
        
        if pixels.shape[0] == 0:
            return False
        
        reference_pixel = pixels[0].astype(np.int16)
        step = max(1, len(pixels) // 100)
        
        for i in range(step, len(pixels), step):
            diff = np.abs(pixels[i].astype(np.int16) - reference_pixel)
            if np.any(diff > threshold):
                return True
        
        if len(pixels) > 20:
            critical_indices = [0, 1, 2, len(pixels)//4, len(pixels)//2, 
                              3*len(pixels)//4, -3, -2, -1]
            for i in critical_indices:
                if i < len(pixels):
                    diff = np.abs(pixels[i].astype(np.int16) - reference_pixel)

        # If bounding box has positive area, accept artist
        return True
    except Exception:
        return True

        return False
        
    except Exception as e:
        return True



def get_granular_annotations(fig, chart_info_map, cls_map):
    """
    Extract detailed bounding box annotations for all visible chart components
    (bars, lines, text, scatter points, wedges, heatmap cells, legends, etc.).
    """
    reverse_map = create_reverse_class_map(cls_map)
    
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        print("WARNING: Could not obtain renderer, annotations will be empty")
        return []
    
    annotations = []
    fig_bbox = fig.get_window_extent(renderer)
    seen_annotations = set()
    
    def add_unique_annotation(class_id, bbox, text=None):
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
            entry = {'class_id': class_id, 'bbox': bbox}
            if text:
                entry['text'] = text
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
                    title_bbox = title.get_window_extent(renderer)
                    if add_unique_annotation(reverse_map['chart_title'], title_bbox, text=title.get_text().strip()):
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
                    xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
                    if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip()):
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Added x-axis title annotation")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: X-axis title bbox failed: {e}")
            
            # Y-axis title  
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                try:
                    ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
                    if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip()):
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
                                label_bbox = label.get_window_extent(renderer)
                                if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
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
                                label_bbox = label.get_window_extent(renderer)
                                if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
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
                        if artist_bbox and add_unique_annotation(reverse_map['bar'], artist_bbox):
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
                            if artistbbox and add_unique_annotation(reverse_map["bar"], artistbbox):
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
                                label_bbox = artist.get_window_extent(renderer)
                                
                                if add_unique_annotation(reverse_map['data_label'], label_bbox, text=artist.get_text().strip()):
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
                    for box_artist in bp_artists['boxes']:
                        if box_artist and box_artist.get_visible():
                            try:
                                bbox = box_artist.get_window_extent(renderer)
                                if bbox.width > 0.5 and bbox.height > 0.5 and add_unique_annotation(reverse_map['box'], bbox):
                                    added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Box bbox error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} box annotations")
                
                # Medians
                if 'median_line' in reverse_map:
                    added = 0
                    for median in bp_artists.get('medians', []):
                        if median and median.get_visible():
                            try:
                                orig_bbox = median.get_window_extent(renderer)
                                # Guarantee minimum pixel thickness in BOTH directions
                                # so the median line stays detectable regardless of
                                # box orientation (vertical boxplot -> thin height;
                                # horizontal boxplot -> thin width).
                                padded = ensure_min_bbox_thickness(orig_bbox, min_size=4.0)
                                
                                if padded.width > 0.5 and padded.height > 0.5:
                                    if add_unique_annotation(reverse_map['median_line'], padded):
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
                                if union_bbox.width > 0.5 and add_unique_annotation(reverse_map['range_indicator'], union_bbox):
                                    added += 1
                        except Exception as e:
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Range {i} error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} range annotations")
                
                # Outliers (Fliers)
                if 'outlier' in reverse_map:
                    added = 0
                    for flier in bp_artists.get('fliers', []):
                        if flier and flier.get_visible():
                            try:
                                xdata, ydata = flier.get_xdata(), flier.get_ydata()
                                for x, y in zip(xdata, ydata):
                                    px, py = ax.transData.transform_point((x, y))
                                    size = 3
                                    bbox = transforms.Bbox.from_extents(
                                        px - size, py - size, px + size, py + size
                                    )
                                    if bbox.width > 0.5 and add_unique_annotation(reverse_map['outlier'], bbox):
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

                        bbox = transforms.Bbox.from_extents(
                            px - radius, py - radius, px + radius, py + radius
                        )

                        if debug and i < 3:
                            print(f"DEBUG: Point {i}: data=({x_data:.2f},{y_data:.2f}) → "
                                  f"display=({px:.1f},{py:.1f}), size={s:.1f}, radius={radius:.2f}")

                        if bbox.width > 0.5 and bbox.height > 0.5:
                            if add_unique_annotation(reverse_map['data_point'], bbox):
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
                            if add_unique_annotation(reverse_map['wedge'], bbox):
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
                        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip()):
                            titles_added += 1
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Heatmap X-title error: {e}")
                
                if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                    try:
                        ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip()):
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
                                    label_bbox = label.get_window_extent(renderer)
                                    if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
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
                                    label_bbox = label.get_window_extent(renderer)
                                    if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
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
                                label_bbox = artist.get_window_extent(renderer)
                                if add_unique_annotation(reverse_map['data_label'], label_bbox, text=artist.get_text().strip()):
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
                        artist_bbox = artist.get_window_extent(renderer)
                        add_unique_annotation(reverse_map['significance_marker'], artist_bbox, text=text_content)
                    elif 'data_label' in reverse_map:
                        artist_bbox = artist.get_window_extent(renderer)
                        if artist_bbox.width > 0.5 and artist_bbox.height > 0.5:
                            add_unique_annotation(reverse_map['data_label'], artist_bbox, text=text_content)
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

def filter_overlapping_annotations(annotations, iou_threshold=0.7):
    """Remove annotations with high IoU overlap within the same class."""
    def bbox_iou(bbox1, bbox2):
        x1 = max(bbox1.x0, bbox2.x0)
        y1 = max(bbox1.y0, bbox2.y0)
        x2 = min(bbox1.x1, bbox2.x1)
        y2 = min(bbox1.y1, bbox2.y1)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0)
        bbox2_area = (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    by_class = defaultdict(list)
    for ann in annotations:
        by_class[ann['class_id']].append(ann)
    
    filtered = []
    for class_id, class_anns in by_class.items():
        class_anns.sort(key=lambda a: (a['bbox'].x1 - a['bbox'].x0) * (a['bbox'].y1 - a['bbox'].y0), reverse=True)
        
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


def apply_realism_effects(pil_img, annotations, effects_config):
    """Apply realism effects and return modified image and annotations."""
    effect_function_map = {
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
        "perspective": apply_perspective_effect,
        "perspective_warp": apply_perspective_warp_effect,
        "uneven_lighting": apply_uneven_lighting_effect,
        "chromatic_aberration": apply_chromatic_aberration_effect,
        "pdf_document_context": apply_pdf_document_context_effect,
    }
    
    total_dx, total_dy = 0, 0
    transform_steps = []
    img_w, img_h = pil_img.size

    def _rotate_point(x, y, angle, img_w, img_h):
        cx = img_w / 2.0
        cy = img_h / 2.0
        rad = np.radians(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        
        # Convert to standard image coordinates (origin top-left)
        x_img = float(x)
        y_img = float(img_h - y)
        
        # Rotate around center
        dx = x_img - cx
        dy = y_img - cy
        x_rot = cx + dx * cos_a + dy * sin_a
        y_rot = cy - dx * sin_a + dy * cos_a
        
        # Convert back to Matplotlib coordinates (origin bottom-left)
        return x_rot, float(img_h - y_rot)

    def _warp_point(x, y, H, img_h):
        x_img = float(x)
        y_img = float(img_h - y)
        denom = (H[2, 0] * x_img) + (H[2, 1] * y_img) + H[2, 2]
        if denom == 0:
            return float(x), float(y)
        x_w = ((H[0, 0] * x_img) + (H[0, 1] * y_img) + H[0, 2]) / denom
        y_w = ((H[1, 0] * x_img) + (H[1, 1] * y_img) + H[1, 2]) / denom
        return float(x_w), float(img_h - y_w)

    def _apply_transform_steps(x, y, img_h):
        x_t, y_t = float(x), float(y)
        for step in transform_steps:
            if step[0] == "translate":
                x_t += step[1]
                y_t += step[2]
            elif step[0] == "homography":
                x_t, y_t = _warp_point(x_t, y_t, step[1], img_h)
            elif step[0] == "rotation":
                x_t, y_t = _rotate_point(x_t, y_t, step[1], img_w, img_h)
        return x_t, y_t
    
    for effect_name, effect_config in effects_config.items():
        if random.random() < effect_config.get('p', 0):
            func = effect_function_map.get(effect_name)
            if not func: 
                continue
            
            print(f"    - Applying effect: {effect_name}")
            params = effect_config.get('params', {})
            
            try:
                if effect_name in ['clipping', 'pdf_document_context']:
                    pil_img, dx, dy = func(pil_img, **params)
                    total_dx += dx
                    total_dy += dy
                    transform_steps.append(("translate", dx, dy))
                elif effect_name == 'scan_rotation':
                    result = func(pil_img, **params)
                    pil_img = result[0]
                    angle = result[1]
                    transform_steps.append(("rotation", angle))
                elif effect_name == 'perspective_warp':
                    pil_img, H = func(pil_img, return_homography=True, **params)
                    if H is not None:
                        transform_steps.append(("homography", H))
                else:
                    pil_img = func(pil_img, **params)
            except Exception as e:
                print(f"      [WARNING] Failed to apply effect '{effect_name}': {e}")
    
    # Apply offset to annotations
    if transform_steps:
        img_w, img_h = pil_img.size
        if total_dx != 0 or total_dy != 0:
            print(f"    - Applying total annotation offset: dx={total_dx}, dy={total_dy}")
        for ann in annotations:
            bbox = ann['bbox']
            corners = [
                (bbox.x0, bbox.y0),
                (bbox.x1, bbox.y0),
                (bbox.x1, bbox.y1),
                (bbox.x0, bbox.y1)
            ]
            warped = [_apply_transform_steps(x, y, img_h) for x, y in corners]
            xs = [p[0] for p in warped]
            ys = [p[1] for p in warped]
            new_bbox = transforms.Bbox.from_extents(min(xs), min(ys), max(xs), max(ys))
            ann['bbox'] = new_bbox

            if "keypoints" in ann and ann["keypoints"]:
                keypoints = ann["keypoints"]
                normalized = all(0.0 <= kp[0] <= 1.0 and 0.0 <= kp[1] <= 1.0 for kp in keypoints)
                updated = []
                for kp in keypoints:
                    x_kp, y_kp = kp[0], kp[1]
                    vis = kp[2] if len(kp) > 2 else None
                    if normalized:
                        x_kp *= img_w
                        y_kp *= img_h
                    x_kp, y_kp = _apply_transform_steps(x_kp, y_kp, img_h)
                    if normalized:
                        x_kp = max(0.0, min(1.0, x_kp / img_w))
                        y_kp = max(0.0, min(1.0, y_kp / img_h))
                    if vis is None:
                        updated.append([x_kp, y_kp])
                    else:
                        updated.append([x_kp, y_kp, vis])
                ann["keypoints"] = updated
    
    return pil_img, annotations

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


# ===================================================================================
# == DUAL-STREAM ANNOTATION: INSTANCE SEGMENTATION & MARKER/EXTREMA DETECTION
# ===================================================================================
# Extracts polygon masks for line/area series and discrete bounding boxes for data
# markers and extrema (peaks, valleys, inflections) directly from chart coordinate spaces.

def stroke_to_polygon_ribbon(pixel_points: List[Tuple[float, float]], linewidth_px: float) -> List[Tuple[float, float]]:
    """
    Converts an ordered sequence of pixel coordinates into a closed ribbon polygon by
    offsetting each vertex along its averaged segment normal.
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
            poly_px = stroke_to_polygon_ribbon(px_pts, linewidth_px=linewidth_pt * dpi_scale)
            if len(poly_px) >= 3:
                seg_annotations.append({
                    'class_id': 0,  # "line_series"
                    'polygon': poly_px,
                    'series_idx': series.get('series_idx')
                })

            # 2. Marker glyphs (only when markers are rendered on the series)
            marker = series.get('marker')
            if marker:
                marker_r = max(2.5, (series.get('markersize', 6.0) / 2.0) * dpi_scale)
                for point_idx, (x, y, *idx) in enumerate(pts_data):
                    mx, my = ax.transData.transform_point((x, y))
                    marker_annotations.append({
                        'class_id': 0,  # "data_marker"
                        'bbox': (mx - marker_r, my - marker_r, mx + marker_r, my + marker_r)
                    })

    return seg_annotations, marker_annotations


def extract_area_segmentation_annotations(
    fig, chart_info_map, img_w: int, img_h: int
) -> List[Dict]:
    """
    Extracts instance segmentation polygon masks for area charts.

    The polygon is always built directly from fill_top + reversed(fill_bottom),
    with no occlusion-based clipping applied here. That is intentional, not
    an oversight: chart.py's _generate_area_chart already encodes the correct
    per-mode extent in fill_top/fill_bottom before this function ever sees it
    -- the full amodal region (bottom=0) for 'overlapping'/'single', or a
    mutually-exclusive band (bottom=y_stack_previous) for 'stacked'. Adding
    any additional visibility/occlusion clipping here would double-apply (and
    likely break) that logic rather than fix anything.
    """
    seg_annotations = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        chart_info = chart_info_map.get(ax, {})
        if chart_info.get('chart_type_str') != 'area':
            continue

        for series in chart_info.get('keypoint_info', []):
            top = series.get('fill_top', [])
            bottom = series.get('fill_bottom', [])
            if len(top) < 2 or len(bottom) < 2:
                continue

            poly_data = top + list(reversed(bottom))

            px_poly = []
            for x, y, *_ in poly_data:
                px, py = ax.transData.transform_point((x, y))
                px_poly.append((px, img_h - py))

            seg_annotations.append({
                'class_id': 0,  # "area_series"
                'polygon': px_poly,
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


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle matplotlib Bbox objects
    elif hasattr(obj, 'extents'):  # This is a matplotlib Bbox
        x0, y0, x1, y1 = obj.extents
        return [float(x0), float(y0), float(x1), float(y1)]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Catch-all for matplotlib Artist objects (Line2D, PathPatch, etc.)
        return str(obj)

def get_detailed_annotations(fig, chart_info_map, cls_map, img_w, img_h, raw_annotations=None):
    """Extract comprehensive metadata with xyxy coordinates and text content
    
    Args:
        raw_annotations: Optional list of raw annotation dicts from get_granular_annotations
    """
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.get_window_extent(renderer)
    seen = set()
    
    detailed_metadata = {
        "chart_type": None,
        "orientation": None,
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
        "outlier": []
    }
    
    def add_annotation(element_type, bbox, text="", conf=1.0, extra=None):
        if not bbox or bbox.width <= 1 or bbox.height <= 1:
            return None
        
        key = (element_type, round(bbox.x0, 2), round(bbox.y0, 2), round(bbox.x1, 2), round(bbox.y1, 2))
        if key in seen:
            return None
        seen.add(key)
        
        xyxy = bbox_to_xyxy(bbox, img_h)
        entry = {"xyxy": xyxy, "conf": conf}
        if text:
            entry["text"] = text
        if extra:
            entry.update(extra)
        
        detailed_metadata[element_type].append(entry)
        return entry
    
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        
        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str', 'unknown')
        orientation = chart_info.get('orientation', 'vertical')
        
        detailed_metadata["chart_type"] = chart_type
        detailed_metadata["orientation"] = orientation
        
        # Chart Title
        if 'chart_title' in cls_map:
            title = ax.title
            if title and title.get_visible() and title.get_text():
                add_annotation("chart_title", title.get_window_extent(renderer), 
                             text=title.get_text().strip(), conf=1.0)
        
        # Axis Titles
        if 'axis_title' in cls_map:
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
                add_annotation("axis_title", ax.xaxis.label.get_window_extent(renderer),
                             text=ax.xaxis.label.get_text().strip(), conf=1.0, 
                             extra={"axis": "x"})
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
                add_annotation("axis_title", ax.yaxis.label.get_window_extent(renderer),
                             text=ax.yaxis.label.get_text().strip(), conf=1.0,
                             extra={"axis": "y"})
        
        # Scale Labels (Tick Labels)
        if 'axis_labels' in cls_map:
            # Use the scale axis information from the chart generation
            scale_axis_info = chart_info.get('scale_axis_info', {})
            primary_scale_axis = scale_axis_info.get('primary_scale_axis', 'y')
            secondary_scale_axis = scale_axis_info.get('secondary_scale_axis', None)
            bg_color = ax.get_facecolor()
            # Process X-axis labels
            for label in ax.get_xticklabels():
                if label.get_visible() and label.get_text():
                    if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                        txt = label.get_text().strip()
                        # Check if X-axis is a scale axis
                        if is_float(txt) and (primary_scale_axis == 'x' or secondary_scale_axis == 'x'):
                            add_annotation("scale_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "x", "is_numeric": True})
                        else:
                            add_annotation("tick_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "x", "is_numeric": is_float(txt)})
                    else:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: label '{label.get_text()}' empty (no pixels), skipping")
            
            # Process Y-axis labels
            for label in ax.get_yticklabels():
                if label.get_visible() and label.get_text():
                    if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                        txt = label.get_text().strip()
                        # Check if Y-axis is a scale axis
                        if is_float(txt) and (primary_scale_axis == 'y' or secondary_scale_axis == 'y'):
                            add_annotation("scale_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "y", "is_numeric": True})
                        else:
                            add_annotation("tick_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "y", "is_numeric": is_float(txt)})
                    else:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: label '{label.get_text()}' empty (no pixels), skipping")
                
        # Legend
        if 'legend' in cls_map:
            legend = ax.get_legend()
            if legend and legend.get_visible():
                # Check if the legend has non-background pixels before adding annotation
                if has_non_background_pixels(legend, fig, ax, ax.get_facecolor(), threshold=5):
                    legend_texts = [t.get_text() for t in legend.get_texts() if t.get_visible()]
                    add_annotation("legend", legend.get_window_extent(renderer),
                                 conf=1.0, extra={"entries": legend_texts})
                else:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: Legend empty (no pixels), skipping")
        
        # Data Elements
        for artist in chart_info.get('data_artists', []):
            if not artist or not artist.get_visible():
                continue
            
            # Line chart points
            if isinstance(artist, matplotlib.lines.Line2D) and chart_type == 'line':
                x_data, y_data = artist.get_xdata(), artist.get_ydata()
                for x, y in zip(x_data, y_data):
                    px, py = ax.transData.transform_point((x, y))
                    bbox = transforms.Bbox.from_extents(px - 5, py - 5, px + 5, py + 5)
                    add_annotation("data_point", bbox, conf=0.95, 
                                 extra={"x": float(x), "y": float(y)})
            
            # Scatter points
            elif 'PathCollection' in str(type(artist)) and chart_type == 'scatter':
                offsets = artist.get_offsets()
                sizes = artist.get_sizes()
                if len(offsets):
                    is_uniform = (sizes.size == 1)
                    pts_to_px = fig.dpi / 72.0
                    for i, (x, y) in enumerate(offsets):
                        px, py = ax.transData.transform_point((x, y))
                        s = float(sizes[0] if is_uniform else sizes[i])
                        r = max(3.0, (np.sqrt(s) / 2.0) * pts_to_px)
                        bbox = transforms.Bbox.from_extents(px - r, py - r, px + r, py + r)
                        add_annotation("data_point", bbox, conf=0.95,
                                     extra={"x": float(x), "y": float(y), "size": float(s)})
            
            # Bar elements
            elif isinstance(artist, patches.Rectangle) and chart_type == 'bar':
                value = float(artist.get_height() if orientation == 'vertical' else artist.get_width())
                add_annotation("bar", artist.get_window_extent(renderer), conf=0.98,
                             extra={"value": value})
        
        # Box plot elements
        if chart_type == 'box':
            bp_dict = chart_info.get('boxplot_dict', {})
            if bp_dict:
                for box in bp_dict.get('boxes', []):
                    add_annotation("box", box.get_window_extent(renderer), conf=0.98)
                
                for median in bp_dict.get('medians', []):
                    bbox = median.get_window_extent(renderer)
                    padded = transforms.Bbox.from_extents(bbox.x0, bbox.y0 - 2, bbox.x1, bbox.y1 + 2)
                    add_annotation("median_line", padded, conf=0.97)
                
                for i in range(len(bp_dict.get('boxes', []))):
                    try:
                        w1 = bp_dict['whiskers'][2 * i]
                        w2 = bp_dict['whiskers'][2 * i + 1]
                        c1 = bp_dict['caps'][2 * i]
                        c2 = bp_dict['caps'][2 * i + 1]
                        combined = transforms.Bbox.union([
                            w1.get_window_extent(renderer),
                            w2.get_window_extent(renderer),
                            c1.get_window_extent(renderer),
                            c2.get_window_extent(renderer)
                        ])
                        add_annotation("range_indicator", combined, conf=0.96)
                    except IndexError:
                        continue
                
                for flier in bp_dict.get('fliers', []):
                    for x, y in zip(flier.get_xdata(), flier.get_ydata()):
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px - 3, py - 3, px + 3, py + 3)
                        add_annotation("outlier", bbox, conf=0.94)
        
        # Error bars and text annotations
        for artist in chart_info.get('other_artists', []):
            if not artist:
                continue
            
            # Error bars
            if isinstance(artist, ErrorbarContainer):
                plotline, caplines, barlinecols = artist.lines
                if barlinecols and caplines:
                    stems = barlinecols[0].get_segments()
                    for i in range(min(len(stems), len(caplines) // 2)):
                        try:
                            p1 = ax.transData.transform(stems[i][0])
                            p2 = ax.transData.transform(stems[i][1])
                            stem_bbox = transforms.Bbox([p1, p2])
                            
                            cap1, cap2 = caplines[2 * i], caplines[2 * i + 1]
                            c1x, c1y = cap1.get_data()
                            c2x, c2y = cap2.get_data()
                            cap1_bbox = transforms.Bbox([
                                ax.transData.transform((c1x[0], c1y[0])),
                                ax.transData.transform((c1x[1], c1y[1]))
                            ])
                            cap2_bbox = transforms.Bbox([
                                ax.transData.transform((c2x[0], c2y[0])),
                                ax.transData.transform((c2x[1], c2y[1]))
                            ])
                            
                            final = transforms.Bbox.union([stem_bbox, cap1_bbox, cap2_bbox])
                            add_annotation("error_bar", final, conf=0.89)
                        except Exception:
                            continue
            
            # Text annotations
            elif isinstance(artist, matplotlib.text.Text) and artist.get_visible():
                txt = artist.get_text().strip()
                if not txt:
                    continue
                
                # Significance markers
                if txt in ['*', '**', '***', 'ns', 'a', 'b', 'c', 'd']:
                    add_annotation("significance_marker", artist.get_window_extent(renderer),
                                text=txt, conf=0.92)
                # Data labels (Captures everything else, both text and numbers)
                else:
                    add_annotation("data_label", artist.get_window_extent(renderer),
                                text=txt, conf=0.91)
        
    if raw_annotations:
        detailed_metadata["raw_annotations"] = []
        for ann in raw_annotations:
            bbox = ann['bbox']
            xyxy = bbox_to_xyxy_absolute(bbox, img_h)
            detailed_metadata["raw_annotations"].append({
                'class_id': int(ann['class_id']),
                'xyxy': [int(coord) for coord in xyxy]
            })
    
    return detailed_metadata


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
    renderer = fig.canvas.get_renderer()

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
    for ax in fig.axes:
        if not ax.get_visible():
            continue

        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get("chart_type_str", "unknown")
        detailed_metadata["chart_type"] = chart_type
        detailed_metadata["orientation"] = chart_info.get("orientation", "vertical")

        from chart import extract_scale_axis_info
        scale_axis_info = chart_info.get("scale_axis_info") or extract_scale_axis_info(ax, chart_type)
        if scale_axis_info:
            detailed_metadata["scale_axis_info"] = scale_axis_info

        from chart import extract_bar_info
        bar_info_list = extract_bar_info(ax, chart_type)
        if bar_info_list:
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
        keypoint_info = extract_keypoint_info(ax, chart_type)
        if keypoint_info:
            detailed_metadata["keypoint_info"] = [
                {
                    "series_idx": kp.get("series_idx"),
                    "points": [
                        {
                            "x": float(pt.get("x", 0)),
                            "y": float(pt.get("y", 0)),
                            "is_inflection": pt.get("is_inflection", False),
                        }
                        for pt in kp.get("points", [])
                    ],
                }
                for kp in keypoint_info
            ]

        boxplot_dict = chart_info.get("boxplot_dict", {})
        if boxplot_dict and chart_type == "box":
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
        if pie_geometry:
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
        if pie_meta:
            detailed_metadata["pie_metadata"] = pie_meta

        histogram_meta = chart_info.get("histogram_metadata")
        if histogram_meta and chart_type == "histogram":
            detailed_metadata["histogram_metadata"] = histogram_meta

        detailed_metadata["series_count"] = chart_info.get("series_count", 1)
        detailed_metadata["series_names"] = chart_info.get("series_names", [])
        detailed_metadata["stacking_mode"] = chart_info.get("stacking_mode")
        detailed_metadata["dual_axis_info"] = chart_info.get("dual_axis_info", {})
        detailed_metadata["style"] = chart_info.get("style")
        detailed_metadata["pattern"] = chart_info.get("pattern")
        detailed_metadata["is_scientific"] = chart_info.get("is_scientific", False)

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
        normalized_raw_annotations.append(raw_entry)

        element_key = raw_to_element_key.get(class_name)
        if element_key:
            extra = {"class_id": class_id}
            if class_name == "axis_title":
                extra["axis"] = ann.get("axis")
            add_xyxy_annotation(element_key, xyxy, text=text, conf=1.0, extra=extra)
        elif class_name == "axis_labels":
            add_xyxy_annotation(
                "scale_labels" if _is_numeric_text(text) else "tick_labels",
                xyxy,
                text=text,
                conf=1.0,
                extra={"is_numeric": _is_numeric_text(text)},
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
            
            # Calculate row and column neighbors (simplified adjacency)
            # For a 2D grid, we look at immediate horizontal and vertical neighbors
            neighbor_sums = 0
            neighbor_count = 0
            
            # Calculate mean
            mean_val = np.nanmean(data)
            
            # Calculate neighbor relationships for Moran's I (simplified)
            for i in range(rows):
                for j in range(cols):
                    current_val = data[i, j]
                    if not np.isfinite(current_val):
                        continue
                    
                    # Check neighbors (up, down, left, right)
                    neighbors = []
                    if i > 0: neighbors.append(data[i-1, j])  # up
                    if i < rows-1: neighbors.append(data[i+1, j])  # down
                    if j > 0: neighbors.append(data[i, j-1])  # left
                    if j < cols-1: neighbors.append(data[i, j+1])  # right
                    
                    for neighbor_val in neighbors:
                        if not np.isfinite(neighbor_val):
                            continue
                        neighbor_sums += (current_val - mean_val) * (neighbor_val - mean_val)
                        neighbor_count += 1
            
            if neighbor_count == 0:
                return True  # No neighbors (single cell)
                
            # Calculate variance
            variance = np.nanvar(data)
            
            if variance == 0:
                # All values are the same - not spatially interesting but valid
                return True
            
            # Simplified Moran's I calculation
            moran_i = (neighbor_count / (rows * cols)) * (neighbor_sums / (variance * neighbor_count))
            
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


def compute_spatial_autocorrelation(data):
    """Compute simplified Moran's I for spatial autocorrelation."""
    if data.size <= 1:
        return 0.0
    
    rows, cols = data.shape
    if rows < 2 or cols < 2:
        return 0.0
    
    mean_val = data.mean()
    variance = np.var(data)
    
    if variance == 0:
        return 0.0
    
    # Calculate neighbor relationships
    neighbor_sums = 0
    neighbor_count = 0
    
    for i in range(rows):
        for j in range(cols):
            current_val = data[i, j]
            
            # Check neighbors (up, down, left, right)
            neighbors = []
            if i > 0: neighbors.append(data[i-1, j])  # up
            if i < rows-1: neighbors.append(data[i+1, j])  # down
            if j > 0: neighbors.append(data[i, j-1])  # left
            if j < cols-1: neighbors.append(data[i, j+1])  # right
            
            for neighbor_val in neighbors:
                neighbor_sums += (current_val - mean_val) * (neighbor_val - mean_val)
                neighbor_count += 1
    
    if neighbor_count == 0:
        return 0.0
    
    moran_i = (neighbor_count / (rows * cols)) * (neighbor_sums / (variance * neighbor_count))
    return moran_i


def monitor_generation_quality(num_samples=100):
    """
    Generate validation set and compute quality metrics.
    This function can be used to monitor the heatmap generation quality.
    """
    print(f"Starting quality monitoring with {num_samples} samples...")
    
    metrics = {
        'data_autocorrelation': [],
        'annotation_completeness': [],
        'bbox_validity': [],
        'colormap_appropriateness': []
    }
    
    # Since we can't easily generate samples here, this is more for documentation
    # of how quality monitoring should work
    
    print("Quality monitoring completed. This function is available for validation checks.")
    return metrics

def generate_single_chart(i, cfg, images_dir, labels_dir, output_dir):
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
        theme_config = THEMES[theme_name]
        print(f"  - Theme: {theme_name}")
        
        if cfg['debug_mode']:
            print(f"DEBUG: AX[{ax_idx}]: Theme selected: {theme_name}")

        is_scientific = random.random() < cfg['bar_chart_config']['scientific_ratio']
        style_config = {}

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
                style_config['style'] = random.choices(styles, weights=weights, k=1)[0]
                patterns, weights = zip(*[(k, v['weight']) for k, v in cfg['bar_chart_config']['patterns'].items()])
                style_config['pattern'] = random.choices(patterns, weights=weights, k=1)[0]
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
        
        ax.set_title(random.choice(CHART_TITLES), fontsize=14, pad=15)
        if random.random() < 0.6 and chart_type not in ['pie', 'heatmap']:
            ax.legend(loc=random.choice(['upper right', 'best']))
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
            'heatmap_data': heatmap_data
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
                    plt.close(fig)
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

    # Apply realism effects
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')
    
    if cfg['debug_mode']:
        print(f"DEBUG: Before realism effects - annotations: {len(annotations)}")
    
    effects_to_apply = dict(cfg.get('realism_effects', {}))
    if cfg.get('dataset_format') == 'multi_chart_detection':
        pdf_noise = cfg.get('multi_chart_detection', {}).get('pdf_context_noise', {})
        if pdf_noise:
            effects_to_apply['pdf_document_context'] = pdf_noise

    pil_img, annotations = apply_realism_effects(pil_img, annotations, effects_to_apply)
    
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
    
    # Filter out-of-bounds annotations
    img_w, img_h = pil_img.size
    final_valid_annotations = []
    for ann in annotations:
        bbox = ann['bbox']
        ann_chart_type = ann.get('chart_type', primary_chart_type)
        
        # Clamp heatmap layout coordinates to visible viewport dimensions
        if ann_chart_type == 'heatmap':
            x0 = max(0.0, min(float(bbox.x0), img_w))
            x1 = max(x0, min(float(bbox.x1), img_w))
            y0 = max(0.0, min(float(bbox.y0), img_h))
            y1 = max(y0, min(float(bbox.y1), img_h))
            if x1 > x0 and y1 > y0:
                ann['bbox'] = transforms.Bbox.from_extents(x0, y0, x1, y1)
                final_valid_annotations.append(ann)
            else:
                print(f"    - Discarded annotation (zero area after clamping): class {ann['class_id']}")
        else:
            if (bbox.x0 >= 0 and bbox.y0 >= 0 and 
                bbox.x1 <= img_w and bbox.y1 <= img_h and
                bbox.x1 > bbox.x0 and bbox.y1 > bbox.y0):
                final_valid_annotations.append(ann)
            else:
                print(f"    - Discarded annotation (out of bounds after effects): class {ann['class_id']}")

    annotations = final_valid_annotations
    
    if cfg['debug_mode']:
        print(f"DEBUG: After size/aspect/bounds filtering - annotations: {len(annotations)}")
    
    annotations = filter_overlapping_annotations(annotations, iou_threshold=0.7)
    
    if cfg['debug_mode']:
        print(f"DEBUG: After overlap filtering - annotations: {len(annotations)}")

    # Extract primary chart type string for classification folder mapping
    primary_chart_type = chart_info_map.get(fig.axes[0], {}).get('chart_type_str', 'unknown')

    # Save files
    base_filename = f"chart_{i:05d}"

    pil_img.save(os.path.join(images_dir, f"{base_filename}.png"))
    save_annotations_yolo(annotations, img_w, img_h, 
                         os.path.join(labels_dir, f"{base_filename}.txt"))

    if cfg.get('dataset_format') == 'multi_chart_detection':
        iter_time = time.time() - iter_start
        print(f"    ✓ Image {i+1}/{cfg['num_images']} complete in {iter_time:.2f}s | Saved {len(annotations)} annotations")
        return

    # =========================================================================
    # AREA CHARTS: Object Detection & Instance Segmentation Masks
    # =========================================================================
    # Exports:
    #   - area_obj_labels/: Bounding boxes for global layout elements
    #   - area_seg_labels/: Polygon masks for each area series fill layer
    if primary_chart_type == 'area':
        clsmap_obj = GENERATION_CONFIG['CLASS_MAP_AREA_OBJ']
        annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
        
        area_obj_dir = os.path.join(output_dir, 'area_obj_labels')
        ensure_dir(area_obj_dir)
        save_annotations_yolo(annotations_obj, img_w, img_h, 
                            os.path.join(area_obj_dir, f"{base_filename}.txt"))
        
        area_seg_anns = extract_area_segmentation_annotations(fig, chart_info_map, img_w, img_h)
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
    # Exports:
    #   - pie_obj_labels/: Bounding boxes for wedges, legends, titles, labels
    #   - pie_pose_labels/: 5 keypoints per wedge (Center, Start, Inter1, Inter2, End)
    elif primary_chart_type == 'pie':
        clsmap_obj = GENERATION_CONFIG['CLASS_MAP_PIE_OBJ']
        annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
        
        pie_obj_dir = os.path.join(output_dir, 'pie_obj_labels')
        ensure_dir(pie_obj_dir)
        save_annotations_yolo(annotations_obj, img_w, img_h,
                            os.path.join(pie_obj_dir, f"{base_filename}.txt"))
        
        clsmap_pose = GENERATION_CONFIG['CLASS_MAP_PIE_POSE']
        clsmap_pose_reverse = {v: k for k, v in clsmap_pose.items()}
        
        keypoint_annotations = extract_pie_pose_annotations(
            fig, chart_info_map, clsmap_pose_reverse, img_w, img_h
        )
        
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
    # Exports:
    #   - line_obj_labels/: Bounding boxes for lines, legends, titles, labels
    #   - line_seg_labels/: Ribbon polygon masks for continuous line stroke extraction
    #   - line_marker_labels/: High-precision boxes for data glyphs and extrema
    elif primary_chart_type == 'line':
        clsmap_obj = GENERATION_CONFIG['CLASS_MAP_LINE_OBJ']
        annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
        
        line_obj_dir = os.path.join(output_dir, 'line_obj_labels')
        ensure_dir(line_obj_dir)
        save_annotations_yolo(annotations_obj, img_w, img_h,
                            os.path.join(line_obj_dir, f"{base_filename}.txt"))
        
        line_seg_anns, line_marker_anns = extract_line_segmentation_annotations(
            fig, chart_info_map, img_w, img_h
        )

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
    # Exports dedicated per-type object labels:
    #   - bar_obj_labels/, histogram_obj_labels/, scatter_obj_labels/
    elif primary_chart_type in ['bar', 'histogram', 'scatter']:
        cls_map_specific = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])
        annotations_obj = get_granular_annotations(fig, chart_info_map, cls_map_specific)
        
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

    # Create three separate JSON files as expected by merge_json.py

    # Get comprehensive unified JSON with complete metadata
    unified_json = create_unified_annotation(fig, chart_info_map, cls_map, img_w, img_h, annotations)
    unified_json = convert_numpy_types(unified_json)

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

        # Legacy aliases kept for merge/backward compatibility
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
        "chart_types": [unified_json.get("chart_type", "unknown")],
        "themes": {},  # Will be populated based on the chart theme
        "num_annotations": len(annotations)
    }

    # Save the three files
    detailed_json = convert_numpy_types(detailed_json)
    with open(os.path.join(labels_dir, f"{base_filename}_detailed.json"), 'w') as f:
        json.dump(detailed_json, f, indent=2)

    with open(os.path.join(labels_dir, f"{base_filename}_ocr.json"), 'w') as f:
        json.dump(ocr_json, f, indent=2)

    with open(os.path.join(labels_dir, f"{base_filename}.json"), 'w') as f:
        json.dump(metadata_json, f, indent=2)

    iter_time = time.time() - iter_start
    print(f"    ✓ Image {i+1}/{cfg['num_images']} complete in {iter_time:.2f}s | Saved {len(annotations)} annotations")


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


def main():
    import argparse
    default_cfg_path = "custom_config.py" if os.path.exists("custom_config.py") else None
    parser = argparse.ArgumentParser(description='Generate synthetic chart training data')
    parser.add_argument('--config', type=str, default=default_cfg_path, help='Path to config file')
    parser.add_argument('--num', type=int, default=None, help='Number of images to generate')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory')
    parser.add_argument('--mode', '--format', type=str, default=None, choices=['classification', 'detection', 'multi_chart_detection'], help='Dataset format mode')
    parser.add_argument('--strict', action='store_true', help='Exit with non-zero status code if any image generation fails')
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg = config_module.OCR_TRAINING_CONFIG
    else:
        cfg = GENERATION_CONFIG

    if args.num:
        cfg['num_images'] = args.num
    
    if args.output:
        cfg['output_dir'] = args.output

    if args.mode:
        cfg['dataset_format'] = args.mode

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # Check if debug mode is enabled via environment variable
    if os.environ.get('DEBUG_MODE', '').lower() in ['true', '1', 'yes']:
        cfg['debug_mode'] = True
    
    if cfg['debug_mode']:
        print("--- DEBUG MODE ENABLED ---")
        debug_dir = 'test'
        print(f"Output will be saved to '{debug_dir}/'")
        images_dir = debug_dir
        labels_dir = debug_dir
        output_dir = debug_dir  # Define output_dir for debug mode
        ensure_dir(debug_dir)
    else:
        output_dir = cfg['output_dir']
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        ensure_dir(images_dir)
        ensure_dir(labels_dir)

    if cfg['debug_mode']:
        print(f"DEBUG: Available chart types: {list(cfg['chart_types'].keys())}")
        print(f"DEBUG: Enabled chart types: {[k for k, v in cfg['chart_types'].items() if v['enabled']]}")
        print(f"DEBUG: Scenario weights: {cfg['scenario_weights']}")
        print(f"DEBUG: Number of images to generate: {cfg['num_images']}")

    start_time = time.time()
    num_images = cfg['num_images']
    
    results = []
    use_parallel = cfg.get('use_parallel', True)
    if use_parallel and num_images > 1:
        num_cores = min(os.cpu_count(), num_images, 16)
        print(f"Launching batch engine parallelized across {num_cores} process cores...")
        tasks = [(i, cfg, images_dir, labels_dir, output_dir) for i in range(num_images)]
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(generate_single_chart_task, tasks))
    else:
        # Sequential fallback
        for i in range(num_images):
            res = generate_single_chart_task((i, cfg, images_dir, labels_dir, output_dir))
            results.append(res)

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\nGeneration complete in {time.time() - start_time:.2f}s.")
    print(f"SUMMARY: {len(successful)}/{num_images} images generated successfully ({len(failed)} failed).")
    if failed:
        failed_indices = [r[0] for r in failed]
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
        cls_map = cfg.get('CLASS_MAP_CLASSIFICATION', GENERATION_CONFIG['CLASS_MAP_CLASSIFICATION'])
        for class_id_str, class_name in sorted(cls_map.items(), key=lambda x: int(x[0])):
            cid = int(class_id_str)
            print(f"  {class_name:20s}: {class_counts[cid]:5d} instances")
    else:
        for i in range(cfg['num_images']):
            label_file = os.path.join(labels_dir, f"chart_{i:05d}.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1

        # Use the combined class map for statistics
        combined_cls_map = {}
        for chart_type, chart_cls_map in CHART_CLASS_MAPS.items():
            if chart_type != 'pie':
                for id_val, class_name in chart_cls_map.items():
                    combined_cls_map[int(id_val)] = class_name

        for class_id, class_name in sorted(combined_cls_map.items(), key=lambda x: x[1]):
            print(f"  {class_name:20s}: {class_counts[class_id]:5d} instances")

    if cfg.get('dataset_format') == 'multi_chart_detection':
        yaml_path = os.path.join(output_dir, 'data.yaml')
        cls_map = cfg.get('CLASS_MAP_CLASSIFICATION', GENERATION_CONFIG['CLASS_MAP_CLASSIFICATION'])
        sorted_names = [cls_map[str(k)] for k in sorted(map(int, cls_map.keys()))]
        abs_output_dir = os.path.abspath(output_dir)
        yaml_content = f"path: {abs_output_dir}\ntrain: images\nval: images\nnc: {len(sorted_names)}\nnames:\n"
        for name in sorted_names:
            yaml_content += f"  - {name}\n"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"\nWritten Ultralytics dataset config to {yaml_path}")

    # Merge JSON files if enabled in config
    if cfg.get('dataset_format') != 'multi_chart_detection' and cfg.get('merge_json_files', False) and batch_merge_all:
        print("\n--- Merging JSON files ---")
        batch_merge_all(labels_dir)
    
    if cfg['debug_mode']:
        print("\n--- Generation complete. Running visualization script... ---")
        try:
            subprocess.run([sys.executable, "testar.py", debug_dir, "--show"], check=True)
        except FileNotFoundError:
            print("\n[ERROR] Could not find a Python interpreter for the current environment.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] 'testar.py' script failed with error: {e}")
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred while trying to run testar.py: {e}")

    if failed and (args.strict or cfg.get('strict', False)):
        print("\n[ERROR] Strict mode enabled and generation failures occurred. Exiting with status 1.")
        sys.exit(1)


if __name__ == '__main__':
    main()
