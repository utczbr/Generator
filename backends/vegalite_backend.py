"""
backends/vegalite_backend.py

Lean functional vector graphics backend using Vega-Lite, vl-convert-python,
and analytical SVG DOM annotation extraction via lxml and svgelements.
Zero calls to Matplotlib get_window_extent() or raster buffer scanning.
"""
import io
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import lxml.etree as etree
import numpy as np
import svgelements as se
import vl_convert as vlc

from themes import THEMES
from config_defaults import ANNOTATION_SCHEMA_VERSION, DATASET_VERSION
from generator import (
    BoundingBox,
    CHART_CLASS_MAPS,
    canonicalize_and_validate_obb,
    ensure_dir,
    json_default_fallback,
    save_annotations_yolo,
    save_annotations_yolo_obb,
)


def build_vegalite_bar_spec(
    categories: List[str],
    values: List[float],
    series: Optional[List[str]] = None,
    stacking: str = "none",  # "none", "grouped", "stacked"
    title: Optional[str] = None,
    x_title: Optional[str] = "Category",
    y_title: Optional[str] = "Value",
    orientation: str = "vertical",
    label_angle: int = 0,
    width: int = 400,
    height: int = 300,
    colors: Optional[List[str]] = None,
    background: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Vega-Lite v5 specification for canonical bar charts."""
    records = []
    if series and len(series) == len(values):
        for c, v, s in zip(categories, values, series):
            records.append({"category": str(c), "value": float(v), "series": str(s)})
    else:
        for c, v in zip(categories, values):
            records.append({"category": str(c), "value": float(v)})

    mark_def: Dict[str, Any] = {"type": "bar"}
    if colors and not series:
        mark_def["color"] = colors[0]

    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": width,
        "height": height,
        "data": {"values": records},
        "mark": mark_def,
    }
    if background:
        spec["background"] = background
    if title:
        spec["title"] = title

    x_enc: Dict[str, Any] = {
        "field": "category",
        "type": "nominal",
        "title": x_title,
    }
    if label_angle != 0:
        x_enc["axis"] = {"labelAngle": label_angle}

    y_enc: Dict[str, Any] = {
        "field": "value",
        "type": "quantitative",
        "title": y_title,
    }

    if series:
        color_enc: Dict[str, Any] = {
            "field": "series",
            "type": "nominal",
            "title": "Series",
        }
        if colors:
            color_enc["scale"] = {"range": colors}

        if stacking == "stacked":
            y_enc["stack"] = "zero"
            spec["encoding"] = {
                "x": x_enc if orientation == "vertical" else y_enc,
                "y": y_enc if orientation == "vertical" else x_enc,
                "color": color_enc,
            }
        else:
            # Grouped bar chart via xOffset channel
            offset_enc = {"field": "series"}
            if orientation == "vertical":
                spec["encoding"] = {
                    "x": x_enc,
                    "y": y_enc,
                    "xOffset": offset_enc,
                    "color": color_enc,
                }
            else:
                spec["encoding"] = {
                    "x": y_enc,
                    "y": x_enc,
                    "yOffset": offset_enc,
                    "color": color_enc,
                }
    else:
        if orientation == "vertical":
            spec["encoding"] = {"x": x_enc, "y": y_enc}
        else:
            spec["encoding"] = {"x": y_enc, "y": x_enc}

    return spec


def build_vegalite_line_spec(
    data_points: List[Dict[str, Any]],
    series_field: Optional[str] = None,
    title: Optional[str] = None,
    x_title: Optional[str] = "X",
    y_title: Optional[str] = "Y",
    width: int = 400,
    height: int = 300,
    colors: Optional[List[str]] = None,
    background: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Vega-Lite v5 specification for canonical line charts."""
    mark_def: Dict[str, Any] = {"type": "line", "point": True}
    if colors and not series_field:
        mark_def["color"] = colors[0]

    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": width,
        "height": height,
        "data": {"values": data_points},
        "mark": mark_def,
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": x_title},
            "y": {"field": "y", "type": "quantitative", "title": y_title},
        },
    }
    if background:
        spec["background"] = background
    if title:
        spec["title"] = title
    if series_field:
        color_enc: Dict[str, Any] = {
            "field": series_field,
            "type": "nominal",
            "title": series_field.capitalize(),
        }
        if colors:
            color_enc["scale"] = {"range": colors}
        spec["encoding"]["color"] = color_enc
    return spec


def build_vegalite_scatter_spec(
    points: List[Dict[str, Any]],
    series_field: Optional[str] = None,
    title: Optional[str] = None,
    x_title: Optional[str] = "X",
    y_title: Optional[str] = "Y",
    width: int = 400,
    height: int = 300,
    colors: Optional[List[str]] = None,
    background: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Vega-Lite v5 specification for canonical scatter charts."""
    mark_def: Dict[str, Any] = {"type": "point", "filled": True, "size": 60}
    if colors and not series_field:
        mark_def["color"] = colors[0]

    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": width,
        "height": height,
        "data": {"values": points},
        "mark": mark_def,
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": x_title},
            "y": {"field": "y", "type": "quantitative", "title": y_title},
        },
    }
    if background:
        spec["background"] = background
    if title:
        spec["title"] = title
    if series_field:
        color_enc: Dict[str, Any] = {
            "field": series_field,
            "type": "nominal",
            "title": series_field.capitalize(),
        }
        if colors:
            color_enc["scale"] = {"range": colors}
        spec["encoding"]["color"] = color_enc
    return spec


def parse_aria_label_dict(aria_label: Optional[str]) -> Dict[str, str]:
    """Parse key-value metadata from Vega-generated aria-label attribute."""
    if not aria_label:
        return {}
    res = {}
    for part in aria_label.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            res[k.strip().lower()] = v.strip()
    return res


def extract_svg_annotations(
    svg_content: str,
    scale_factor: float = 1.0,
    chart_type: str = "bar",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, int]:
    """
    Parse SVG DOM analytically using lxml and svgelements.
    Extract exact mark geometries directly from XML node attributes without raster inspection.
    Maps scene-graph marks directly to dataset semantic classes.
    """
    svg_obj = se.SVG.parse(io.StringIO(svg_content))
    svg_w = float(svg_obj.width or 400.0)
    svg_h = float(svg_obj.height or 300.0)

    img_w = int(round(svg_w * scale_factor))
    img_h = int(round(svg_h * scale_factor))

    # Parse with lxml for hierarchical role/class inspection
    root = etree.fromstring(svg_content.encode("utf-8"))

    cls_map = CHART_CLASS_MAPS.get(chart_type, CHART_CLASS_MAPS["bar"])
    name_to_class_id = {v: int(k) for k, v in cls_map.items() if str(k).isdigit()}

    annotations: List[Dict[str, Any]] = []
    detailed_metadata: Dict[str, Any] = {
        "chart_type": chart_type,
        "orientation": "vertical",
        "resolution": [img_w, img_h],
        "scale_labels": [],
        "tick_labels": [],
        "chart_title": [],
        "axis_title": [],
        "legend": [],
        "bar": [],
        "data_point": [],
        "line_segment": [],
        "bar_info": [],
        "series_names": [],
        "series_count": 1,
        "stacking_mode": None,
    }

    # 1. Extract Bars (<path role="bar"> or <rect role="bar">)
    raw_bars = []
    for elem in svg_obj.elements():
        vals = getattr(elem, "values", {})
        role_desc = vals.get("aria-roledescription")
        if role_desc == "bar":
            bbox = elem.bbox()
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            px_x0 = x0 * scale_factor
            px_y0 = y0 * scale_factor
            px_x1 = x1 * scale_factor
            px_y1 = y1 * scale_factor

            # Clamp coordinates to viewport
            px_x0 = max(0.0, min(float(img_w), px_x0))
            px_x1 = max(px_x0, min(float(img_w), px_x1))
            px_y0 = max(0.0, min(float(img_h), px_y0))
            px_y1 = max(px_y0, min(float(img_h), px_y1))

            aria_dict = parse_aria_label_dict(vals.get("aria-label"))
            series_name = (
                aria_dict.get("series")
                or aria_dict.get("group")
                or aria_dict.get("color")
                or ""
            )
            cat_name = aria_dict.get("category") or aria_dict.get("cat") or ""
            val_str = aria_dict.get("value") or aria_dict.get("val") or "0"
            try:
                numeric_val = float(val_str)
            except ValueError:
                numeric_val = 0.0

            raw_bars.append(
                {
                    "px_box": (px_x0, px_y0, px_x1, px_y1),
                    "series_name": series_name,
                    "category": cat_name,
                    "value": numeric_val,
                }
            )

    # Establish series names and indexing for bars
    unique_series = list(
        dict.fromkeys([b["series_name"] for b in raw_bars if b["series_name"]])
    )
    if unique_series:
        detailed_metadata["series_names"] = unique_series
        detailed_metadata["series_count"] = len(unique_series)

    for b in raw_bars:
        px_x0, px_y0, px_x1, px_y1 = b["px_box"]
        s_name = b["series_name"]
        s_idx = unique_series.index(s_name) if s_name in unique_series else 0

        # Create annotation with bottom-left BoundingBox for generator pipeline compatibility
        mat_y0 = img_h - px_y1
        mat_y1 = img_h - px_y0
        bar_ann = {
            "class_id": name_to_class_id.get("bar", 1),
            "class_name": "bar",
            "bbox": BoundingBox(px_x0, mat_y0, px_x1, mat_y1),
            "obb": [
                (px_x0, px_y0),
                (px_x1, px_y0),
                (px_x1, px_y1),
                (px_x0, px_y1),
            ],
            "series_idx": s_idx,
            "series_name": s_name,
            "category": b["category"],
            "value": b["value"],
            "attrs": {
                "series_idx": s_idx,
                "series_name": s_name,
                "category": b["category"],
                "value": b["value"],
            },
        }
        annotations.append(bar_ann)

        bar_record = {
            "xyxy": [int(px_x0), int(px_y0), int(px_x1), int(px_y1)],
            "series_idx": s_idx,
            "series_name": s_name,
            "category": b["category"],
            "value": b["value"],
            "conf": 1.0,
        }
        detailed_metadata["bar"].append(bar_record)
        detailed_metadata["bar_info"].append(bar_record)

    # 2. Extract Data Points (Scatter / Symbol marks)
    for elem in svg_obj.elements():
        vals = getattr(elem, "values", {})
        role_desc = vals.get("aria-roledescription")
        # Exclude legend symbols which have 'role-legend' ancestor
        if role_desc in ("point", "symbol"):
            bbox = elem.bbox()
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            px_x0 = max(0.0, min(float(img_w), x0 * scale_factor))
            px_y0 = max(0.0, min(float(img_h), y0 * scale_factor))
            px_x1 = max(px_x0, min(float(img_w), x1 * scale_factor))
            px_y1 = max(px_y0, min(float(img_h), y1 * scale_factor))

            mat_y0 = img_h - px_y1
            mat_y1 = img_h - px_y0
            pt_ann = {
                "class_id": name_to_class_id.get("data_point", 1),
                "class_name": "data_point",
                "bbox": BoundingBox(px_x0, mat_y0, px_x1, mat_y1),
                "obb": [
                    (px_x0, px_y0),
                    (px_x1, px_y0),
                    (px_x1, px_y1),
                    (px_x0, px_y1),
                ],
            }
            annotations.append(pt_ann)
            detailed_metadata["data_point"].append(
                {
                    "xyxy": [int(px_x0), int(px_y0), int(px_x1), int(px_y1)],
                    "conf": 1.0,
                }
            )

    # 3. Extract Line marks and vertices
    for elem in svg_obj.elements():
        vals = getattr(elem, "values", {})
        role_desc = str(vals.get("aria-roledescription", ""))
        if isinstance(elem, se.Path) and role_desc == "line mark":
            bbox = elem.bbox()
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            px_x0 = max(0.0, min(float(img_w), x0 * scale_factor))
            px_y0 = max(0.0, min(float(img_h), y0 * scale_factor))
            px_x1 = max(px_x0, min(float(img_w), x1 * scale_factor))
            px_y1 = max(px_y0, min(float(img_h), y1 * scale_factor))

            # Extract transformed vertices directly from path segments
            vertices = []
            for seg in elem.segments():
                if hasattr(seg, "end") and seg.end is not None:
                    vx = float(seg.end.x) * scale_factor
                    vy = float(seg.end.y) * scale_factor
                    vertices.append((vx, vy))

            mat_y0 = img_h - px_y1
            mat_y1 = img_h - px_y0
            line_ann = {
                "class_id": name_to_class_id.get("line_segment", 1),
                "class_name": "line_segment",
                "bbox": BoundingBox(px_x0, mat_y0, px_x1, mat_y1),
                "obb": [
                    (px_x0, px_y0),
                    (px_x1, px_y0),
                    (px_x1, px_y1),
                    (px_x0, px_y1),
                ],
                "vertices": vertices,
            }
            annotations.append(line_ann)
            detailed_metadata["line_segment"].append(
                {
                    "xyxy": [int(px_x0), int(px_y0), int(px_x1), int(px_y1)],
                    "vertices": vertices,
                    "conf": 1.0,
                }
            )

    # 4. Extract Legend Group
    for elem in svg_obj.elements():
        vals = getattr(elem, "values", {})
        cls_name = str(vals.get("class", ""))
        role_desc = str(vals.get("aria-roledescription", ""))
        if "role-legend" in cls_name or role_desc == "legend":
            bbox = elem.bbox()
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                px_x0 = max(0.0, min(float(img_w), x0 * scale_factor))
                px_y0 = max(0.0, min(float(img_h), y0 * scale_factor))
                px_x1 = max(px_x0, min(float(img_w), x1 * scale_factor))
                px_y1 = max(px_y0, min(float(img_h), y1 * scale_factor))

                if px_x1 > px_x0 and px_y1 > px_y0:
                    mat_y0 = img_h - px_y1
                    mat_y1 = img_h - px_y0
                    leg_ann = {
                        "class_id": name_to_class_id.get("legend", 5),
                        "class_name": "legend",
                        "bbox": BoundingBox(px_x0, mat_y0, px_x1, mat_y1),
                        "obb": [
                            (px_x0, px_y0),
                            (px_x1, px_y0),
                            (px_x1, px_y1),
                            (px_x0, px_y1),
                        ],
                    }
                    annotations.append(leg_ann)
                    detailed_metadata["legend"].append(
                        {
                            "xyxy": [int(px_x0), int(px_y0), int(px_x1), int(px_y1)],
                            "conf": 1.0,
                        }
                    )
            break

    # 5. Extract Text Elements (Titles, Axis Labels, Tick Labels)
    # Match svgelements Text with lxml Text nodes in document order
    text_se_list = [e for e in svg_obj.elements() if isinstance(e, se.Text)]
    text_lx_list = root.xpath('//*[local-name()="text"]')

    for se_elem, lx_elem in zip(text_se_list, text_lx_list):
        text_str = str(se_elem.text or "").strip()
        if not text_str:
            continue

        parent = lx_elem.getparent()
        parent_cls = parent.get("class", "") if parent is not None else ""

        # Map semantic role from SVG parent classes
        if "role-title-text" in parent_cls:
            semantic_role = "chart_title"
            class_id = name_to_class_id.get("chart_title", 6)
        elif "role-axis-title" in parent_cls:
            semantic_role = "axis_title"
            class_id = name_to_class_id.get("axis_title", 2)
        elif "role-axis-label" in parent_cls:
            semantic_role = "axis_labels"
            class_id = name_to_class_id.get("axis_labels", 8)
        elif "role-legend" in parent_cls:
            semantic_role = "legend"
            class_id = name_to_class_id.get("legend", 5)
        else:
            semantic_role = "text"
            class_id = name_to_class_id.get("data_label", 7)

        # Compute text dimensions and local box corners
        fs = float(se_elem.font_size or 10.0)
        anchor = str(se_elem.anchor or "start").lower()
        approx_w = len(text_str) * fs * 0.60
        if anchor == "end":
            lx0, lx1 = -approx_w, 0.0
        elif anchor == "middle":
            lx0, lx1 = -approx_w / 2.0, approx_w / 2.0
        else:
            lx0, lx1 = 0.0, approx_w

        ly0, ly1 = -0.80 * fs, 0.20 * fs
        local_corners = [(lx0, ly0), (lx1, ly0), (lx1, ly1), (lx0, ly1)]

        # Transform corners through composed SVG transformation matrix
        mat = se_elem.transform
        tf_corners = [se.Point(x, y) * mat for x, y in local_corners]
        scaled_corners = [
            (pt.x * scale_factor, pt.y * scale_factor) for pt in tf_corners
        ]

        # Canonicalize to clockwise winding starting from top-left
        canonical_obb = canonicalize_and_validate_obb(scaled_corners, y_down=True)
        if canonical_obb is None:
            canonical_obb = scaled_corners

        px_x0 = max(0.0, min(float(img_w), min(c[0] for c in canonical_obb)))
        px_x1 = max(px_x0, min(float(img_w), max(c[0] for c in canonical_obb)))
        px_y0 = max(0.0, min(float(img_h), min(c[1] for c in canonical_obb)))
        px_y1 = max(px_y0, min(float(img_h), max(c[1] for c in canonical_obb)))

        mat_y0 = img_h - px_y1
        mat_y1 = img_h - px_y0
        text_ann = {
            "class_id": class_id,
            "class_name": semantic_role,
            "text": text_str,
            "bbox": BoundingBox(px_x0, mat_y0, px_x1, mat_y1),
            "obb": canonical_obb,
        }
        annotations.append(text_ann)

        json_rec = {
            "xyxy": [int(px_x0), int(px_y0), int(px_x1), int(px_y1)],
            "text": text_str,
            "conf": 1.0,
        }
        if semantic_role == "chart_title":
            detailed_metadata["chart_title"].append(json_rec)
        elif semantic_role == "axis_title":
            detailed_metadata["axis_title"].append(json_rec)
        elif semantic_role == "axis_labels":
            detailed_metadata["tick_labels"].append(json_rec)
            detailed_metadata["scale_labels"].append(json_rec)
        elif semantic_role == "legend":
            detailed_metadata["legend"].append(json_rec)

    return annotations, detailed_metadata, img_w, img_h


def generate_single_vegalite_chart(
    i: int,
    cfg: Dict[str, Any],
    images_dir: str,
    labels_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Generate a single chart using Vega-Lite vector rendering backend.
    Saves image (PNG), YOLO AABB, YOLOv8-OBB (additive), detailed.json, ocr.json, and metadata.
    """
    iter_start = time.time()
    chart_type = cfg.get("chart_type")
    if not chart_type or chart_type not in ("bar", "line", "scatter"):
        # Select first enabled supported canonical chart type
        chart_types_cfg = cfg.get("chart_types", {})
        candidates = [
            k for k in ("bar", "line", "scatter")
            if chart_types_cfg.get(k, {}).get("enabled", True)
        ]
        chart_type = candidates[0] if candidates else "bar"

    # Sample visual theme styling
    theme_name = cfg.get("theme")
    if not theme_name or theme_name not in THEMES:
        available_themes = [k for k in THEMES.keys() if k not in ("heatmap", "dark_mode")]
        theme_name = random.choice(available_themes) if available_themes else "default"

    theme_cfg = THEMES.get(theme_name, {})
    background = theme_cfg.get("facecolor", "#ffffff")
    raw_palette = theme_cfg.get("palette", ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b"])
    if isinstance(raw_palette, list):
        palette = list(raw_palette)
    else:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            cmap = plt.get_cmap(raw_palette)
            palette = [mcolors.to_hex(cmap(v)) for v in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]]
        except Exception:
            palette = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b"]

    scale_factor = float(cfg.get("vegalite_scale", 2.0))
    base_filename = f"chart_{i:05d}"

    syn_table = None

    # Build chart specification
    use_synthetic = cfg.get("use_synthetic_data_engine", False)
    raw_syn_domain = cfg.get("synthetic_domain", None)
    domain_map = {"financial": "business", "sensor_telemetry": "engineering"}
    syn_domain = domain_map.get(raw_syn_domain, raw_syn_domain)

    title = f"Synthesized {chart_type.capitalize()} Chart"
    if chart_type == "bar":
        # Determine bar mode: single, grouped, or stacked
        bar_style = cfg.get("bar_style", "standard")
        is_multi = bar_style in ("grouped", "stacked", "compare_side_by_side")

        if use_synthetic:
            from synth.tabular import sample_multivariate_table
            num_cats = random.randint(3, 5)
            syn_table = sample_multivariate_table(num_rows=num_cats, num_series=2 if is_multi else 1, domain=syn_domain)
            title = syn_table["title"]
            x_title = syn_table["x_label"]
            y_title = syn_table["y_label"]
            if is_multi:
                cat_list = []
                val_list = []
                ser_list = []
                for row_idx, c in enumerate(syn_table["categories"]):
                    for s_idx, s in enumerate(syn_table["series_names"]):
                        cat_list.append(c)
                        ser_list.append(s)
                        val_list.append(round(float(syn_table["data"][row_idx, s_idx]), 1))
                stacking = "stacked" if bar_style == "stacked" else "grouped"
                spec = build_vegalite_bar_spec(
                    categories=cat_list,
                    values=val_list,
                    series=ser_list,
                    stacking=stacking,
                    title=title,
                    x_title=x_title,
                    y_title=y_title,
                    label_angle=cfg.get("label_angle", 0),
                    colors=palette,
                    background=background,
                )
            else:
                spec = build_vegalite_bar_spec(
                    categories=syn_table["categories"],
                    values=[round(float(v), 1) for v in syn_table["data"][:, 0]],
                    title=title,
                    x_title=x_title,
                    y_title=y_title,
                    label_angle=cfg.get("label_angle", 0),
                    colors=palette,
                    background=background,
                )
        else:
            cat_pool = ["Category A", "Category B", "Category C", "Category D", "Category E"]
            num_cats = random.randint(3, 5)
            selected_cats = cat_pool[:num_cats]

            if is_multi:
                series_names = ["Alpha", "Beta"]
                cat_list = []
                val_list = []
                ser_list = []
                for c in selected_cats:
                    for s in series_names:
                        cat_list.append(c)
                        ser_list.append(s)
                        val_list.append(round(random.uniform(10.0, 95.0), 1))
                stacking = "stacked" if bar_style == "stacked" else "grouped"
                spec = build_vegalite_bar_spec(
                    categories=cat_list,
                    values=val_list,
                    series=ser_list,
                    stacking=stacking,
                    title=title,
                    label_angle=cfg.get("label_angle", 0),
                    colors=palette,
                    background=background,
                )
            else:
                val_list = [round(random.uniform(15.0, 90.0), 1) for _ in selected_cats]
                spec = build_vegalite_bar_spec(
                    categories=selected_cats,
                    values=val_list,
                    title=title,
                    label_angle=cfg.get("label_angle", 0),
                    colors=palette,
                    background=background,
                )

    elif chart_type == "line":
        if use_synthetic:
            from synth.tabular import sample_multivariate_table
            syn_table = sample_multivariate_table(num_rows=7, num_series=1, domain=syn_domain)
            title = syn_table["title"]
            points = [{"x": x_val, "y": round(float(syn_table["data"][x_val - 1, 0]), 1)} for x_val in range(1, 8)]
            spec = build_vegalite_line_spec(
                points,
                title=title,
                x_title=syn_table["x_label"],
                y_title=syn_table["y_label"],
                colors=palette,
                background=background,
            )
        else:
            points = []
            for x_val in range(1, 8):
                points.append({"x": x_val, "y": round(random.uniform(10.0, 80.0), 1)})
            spec = build_vegalite_line_spec(
                points,
                title=title,
                colors=palette,
                background=background,
            )

    else:  # scatter
        if use_synthetic:
            from synth.tabular import sample_multivariate_table
            syn_table = sample_multivariate_table(num_rows=20, num_series=2, domain=syn_domain)
            title = syn_table["title"]
            points = [
                {"x": round(float(syn_table["data"][r, 0]), 1), "y": round(float(syn_table["data"][r, 1]), 1)}
                for r in range(20)
            ]
            spec = build_vegalite_scatter_spec(
                points,
                title=title,
                x_title=syn_table["series_names"][0],
                y_title=syn_table["series_names"][1],
                colors=palette,
                background=background,
            )
        else:
            points = []
            for _ in range(20):
                points.append({
                    "x": round(random.uniform(5.0, 100.0), 1),
                    "y": round(random.uniform(5.0, 100.0), 1),
                })
            spec = build_vegalite_scatter_spec(
                points,
                title=title,
                colors=palette,
                background=background,
            )

    # 1. Compile to SVG string
    svg_content = vlc.vegalite_to_svg(spec)

    # 2. Compile to raster PNG
    png_bytes = vlc.vegalite_to_png(spec, scale=scale_factor)

    # 3. Extract exact annotations analytically from SVG DOM
    annotations, detailed_metadata, img_w, img_h = extract_svg_annotations(
        svg_content, scale_factor=scale_factor, chart_type=chart_type
    )

    if use_synthetic and syn_table is not None:
        domain_map = {"financial": "business", "sensor_telemetry": "engineering"}
        raw_domain = syn_table.get("domain", syn_domain or "business")
        domain = domain_map.get(raw_domain, raw_domain)
        detailed_metadata["semantic_domain"] = domain
        from synth.semantics.loader import registry
        if domain in registry.domains:
            detailed_metadata["is_scientific"] = registry.domains[domain].is_scientific
        else:
            detailed_metadata["is_scientific"] = (domain in ("biomedical", "engineering"))
    else:
        detailed_metadata["semantic_domain"] = None
        detailed_metadata["is_scientific"] = False

    detailed_metadata["theme"] = theme_name
    detailed_metadata["schema_version"] = ANNOTATION_SCHEMA_VERSION
    detailed_metadata["dataset_version"] = DATASET_VERSION

    # Save rendered PNG image
    ensure_dir(images_dir)
    img_path = os.path.join(images_dir, f"{base_filename}.png")
    with open(img_path, "wb") as f:
        f.write(png_bytes)

    # Optional: save SVG file
    if cfg.get("save_svg", False) or cfg.get("export_svg", False):
        svg_dir = os.path.join(output_dir, "svgs")
        ensure_dir(svg_dir)
        with open(os.path.join(svg_dir, f"{base_filename}.svg"), "w", encoding="utf-8") as f:
            f.write(svg_content)

    # Save YOLO AABB annotations
    ensure_dir(labels_dir)
    save_annotations_yolo(
        annotations, img_w, img_h, os.path.join(labels_dir, f"{base_filename}.txt")
    )

    # Save additive YOLOv8-OBB annotations if requested
    if cfg.get("export_obb", False) or cfg.get("export_labels_obb", False):
        labels_obb_dir = os.path.join(output_dir, "labels_obb")
        ensure_dir(labels_obb_dir)
        save_annotations_yolo_obb(
            annotations, img_w, img_h, os.path.join(labels_obb_dir, f"{base_filename}.txt")
        )

    # Save detailed.json
    detailed_metadata["image_id"] = base_filename
    with open(os.path.join(labels_dir, f"{base_filename}_detailed.json"), "w") as f:
        json.dump(detailed_metadata, f, indent=2, default=json_default_fallback)

    if cfg.get("export_legacy_json", False):
        # Save ocr.json
        ocr_records = [
            {"text": ann["text"], "xyxy": ann.get("obb") or [ann["bbox"].x0, ann["bbox"].y0]}
            for ann in annotations
            if "text" in ann
        ]
        ocr_json = {
            "ocr_annotations": ocr_records,
            "effects_applied": [],
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "dataset_version": DATASET_VERSION,
        }
        with open(os.path.join(labels_dir, f"{base_filename}_ocr.json"), "w") as f:
            json.dump(ocr_json, f, indent=2, default=json_default_fallback)

        # Save metadata.json
        metadata_json = {
            "image_id": base_filename,
            "engine": "vegalite",
            "theme": theme_name,
            "resolution": [int(img_w), int(img_h)],
            "chart_types": [chart_type],
            "num_annotations": len(annotations),
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "dataset_version": DATASET_VERSION,
        }
        with open(os.path.join(labels_dir, f"{base_filename}.json"), "w") as f:
            json.dump(metadata_json, f, indent=2, default=json_default_fallback)

    iter_time = time.time() - iter_start
    if cfg.get("debug_mode", False):
        print(f"    ✓ Vega-Lite Chart {i+1} complete in {iter_time:.2f}s | Saved {len(annotations)} annotations")

    return {
        "image_id": base_filename,
        "chart_type": chart_type,
        "annotations_count": len(annotations),
        "img_dimensions": (img_w, img_h),
    }
