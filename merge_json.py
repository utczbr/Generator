import json
import os

def merge_json_files(base_filename, labels_dir="labels"):
    """
    Merge 3 JSON files into 1 comprehensive JSON with complete chart metadata.
    Uses (x0,y0,x1,y1) coordinate pattern for ALL systems including OCR.
    Includes full chart generation metadata for reconstruction and analysis.
    Deletes original 3 files after successful merge.
    """
    # Define paths
    detailed_path = os.path.join(labels_dir, f"{base_filename}_detailed.json")
    ocr_path = os.path.join(labels_dir, f"{base_filename}_ocr.json")
    metadata_path = os.path.join(labels_dir, f"{base_filename}.json")

    # Load all JSON files
    with open(detailed_path, 'r') as f:
        detailed = json.load(f)
    with open(ocr_path, 'r') as f:
        ocr = json.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    def normalize_ocr_bbox(bbox_data):
        """Convert any bbox format to [x0, y0, x1, y1]"""
        if isinstance(bbox_data, dict):
            # If dict format {"x0": ..., "y0": ...}
            return [
                bbox_data.get("x0", 0),
                bbox_data.get("y0", 0),
                bbox_data.get("x1", 0),
                bbox_data.get("y1", 0)
            ]
        elif isinstance(bbox_data, (list, tuple)):
            # Already list/tuple - ensure 4 values
            return list(bbox_data[:4])
        return [0, 0, 0, 0]

    def pick_list(*keys):
        """Pick the first non-empty list among possible key aliases."""
        for key in keys:
            value = detailed.get(key)
            if isinstance(value, list) and value:
                return value
        for key in keys:
            value = detailed.get(key)
            if isinstance(value, list):
                return value
        return []

    def pick_value(*keys, default=None):
        """Pick first non-None value among key aliases."""
        for key in keys:
            if key in detailed and detailed.get(key) is not None:
                return detailed.get(key)
        return default

    def normalize_raw_annotation(entry):
        if not isinstance(entry, dict):
            return None

        xyxy = entry.get("xyxy")
        if xyxy is None:
            xyxy = entry.get("bbox")
        if isinstance(xyxy, dict):
            xyxy = [xyxy.get("x0", 0), xyxy.get("y0", 0), xyxy.get("x1", 0), xyxy.get("y1", 0)]
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            return None

        normalized = {
            "xyxy": [int(round(v)) for v in xyxy[:4]],
            "class_id": entry.get("class_id"),
        }
        if "class_name" in entry:
            normalized["class_name"] = entry.get("class_name")
        if "semantic_role" in entry:
            normalized["semantic_role"] = entry.get("semantic_role")
        if "text" in entry and entry.get("text") is not None:
            normalized["text"] = entry.get("text")
        return normalized

    normalized_raw_annotations = []
    for entry in detailed.get("raw_annotations", []):
        normalized = normalize_raw_annotation(entry)
        if normalized:
            normalized_raw_annotations.append(normalized)

    bars = pick_list("bar")
    datapoints = pick_list("data_point", "datapoint")
    boxes = pick_list("box")
    medianlines = pick_list("median_line", "medianline")
    outliers = pick_list("outlier")
    wedges = pick_list("wedge")
    line_segments = pick_list("line_segment", "linesegment")
    area_boundaries = pick_list("area_boundary", "areaboundary")
    cells = pick_list("cell")
    chart_titles = pick_list("chart_title", "charttitle")
    axis_titles = pick_list("axis_title", "axistitle")
    data_labels = pick_list("data_label", "datalabel")
    legends = pick_list("legend")
    scale_labels = pick_list("scale_labels", "scalelabels")
    tick_labels = pick_list("tick_labels", "ticklabels")
    error_bars = pick_list("error_bar", "errorbar")
    significance_markers = pick_list("significance_marker", "significancemarker")
    range_indicators = pick_list("range_indicator", "rangeindicator")
    color_bars = pick_list("color_bar", "colorbar")
    connector_lines = pick_list("connector_line", "connectorline")

    # Create unified structure with complete chart information
    unified = {
        # ===== IMAGE METADATA =====
        "image_metadata": {
            "image_id": metadata.get("image_id"),
            "resolution": {
                "width": metadata["resolution"][0],
                "height": metadata["resolution"][1]
            },
            "chart_types": metadata.get("chart_types", []),
            "theme": metadata.get("themes", {}),
            "effects_applied": ocr.get("effects_applied", [])
        },

        # ===== CHART ANALYSIS =====
        "chart_analysis": {
            "chart_type": detailed.get("chart_type"),
            "orientation": detailed.get("orientation"),
            "num_annotations": metadata.get("num_annotations", 0)
        },

        # ===== CHART GENERATION METADATA (NEW) =====
        "chart_generation_metadata": {
            # Scale axis information
            "scale_axis_info": detailed.get("scale_axis_info", {}),

            # Bar chart specific metadata
            "bar_info": detailed.get("bar_info", []),

            # Keypoint information (for line, area, pie charts)
            "keypoint_info": detailed.get("keypoint_info", []),

            # Boxplot specific metadata
            "boxplot_metadata": detailed.get("boxplot_metadata", {}),

            # Pie chart geometry
            "pie_geometry": detailed.get("pie_geometry", {}),

            # Data series information
            "series_info": {
                "count": detailed.get("series_count", 1),
                "names": detailed.get("series_names", []),
                "stacking_mode": detailed.get("stacking_mode"),
                "dual_axis": detailed.get("dual_axis_info", {})
            },

            # Style and pattern information
            "visual_style": {
                "style": detailed.get("style"),
                "pattern": detailed.get("pattern"),
                "is_scientific": detailed.get("is_scientific", False)
            }
        },

        # ===== RAW ANNOTATIONS (XYXY) =====
        "raw_annotations": normalized_raw_annotations,

        # ===== GNN TRAINING DATA (Bar-to-Baseline Graph Topology) =====
        "baselines": detailed.get("baselines", []),
        "bars_with_baseline": detailed.get("bars_with_baseline", []),
        "baseline_keypoints": detailed.get("baseline_keypoints", []),

        # ===== ELEMENT-SPECIFIC ANNOTATIONS =====
        "annotations_by_element": {
            # Visual Data Elements
            "data_elements": {
                "bars": bars,
                "datapoints": datapoints,
                "boxes": boxes,
                "medianlines": medianlines,
                "outliers": outliers,
                "wedges": wedges,
                "line_segments": line_segments,
                "area_boundaries": area_boundaries,
                "cells": cells
            },

            # Text Elements
            "text_elements": {
                "chart_title": chart_titles,
                "axis_titles": axis_titles,
                "data_labels": data_labels,
                "legend": legends
            },

            # Scale and Tick Elements
            "scale_elements": {
                "scale_labels": scale_labels,
                "tick_labels": tick_labels
            },

            # Statistical Elements
            "statistical_elements": {
                "error_bars": error_bars,
                "significance_markers": significance_markers,
                "range_indicators": range_indicators
            },

            # Additional Elements
            "additional_elements": {
                "colorbar": color_bars,
                "connector_lines": connector_lines
            }
        },

        # ===== OCR GROUND TRUTH (x0,y0,x1,y1) =====
        "ocr_ground_truth": [
            {
                "xyxy": normalize_ocr_bbox(ann["bbox"]),
                "text": ann["text"],
                "type": ann["type"],
                "is_numeric": ann["is_numeric"]
            }
            for ann in ocr.get("ocr_annotations", [])
        ],

        # ===== STATISTICS =====
        "statistics": {
            "total_annotations": metadata.get("num_annotations", 0),
            "annotation_counts_by_type": {
                "bars": len(bars),
                "datapoints": len(datapoints),
                "boxes": len(boxes),
                "wedges": len(wedges),
                "line_segments": len(line_segments),
                "cells": len(cells),
                "scale_labels": len(scale_labels),
                "tick_labels": len(tick_labels),
                "text_annotations": len(ocr.get("ocr_annotations", []))
            },
            "bar_info": pick_value("bar_info", default={})
        }
    }

    # Save unified JSON
    output_path = os.path.join(labels_dir, f"{base_filename}_unified.json")
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2)

    # Delete the 3 original files after successful merge
    try:
        os.remove(detailed_path)
        os.remove(ocr_path)
        os.remove(metadata_path)
        print(f"✓ Merged & deleted 3 JSON files → {output_path}")
    except OSError as e:
        print(f"✓ Merged JSON saved, but failed to delete originals: {e}")

    return unified


def batch_merge_all(labels_dir="labels"):
    """Merge all JSON files in directory and delete originals"""
    processed = set()

    for filename in os.listdir(labels_dir):
        if filename.endswith("_detailed.json"):
            base = filename.replace("_detailed.json", "")
            if base not in processed:
                try:
                    merge_json_files(base, labels_dir)
                    processed.add(base)
                except Exception as e:
                    print(f"✗ Failed to merge {base}: {e}")

    print(f"\n✓ Processed {len(processed)} image annotations")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", type=str, default="labels")
    parser.add_argument("--base_filename", type=str, default=None)
    parser.add_argument("--batch", action="store_true")

    args = parser.parse_args()

    if args.batch:
        batch_merge_all(args.labels_dir)
    elif args.base_filename:
        merge_json_files(args.base_filename, args.labels_dir)
    else:
        print("Please specify either --base_filename or --batch")
