# Annotation Schema Specification

This document specifies the dataset output schemas produced by the chart synthesis pipeline, versioned via `ANNOTATION_SCHEMA_VERSION`.

## Schema Overview

| Schema Version | Primary Bounding Box | OBB (`.txt`) | Topology & Keypoints | Modal / Amodal Masks | Semantic & Composites | Compatibility |
|---|---|---|---|---|---|---|
| **v1** (Legacy) | Axis-Aligned Bounding Box (AABB) | None | Ribbon-only (no graph) | Collapsed extents | None | YOLOv8/v11 Detection, GNN v1 |
| **v2** (Modern) | Additive: AABB + OBB | YOLOv26-OBB (8 coords) | Directed keypoint graph | Dual `amodal_polygon` / `modal_polygon` | None | YOLOv26-OBB, GNN v2, Vision-Language |
| **v2.1** (Semantic) | Additive: AABB + OBB | YOLOv26-OBB (8 coords) | Directed keypoint graph | Dual `amodal_polygon` / `modal_polygon` | `semantic_domain`, `subplots`, `is_composite` | Full backwards compatibility with v2 & v1 |
| **v3.0** (Declarative) | Additive: AABB + OBB | YOLOv26-OBB (8 coords) | Directed keypoint graph | Dual `amodal_polygon` / `modal_polygon` | Declarative YAML manifests, direct domain naming, purged legacy constants | Full backwards compatibility with v2.1, v2 & v1 |
| **v4.0** (Unified) | Unified single-list `annotations` (pixel `xyxy` + `attrs`) | YOLOv26-OBB (8 coords) | Directed keypoint graph & topology | Dual `amodal_polygon` / `modal_polygon` | Dynamic registry, single-list annotations, pre-effects coordinate preservation, `subplots` | Production default for matplotlib engine; supersedes v3.0 |
| **v4.1** (Hardened) | Unified single-list `annotations` (pixel `xyxy` + `attrs`) | YOLOv26-OBB (8 coords) | Directed keypoint graph & topology | Dual `amodal_polygon` / `modal_polygon` | Top-level `filter_stats` dictionary, error bar annotations, multi-axis deduplication | Additive extension to v4.0; production default |

---

## Version 1: Legacy Schema (`v1`)

### 1. Detection Label (`<image_id>.txt`)
Standard normalized axis-aligned bounding boxes (AABB) format:
```
<class_id> <x_center> <y_center> <width> <height>
```
* **Coordinate space**: Normalized $[0.0, 1.0]$ relative to image dimensions.
* **Origin**: Top-left corner $(0, 0)$.

### 2. Detailed Metadata (`<image_id>_detailed.json`)
```json
{
  "chart_type": "bar",
  "raw_annotations": [
    {
      "class_id": 1,
      "class_name": "bar",
      "bbox": [x0, y0, x1, y1],
      "text": null
    }
  ]
}
```

---

## Version 2: Modern Vector-Faithful Schema (`v2`)

Introduced under Feature `001-chart-pipeline-modernization`. All `v1` outputs are preserved additively to guarantee downstream consumers continue to function without breaking.

### 1. Detection Label (`<image_id>.txt`)
Identical to `v1` for full backwards compatibility.

### 2. Oriented Bounding Box Label (`<image_id>_obb.txt`)
YOLOv26-OBB standard format for oriented bounding boxes:
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```
* **Coordinates**: 8 floats normalized to $[0.0, 1.0]$.
* **Winding**: Clockwise order $(x_1, y_1) \to (x_2, y_2) \to (x_3, y_3) \to (x_4, y_4)$, starting at the top-left vertex.
* **Convexity**: Guaranteed convex and non-self-intersecting (verified via sign of consecutive edge cross-products).

### 3. Detailed Metadata (`<image_id>_detailed.json`)
Extends `v1` with exact geometric, topological, and occlusion metadata:

```json
{
  "chart_type": "line",
  "schema_version": "v2",
  "raw_annotations": [
    {
      "class_id": 1,
      "class_name": "line_segment",
      "bbox": [x0, y0, x1, y1],
      "obb": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "visibility": 1,
      "occluded": false,
      "text": null
    },
    {
      "class_id": 2,
      "class_name": "data_marker",
      "bbox": [x0, y0, x1, y1],
      "obb": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "visibility": 0,
      "occluded": true,
      "text": null
    }
  ],
  "bar_info": [
    {
      "bar_idx": 0,
      "series_idx": 1,
      "bottom": 0.0,
      "top": 45.2,
      "value": 45.2
    }
  ],
  "baselines": [
    {
      "id": "baseline_0",
      "y_pixel": 360.0,
      "x_range": [60.0, 540.0],
      "width": 480.0,
      "type": "primary"
    }
  ],
  "bars_with_baseline": [
    {
      "bar_index": 0,
      "series_idx": 1,
      "data_value": 45.2,
      "xyxy": [100.0, 120.0, 140.0, 360.0],
      "baseline_id": "baseline_0"
    }
  ],
  "keypoints": [
    [x, y, 1, "endpoint"],
    [x, y, 1, "peak"],
    [x, y, 0, "valley"]
  ],
  "line_topology": {
    "edges": [
      {"from": 0, "to": 1, "role_transition": "endpoint_to_peak"}
    ]
  },
  "amodal_polygon": [[x1, y1], [x2, y2], "..."],
  "modal_polygon": [[x1, y1], [x2, y2], "..."]
}
```

### 4. Categorical Keypoint Roles
For continuous series (lines, area contours), points are labeled with topological roles:
* `endpoint`: First or last vertex in series.
* `peak`: Local maximum ($\frac{dy}{dx} = 0, \frac{d^2y}{dx^2} < 0$).
* `valley`: Local minimum ($\frac{dy}{dx} = 0, \frac{d^2y}{dx^2} > 0$).
* `inflection`: Curvature zero-crossing ($\frac{d^2y}{dx^2} = 0$).
* `vertex`: General polyline segment knot.

### 5. Occlusion & Amodal Representation
* Elements covered $\ge 85\%$ by an opaque foreground element (such as a legend) have `"visibility": 0` and `"occluded": true`.
* Amodal ground-truth is retained: occluded marks are not culled from detection targets, preserving complete structure for GNN and layout models.
* For area series, `amodal_polygon` records the unoccluded data polygon, while `modal_polygon` records the post-occlusion visible shape.

---

## Version 2.1: Semantic-Coherent & Composite-Aware Schema (`v2.1`)

Introduced under Feature `002-semantic-storage-registry`. Schema Version `v2.1` (`dataset_version = "2.1.0"`) adds end-to-end domain tracking, multi-subplot composite figure representations, and eliminates semantic incoherence across synthetic charts while preserving 100% backward compatibility for visual element detection and graph classification models.

### 1. New Fields in Detailed Metadata (`<image_id>_detailed.json` and `<image_id>_unified.json`)

```json
{
  "chart_type": "bar",
  "semantic_domain": "biomedical",
  "is_scientific": true,
  "schema_version": "v2.1",
  "dataset_version": "2.1.0",
  "is_composite": false,
  "subplots": [
    {
      "chart_type": "bar",
      "semantic_domain": "biomedical",
      "is_scientific": true,
      "orientation": "vertical",
      "series_count": 2,
      "series_names": ["Control", "Treated"],
      "stacking_mode": null,
      "style": "side_by_side",
      "pattern": null,
      "dual_axis_info": {},
      "scale_axis_info": {"primary_scale_axis": "y"}
    }
  ],
  "composite_chart_types": ["bar"],
  "composite_domains": ["biomedical"]
}
```

### 2. Multi-Subplot Composite Tracking Rules
* **No Root Scalar Widening**: To prevent downstream visual classifiers from failing, root scalar `chart_type` and `semantic_domain` **strictly reflect the primary subplot (`axes[0]`)**. They never emit `"composite"` or `"mixed"`.
* **`is_composite` Flag**: Set to `true` if `len(subplots) > 1`, and `false` when single-plot.
* **`subplots` Array**: Captures per-axis metadata for every visible axis in the figure.
* **`composite_chart_types` & `composite_domains`**: Additive arrays listing the individual chart types and domains of all subplots in order.
* **`metadata_json["chart_types"]`**: Populated with `composite_chart_types` or falls back to `[chart_type]`.

---

## Version 3.0: Declarative Semantic Manifests & Procedural Purge (`v3.0`)

Introduced under Feature `003-legacy-decommissioning-declarative-manifests`. Schema Version `v3.0` (`dataset_version = "3.0.0"`) promotes dataset annotations to `v3.0`, migrates all semantic vocabulary to declarative YAML manifests under `synth/semantics/domains/`, aligns tabular engine presets directly to canonical domains (`business`, `engineering`), and completely purges legacy procedural vocabulary constants from `themes.py`.

### 1. Metadata Schema Versioning
Output metadata files (`<image_id>_detailed.json` and `<image_id>_unified.json`) now emit:
```json
{
  "schema_version": "v3.0",
  "dataset_version": "3.0.0"
}
```

### 2. Declarative Domain Manifests (`synth/semantics/domains/`)
All axis metric definitions, titles, and comparative pairs are authored in human-readable, schema-validated YAML files:
* `biomedical.yaml`: Clinical, pharmacokinetics, molecular biology, and physiological terms.
* `engineering.yaml`: Materials science, mechanics, control systems, computing, ML, and telemetry terms.
* `business.yaml`: Financial, SaaS, marketing, A/B testing, and operational terms.
* `demographic.yaml`: Population, geography, education, and demographic dimensions.
* `common.yaml`: Canonical independent variables (Date, Year, Month, Step, Depth, etc.), frequency/density metrics, and generic titles.

Each manifest is validated against the Pydantic v2 `DomainManifest` schema:
```yaml
domain_id: biomedical
display_name: Biomedical & Life Sciences
is_scientific: true
default_weight: 1.0
metrics:
  - label: "Plasma Concentration (ng/mL)"
    scale_type: continuous
    role: dependent
    concept_stem: plasma_concentration
    unit: ng/mL
    concept_group: pharmacokinetics
    domain_tags: [biomedical, pk]
titles:
  - title: "Pharmacokinetic Exposure Profile"
    allowed_chart_types: [line, scatter, bar]
    domain_tags: [biomedical]
comparative_pairs:
  - control: "Vehicle"
    treatment: "Test Article"
    domain_category: biomedical
```

### 3. Procedural Purge of Legacy Constants
All procedural vocabulary lists and legacy dictionaries in `themes.py` have been purged:
* Purged from `themes.py`: `SCIENTIFIC_X_LABELS`, `SCIENTIFIC_Y_LABELS`, `BUSINESS_X_LABELS`, `BUSINESS_Y_LABELS`, `CHART_TITLES`, `COMPARATIVE_LABELS`, `HISTOGRAM_Y_LABELS`, `SCIENTIFIC_DOMAIN_DICT`, `BUSINESS_DOMAIN_DICT`, `HEATMAP_*`, `COLORBAR_*`, `CONTEXT_CONFIGURATIONS`, `STRUCTURAL_THEMES`.
* Preserved in `themes.py`: Strictly visual styling profiles (`THEMES`, `PUBLICATION_THEMES`, `FONT_FAMILIES`).
* Tabular direct naming: `synth/tabular.py` directly references `DOMAIN_PRESETS["business"]` and `DOMAIN_PRESETS["engineering"]` with zero alias shims (`DOMAIN_ALIAS_TO_PRESET` and `PRESET_TO_CANONICAL` deleted).

---

## Version 4.0: Unified Single-List Annotations & Production Default (`v4.0`)

Schema Version `v4.0` (`dataset_version = "4.0.0"`) is the production default for the Matplotlib rendering engine (`annotation_schema_version = "v4.0"` in `config_defaults.py`). It replaces the multi-list legacy structure with a **single unified `annotations` list**, ensuring all visual elements, geometric bounding boxes, and domain/data attributes reside in a single consistent coordinate frame.

### 1. Key Design Principles
* **Single Annotations List (`annotations`)**: Every detected chart element (bars, line segments, markers, titles, axes labels, legends, baselines, data points) is represented as an entry in a flat list with unique integer `id`, `class_name`, `xyxy` bounding box in image pixel space, and visibility/occlusion flags. There are no parallel per-class dictionary keys (`bar`, `line_segment`, `chart_title`).
* **Pre-Effects Coordinate Frame**: All pixel coordinates are computed before realism effects and tracked through realism transforms (affine warps, perspective transforms, rotations, lighting), ensuring pixel ground truth remains aligned with final saved raster images (`.png`).
* **Attributes Pass-Through (`attrs`)**: Specialized metadata (e.g. `baseline_id`, `data_value`, `series_idx`, `keypoints`) is stored under an optional `attrs` mapping within each annotation rather than separate parallel arrays (`bars_with_baseline`, `bar_info`).
* **Subplot Architecture (`subplots`)**: Multi-axes and composite charts record per-axis configuration in `subplots` (one record per visible axes). Root scalars (`chart_type`, `semantic_domain`, `is_scientific`) reflect the primary subplot (`axes[0]`).
* **Image Dimensions (`image`)**: Top-level `image: {"width": W, "height": H}` explicitly documents canvas dimensions.

### 2. Output Format (`<image_id>_detailed.json`)

```json
{
  "schema_version": "v4.0",
  "dataset_version": "4.0.0",
  "image_id": "chart_00000",
  "image": {
    "width": 800,
    "height": 600
  },
  "chart_type": "bar",
  "semantic_domain": "business",
  "is_scientific": false,
  "is_composite": false,
  "composite_chart_types": [],
  "composite_domains": [],
  "subplots": [
    {
      "chart_type": "bar",
      "semantic_domain": "business",
      "is_scientific": false,
      "orientation": "vertical",
      "series_count": 2,
      "series_names": ["Q1", "Q2"],
      "stacking_mode": null,
      "style": "flat",
      "pattern": null,
      "dual_axis_info": {},
      "scale_axis_info": {"primary_scale_axis": "y"}
    }
  ],
  "annotations": [
    {
      "id": 0,
      "class_name": "bar",
      "xyxy": [120.0, 150.0, 160.0, 480.0],
      "visibility": 1,
      "occluded": false,
      "attrs": {
        "series_idx": 0,
        "bar_idx": 0,
        "data_value": 42.5,
        "baseline_id": 4
      }
    },
    {
      "id": 1,
      "class_name": "chart_title",
      "xyxy": [250.0, 30.0, 550.0, 65.0],
      "visibility": 1,
      "occluded": false,
      "text": "Quarterly Revenue Breakdown"
    },
    {
      "id": 2,
      "class_name": "x_axis_label",
      "xyxy": [380.0, 520.0, 460.0, 545.0],
      "visibility": 1,
      "occluded": false,
      "text": "Quarter"
    },
    {
      "id": 3,
      "class_name": "y_axis_label",
      "xyxy": [45.0, 270.0, 75.0, 360.0],
      "visibility": 1,
      "occluded": false,
      "text": "Revenue ($M)"
    },
    {
      "id": 4,
      "class_name": "baseline",
      "xyxy": [100.0, 479.0, 700.0, 481.0],
      "visibility": 1,
      "occluded": false,
      "attrs": {
        "type": "primary",
        "axis": "y"
      }
    }
  ]
}
```

### 3. Annotation Record Schema
Each element in `annotations` conforms to:
* `id` (`int`): Unique sequential index within the document.
* `class_name` (`str`): Target visual or semantic element class (e.g. `bar`, `line_segment`, `data_marker`, `chart_title`, `x_axis_label`, `y_axis_label`, `legend`, `baseline`, `data_point`).
* `xyxy` (`List[float]`): Bounding box `[x_min, y_min, x_max, y_max]` in final image pixel coordinates.
* `visibility` (`int`): `1` if visible, `0` if occluded $\ge 85\%$.
* `occluded` (`bool`): `true` if covered $\ge 85\%$ by foreground elements.
* `text` (`Optional[str]`): Text content for OCR targets (labels, titles, legend entries, tick marks).
* `obb` (`Optional[List[List[float]]]`): YOLOv26-OBB oriented bounding box vertices `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`.
* `attrs` (`Optional[Dict[str, Any]]`): Element-specific attributes (`baseline_id`, `data_value`, `series_idx`, `keypoints`, `line_topology`).
* `amodal_polygon` / `modal_polygon` (`Optional[List[List[float]]]`): Polygon boundary coordinates for area charts.

---

## Version 4.1: Filter Accounting & Baseline Hardening (`v4.1`)

Introduced under Feature `005-domain-gap-closure` (baseline-2).
Extends `v4.0` additively:
- `schema_version` is bumped from `"v4.0"` to `"v4.1"`.
- `dataset_version` is bumped from `"4.0.0"` to `"4.1.0"`.
- Adds top-level `filter_stats` dictionary recording discarded annotations by filter reason and class name:

```json
{
  "schema_version": "v4.1",
  "dataset_version": "4.1.0",
  "filter_stats": {
    "size": { "bar": 1 },
    "aspect": { "axis_labels": 2 },
    "viewport": {},
    "duplicate": { "axis_labels": 1 },
    "overlap": {}
  },
  "image": { "width": 800, "height": 600 },
  "chart_type": "bar",
  "annotations": [...]
}
```

### Filter Reason Taxonomy
* `size`: Bounding box width or height below minimum threshold (`MIN_BBOX_SIZE`).
* `aspect`: Bounding box aspect ratio exceeding `MAX_ASPECT_RATIO`.
* `viewport`: Bounding box or polygon clipped to degenerate area (< 4 px² or < 2 px in either dimension) at canvas boundaries.
* `duplicate`: Duplicate annotation detected during layout traversal, axis deduplication, or cross-stream merge.
* `overlap`: Same-class bounding boxes exceeding high-IoU suppression threshold (`iou_threshold >= 0.7`).


