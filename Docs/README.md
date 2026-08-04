# Synthetic Chart Generation Suite

**Location:** `src/train/gerador_charts/`

This directory contains a highly configurable, end-to-end synthetic data generation pipeline that produces realistic chart images paired with rich, multi-format annotations. The output is purpose-built for training computer vision models — specifically **YOLO object detection**, **YOLO pose estimation** (keypoints), **OCR models**, and **Graph Neural Networks (GNNs)**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [File Reference](#file-reference)
   - [chart-2.py — Core Rendering Engine](#chart-2py--core-rendering-engine)
   - [generator-3.py — Orchestrator & Annotation Extractor](#generator-3py--orchestrator--annotation-extractor)
   - [themes.py — Visual Styles & Text Vocabularies](#themespy--visual-styles--text-vocabularies)
   - [effects.py — Image Degradation Effects](#effectspy--image-degradation-effects)
   - [custom_config.py — Default Dataset Configuration](#custom_configpy--default-dataset-configuration)
   - [ocr_training_config.py — OCR-Specific Configuration](#ocr_training_configpy--ocr-specific-configuration)
   - [generate_whisker_training_data.py — Box Plot Regression Labels](#generate_whisker_training_datapy--box-plot-regression-labels)
   - [merge_json.py — JSON Unification](#merge_jsonpy--json-unification)
   - [package_training_data.py — Dataset Packaging](#package_training_datapy--dataset-packaging)
4. [Data Flow & Pipeline Stages](#data-flow--pipeline-stages)
5. [Annotation Format Specification](#annotation-format-specification)
   - [YOLO Object Detection Labels (.txt)](#yolo-object-detection-labels-txt)
   - [YOLO Pose Estimation Labels (.txt)](#yolo-pose-estimation-labels-txt)
   - [Keypoint Schemas Per Chart Type](#keypoint-schemas-per-chart-type)
   - [JSON Metadata Structure](#json-metadata-structure)
6. [Class Maps](#class-maps)
7. [Configuration Keys Reference](#configuration-keys-reference)
8. [Chart Types & Data Generation Patterns](#chart-types--data-generation-patterns)
9. [Keypoint Pipeline Deep Dive](#keypoint-pipeline-deep-dive)
10. [How to Modify the Pipeline](#how-to-modify-the-pipeline)
11. [Debugging Guide](#debugging-guide)
12. [Common Pitfalls & Known Gotchas](#common-pitfalls--known-gotchas)
13. [Dependencies](#dependencies)

---

## Architecture Overview

The pipeline operates in three sequential stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DATA & STYLE GENERATION                                  │
│  chart-2.py generates statistically realistic data (scientific or   │
│  business domain). themes.py supplies color palettes, fonts, grid   │
│  styles, and vocabulary for axis/chart titles.                      │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  matplotlib Figure + Axes objects
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — RENDERING & ANNOTATION EXTRACTION                        │
│  generator-3.py renders the figure in memory (Agg backend). After  │
│  drawing, it iterates over matplotlib Artist objects to extract     │
│  pixel-precise bounding boxes (get_window_extent()) and YOLO-pose  │
│  keypoints via arc-length or iterative bisection resampling.        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  PIL Image + annotation dicts
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — POST-PROCESSING & PACKAGING                              │
│  effects.py degrades the image (scan artifacts, blur, watermarks).  │
│  generator-3.py writes YOLO .txt label files. merge_json.py        │
│  consolidates per-image JSON outputs. package_training_data.py      │
│  bundles the final dataset for cloud training environments.         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
gerador_charts/
├── chart-2.py                        # Core matplotlib rendering engine
├── generator-3.py                    # Main orchestrator + annotation extractor
├── themes.py                         # Color palettes, fonts, vocabulary lists
├── effects.py                        # PIL/NumPy image degradation library
├── custom_config.py                  # Default generation configuration
├── ocr_training_config.py            # Alt config tuned for OCR training
├── generate_whisker_training_data.py # Box-plot Five-Number-Summary extractor
├── merge_json.py                     # JSON consolidation utility
├── package_training_data.py          # Dataset packaging for Colab/cloud
└── README.md                         # This file
```

---

## File Reference

### `chart-2.py` — Core Rendering Engine

**Purpose:** All chart drawing and domain-specific data generation live here. This file is stateless — it receives a config dict, draws onto a provided `matplotlib.axes.Axes` object, and returns annotated artist objects.

#### Public API (functions called by `generator-3.py`)

| Function | Signature | Description |
|---|---|---|
| `generate_realistic_data` | `(num_points, max_scale, allow_negative, pattern_type, domain)` | Generates statistically plausible data arrays |
| `apply_chart_theme` | `(ax, theme_name, orientation)` | Applies theme from `THEMES` dict to axes |
| `apply_typography_variation` | `(ax, domain)` | Randomizes fonts, sizes, and label rotation |
| `apply_axis_scaling` | `(ax, data_min, orientation, scale_type)` | Applies log/symlog/linear axis scaling |
| `_generate_bar_chart` | `(ax, style_config, theme_config, debug_mode)` | Draws bar chart, returns `(data_artists, bar_info_list, axis_related_artists, significance_artists)` |
| `_generate_line_chart` | `(ax, style_config, theme_config, debug_mode)` | Draws line chart, returns series keypoint data |
| `_generate_scatter_chart` | `(ax, style_config, theme_config, debug_mode)` | Draws scatter chart |
| `_generate_boxplot_chart` | `(ax, style_config, theme_config, debug_mode)` | Draws box plots with optional outliers |
| `_generate_heatmap_chart` | `(ax, style_config, theme_config, debug_mode)` | Draws heatmap with QuadMesh cells |
| `_generate_pie_chart` | `(ax, style_config, theme_config, debug_mode)` | Draws pie/donut chart with wedge artists |
| `_generate_area_chart` | `(ax, style_config, theme_config, debug_mode)` | Draws filled area chart (PolyCollection) |
| `_generate_histogram` | `(ax, style_config, theme_config, debug_mode)` | Draws histogram with configurable bins |
| `add_data_labels` | `(ax, bar_info_list, orientation)` | Annotates bars with numeric labels |
| `add_significance_markers` | `(ax, bar_info, y_max, orientation, error_tops)` | Adds `*`, `**`, `***`, `ns` brackets/letters |

#### `generate_realistic_data` — Pattern Reference

The function selects a pattern probabilistically based on the `domain` parameter.

**Scientific domain patterns** (default weights):

| Pattern | Weight | Model | Key Parameters |
|---|---|---|---|
| `dose_response` | 25% | Hill equation | `ec50 ∈ [-8.5, -4.5]`, `hill_slope ∈ [0.7, 3.5]` |
| `replicates` | 20% | Log-normal | CV: qPCR 5–15%, Western 10–25%, Cell assay 15–35% |
| `exponential_decay` | 15% | Pharmacokinetic decay | `half_life ∈ [0.5, 72]` hours |
| `power_law` | 10% | Power function | — |
| `sigmoid_growth` | 10% | Logistic | — |
| `linear_regression` | 10% | Linear + noise | CV: 5–15% |
| `gaussian_peak` | 5% | Gaussian + Poisson noise | `σ ∈ [5%, 20%]` of range |
| `enzyme_kinetics` | 5% | Michaelis-Menten | `Km ∈ [0.5, 50]`, substrate logspace |

**Business domain patterns** (default weights):

| Pattern | Weight | Model |
|---|---|---|
| `seasonal_trend` | 30% | Multi-component cosine + linear trend |
| `pareto_distribution` | 25% | Pareto distribution, sorted descending |
| `exponential_growth` | 15% | Logistic growth with carrying capacity |
| `market_saturation` | 15% | — |
| `random_walk_drift` | 10% | Cumulative random walk with drift |
| `step_intervention` | 5% | Plateau function |

> **Critical:** Noise is heteroscedastic (varies with data magnitude) for most scientific patterns, matching real instrument error characteristics. All data is post-processed to enforce measurement precision rounding (instrument precision is simulated by rounding to 0–3 decimal places based on `max_scale`).

#### `bar_info_list` Structure

Each entry is a `dict` that `generator-3.py` uses to locate bars for bounding-box and keypoint extraction:

```python
{
    "center":     float,  # Bar center position on primary axis (x for vertical, y for horizontal)
    "height":     float,  # Data value of this bar segment
    "width":      float,  # Bar width in data coordinates
    "bottom":     float,  # Bottom of this segment in data coords (0 for non-stacked; y_values1[i] for stacked top)
    "top":        float,  # Absolute top of this segment in data coords
    "series_idx": int,    # (optional) Index of the data series (0, 1, ...)
    "bar_idx":    int     # (optional) Index within the group
}
```

---

### `generator-3.py` — Orchestrator & Annotation Extractor

**Purpose:** The main entry point for dataset generation. Manages the full lifecycle: config loading → chart generation → annotation extraction → effect application → file writing.

#### Top-Level Execution Flow

```
generate_dataset(config)
    └── for each image:
            ├── select_chart_type(config)
            ├── select_theme_and_style(config)
            ├── fig, ax = plt.subplots(...)
            ├── draw_chart(ax, chart_type, style_config, theme_config)  ← calls chart-2.py
            ├── fig.canvas.draw()  ← forces renderer to calculate bounding boxes
            ├── extract_annotations(fig, ax, artists)
            │       ├── extract_bounding_boxes()          ← get_window_extent()
            │       ├── extract_pose_keypoints()           ← chart-type specific
            │       ├── filter_overlapping_annotations()
            │       └── add_unique_annotation()
            ├── apply_effects(pil_image, config)           ← calls effects.py
            ├── write_yolo_label_file(annotations)
            ├── write_json_metadata(annotations)
            └── plt.close(fig)
```

#### Important Internal Functions

| Function | Description |
|---|---|
| `validate_coordinates(coords, context)` | Validates coord lists for finite values and correct tuple structure. Controlled by `debug_coords` config flag. |
| `verify_pose_format(annotations, context)` | Verifies full pose annotation dicts: checks `class_id`, `bbox` (4 values, normalized), `keypoints` (51 or 5 per type), visibility flags (0 or 2 only). |
| `resample_keypoints(points, target_count)` | Arc-length parameterized resampling using `scipy.interpolate.interp1d`. Used for downsampling or smooth upsampling. |
| `resample_keypoints_adaptive(points, target, anchors_idx)` | Curvature-weighted arc-length resampling. Preserves semantically important anchor points (peaks, valleys, inflections). |
| `resample_keypoints_iterative(points, target)` | Max-heap based iterative bisection of the longest segment. **Primary upsampler** — guarantees all original points are preserved. |
| `build_51_from_plotted(points)` | Master function that selects upsampling vs. downsampling strategy and returns exactly 51 `(x, y, idx)` tuples sorted left-to-right. |
| `pad_keypoints(keypoints, target_count, pad_value)` | Pads a keypoint list to `target_count` with invisible keypoints `(0.0, 0.0, 0)`. |
| `order_left_to_right(points)` | Sorts `(x, y, idx)` points by x ascending. |
| `curvature_importance(points)` | Returns per-vertex importance scores (turn angle + arc length). Used for importance-weighted downsampling. |
| `filter_overlapping_annotations(annotations, iou_threshold)` | Removes duplicate/overlapping annotations using IoU. |
| `add_unique_annotation(annotations_list, new_ann)` | Adds an annotation only if it doesn't duplicate an existing one (by class + spatial proximity). |

#### Keypoint Data Structures (Internal)

Points are stored as `List[Tuple[float, float, int]]` where the `int` is the original draw index for traceability through resampling operations.

**`KeypointConfig` dataclass:**
```python
@dataclass
class KeypointConfig:
    num_keypoints: int
    skeleton: List[Tuple[int, int]]  # Keypoint connection pairs for YOLO skeleton
    keypoint_names: List[str]
```

Three global configs are defined:
- `LINE_KEYPOINT_CONFIG` — 51 keypoints
- `AREA_KEYPOINT_CONFIG` — 51 keypoints
- `PIE_KEYPOINT_CONFIG` — 17 keypoints (1 center + 1 wedge center + 15 arc points)

> **Note:** The active YOLO pose format uses **51 keypoints for line/area** and **5 keypoints for pie**. The `PIE_KEYPOINT_CONFIG` dataclass defines 17 but the active YOLO label writer uses 5. Verify `verify_pose_format()` for the currently enforced count.

#### `GENERATION_CONFIG` Dictionary

This dict is the live configuration object inside `generator-3.py`. It is populated at startup from either `custom_config.py` or `ocr_training_config.py`. See [Configuration Keys Reference](#configuration-keys-reference) for full documentation.

---

### `themes.py` — Visual Styles & Text Vocabularies

**Purpose:** Centralized library of all visual styling and all text content used in generated charts. Changing anything here affects the appearance and label vocabulary across the entire dataset.

#### Exported Constants

| Constant | Type | Description |
|---|---|---|
| `THEMES` | `dict` | Keyed by theme name; each entry defines `facecolor`, `grid_color`, `grid_style`, `grid_linewidth`, `font`, `spines` (dict of visibility), `spine_width`, `tick_direction`, `palette` |
| `SCIENTIFIC_Y_LABELS` | `list[str]` | Y-axis titles for scientific charts (e.g., "Relative Expression (A.U.)") |
| `SCIENTIFIC_X_LABELS` | `list[str]` | X-axis titles for scientific charts |
| `BUSINESS_Y_LABELS` | `list[str]` | Y-axis titles for business charts |
| `BUSINESS_X_LABELS` | `list[str]` | X-axis titles for business charts |
| `COMPARATIVE_LABELS` | `list[str]` | Labels used in grouped/comparative bar charts |
| `HISTOGRAM_Y_LABELS` | `list[str]` | Y-axis labels specific to histograms |
| `CHART_TITLES` | `list[str]` | Pool of chart title strings |
| `FONT_FAMILIES` | `dict` | Keys: `'sans-serif'`, `'serif'`; values: lists of font name strings |
| `HEATMAP_XLABELS_SCIENTIFIC` | `list[str]` | Heatmap column labels for scientific domain |
| `HEATMAP_YLABELS_SCIENTIFIC` | `list[str]` | Heatmap row labels for scientific domain |
| `HEATMAP_XLABELS_BUSINESS` | `list[str]` | Heatmap column labels for business domain |
| `HEATMAP_YLABELS_BUSINESS` | `list[str]` | Heatmap row labels for business domain |
| `COLORBAR_TITLES_SCIENTIFIC` | `list[str]` | Colorbar axis labels for scientific heatmaps |
| `COLORBAR_TITLES_BUSINESS` | `list[str]` | Colorbar axis labels for business heatmaps |
| `HEATMAP_CHART_TITLES` | `list[str]` | Chart title strings specific to heatmaps |
| `HEATMAP_ANNOTATION_FORMATS` | `list[str]` | Format strings for heatmap cell annotations |

#### Available Theme Names

Built-in themes include: `ggplot`, `excel`, `prism`, `minimal`, `dark`, `scientific_white`, `nature`, `default`. Each defines a complete visual profile. To add a new theme, add a new key to the `THEMES` dict with the required sub-keys.

---

### `effects.py` — Image Degradation Effects

**Purpose:** Applies realistic visual degradation to clean matplotlib renders to simulate real-world acquisition scenarios (scans, screen captures, print-outs, low-quality web images).

**Dependencies:** `Pillow (PIL)`, `NumPy`, optionally `OpenCV (cv2)`.

#### Available Effects

| Function | Description | Key Parameters |
|---|---|---|
| `apply_jpeg_compression_effect` | Re-encodes image at low JPEG quality | `quality ∈ [20, 85]` |
| `apply_noise_effect` | Adds Gaussian or salt-and-pepper noise | `noise_type`, `intensity` |
| `apply_blur_effect` | Gaussian blur | `radius` |
| `apply_motion_blur_effect` | Directional motion blur kernel | `angle`, `distance` |
| `apply_low_res_effect` | Downscale then upscale (pixelation) | `scale_factor` |
| `apply_pixelation_effect` | Block pixelation | `block_size` |
| `apply_posterize_effect` | Reduces bit depth | `bits` |
| `apply_color_variation_effect` | Hue/saturation shift | `hue_shift`, `saturation_factor` |
| `apply_ui_chrome_effect` | Adds simulated browser/app chrome around chart | — |
| `apply_watermark_effect` | Semi-transparent text watermark overlay | `text`, `opacity` |
| `apply_vignette_effect` | Dark radial gradient vignette | `intensity` |
| `apply_scanner_streaks_effect` | Horizontal/vertical scanner artifact lines | `num_streaks`, `intensity` |
| `apply_clipping_effect` | Randomly crops image edges | `margin` |
| `apply_printing_artifacts_effect` | Banding, halftoning, registration errors | — |
| `apply_mouse_cursor_effect` | Overlays a simulated cursor icon | `position` |
| `apply_text_degradation_effect` | Blurs/distorts text regions specifically | `intensity` |
| `apply_grid_occlusion_effect` | Adds rectangular occlusion patches | `num_patches` |
| `apply_scan_rotation_effect` | Slight rotation with white/gray fill | `max_angle` |
| `apply_grayscale_effect` | Converts to grayscale (RGB values equal) | — |
| `apply_perspective_effect` | Perspective warp | `intensity` |

> **To add a new effect:** Define a function `apply_<name>_effect(image: PIL.Image, **kwargs) -> PIL.Image`. Then register it in the `effect_function_map` dictionary inside `generator-3.py` and add it to the config probability dict in `custom_config.py`.

---

### `custom_config.py` — Default Dataset Configuration

See [Configuration Keys Reference](#configuration-keys-reference) for all keys.

---

### `ocr_training_config.py` — OCR-Specific Configuration

An alternate config that overrides `custom_config.py` for OCR model training. Key differences:
- `force_numeric_labels: True` — forces axis tick labels to be numeric
- Dramatically increased probabilities for text-degrading effects: `apply_blur_effect`, `apply_jpeg_compression_effect`, `apply_pixelation_effect`, `apply_low_res_effect`
- Reduced chart variety to maximize text-heavy layouts

**Usage:** In your entry point, replace the import of `custom_config` with `ocr_training_config`.

---

### `generate_whisker_training_data.py` — Box Plot Regression Labels

**Purpose:** Post-processes generated boxplot images to extract the **Five-Number Summary** as numerical regression targets for a specialized `WhiskerRegressionNet` model.

#### Extracted Values Per Box

```python
{
    "q1":            float,  # First quartile Y coordinate (normalized)
    "q3":            float,  # Third quartile Y coordinate (normalized)
    "median":        float,  # Median line Y coordinate (normalized)
    "upper_whisker": float,  # Upper whisker tip Y coordinate (normalized)
    "lower_whisker": float,  # Lower whisker tip Y coordinate (normalized)
    "outliers":      list[float]  # Y coordinates of outlier markers (normalized)
}
```

**Implementation detail:** Values are extracted by inspecting `matplotlib.patches.PathPatch` vertices from the rendered box plot. If you change the drawing implementation of boxes, whiskers, or outlier markers in `chart-2.py`, you **must** update `extract_five_number_summary()` here to match the new artist structure.

---

### `merge_json.py` — JSON Unification

**Purpose:** The generation pipeline writes multiple JSON fragments per image (metadata, OCR output, spatial annotation details). `merge_json_files()` consolidates these into a single `<image_name>.json` per image.

#### Unified JSON Schema

```json
{
  "image_id":      "string",
  "image_path":    "string",
  "chart_type":    "string",
  "theme":         "string",
  "domain":        "scientific | business",
  "width_px":      "int",
  "height_px":     "int",
  "annotations":   [ ... ],  // from YOLO label file, reprojected to absolute pixel coords
  "ocr":           { ... },  // text and bounding boxes for each text element
  "detailed":      { ... }   // raw spatial metadata from generator
}
```

**Usage:** Call `batch_merge_all(output_dir)` after generation completes.

---

### `package_training_data.py` — Dataset Packaging

**Purpose:** Filters the raw output directory to extract only the subset of data required for GNN and keypoint model training, then zips it for upload.

**Key function:** `create_gnn_training_sample(image_id, annotations)` — Extracts bar-to-baseline node/edge connections for GNN training. Edit this function if your GNN architecture requires different graph topologies (e.g., bar-to-data-label edges, or bar-to-axis-label edges).

---

## Data Flow & Pipeline Stages

```
custom_config.py
       │
       ▼
generator-3.py  ──────────► chart-2.py ──────────► themes.py
       │                       │
       │                 matplotlib Figure
       │                 + artist metadata
       │                       │
       ▼                       ▼
  effects.py             Annotation Extraction
  (PIL Image)            (bbox + keypoints)
       │                       │
       └──────────────┬────────┘
                      ▼
              YOLO .txt label files
              Per-image JSON fragments
                      │
                      ▼
               merge_json.py
              (unified .json files)
                      │
                      ▼
          package_training_data.py
              (training .zip)
```

---

## Annotation Format Specification

### YOLO Object Detection Labels (`.txt`)

One `.txt` file per image. Each line is one annotation:

```
<class_id> <cx> <cy> <w> <h>
```

All values are **normalized to [0.0, 1.0]** relative to image width/height. `cx`, `cy` are the bounding box center.

### YOLO Pose Estimation Labels (`.txt`)

One `.txt` file per image. Each line is one annotation:

```
<class_id> <cx> <cy> <w> <h> <kx1> <ky1> <vis1> <kx2> <ky2> <vis2> ... <kxN> <kyN> <visN>
```

- `cx, cy, w, h` — normalized bounding box (same as object detection)
- `kxN, kyN` — normalized keypoint coordinate [0.0, 1.0]
- `visN` — visibility flag: `0` = not labeled/padded, `2` = visible and labeled

### Keypoint Schemas Per Chart Type

#### Line Chart & Area Chart — 51 Keypoints

```
Index 0:      start          (leftmost point of the line)
Index 1–25:   boundary_0 ... boundary_24  (25 evenly distributed path points)
Index 26–45:  inflection_0 ... inflection_19 (up to 20 inflection points, padded with vis=0 if fewer)
Index 46:     peak_1
Index 47:     peak_2
Index 48:     valley_1
Index 49:     valley_2
Index 50:     end            (rightmost point of the line)
```

**Ordering:** Final keypoints are sorted left-to-right by x coordinate after resampling.

**Monotonicity:** x coordinates of visible keypoints are expected to be non-decreasing. `verify_pose_format()` logs a warning (not error) if this is violated, since axes can be inverted.

#### Area Chart — 51 Keypoints

```
Index 0:      start
Index 1–25:   top_0 ... top_24    (25 points along the top fill boundary)
Index 26–49:  bottom_0 ... bottom_23 (24 points along the bottom boundary)
Index 50:     end
```

Skeleton connects top boundary as a chain and bottom boundary as a separate chain.

#### Pie Chart — 5 Keypoints (active format)

```
Index 0: center       (center of the full pie)
Index 1: wedge_center (centroid of this specific wedge/slice)
Index 2: arc_start    (start of the arc boundary)
Index 3: arc_mid      (midpoint of the arc)
Index 4: arc_end      (end of the arc boundary)
```

> Note: `PIE_KEYPOINT_CONFIG` in the code defines 17 keypoints but the active YOLO label writer enforces **5** keypoints per pie slice. `verify_pose_format()` explicitly accepts both 51 and 5 as valid counts.

### JSON Metadata Structure

Each image produces (before merging) the following JSON types:
- `<id>_metadata.json` — theme, domain, chart type, dimensions, generation seed
- `<id>_ocr.json` — all text elements with pixel bounding boxes and string content
- `<id>_detailed.json` — raw annotation data before normalization, bar_info_list, keypoint arrays

After `merge_json.py`, all three are merged into `<id>.json`.

---

## Class Maps

These map integer class IDs to semantic category names for each YOLO task. All are defined in `GENERATION_CONFIG` inside `generator-3.py` (and mirrored in `custom_config.py`).

### Bar Chart (`CLASS_MAP_BAR`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | bar |
| 2 | axis_title |
| 3 | significance_marker |
| 4 | error_bar |
| 5 | legend |
| 6 | chart_title |
| 7 | data_label |
| 8 | axis_labels |

### Pie Chart — Object (`CLASS_MAP_PIE_OBJ`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | wedge |
| 2 | legend |
| 3 | chart_title |
| 4 | data_label |
| 5 | connector_line |

### Pie Chart — Pose (`CLASS_MAP_PIE_POSE`)
| ID | Class |
|---|---|
| 0 | slice_boundary |

### Line Chart — Object (`CLASS_MAP_LINE_OBJ`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | line_segment |
| 2 | axis_title |
| 3 | legend |
| 4 | chart_title |
| 5 | data_label |
| 6 | axis_labels |

### Line Chart — Pose (`CLASS_MAP_LINE_POSE`)
| ID | Class |
|---|---|
| 0 | line_boundary |

### Scatter Chart (`CLASS_MAP_SCATTER`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | data_point |
| 2 | axis_title |
| 3 | significance_marker |
| 4 | error_bar |
| 5 | legend |
| 6 | chart_title |
| 7 | data_label |
| 8 | axis_labels |

### Box Plot (`CLASS_MAP_BOX`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | box |
| 2 | axis_title |
| 3 | significance_marker |
| 4 | range_indicator |
| 5 | legend |
| 6 | chart_title |
| 7 | median_line |
| 8 | axis_labels |
| 9 | outlier |

### Histogram (`CLASS_MAP_HISTOGRAM`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | bar |
| 2 | axis_title |
| 3 | legend |
| 4 | chart_title |
| 5 | data_label |
| 6 | axis_labels |

### Heatmap (`CLASS_MAP_HEATMAP`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | cell |
| 2 | axis_title |
| 3 | color_bar |
| 4 | legend |
| 5 | chart_title |
| 6 | data_label |
| 7 | axis_labels |
| 8 | significance_marker |

### Area Chart — Object (`CLASS_MAP_AREA_OBJ`)
| ID | Class |
|---|---|
| 0 | chart |
| 1 | axis_title |
| 2 | legend |
| 3 | chart_title |
| 4 | data_label |
| 5 | axis_labels |

### Area Chart — Pose (`CLASS_MAP_AREA_POSE`)
| ID | Class |
|---|---|
| 0 | area_boundary |

---

## Configuration Keys Reference

All keys live in the top-level `GENERATION_CONFIG` dict (in `generator-3.py`) or equivalently in `custom_config.py` / `ocr_training_config.py`.

| Key | Type | Default | Description |
|---|---|---|---|
| `debug_mode` | `bool` | `False` | Enables verbose generation logs |
| `debug_annotations` | `bool` | `False` | Logs detailed annotation processing steps |
| `debug_artists` | `bool` | `False` | Logs per-artist processing |
| `debug_coords` | `bool` | `False` | Logs every coordinate transformation from data → pixel → normalized |
| `num_images` | `int` | `20` | Number of images to generate per run |
| `output_dir` | `str` | `"train"` | Output directory for images and labels |
| `seed` | `int` | `42` | Global random seed for reproducibility |
| `scenario_weights` | `dict` | `{"single": 100, "overlay": 0, "multi": 0}` | Weights for single-panel, overlay, and multi-panel layouts |
| `CLASS_MAP_BAR` | `dict` | see above | ID-to-name mapping for bar chart classes |
| `CLASS_MAP_PIE_OBJ` | `dict` | see above | ID-to-name mapping for pie object classes |
| `CLASS_MAP_PIE_POSE` | `dict` | see above | ID-to-name mapping for pie pose classes |
| `CLASS_MAP_LINE_OBJ` | `dict` | see above | ID-to-name mapping for line object classes |
| `CLASS_MAP_LINE_POSE` | `dict` | see above | ID-to-name mapping for line pose classes |
| `CLASS_MAP_SCATTER` | `dict` | see above | ID-to-name mapping for scatter classes |
| `CLASS_MAP_BOX` | `dict` | see above | ID-to-name mapping for box plot classes |
| `CLASS_MAP_HISTOGRAM` | `dict` | see above | ID-to-name mapping for histogram classes |
| `CLASS_MAP_HEATMAP` | `dict` | see above | ID-to-name mapping for heatmap classes |
| `CLASS_MAP_AREA_OBJ` | `dict` | see above | ID-to-name mapping for area chart object classes |
| `CLASS_MAP_AREA_POSE` | `dict` | see above | ID-to-name mapping for area chart pose classes |

### Chart Type Weights

Configured under the `"chart_types"` key as a dict of `{chart_name: {"weight": int}}`. Example:

```python
"chart_types": {
    "bar":       {"weight": 60},
    "line":      {"weight": 20},
    "scatter":   {"weight": 10},
    "pie":       {"weight": 5},
    "box":       {"weight": 5},
    "heatmap":   {"weight": 0},
    "area":      {"weight": 0},
    "histogram": {"weight": 0},
}
```

To generate **only bar charts**, set `"bar": {"weight": 100}` and all others to `0`.

### Effect Probabilities

Each effect is gated by a probability float (0.0–1.0) in the config. Example structure:

```python
"effects": {
    "jpeg_compression": {"prob": 0.4, "quality_range": [30, 75]},
    "blur":             {"prob": 0.3, "radius_range": [0.5, 2.0]},
    "scan_rotation":    {"prob": 0.2, "max_angle": 3.0},
    "watermark":        {"prob": 0.1},
    ...
}
```

---

## Chart Types & Data Generation Patterns

| Chart Type | Styles Available | Orientation | Domain |
|---|---|---|---|
| Bar | `default`, `side_by_side`, `stacked`, `touching`, `3d_effect`, `scientific` (hatched grouped) | vertical / horizontal | scientific + business |
| Line | single series, multi-series, with/without markers, step | — | scientific + business |
| Scatter | with/without trend line, error bars, significance markers | — | scientific |
| Box Plot | standard, notched, with outliers, with significance brackets | vertical | scientific |
| Heatmap | annotated cells, colorbar, significance markers | — | scientific + business |
| Pie | standard, donut; label strategies: `default_leader`, `outside_pct_only`, `inside_pct_only`, `none` | — | business |
| Area | stacked, filled single series | — | business |
| Histogram | standard bins, KDE overlay | vertical | scientific |

---

## Keypoint Pipeline Deep Dive

The keypoint extraction pipeline converts matplotlib artist pixel coordinates into normalized YOLO pose keypoints. Understanding this is critical before modifying any chart element.

### Step 1 — Raw Point Extraction

After `fig.canvas.draw()`, artist objects are inspected to extract pixel coordinates. Different artist types require different extraction methods:

- `matplotlib.lines.Line2D` → `line.get_xydata()` (returns data coords) → transformed via `ax.transData`
- `matplotlib.patches.PathPatch` (PolyCollection for areas) → `path.vertices`
- `matplotlib.patches.Wedge` (pie slices) → bounding box + arc angle computation
- `matplotlib.patches.Rectangle` (bars) → `get_xy()`, `get_width()`, `get_height()`

### Step 2 — Coordinate Transformation

Data coordinates → display (pixel) coordinates → normalized [0,1]:

```python
# Data → pixel
display_coords = ax.transData.transform(data_coords)

# Pixel → normalized (YOLO format, y-axis flipped)
norm_x = pixel_x / fig_width_px
norm_y = 1.0 - (pixel_y / fig_height_px)  # CRITICAL: y is flipped for YOLO
```

### Step 3 — Resampling to Fixed Count

The pipeline must produce **exactly 51 keypoints** for line/area charts. The `build_51_from_plotted()` function handles this:

```
n < 51  →  resample_keypoints_iterative()   (bisects longest segment, preserves originals)
n > 51  →  curvature_importance() downsampling (keeps high-curvature points, always keeps endpoints)
n == 51 →  order_left_to_right() only
```

### Step 4 — Anchor Preservation

`resample_keypoints_adaptive()` accepts an `anchors_idx` list of original point indices that must appear in the output. These are sourced from `_collect_anchors_from_series()`, which collects indices of peaks, valleys, and inflection points detected by `scipy.signal.find_peaks`.

### Step 5 — Validation

`verify_pose_format()` is called in debug mode (`debug_coords: True`) and checks:
- `bbox` has exactly 4 values, all normalized
- `keypoints` has exactly 51 or 5 entries
- Each keypoint has exactly 3 elements `(x, y, vis)`
- Visibility is `0` or `2` only
- Visible keypoint coordinates are within [0, 1]
- x-coordinates of visible points are monotonically non-decreasing (warning only)

---

## How to Modify the Pipeline

### Adding a New Chart Type (e.g., Radar Chart)

1. **`chart-2.py`** — Add `_generate_radar_chart(ax, style_config, theme_config, debug_mode)`. Return `(data_artists, metadata_dict)`.
2. **`generator-3.py`** — Add a branch in the chart dispatch logic. Implement keypoint extraction for radar chart artists. Define a `CLASS_MAP_RADAR` dict.
3. **`custom_config.py`** — Add `"radar": {"weight": N}` to `chart_types` and define `CLASS_MAP_RADAR`.
4. **`themes.py`** — Optionally add radar-specific title vocabulary.

### Adding a New Annotated Visual Element (e.g., Reference Line)

1. **`chart-2.py`** — Draw the line using `ax.axhline()` or `ax.plot()`. Store the returned `Line2D` artist in a returned list.
2. **`generator-3.py`** — In the annotation extraction loop for that chart type, intercept the `Line2D` artist, call `artist.get_window_extent(renderer)` to get the bounding box, and call `add_unique_annotation(annotations, new_ann)` with the appropriate new `class_id`.
3. **`custom_config.py`** — Add the new class name and ID to the relevant `CLASS_MAP_*` dict. **Note:** All class IDs within a given map must be contiguous from 0. Adding a new class changes the total class count — update any YOLO training `.yaml` file accordingly.

### Changing Chart Type Distribution

Edit the `"chart_types"` weights in `custom_config.py`:
```python
"chart_types": {
    "bar":  {"weight": 100},  # 100% bar charts
    "line": {"weight": 0},
    ...
}
```

### Adding a New Image Degradation Effect

1. **`effects.py`** — Define `apply_coffee_stain_effect(image: PIL.Image, intensity: float) -> PIL.Image`.
2. **`generator-3.py`** — Add `"coffee_stain": apply_coffee_stain_effect` to `effect_function_map`.
3. **`custom_config.py`** — Add `"coffee_stain": {"prob": 0.05, "intensity": 0.3}` to the effects config block.

### Adding a New Text Vocabulary Domain (e.g., Medical)

1. **`themes.py`** — Add `MEDICAL_Y_LABELS`, `MEDICAL_X_LABELS` lists with appropriate terminology.
2. **`chart-2.py`** — In the title/label selection logic, add a `'medical'` branch that samples from the new lists.
3. **`custom_config.py`** — Add `"domain_weights": {"scientific": 30, "business": 30, "medical": 40}`.

### Generating OCR Training Data

Replace `custom_config.py` import with `ocr_training_config.py` in your entry point. This automatically:
- Increases text-degradation effect probabilities
- Forces numeric axis labels for digit recognition training
- Increases image resolution variation

---

## Debugging Guide

### Enable Debug Logging

In `GENERATION_CONFIG` (or your config file):
```python
"debug_mode": True,         # General pipeline logs
"debug_annotations": True,  # Annotation processing details
"debug_artists": True,      # Per-artist inspection logs
"debug_coords": True,       # Full coordinate transformation trace
```

### Debug Log Prefixes

| Prefix | Source Function | What It Shows |
|---|---|---|
| `DEBUG [COORD-VALIDATION]` | `validate_coordinates()` | Per-point type/finiteness checks |
| `DEBUG [POSE-VERIFICATION]` | `verify_pose_format()` | Per-annotation bbox/keypoint format checks |
| `DEBUG [UPSAMPLE-VERIFY-ITERATIVE]` | `build_51_from_plotted()` | Upsampling preservation stats |
| `DEBUG:` (general) | Various | Chart type, style, orientation, scale choices |

### Common Debug Scenarios

**Problem: Annotations appear offset from visual elements**
→ Enable `debug_coords`. Check that `fig.canvas.draw()` is called before `get_window_extent()`. The renderer must be initialized for bbox extraction to work.

**Problem: Wrong number of keypoints in output file**
→ Enable `debug_coords` and look for `[POSE-VERIFICATION]` lines reporting unexpected keypoint counts. Check the `build_51_from_plotted()` return value for that chart type.

**Problem: Duplicate annotations in output**
→ Lower the IoU threshold in `filter_overlapping_annotations()` or add stricter conditions to `add_unique_annotation()`.

**Problem: Empty/zero bounding boxes**
→ An artist may not be rendered (e.g., empty legend, zero-width bar). `get_window_extent()` returns a zero-size box. The pipeline should already filter these; check the `min_bbox_area` threshold in the filtering logic.

---

## Common Pitfalls & Known Gotchas

1. **`fig.canvas.draw()` must be called before bbox extraction.** The Agg renderer is lazy — bounding boxes are only computed after an explicit draw call. This is already handled in the main loop but is easy to break if you restructure the generation order.

2. **Y-axis is flipped in YOLO coordinates.** matplotlib uses bottom-left origin; YOLO uses top-left origin. The conversion `norm_y = 1.0 - (pixel_y / fig_height_px)` is applied in `generator-3.py`. Any new coordinate extraction code must include this flip.

3. **Stacked bars have non-zero `bottom`.** The `bar_info_list` `"bottom"` field is critical for correct bounding box height of stacked segments. A stacked top segment has `bottom = y_values1[i]`, not `0`. Incorrectly assuming `bottom=0` will produce bounding boxes that span from the image baseline instead of from the segment base.

4. **`resample_keypoints_iterative` sorts input by `p[2]` (original draw index).** If draw indices are not monotonically increasing along the path (e.g., if you add points in a non-sequential order), the resampler will produce incorrect interpolations.

5. **Class IDs must be contiguous from 0 within each CLASS_MAP.** YOLO requires contiguous class IDs starting at 0. Gaps will cause errors during training. When adding a new class, always use `max(existing_ids) + 1`.

6. **Pie chart keypoints use 5, not 17.** `PIE_KEYPOINT_CONFIG` defines 17 but the active label writer enforces 5. If you change this, update `verify_pose_format()` to accept the new count.

7. **Horizontal bar charts swap x/y semantics.** In `bar_info_list`, `"height"` always refers to the data value (not necessarily the pixel height), and `"center"` refers to the position along the categorical axis. For horizontal bars, `"center"` is a y-coordinate and `"height"` maps to x-extent. The significance marker code in `chart-2.py` handles this with an `orientation` branch.

8. **`apply_axis_scaling` with `log` scale will silently fall back to `symlog`** if any data value ≤ 0. This is by design to prevent crashes, but can produce unexpected scale behavior. Check `data_min` before calling.

9. **`batch_merge_all()` in `merge_json.py` will fail gracefully if the import fails.** The `try/except ImportError` in `generator-3.py` sets `batch_merge_all = None`. Always check that `merge_json.py` is present in the same directory before running packaging steps.

10. **Matplotlib `rcParams` font changes are global.** `apply_chart_theme()` modifies `rcParams['font.sans-serif']` which persists across figures in the same process. If you generate multiple chart types in the same process, ensure themes are re-applied per figure or explicitly reset `rcParams` between runs.

---

## Dependencies

| Library | Version | Usage |
|---|---|---|
| `matplotlib` | ≥3.6 | Chart rendering (Agg backend, `use('Agg')` set globally) |
| `numpy` | ≥1.23 | Numerical data generation, coordinate math |
| `scipy` | ≥1.9 | Signal processing (`find_peaks`), interpolation (`interp1d`), statistics |
| `Pillow (PIL)` | ≥9.0 | Image manipulation, effects |
| `opencv-python` | ≥4.6 (optional) | Advanced effects (falls back to PIL if absent) |
| `Python` | ≥3.9 | `dataclasses`, `typing`, `collections.namedtuple` |

All imports at the top of `generator-3.py` are guarded with `try/except` for optional dependencies (notably `cv2`). The pipeline is fully functional without OpenCV, with some effects falling back to PIL equivalents.

---

*This README was auto-generated from source analysis of `chart-2.py` and `generator-3.py`. Keep it updated whenever new chart types, class maps, configuration keys, or keypoint schemas are added.*
