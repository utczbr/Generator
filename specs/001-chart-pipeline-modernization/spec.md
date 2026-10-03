# Feature Specification: Vector-Faithful Annotation & Rendering Pipeline

**Feature Branch**: `001-chart-pipeline-modernization`
**Status**: Complete — P0, P1, and P2 Completed and Verified (43/43 Tests Passing)
**Created**: 2026-09-17
**Input**: Engineering audit of the synthetic chart-generation pipeline (`chart.py`, `generator.py`, `effects.py`, `themes.py`), corroborated against source in `research.md`
**Primary artifacts touched**: `generator.py`, `effects.py`, `chart.py` (read-mostly), `themes.py` (P2 only)

## Overview

The pipeline generates synthetic chart images with pixel-level ground-truth annotations to train visual chart-understanding models (object detectors, instance segmenters, GNN-based structure parsers, VLMs). `chart.py` already implements statistically rich, literature-grounded data synthesis (Hill/Michaelis-Menten curves, AR(1) time series, compositional ALR transforms, spectral matrix seriation — see `research.md` §4). The gap is downstream of that: the annotation-extraction and realism-augmentation layer (`generator.py`, `effects.py`) does not always preserve geometric or semantic truth between what is drawn and what is labeled. This feature closes that gap in three priority waves (P0/P1/P2) without touching the statistical generators in `chart.py`.

## User Scenarios & Testing *(mandatory)*

### Primary User Story

As an ML engineer training a chart-element detector, segmenter, or graph-structure parser on this synthetic dataset, I need every emitted annotation — bounding box, polygon, keypoint, or series/baseline link — to be an exact description of what is actually rasterized, including after realism augmentation, so that supervised loss (GIoU/CIoU, mask IoU, keypoint OKS, GNN edge loss) trains against clean labels rather than noise the generator itself introduced.

### Acceptance Scenarios

1. **Given** a chart element with an axis-aligned annotation, **When** a rotation, perspective, or homographic realism effect is applied to the rendered image, **Then** the exported box for that element is the tightest quadrilateral containing only the transformed element's true pixels — not an axis-aligned box expanded by the transform.
2. **Given** any realism effect that geometrically warps the image (including ones added in the future), **When** that effect fires, **Then** every existing annotation for that image is compensated by the same transform — no effect can silently leave annotations untouched while warping pixels.
3. **Given** an axis tick or axis label, **When** the pipeline decides whether to keep or drop that label as an annotation, **Then** the decision is made analytically from the label's text, visibility, and axis-limit state — not by rasterizing the figure and sampling pixel colors.
4. **Given** a grouped or stacked bar chart with two or more series, **When** bar-level annotations are exported, **Then** each bar's `series_idx` correctly identifies which series it belongs to.
5. **Given** a chart rendered on a secondary (twinx) axis, **When** bars are linked to a baseline for GNN supervision, **Then** each bar links to the baseline of the axis it was actually plotted against.
6. **Given** two line series that visually cross, **When** line-segmentation polygons are exported, **Then** the crossing is represented (via an occlusion-aware mask or a topological keypoint graph) rather than two independently self-intersecting ribbons.
7. **Given** the same logical chart definition, **When** it is rendered once through the existing Matplotlib backend and once through the new vector backend, **Then** the two backends' bounding boxes/polygons for equivalent elements agree within an explicitly defined tolerance, so a chart type can be migrated without changing the downstream label schema.
8. **Given** a chart's x-axis label, y-axis label, and title are sampled for a synthetic table, **When** the chart is rendered, **Then** all three belong to the same semantic domain (e.g., a pharmacokinetic y-label is never paired with a fiscal-quarter x-label).
9. **Given** a line, pie, or area chart with a geometry-changing realism effect applied, **When** its segmentation polygons, pose keypoints, or marker boxes are exported alongside its primary annotation boxes, **Then** every stream reflects the same transform — none is left in pre-effect coordinates.
10. **Given** a realism-effect pipeline where a canvas-resizing effect (e.g. `pdf_document_context`) runs after a rotation or perspective effect, **When** annotations are transformed, **Then** each transform step is applied using the canvas size that was current when that step was computed, not the final canvas size.
11. **Given** an annotation whose box extends past the canvas edge after realism effects, **When** annotations are finalized for export, **Then** the box is clamped to the canvas boundary and retained (provided the clamped remainder still exceeds the minimum-size threshold) rather than discarded outright.
12. **Given** an axis tick label rendered with a non-zero rotation (e.g. 45° or 90°), **When** its bounding geometry is exported, **Then** the exported box is the tight oriented box of the rendered text, not Matplotlib's axis-aligned envelope of the rotated glyphs.
13. **Given** an annotation's 4 corners after a homography or rotation is applied, **When** the resulting quadrilateral is exported, **Then** it is verified convex and consistently wound (or corrected/dropped if not) — never exported as a self-intersecting or inverted-winding shape.
14. **Given** any value written to a JSON annotation file (not only the primary detailed-annotations file), **When** that value originates from a NumPy computation (including a NumPy boolean), **Then** it is serialized as the equivalent native JSON type — never silently stringified or left to crash the export.
15. **Given** an area chart whose underlying data can reach zero or negative values, **When** a logarithmic y-scale is selected, **Then** the system falls back to `symlog` based on the chart's actual plotted data — never a hardcoded placeholder that bypasses the check.
16. **Given** a legend rendered on top of a data element (bar, marker, or line segment), **When** the legend's box covers a large majority of that element's box, **Then** the covered element is flagged as occluded rather than exported as a fully-visible ground-truth annotation.
17. **Given** a line or marker whose geometry touches or crosses an axes spine, **When** its stroke-expanded polygon or radius-expanded box is exported, **Then** it is clipped to the axes' visible viewport, matching what Matplotlib actually rendered under its default clipping.
18. **Given** the golden-master regression harness runs on two different machines with identical seeds, **When** their outputs are compared, **Then** the pass/fail decision is based on annotation-field and bounding-box agreement, not a raw image hash that cross-platform font rasterization can perturb.

### Edge Cases

- What happens when **two or more** geometric effects compose in the same image (e.g., `scan_rotation` + `perspective_warp`)? Annotation compensation must apply all transforms in the order they were applied, not just the last one (existing `transform_steps` list in `generator.py:1519` already models this ordering for the compensated path; the fix must not regress it). Additionally, per FR-021, each queued step must retain the canvas size it was computed against — a later canvas-*resizing* effect (`clipping`, `pdf_document_context`) changes `img_w`/`img_h` before compensation is applied, and today's implementation silently uses the wrong size for every step recorded before it (`research.md` §7.3).
- What happens to an OBB whose element is rotated fully or partially **off-canvas**? The exported polygon must be the true (possibly clipped) shape, not silently dropped or clamped to a degenerate box.
- What happens when a bar's height rounds to **near-zero** (e.g., a near-empty category)? The minimum-thickness padding (`ensure_min_bbox_thickness`, `generator.py:147`) must still apply, now operating on an OBB rather than an AABB.
- What happens on a **stacked** area/bar chart where one series fully occludes another? Modal (visible-only) and amodal (full true extent) masks must both be derivable from the same annotation record.
- What happens when the **vector backend does not yet support** a given chart type? The system must fall back to the Matplotlib backend for that type without producing a mixed or inconsistent annotation schema.
- What happens to a non-heatmap annotation (tick label, title, margin-adjacent bar) that extends a few pixels past the canvas edge after effects? Per FR-022, it must be clamped to the canvas boundary and kept — today it is unconditionally discarded, which systematically strips exactly this category of annotation (`research.md` §7.4).
- What happens when `merge_json_files: True` is configured but the merge utility it depends on isn't wired up, or `custom_config.py` is absent from the deployment? Per FR-023, the system must either perform the merge or clearly skip it — it must never hard-fail with an undefined-name error, and must run end-to-end on a self-contained default configuration.
- What happens when two or more subplots in the same composite (`multi_chart_detection`) image are assigned different themes? Per FR-024, each subplot must render text in its own assigned font; theme state must not leak from one subplot into another, or from one worker-process iteration into the next.
- What happens to a rotated axis tick label (45°/90°, `chart.py:432-434`) when its bounding geometry is extracted? Per FR-025, the exported box must be the tight rotated box derived from the label's unrotated font-metric size and its actual rotation/anchor, not the loose axis-aligned envelope `get_window_extent()` returns for rotated text.
- What happens when a homography's warped corner points land near or across the vanishing line (an extreme `distortion_factor`, or a very large annotation like a full-width legend)? Per FR-026, the exported quadrilateral must be verified convex and consistently wound; a self-intersecting or inverted result must be corrected via a dependency-free convex-hull pass or the annotation dropped — never exported as-is.
- What happens when a NumPy boolean (`np.bool_`) or any value in the OCR/metadata JSON files (not just the primary detailed-annotations file) needs serializing? Per FR-027, it must convert to the equivalent native Python/JSON type at every export boundary, not just the one currently sanitized.
- What happens when an area chart's underlying data dips to zero or below and a logarithmic scale is randomly selected? Per FR-028, the fallback to `symlog` must be driven by the chart's actual data, not a hardcoded stand-in that always looks positive regardless of what was plotted.
- What happens when a legend (or, in the future, any other opaque foreground element) is placed directly over a bar, marker, or line? Per FR-029, the covered element must be flagged as occluded rather than exported as if fully visible — consistent with how stacked-series occlusion is already handled for FR-011/FR-012.
- What happens to a thick-stroked line or a large marker whose center sits exactly on an axes spine? Per FR-030, the exported stroke/marker geometry must be clipped to `ax.bbox`, matching Matplotlib's own default `clip_on=True` rendering, rather than extending into margin space where no pixels were drawn.
- What happens when the same fixed-seed golden-master fixture is generated on two different operating systems? Per FR-031, the harness's pass/fail signal must not depend on exact pixel/image-hash agreement, since FreeType/font-hinting differences are expected and legitimate there.
- What happens if a future chart type introduces a `twinx()`/`twiny()` axis pair whose content genuinely overlaps in screen space (unlike the current dual-Y-axis bar chart, which avoids this by construction)? The annotation layer must not assume artist `zorder` is comparable across different `Axes` instances — occlusion/layering across twin axes needs an explicit, axes-aware priority, not a same-axis-only comparison (`research.md` §9.1; no current chart type requires this, so no FR is opened yet).

## Requirements *(mandatory)*

Requirements are grouped by priority wave. Each is independently testable.

### P0 — Geometric & Annotation Correctness (no visual/schema changes to chart output itself; fixes what is exported as ground truth)

- **FR-001**: The system MUST export, for every annotated element, an oriented bounding box (4 clockwise vertices) in addition to (or in place of, per FR-016) the current axis-aligned box.
- **FR-002**: When a homography, rotation, or other geometric realism effect is applied to an image, the system MUST transform each of an element's 4 OBB corners individually through the exact same transform (reusing the existing per-point mapping already used for keypoints, `generator.py:1522-1562`) and MUST NOT re-derive an axis-aligned box from `min`/`max` of those points as the primary exported geometry.
- **FR-003**: Every registered realism effect that changes pixel geometry (translation, rotation, or homography) MUST supply enough information (an explicit transform, not just a probability) for annotation compensation to be applied; the system MUST NOT allow a geometry-changing effect to fire without a corresponding annotation transform being appended to `transform_steps`.
- **FR-004**: The label-visibility decision currently made by rasterizing the figure and sampling pixel colors (`has_non_background_pixels`) MUST be replaced by an analytic check driven by the label's own state (text content, `Text.get_visible()`, and whether its data coordinate falls within the axis's current view limits).
- **FR-005**: The system MUST correctly assign `series_idx` for every bar in a grouped or stacked bar chart, reflecting which data series the bar belongs to (not a constant placeholder value).
- **FR-006**: The system MUST link each bar to the baseline of the axis (primary or secondary/twinx) it was actually plotted against, and this linkage MUST remain correct in the presence of axis margins.
- **FR-007**: The system MUST establish a measured performance baseline (wall-clock per image, broken down by annotation-extraction phase) before any performance-motivated change is merged, so improvement claims are verifiable rather than assumed.
- **FR-008**: All P0 changes MUST be verified against a fixed-seed regression corpus so that non-geometry-related annotation fields do not change value or ordering as a side effect.

### P1 — Representation & Backend Upgrades

- **FR-009**: The system MUST provide a vector rendering backend, selectable per chart type, whose emitted geometry (bounding boxes, polygons) can be extracted analytically from the scene description rather than from rasterized pixel inspection.
- **FR-010**: For any chart type migrated to the vector backend, the system MUST continue to also support the Matplotlib backend, selectable by configuration, until the vector backend's output has been validated for that chart type (FR-013).
- **FR-011**: Line-series segmentation MUST represent each series as a topological structure of typed keypoints (at minimum: start, vertex, inflection, peak, valley, end) with a visibility state (visible / occluded-by-another-series / invisible), rather than solely as an independently generated ribbon polygon.
- **FR-012**: Area-chart and any newly-covered occluded elements MUST expose both a modal (currently visible) and an amodal (true full extent) representation, extending the pattern already implemented for area-chart fills (`generator.py:1791-1833`) to occluded bars, error bars, and overlapping labels.
- **FR-013**: The system MUST provide an automated check that, for a shared chart definition, compares vector-backend geometry against Matplotlib-backend geometry for the same element and flags any disagreement beyond an agreed tolerance.
- **FR-014**: Polygon construction for filled regions (area fills, ribbons) MUST use an analytic clipping algorithm against the axis boundary rather than naive coordinate clamping, so that off-canvas vertices do not distort the visible contour.

### P2 — Semantic & Robustness Improvements

- **FR-015**: Sampled axis labels, titles, and units for a given chart MUST be constrained to a single coherent semantic domain (e.g., a chart cannot pair a pharmacokinetic y-axis label with a fiscal-period x-axis label), replacing independent random sampling across `themes.py`'s unpaired vocabulary lists.
- **FR-016**: Tabular values feeding multi-series charts SHOULD support a configurable multivariate dependency structure (copula-based) so correlated series can be generated without violating each series' own marginal distribution and physical bounds (non-negativity, valid percentage ranges, etc.).
- **FR-017**: Scan/photograph realism effects SHOULD support non-rigid (mesh-based) deformation in addition to today's rigid/homographic effects, with annotation compensation extended to match (reusing the per-point transform approach from FR-002/FR-003).

### Backward Compatibility (cross-cutting)

- **FR-018**: The annotation schema version MUST be explicit and included in exported metadata, so that a downstream training pipeline can detect whether it is consuming AABB-only (legacy) or OBB-aware (new) annotations. AABB export is preserved additively alongside OBB in `v2` so downstream YOLO and GNN consumers continue functioning without breakage.

### P0 — Addendum (Second-Pass Findings, confirmed 2026-09-17)

Found after an external review proposed them and each was independently re-verified against source (`research.md` §7). Numbered to continue the FR sequence without disturbing existing cross-references in `plan.md`/`tasks.md`; all are P0 priority (pure correctness, no new dependency) despite the non-contiguous numbering.

- **FR-019 (Single Transformed Source of Truth)**: Every exported annotation stream for an image — primary bounding boxes, OBB vertices, keypoints, segmentation polygons, pose keypoints, marker boxes, and any chart-type-specific object boxes — MUST be derived from the same post-effects transform applied by `apply_realism_effects`. No annotation-producing function may re-query the pre-effect Matplotlib figure (`ax.transData`, `fig.canvas`) once effects have been applied to that image.
- **FR-020 (Bar Metadata Passthrough)**: The `series_idx`, `bar_idx`, `bottom`, and `top` values already computed at chart-synthesis time by `_generate_bar_chart` MUST be carried through `chart_info_map` and consumed directly by the annotation layer. The annotation layer MUST NOT re-derive this metadata by introspecting rendered patches after the fact. *(Deepens FR-005: FR-005 establishes that `series_idx` must be correct; FR-020 establishes how — passthrough of data the generator already has, not a smarter post-hoc scraper.)*
- **FR-021 (Transform State Consistency)**: The canvas dimensions used to compute a geometric transform (rotation angle or homography) MUST be the same canvas dimensions used when that transform is later applied to annotations, even when a subsequent effect in the same pipeline run changes the canvas size.
- **FR-022 (Boundary Clipping Instead of Discarding)**: An annotation that extends beyond the canvas after realism effects MUST be clamped to the canvas boundary and retained, provided the clamped box still exceeds the existing minimum-size threshold, rather than discarded outright. This generalizes the clamp already applied to heatmap annotations to every chart type; it requires no new dependency, since clamping an axis-aligned box to a rectangle is `min`/`max` arithmetic, not the general polygon clipping FR-014 introduces for filled regions.
- **FR-023 (Config Fallback & Import Safety)**: The system MUST run end-to-end without requiring the external `custom_config.py` to be present, via a self-contained default configuration. Every symbol referenced at runtime (e.g. a merge utility invoked by an optional config flag) MUST be either defined, imported, or have its dead code path removed — no config combination may trigger a `NameError`/`ImportError`.
- **FR-024 (Theme Isolation)**: Per-chart-type font/theme selection MUST NOT depend on process-global mutable state (e.g. Matplotlib `rcParams`) that a sibling subplot's theme, or a later image generated by the same worker process, can silently overwrite.

### P0 — Addendum II (Third-Pass Findings, confirmed 2026-09-18)

A third, unattributed review proposed seven further claims (`research.md` §8); three were confirmed as new defects, one was confirmed but already mitigated by FR-022 (no new requirement needed), and three did not hold up against source (recorded in `research.md` §8 for the audit trail, not repeated here). Continues the FR sequence from the first addendum rather than reusing FR-019/FR-020.

- **FR-025 (Rotated Text OBB Fidelity)**: For any text artist rendered with a non-zero rotation (tick labels, rotated titles), the exported bounding geometry MUST be the tight oriented box of the rendered text — derived from the artist's unrotated font-metric box plus its actual rotation angle and anchor point — not the axis-aligned envelope `get_window_extent()` returns for rotated text.
- **FR-026 (OBB Convexity & Winding Guarantee)**: After a homography or rotation is applied to an annotation's 4 corner points, the resulting quadrilateral MUST be verified convex and consistently wound before export. A degenerate (self-intersecting or collinear) result MUST be corrected via a convex-hull reordering or the annotation dropped. This check MUST be implemented without a new dependency, consistent with P0's no-new-dependency constraint (it protects FR-001/FR-002, which are themselves P0).
- **FR-027 (Uniform NumPy/Boolean JSON Safety)**: Every JSON export boundary — not only the primary detailed-annotations file — MUST pass through a scalar sanitizer that correctly converts `np.bool_` (which is not a subclass of Python `bool`) and any other NumPy scalar/array type to its native Python equivalent before serialization. No JSON export path may bypass this sanitizer.

### P0 — Addendum III (Fourth-Pass Findings, confirmed 2026-09-18)

A fourth, unattributed review proposed seven further claims (`research.md` §9); four were confirmed (one narrower than stated), two were refuted outright (both already mitigated in existing code), and one is a valid general risk with no current manifestation, recorded in `research.md` §9.1 and `plan.md`'s risk register rather than as a requirement here. Continues the FR sequence from the first two addenda.

- **FR-028 (Data-Driven Log-Scale Guard)**: Every call site that selects a logarithmic axis scale MUST derive its positivity check from the chart's actual final plotted data, never from a hardcoded placeholder value. (Narrows and completes FR-related coverage: `apply_axis_scaling`'s guard itself is correct — this closes the one call site, the area chart, that bypasses it.)
- **FR-029 (Cross-Class Occlusion Flagging)**: An annotation whose box is substantially covered by a different-class opaque element (e.g. a legend) in the same image MUST be flagged as occluded (visibility = 0) rather than exported as if fully visible, consistent with how FR-011/FR-012 already treat stacked-series occlusion within a class.
- **FR-030 (Stroke/Marker Viewport Clipping)**: Stroke-expanded line polygons and radius-expanded marker boxes MUST be clipped to the axes' visible viewport (`ax.bbox`) before export, matching Matplotlib's default `clip_on=True` rendering, so no exported geometry extends into margin space where no pixels were drawn.
- **FR-031 (Cross-Platform Golden-Master Determinism)**: The P0 regression harness MUST NOT rely on raw image-hash equality as its primary pass/fail signal, since Matplotlib's text rendering legitimately differs across FreeType versions/font installations even with identical seeds. It MUST pin a bundled font and use annotation-field/bounding-box agreement as the primary signal, with an image-hash comparison retained only as a secondary, non-blocking diagnostic.

## Key Entities

- **OrientedBoundingBox (OBB)**: 4 vertices `(x₁,y₁)…(x₄,y₄)` in clockwise order, plus derived `(cx, cy, w, h, θ)`. Replaces/augments the current `(x0,y0,x1,y1)` AABB for any element that can be rotated or perspective-warped.
- **Keypoint**: `(x, y, visibility, role)` where `visibility ∈ {invisible, occluded, visible}` and `role ∈ {start, vertex, inflection, peak, valley, end}`. Used for line/area topology.
- **SeriesLink**: association between a rendered element (bar, line, area) and its logical `series_idx`, distinct from **BaselineLink**, the association between a bar and the `baseline_id` (primary/secondary axis) it is measured against.
- **ModalAmodalMask**: a pair `(amodal_polygon, modal_rle)` — the full true extent of an element and its currently-visible-pixel mask, for any element that can be partially occluded.
- **SemanticDomain**: a tag (e.g., `pharmacokinetic`, `fiscal`, `clinical`) that constrains which label pools an x-label, y-label, and title may jointly be sampled from for one chart.
- **TransformStep**: `(kind, params…, canvas_w, canvas_h)` — one entry in the ordered list of geometric operations applied to an annotation. Per FR-021, `canvas_w`/`canvas_h` are recorded at the moment the step is appended (reflecting the canvas size that step's parameters were computed against), not read from shared mutable state when the step is later applied.

## Non-Functional Requirements

- **Determinism**: for a fixed random seed, the generated dataset (images + annotations) must be reproducible before and after each phase's changes, except for the specific fields that phase intentionally changes. This is the regression safety net for the whole modernization.
- **Backward compatibility**: no change may silently alter the meaning of an existing annotation field name without a schema version bump (FR-018).
- **Incrementality**: each chart type may be migrated to the vector backend independently; the system must never be in a state where some elements of a single image use raster-derived geometry and others use vector-derived geometry without both having passed the parity check (FR-013).

## Out of Scope

- Retraining or evaluating any downstream detection/segmentation/VLM model on the new annotations (validating the *hypothesis* that fidelity improves mAP is a follow-on effort, not part of this feature).
- Changing chart visual themes, color palettes, or the statistical/data-generation logic in `chart.py` (Hill/Michaelis-Menten/AR(1)/copula-for-pies/etc. are explicitly preserved as-is).
- Migrating the project off Python, or replacing NumPy/SciPy/Pillow.
- Building a hosted/web rendering service; the vector backend is a local, headless compilation step (Python → SVG/Vega-Lite spec → raster), not a browser-based renderer.

## Success Criteria

- **SC-001**: Under a synthetic rotation/homography test sweep, the ratio of exported-box area to true-element area for affected elements drops from the current dilation (frequently >1.3× for rotated/perspective elements, per `research.md` §2.1) to ≤1.05× for the OBB path.
- **SC-002**: Zero images in a 10,000-image regression run have a geometry-changing effect applied without a corresponding annotation transform recorded (closes the gap in `research.md` §2.2).
- **SC-003**: 100% of bars in a grouped/stacked-bar regression corpus (≥2 series) have a `series_idx` that matches the series used to generate that bar's value (verifiable because the generator knows ground truth at synthesis time).
- **SC-004**: Per-image annotation-extraction wall-clock time is measured before and after P0 (FR-007) and the delta is reported — the target multiplier from the source audit is treated as a hypothesis to confirm, not a guarantee.
- **SC-005**: For at least one migrated chart type, vector-backend and Matplotlib-backend geometry agree within the tolerance defined by FR-013 on a shared regression corpus.
- **SC-006**: A domain-coherence check run over a sample of generated charts shows 0 cross-domain label pairings (e.g., no pharmacokinetic/fiscal mismatches) once FR-015 ships.
- **SC-007**: Zero images in a regression run where a segmentation, pose, marker, or re-derived object-box stream disagrees with the primary annotation's transform for the same visual element (closes `research.md` §7.1).
- **SC-008**: Zero images in a regression run where a chained rotation/homography + canvas-resize effect produces annotation coordinates outside a documented tolerance of the hand-computed expected value (closes `research.md` §7.3).
- **SC-009**: The rate of annotations discarded solely for extending past the canvas boundary after effects drops to zero for boxes whose clamped remainder exceeds the minimum-size threshold (closes `research.md` §7.4).
- **SC-010**: `generator.py` completes a full run with `custom_config.py` absent and `merge_json_files: True` set, with zero `NameError`/`ImportError` occurrences (closes `research.md` §7.5).
- **SC-011**: A composite multi-subplot regression sample shows 0 instances of a subplot's rendered font not matching its assigned theme (closes `research.md` §7.6).
- **SC-012**: For a regression sample containing rotated tick labels, the exported OBB's area is within a documented tolerance of the label's true rotated footprint (computed independently from font metrics), not the loose axis-aligned envelope (closes `research.md` §8.3).
- **SC-013**: Zero images in a regression run export a self-intersecting or inconsistently-wound 4-vertex OBB (closes `research.md` §8.5).
- **SC-014**: A regression run including a value derived from `np.bool_` produces a valid JSON boolean (not the string `"True"`/`"False"`) in every export file, including `ocr_*.json` and the base metadata `.json` (closes `research.md` §8.6).
- **SC-015**: An area-chart regression fixture with data reaching ≤0 never renders with a `'log'` y-scale — only `'linear'` or `'symlog'` (closes `research.md` §9.4).
- **SC-016**: Zero images in a regression run export a >85%-legend-covered annotation without a visibility=0 flag (closes `research.md` §9.5).
- **SC-017**: Zero images in a regression run export a stroke/marker polygon whose extent falls outside `ax.bbox` (closes `research.md` §9.6).
- **SC-018**: The golden-master harness produces identical pass/fail results when run on two different operating systems against the same fixed-seed fixture, using the annotation-field/bbox comparison as ground truth (closes `research.md` §9.7).

## Clarifications

### Session 2026-09-17 / Phase 1 Setup (Resolved)

- Q: Does the exported AABB format need to remain byte-for-byte available for existing downstream consumers, or can it be replaced outright once OBB ships? → **Resolved**: Production consumers (YOLO object detection models, `merge_json.py`, and downstream GNN structure parsers) strictly depend on normalized AABB `<class_id> <x_center> <y_center> <w> <h>` `.txt` labels and `detailed.json`'s `raw_annotations` (`xyxy`/`bbox`). OBB MUST be introduced additively alongside AABB under `ANNOTATION_SCHEMA_VERSION = "v2"`, never replacing AABB outright.
- Q: What is the actual configured probability of the uncompensated `"perspective"` effect (§2.2 in `research.md`) in the live `custom_config.py`? → **Resolved**: In `custom_config.py:316`, `realism_effects['perspective']['p'] = 0.0` (with `params: {"magnitude": 0.5}`). `"perspective_warp"` is not present in `custom_config.py`.
- Q: Does `batch_merge_all` exist in the target deployment? → **Resolved**: Yes, `batch_merge_all(labels_dir="labels")` is defined in `merge_json.py:260`. `generator.py:4110-4112` references it without an import statement; resolving it requires importing from `merge_json`.
- Q: Should the vector backend target full parity across all 8 chart types before any chart type ships to production data generation, or can chart types graduate independently? → **Resolved**: Chart types will graduate independently (per `plan.md`).
- Q: Is a local LLM already available in the target deployment environment for FR-015/schema generation, or must domain-coherent labels be achieved with a simpler rule-based domain tag on existing `themes.py` lists first? → **Resolved**: Rule-based domain tagging (`synth/domain_tags.py`) precedes any LLM integration.
