# Tasks: Vector-Faithful Annotation & Rendering Pipeline

**Input**: `spec.md`, `plan.md`, `research.md`
**Legend**: `[P]` = can be done in parallel with other `[P]` tasks in the same phase (touches different files / no shared state). Tasks without `[P]` are sequential within their phase. Every task lists **Done when**, a concrete, checkable acceptance condition — no task is complete on "looks right."

---

## Phase 1: Setup

- **T001 [DONE]** Read `custom_config.py` (`OCR_TRAINING_CONFIG`) in full and record: (a) the configured probability `p` for the `"perspective"` effect key, (b) the exact schema of each `CLASS_MAP_*`, (c) whether any documented consumer depends on the current AABB `.txt`/JSON field names, (d) whether a `batch_merge_all` implementation exists anywhere in the target deployment outside the files reviewed here (referenced but never defined/imported at `generator.py:4110-4112`, per `research.md` §7.5).
  **Done when**: answers to `spec.md`'s three config-dependent `[NEEDS CLARIFICATION]` items are written back into `spec.md`'s Clarifications section, replacing the `[NEEDS CLARIFICATION]` marker with the resolved answer; item (d)'s answer is recorded for T030 to consume.

- **T002 [DONE]** Add `pytest` (and `pytest-regressions` or equivalent snapshot helper) to the project's dependency manifest.
  **Done when**: `pytest` runs a trivial smoke test in CI/locally.

- **T003 [DEFERRED / PRUNED per Ponytail]** Create the `geometry/`, `annotate/`, `backends/`, `synth/`, `tests/` package skeletons per `plan.md`'s Project Structure, each with an `__init__.py` and a one-line module docstring; no logic yet. *(Pruned during Phase 1 under Ponytail compliance to avoid speculative scaffolding; `tests/` created, logic kept surgical in existing modules).*
  **Done when**: `import geometry, annotate` succeeds with no errors.

- **T004 [DONE]** Verify the P1 dependency set (`shapely`, `svgelements`, `lxml`, `altair`, `vl-convert-python`) installs cleanly in the target environment (verified and pinned in `requirements.txt`).
  **Done when**: dependencies install with zero conflicts against `numpy`/`scipy`/`Pillow`/`matplotlib` and pass test execution.

---

## Phase 2: Foundational (blocks all user stories — build the regression safety net before touching any behavior)

- **T005 [DONE]** Build the golden-master regression harness: given a list of fixed seeds, run the current (unmodified) `generate_single_chart` for each, and snapshot both the output image (hash) and the full annotation JSON/YOLO-txt to `tests/golden/`.
  **Done when**: running the harness twice against unmodified code produces zero diffs.

- **T006 [DONE]** Add a harness assertion mode: diff a new run's annotations against the golden snapshot field-by-field and report which fields changed, rather than a blunt pass/fail.
  **Done when**: intentionally mutating one annotation field in a scratch copy produces a report naming exactly that field.

- **T007 [DONE]** Instrument `generate_single_chart` (or its call sites) with coarse timers around (a) chart drawing, (b) `get_granular_annotations`, (c) `get_detailed_annotations`, (d) `apply_realism_effects`, (e) file I/O — and log per-image timings.
  **Done when**: a run of 100 images produces a timing breakdown table; this is the FR-007 baseline that every later performance claim is compared against.

- **T008 [PARTIAL: SURGICAL IN-PLACE]** Extract `_rotate_point`, `_warp_point`, `_apply_transform_steps` from inside `apply_realism_effects` (`generator.py:1522-1562`) into `geometry/transforms.py` as top-level pure functions with the same signatures; update `apply_realism_effects` to import and call them. *(Transformed functions updated with explicit canvas dimension arguments and tested directly in `generator.py`; modular package extraction deferred per Ponytail rules).*
  **Done when**: the Phase 2 golden-master harness (T005/T006) shows **zero** diffs — this is a pure refactor, not a behavior change.

- **T009 [PARTIAL: SURGICAL IN-PLACE]** Implement `geometry/obb.py`: an `OrientedBoundingBox` type supporting construction from 4 corner points, conversion to/from `(cx, cy, w, h, θ)`, an `.area` property, and a `.to_aabb()` method (for legacy export) that is clearly named as a lossy operation, never called implicitly. *(Canonical OBB construction, winding, and hull validation implemented directly in `generator.py:canonicalize_and_validate_obb` and `get_text_obb_and_bbox`; separate file deferred per Ponytail rules).*
  **Done when**: unit tests in `tests/test_geometry_transforms.py` confirm `OrientedBoundingBox` round-trips corners → `(cx,cy,w,h,θ)` → corners within floating-point tolerance for at least axis-aligned, 45°-rotated, and arbitrary-perspective-warped test cases.

---

## Phase 3: User Story 1 — Effects never dilate or skip annotations (P0a) — FR-001, FR-002, FR-003

**Story**: As an ML engineer, when a realism effect geometrically warps an image, I need the exported box to be the exact warped shape, and I need every geometry-changing effect to be compensated, with no exceptions.

- **T010 [DONE]** Write a failing test in `tests/test_obb_propagation.py` (implemented in `tests/test_p0_annotations.py::test_rotated_text_obb_tighter_than_aabb` and `tests/test_p0_transforms.py`): apply a known rotation to a known rectangle's 4 corners via `apply_realism_effects`, and assert the exported box is a rotated OBB (`OBB.area / true_area ≤ 1.05`), not an AABB with a much larger area.
  **Done when**: the test fails against unmodified `generator.py:1598-1610` with an area ratio inconsistent with the assertion.

- **T011 [DONE]** Change `apply_realism_effects` (`generator.py:1593-1610`) to store the 4 individually-warped corners as an `OrientedBoundingBox` (via `generator.py:canonicalize_and_validate_obb`) on `ann['obb']`'s replacement/companion field, instead of collapsing them to `transforms.Bbox.from_extents(min(xs), min(ys), max(xs), max(ys))`.
  **Done when**: T010's test passes, and the golden-master harness shows diffs **only** in the newly-added OBB field(s) — no unrelated field changes.

- **T012 [DONE]** Write a failing test asserting: for every key in `effect_function_map` (`generator.py:1491-1516`) that is geometry-changing (translation, rotation, or homography), the dispatch logic in `apply_realism_effects` (`generator.py:1564-1591`) appends a corresponding entry to `transform_steps`. Parametrize over `"perspective"` specifically, since `research.md` §2.2 shows it currently does not.
  **Done when**: the test fails for the `"perspective"` key against unmodified code.

- **T013 [DONE]** Resolve the `"perspective"` vs `"perspective_warp"` split per T001's finding: make `apply_perspective_effect` also return and propagate a homography, unifying it with the `"perspective_warp"` dispatch branch, and resolve keyword parameter collision (`magnitude` vs `distortion_factor`).
  **Done when**: T012's test passes for all geometry-changing effects, and the golden-master harness confirms no non-geometric effect's behavior changed.

- **T014 [DONE]** Add the "effect-compensation completeness" check from `plan.md`'s Testing Strategy as a permanent, always-run test (`tests/test_p0_transforms.py::test_perspective_distortion_shifts_all_annotation_types`): it must fail automatically if a future contributor adds a new geometry-changing effect to `effect_function_map` without wiring compensation.
  **Done when**: adding a deliberately uncompensated dummy geometric effect in a scratch test causes this check to fail.

- **T015 [DONE]** Update `save_annotations_yolo`/`save_annotations_pose`/`save_annotations_yolo_seg` (`generator.py:1634-1836`) to optionally emit the OBB fields (8 normalized coordinates, per the YOLO-OBB convention) alongside existing AABB fields, gated by `ANNOTATION_SCHEMA_VERSION` (resolves FR-018, informed by T001's backward-compatibility answer).
  **Done when**: with the schema flag at `v1`, output is byte-identical to today's; at `v2`, output additionally contains valid OBB rows that pass T009's round-trip test when re-parsed.

---

## Phase 4: User Story 2 — Analytic label visibility & measured performance (P0b) — FR-004, FR-007

**Story**: As an ML engineer, I need label-inclusion decisions and performance claims to be based on the label's own state and on real measurements, not on rasterized pixel sampling or assumption.

- **T016 [DONE]** Write analytical label visibility logic (`generator.py::has_non_background_pixels` analytical overhaul) that returns `True`/`False` using only: `text_artist.get_visible()`, whether `text_artist.get_text()` is non-empty, and whether the label's associated data coordinate falls within `ax`'s current view limits (`ax.get_xlim()`/`get_ylim()`). No canvas draw, no pixel buffer read.
  **Done when**: a unit test covers at least: an empty-string label (→ `False`), an out-of-view-limits label (→ `False`), and a normal in-range label (→ `True`).

- **T017 [DONE]** Replace all call sites of `has_non_background_pixels` (`generator.py:195-266`) with analytical checks; delete unreachable dead branches (`generator.py:259-264`) and eliminate RGB buffer pixel scanning.
  **Done when**: golden-master harness shows the same set of labels is kept/dropped as before on the regression corpus (or, where it legitimately differs, each difference is manually reviewed and confirmed to be a *correction* of a previously-wrong pixel-based decision, not a regression); `fig.canvas.buffer_rgba()` no longer appears anywhere in `generator.py`.

- **T018 [DONE]** Re-run the T007 timing instrumentation on the same 100-image sample used for the baseline.
  **Done when**: a before/after report exists showing the measured delta for the label-visibility phase specifically (this is the FR-007/SC-004 deliverable — report the real number, whatever it is, rather than asserting the audit's 3–5× figure).

---

## Phase 5: User Story 3 — Correct series and baseline linkage (P0c) — FR-005, FR-006, FR-020

**Story**: As an ML engineer training a GNN on chart structure, I need every bar's series and axis linkage to be genuinely correct, not a placeholder or a fragile heuristic.

- **T019 [DONE]** Write a failing test: generate a grouped bar chart with 3 series, and assert `extract_bar_info`'s returned `series_idx` values (`chart.py:4637-4671`) actually distinguish the 3 series (not all `0`).
  **Done when**: the test fails against unmodified `chart.py:~4658`.

- **T020 [DONE]** *(Revised per `research.md` §7.2 — fixes the root cause, not the symptom T019 exercises)* `_generate_bar_chart` (`chart.py:1511-1620`) already computes correct per-bar `series_idx`/`bar_idx`/`bottom`/`top` and returns it as `bar_info_list`; `generator.py` captures this locally (`~3154/3182/3184`) and stores it in the `chart_info_map[ax] = {...}` dict literal that follows (`~3222-3238`). Add `chart_info_map[ax]['bar_info'] = bar_info_list` there, and change `create_unified_annotation` (`generator.py:2585`) to read `chart_info_map[ax].get('bar_info')` directly when present, calling `extract_bar_info(ax, chart_type)` only as a fallback for chart-generation paths that don't populate it.
  **Done when**: T019's test passes using the real synthesis-time `series_idx` (not a value inferred by `extract_bar_info`); golden-master harness confirms single-series charts (where the fallback path already trivially returned `series_idx: 0`) show no change; a new fixture with a grouped 3-series bar chart shows `series_idx ∈ {0,1,2}` correctly distributed and matching each bar's true generating series (FR-020).

- **T021 [PARTIAL: SURGICAL IN-PLACE]** Write `annotate/series_link.py` to house the fixed series-linkage logic, reading from `chart_info_map[ax]['bar_info']` (per T020) rather than from `extract_bar_info`'s output, so the logic is unit-testable independent of the ~900-line `generate_single_chart`. *(Logic embedded and tested directly in `generator.py`; standalone module extraction deferred per Ponytail rules).*
  **Done when**: existing call sites (`generator.py:2314`, `2585-2607`) are updated to import from `annotate/series_link.py` with no behavior change beyond T020's fix.

- **T022 [DONE]** Harden `add_graph_topology_metadata`'s baseline-matching (`generator.py:3021-3033`): added a maximum-plausible-distance cutoff (`max_baseline_dist = 1.5 * img_h`) in addition to the existing x-range containment check, rejecting implausibly distant baselines under axis distortion or twinx margins.
  **Done when**: the new test passes and golden-master shows no change for charts without twinx/large margins.

---

## Phase 6: User Story 9 — Cross-cutting correctness fixes from second-pass review (P0d) — FR-019, FR-021, FR-022, FR-023, FR-024

**Story**: As an ML engineer, I need every annotation stream — not just the primary bounding boxes — to reflect the same realism-effect transform; every geometric transform to be applied using the canvas size it was actually computed against; every out-of-bounds annotation clipped rather than silently dropped; the generator to run without crashing on a documented config option; and per-chart theming to never leak between subplots or across a worker's lifetime. These five were confirmed by direct source review (`research.md` §7) after an external reviewer proposed them without citations; none require a new dependency. (The sixth second-pass finding, the `bar_info_list`/`chart_info_map` disconnect, is folded into Phase 5's T020 since it deepens that phase's existing story rather than starting a new one.)

*Depends on: Phase 2 (T008's extracted `geometry/transforms.py`, which T026 below modifies further) and Phase 5 (T020's `chart_info_map` bar-metadata passthrough pattern, which T024 below reuses for the secondary annotation streams). Independent of Phases 3-4's remaining tasks.*

- **T023 [DONE]** Write a failing test: generate a line, pie, or area chart with a geometry-changing realism effect enabled (e.g. `scan_rotation`), and assert that the exported segmentation polygon / pose keypoints / marker boxes (`extract_line_segmentation_annotations`, `extract_pie_pose_annotations`, `extract_area_segmentation_annotations`, and the re-derived `*_obj_labels`) land at the *same* transformed coordinates as the primary annotation for the same visual element, not at the untransformed `ax.transData` coordinates.
  **Done when**: the test fails against unmodified `generator.py` — the secondary-stream coordinates and the primary annotation disagree by the full magnitude of the effect (`research.md` §7.1).

- **T024 [DONE]** Fix the desync: give `apply_realism_effects` (`generator.py:1489-1632`) a way to expose its final `transform_steps` list (via `extra_annotation_sets` parameter) instead of keeping it function-local, and update `extract_pie_pose_annotations`, `extract_line_segmentation_annotations`, `extract_area_segmentation_annotations`, and the `area_obj`/`pie_obj`/`line_obj` re-extraction calls (`generator.py:3493`, `3518`, `3550`) to apply that same `transform_steps` to each point they read from `ax.transData`, rather than exporting raw pre-effect coordinates.
  **Done when**: T023's test passes, and the golden-master harness shows every affected secondary-stream file (`line_seg_labels/`, `pie_pose_labels/`, `area_seg_labels/`, `line_marker_labels/`, and the three re-derived `*_obj_labels/`) changes *only* on images where a geometry-changing effect actually fired (FR-019).

- **T025 [DONE]** Write a failing test: chain a `scan_rotation` (or `perspective_warp`) effect with a subsequent `pdf_document_context` effect (which grows the canvas), and assert that the exported annotation coordinates match a hand-computed expected value using the canvas size that was in effect *when the rotation/homography was computed* — not the final, padded canvas size.
  **Done when**: the test fails against unmodified `generator.py:1520-1622`, reproducing the closure-variable bug from `research.md` §7.3.

- **T026 [DONE]** Fix the closure bug: change each entry appended to `transform_steps` (`generator.py:1578, 1583, 1587`) to carry the canvas `(w, h)` that was current *at the moment that step was recorded*, and change `_apply_transform_steps`/`_rotate_point`/`_warp_point` to use each step's own recorded `(w, h)` for its origin-flip math, instead of reading the enclosing function's `img_w`/`img_h` variables (which get reassigned to the final post-all-effects size before any point is actually warped).
  **Done when**: T025's test passes; golden-master shows no change for any image where no canvas-resizing effect (`clipping`, `pdf_document_context`) is chained after a rotation/homography effect — the orderings that were already consistent before this fix (FR-021).

- **T027 [DONE]** Write a failing test: construct an annotation (e.g. an axis tick label) whose box extends a few pixels past the canvas edge after a realism effect, and assert it is present in the final output with a clamped box — not absent.
  **Done when**: the test fails against unmodified `generator.py:3452-3458`, confirming the annotation is silently discarded (`research.md` §7.4).

- **T028 [DONE]** Fix: replace the discard branch at `generator.py:3452-3458` with the same axis-aligned clamp already used for heatmaps (`generator.py:3442-3451`), generalized to every chart type — `x0/y0/x1/y1` are clamped to `[0, img_w]`/`[0, img_h]`, and the annotation is kept if the clamped box still exceeds the existing `MIN_BBOX_SIZE` threshold, dropped otherwise. No new dependency: this is a rectangle-to-rectangle clamp (`min`/`max`), not the general polygon clipping FR-014 introduces in P1 for filled regions.
  **Done when**: T027's test passes; golden-master shows previously-discarded near-boundary annotations (especially axis tick labels/titles) now present and clamped, with no change to annotations that were already fully in-bounds (FR-022).

- **T029 [DONE]** Write a failing test: call the generator's end-of-run merge step with `merge_json_files: True` and `dataset_format != 'multi_chart_detection'`, and assert it does not raise `NameError`.
  **Done when**: the test fails against unmodified `generator.py:4110-4112` with `NameError: name 'batch_merge_all' is not defined`.

- **T030 [DONE]** Resolve `batch_merge_all` per T001(d)'s finding: imported cleanly via `try: from merge_json import batch_merge_all except ImportError: batch_merge_all = None`. Fallback default configuration provided in `config_defaults.py` ensuring `generator.py` runs cleanly when `custom_config.py` is absent (completes FR-023).
  **Done when**: T029's test passes; running `generator.py` end-to-end in a scratch environment with no `custom_config.py` on the path completes without an import error (FR-023).

- **T031 [DONE]** Write a failing test: render a 2-subplot composite figure (as used by `dataset_format: multi_chart_detection`) with two visually distinct theme fonts assigned to the two subplots, and assert both subplots render text in their own assigned font — not both in whichever theme was applied last.
  **Done when**: the test fails against unmodified `chart.py:374`, confirming both subplots render in the second theme's font (`research.md` §7.6).

- **T032 [DONE]** Fix: scope `apply_chart_theme`'s font selection to the axis being themed instead of mutating the global `rcParams['font.sans-serif']` — wrap the theming and subsequent draw calls for that axis in `matplotlib.rc_context({...})`, or pass an explicit `fontproperties` down to each `Text`/label/tick call so font resolution does not depend on shared global state.
  **Done when**: T031's test passes; golden-master shows no font change for any single-subplot image — single-theme figures were already correct by construction, so this is provably a no-op there (FR-024).

---

## Phase 7: User Story 10 — Third-pass correctness fixes (P0e) — FR-025, FR-026, FR-027

**Story**: As an ML engineer, I need rotated tick labels to export a tight oriented box instead of Matplotlib's loose rotated-AABB, every homography-warped OBB to be a valid (convex, consistently-wound) quadrilateral even near the vanishing line, and every JSON export file — not only the primary one — to serialize NumPy values (including booleans) correctly. These three were confirmed by direct source review (`research.md` §8) after a third, unattributed reviewer proposed seven claims; four of the seven did not hold up against source and required no task (recorded in `research.md` §8 for the audit trail). None of these three require a new dependency — T036's fix in particular was deliberately rescoped off the reviewing claim's own suggested `shapely` approach to keep it P0-compatible.

*Depends on: Phase 2 (T008's `geometry/transforms.py`, which produces the warped corners T035 validates) and Phase 3 (T012-T013's OBB corner propagation, which T034's rotated-text fix extends to text artists specifically). Independent of Phase 6.*

- **T033 [DONE]** Write a failing test: render a chart with a 45°-rotated x-tick label (`chart.py:1392`), independently compute the label's true tight rotated box from its font metrics and rotation angle, and assert the currently-exported box's area is significantly larger (the loose axis-aligned envelope of the rotated glyphs).
  **Done when**: the test fails against unmodified `generator.py`, confirming no rotation compensation exists — `get_rotation()` is not read anywhere in the label-extraction call sites (`research.md` §8.3).

- **T034 [DONE]** Fix: for any text artist where `label.get_rotation() != 0` at each of the ~30 `get_window_extent(renderer)` call sites in `generator.py` (407, 421, 568, 581, 927, 940, and others), compute the tight box with rotation temporarily zeroed, then derive the 4-corner OBB by rotating that box's corners around the artist's anchor point (accounting for `ha`/`va`) at its actual angle via `get_text_obb_and_bbox`, and export those corners instead of the raw `get_window_extent()` result.
  **Done when**: T033's test passes; golden-master shows no change for any unrotated label (rotation-compensation is provably a no-op when `get_rotation() == 0`) (FR-025).

- **T035 [DONE]** Write a failing test: construct a homography (via `apply_perspective_warp_effect` with an exaggerated `distortion_factor`, or a hand-built `H`) such that one corner of a large annotation lands on the far side of the vanishing line from the others, and assert the currently-exported 4-vertex OBB is self-intersecting or has inconsistent winding (e.g. via a signed-area/cross-product check).
  **Done when**: the test fails against unmodified `apply_realism_effects`, confirming no convexity/winding validation exists on the warped corners (`research.md` §8.5).

- **T036 [DONE]** Fix: add a dependency-free convexity/winding check (`canonicalize_and_validate_obb` in `generator.py`) using the sign of consecutive edge cross-products around the 4 warped corners run immediately after `_apply_transform_steps` warps them; on failure, reorder the points via a pure-NumPy 4-point convex-hull pass, or drop the annotation if fewer than 4 points survive the hull (genuine self-intersection).
  **Done when**: T035's test passes; golden-master shows no change for any image where no corner crosses the vanishing line (the common case, where the 4 warped corners were already convex and correctly wound) (FR-026).

- **T037 [DONE]** Write a failing test: pass an `np.bool_` value through `convert_numpy_types` (`generator.py:1883`) and assert the result is a native Python `bool`, not the string `"True"`/`"False"`; separately, construct an `ocr_json`/`metadata_json` payload containing a NumPy-derived value and assert `json.dump` does not raise `TypeError`.
  **Done when**: the test fails against unmodified `generator.py` — `np.bool_` falls through to `convert_numpy_types`'s string catch-all, and `ocr_json`/`metadata_json` (3947-3953) have no sanitizer applied at all (`research.md` §8.6).

- **T038 [DONE]** Fix: add an `isinstance(obj, (bool, np.bool_))` branch to `convert_numpy_types` returning `bool(obj)`, ordered before the existing catch-all; apply `json_default_fallback` and `convert_numpy_types` to `ocr_json` and `metadata_json` at their `json.dump` call sites (`generator.py:4315-4323`), matching what `detailed_json` already does.
  **Done when**: T037's test passes; golden-master shows no change to any existing (non-boolean, non-NumPy) field in any of the three JSON files (FR-027).

---

## Phase 8: User Story 11 — Fourth-pass correctness fixes (P0f) — FR-028, FR-029, FR-030, FR-031

**Story**: As an ML engineer, I need the area chart's log-scale guard to key off real data instead of a hardcoded placeholder, elements fully covered by a legend to be flagged occluded instead of exported as visible, stroke/marker geometry to respect the same axes-boundary clipping Matplotlib itself applies, and the golden-master harness to give the same pass/fail verdict on any machine. These four were confirmed by direct source review (`research.md` §9) after a fourth, unattributed reviewer proposed seven claims; three of the seven did not hold up (two refuted outright, one recorded as a forward-looking risk with no current fix — see `plan.md`'s risk register). None of these four require a new dependency; T042's fix in particular uses a dependency-free Sutherland–Hodgman clip rather than the reviewing claim's implied general polygon-clipping library, to stay inside P0's constraint.

*Depends on: Phase 2 (T005/T006's harness, which T045-T046 modify) and Phase 3 (T012's stroke/marker geometry, which T041-T042 clip). Independent of Phases 6-7.*

- **T039 [DONE]** Write a failing test: build an area-chart fixture whose underlying data reaches ≤0 (e.g. via the same AR(1)/mean-zero noise path used elsewhere in `chart.py`), force `scale_type='log'`, and assert the current code still calls `ax.set_yscale('log')` instead of falling back to `symlog`.
  **Done when**: the test fails against unmodified `chart.py:3044`, confirming the hardcoded `data_min=0.01` bypasses `apply_axis_scaling`'s own positivity guard (`research.md` §9.4).

- **T040 [DONE]** Fix: replace `apply_axis_scaling(ax, data_min=0.01, orientation='vertical')` (`chart.py:3044`) with a real `data_min` computed from `boundary_y`/`y_stack` (`chart.py:3044-3046`), matching the pattern already used correctly at `chart.py:1826` and `2066`.
  **Done when**: T039's test passes; golden-master shows no change for any area-chart fixture whose data was already positive throughout (FR-028).

- **T041 [DONE]** Write a failing test: render a chart with a legend forced to fully cover a known bar or marker (e.g. via a fixed `loc` and a small axes), and assert the covered element's exported annotation has no occlusion/visibility flag distinguishing it from a fully-visible one.
  **Done when**: the test fails against unmodified `generator.py`, confirming `filter_overlapping_annotations` (1277-1311) only compares same-class boxes and nothing else checks legend-vs-other-class overlap (`research.md` §9.5).

- **T042 [DONE]** Fix: after an axis's annotations are assembled and a legend annotation exists for that axis, compute `area(bbox ∩ legend_bbox) / area(bbox)` for every other annotation on that axis; where this is ≥ 0.85, set that annotation's visibility field to `0` (occluded) and retain it for amodal representation (`cull_occluded=False` by default in `filter_overlapping_annotations`, completing FR-029).
  **Done when**: T041's test passes; golden-master shows no change for any image with no legend, or with a legend that doesn't materially overlap another annotation (FR-029).

- **T043 [DONE]** Write a failing test: construct a line series with a thick `linewidth` whose path touches an axes spine (e.g. a value at `y=0` on an axis with `ylim` starting at 0), and assert the exported `stroke_to_polygon_ribbon` polygon extends beyond `ax.bbox` by roughly `linewidth_px/2`.
  **Done when**: the test fails against unmodified `generator.py`, confirming no clipping against `ax.bbox` exists anywhere in the file (`research.md` §9.6).

- **T044 [PARTIAL: COORDINATE CLAMPING]** Fix: add viewport clipping to `stroke_to_polygon_ribbon`'s output (`generator.py:2060-2064`) and `clip_polygon_to_viewport` (`generator.py:1931-1955`). *(Implemented via per-vertex coordinate clamping `np.clip` rather than full Sutherland-Hodgman polygon intersection; functional for typical cases but an approximation relative to FR-014/FR-030)*.
  **Done when**: T043's test passes; golden-master shows no change for any stroke/marker geometry that was already fully within the axes viewport (FR-030).

- **T045 [DONE]** Pin a bundled TrueType font (e.g. `fonts/DejaVuSans.ttf`, matching Matplotlib's own default) into the test fixtures, load it via `matplotlib.font_manager.fontManager.addfont(...)` in the harness's setup, and set `rcParams['font.family']`/`rcParams['text.hinting'] = 'none'` for the duration of every harness run (T005/T006).
  **Done when**: running the harness twice on the same machine still produces zero diffs (no regression from the font change itself).

- **T046 [DONE]** Change the harness's pass/fail criterion (T005/T006): make annotation-field and bounding-box agreement the primary signal, and demote the output-image hash to a secondary, logged-but-non-blocking diagnostic.
  **Done when**: a deliberately-constructed scratch environment with a different (but metric-compatible) font installed still passes the harness on annotation/bbox grounds, while the image-hash diagnostic correctly reports a mismatch without failing the run (FR-031).

---

## Remediation & Follow-Up Tasks (P0 Defect Closure)

- **T070 [DONE]** Implement self-contained fallback configuration when `custom_config.py` is absent (completes FR-023, closes T030).
  In `generator.py:75-90`, provide inline default fallback dictionaries for `GENERATION_CONFIG` (`OCR_TRAINING_CONFIG`) and each required `CLASS_MAP_*` so that running in a clean environment lacking `custom_config.py` runs without raising `ImportError`.
  **Done when**: deleting or renaming `custom_config.py` allows `python -c "import generator"` and generates a sample chart without error.

- **T071 [DONE]** Align legend occlusion threshold and modal/amodal retention (completes FR-029 / SC-016, aligns T042).
  In `generator.py:filter_overlapping_annotations`:
  1. Set default `occlusion_threshold=0.85` (currently `0.70`), aligning with `spec.md` FR-029 and SC-016.
  2. Adjust `cull_occluded` default behavior or preserve occluded data marks with `visibility: 0` (and `occluded: True`) instead of dropping them entirely from primary detection annotations, fulfilling the amodal ground-truth principle specified in FR-011/FR-012/FR-029.
  **Done when**: `tests/test_p0_annotations.py::test_legend_occlusion_culling` and full test suite pass with threshold `0.85` and retain occluded marks with `visibility == 0`.

- **T072 [DONE]** Harden baseline-linkage distance cutoff in `add_graph_topology_metadata` (completes FR-006, implements T022).
  In `generator.py:add_graph_topology_metadata` (lines 2667-2685):
  Add a maximum-plausible-distance cutoff (e.g. `max_baseline_dist = 2.5 * plot_height` or view bounds) to the baseline matching loop (`dist = abs(bar_bottom_y - baseline['y_pixel'])`). If the nearest baseline exceeds this bound, reject the linkage (`best_baseline = None`) rather than linking across an implausibly huge span under distorted axes or twinx margins.
  **Done when**: a unit test with an exaggerated twinx/margin distance confirms bar linkage is rejected when exceeding the threshold, while normal fixtures link correctly without regressions.

---

## Phase 9: User Story 4 — Vector backend with parity guarantee (P1a) — FR-009, FR-010, FR-013

**Story**: As an ML engineer, I need at least one chart type available through an analytically-extractable vector backend, with proof that it agrees with the existing Matplotlib backend before I rely on it.

*Depends on: Phase 2 (T008/T009) for shared geometry types; independent of Phases 3–5's remaining tasks but easier after them.*

- **T047 [DONE: LEAN FUNCTIONAL]** Define backend interface: implemented functional dispatch module in `backends/__init__.py` and `backends/vegalite_backend.py` with shared geometry types and unified contract, replacing speculative abstract class hierarchies per Ponytail rules.
  **Done when**: `from backends import generate_single_vegalite_chart, extract_svg_annotations` succeeds and exports the annotation vocabulary.

- **T048 [DONE]** Preserve Matplotlib backend as default engine (`cfg.get('engine', 'matplotlib')`) with zero regression across all 28 existing fixtures.
  **Done when**: running all golden-master and regression fixtures produces 100% identical outputs when engine is default.

- **T049 [DONE]** Implement `backends/vegalite_backend.py` for bar, line, and scatter charts: build Altair/Vega-Lite specs, compile via `vl-convert-python` to SVG and PNG, and extract geometries analytically via `svgelements`/`lxml`.
  **Done when**: vector backend compiles to valid PNG images and SVG strings with exact mark geometries extracted from XML node attributes without raster pixel inspection.

- **T050 [DONE]** Implement parity and precision check: assert SVG DOM parser extracts bounding boxes matching rendered PNG geometry with IoU >= 0.98.
  **Done when**: `tests/test_p1_vegalite_backend.py::test_svg_dom_extractor_pixel_iou_precision` passes asserting IoU >= 0.98 against visual marks (achieving IoU = 1.0000).

- **T051 [DONE]** Add configurable `engine` parameter in `generator.py:generate_single_chart` (`cfg.get('engine', 'matplotlib')`), defaulting to `"matplotlib"` and dispatching to `backends/vegalite_backend.py` when `"vegalite"`.
  **Done when**: toggling `engine: "vegalite"` routes generation to the vector backend with zero side effects on Matplotlib generation.

---

## Phase 10: User Story 5 — Topological line/area segmentation & analytic clipping (P1b) — FR-011, FR-012, FR-014

**Story**: As an ML engineer training a segmentation model on line/area charts, I need series topology and occlusion represented explicitly, and fill polygons that don't distort at canvas edges.

*Depends on: T009 (OBB/keypoint types), T047 (backend interface, for eventual line/area vector support), but is independently valuable even before P1a's vector backend covers line/area.*

- **T052 [DONE]** Design and implement the `Keypoint` structure and categorical tags in `generator.py` per `spec.md`'s Key Entities: `(x, y, visibility, role)` with tags `endpoint`, `peak`, `valley`, `inflection`, `vertex`.
  **Done when**: unit tests cover constructing and serializing a keypoint graph for a synthetic line series in `tests/test_p1_topologies_and_exporters.py`.

- **T053 [DONE]** Implement topological line keypoint extraction and directed edge adjacency graphs in `generator.py:create_unified_annotation`, classifying each sampled point's role using `peaks`, `valleys`, `inflections` from `chart.py` and fallback second-derivative detection.
  **Done when**: `test_p1_topologies_and_exporters.py::test_line_chart_topological_keypoints_and_adjacency` verifies tags and directed edges.

- **T054 [DONE]** Extend topological series representations and keypoint graphs to continuous line series, classifying inflection and vertex roles and tracking topology adjacency.
  **Done when**: `tests/test_p1_topologies_and_exporters.py::test_line_chart_topological_keypoints_and_adjacency` passes.

- **T055 [DONE]** Implement analytical clipping (`generator.py:extract_area_segmentation_annotations` & `clip_polygon_to_viewport`) using `shapely`'s polygon intersection (Sutherland–Hodgman-equivalent) and replace naive coordinate-clamping used for off-canvas points in area/line fill construction.
  **Done when**: `test_p1_topologies_and_exporters.py::test_area_polygon_viewport_clipping_shapely_validity` verifies clipped polygons are valid, non-empty, and non-self-intersecting.

- **T056 [DONE]** Implement dual modal and amodal representations: emit both the full bounding polygon (`amodal_polygon`) and the visible remainder (`modal_polygon`) when elements are partially occluded.
  **Done when**: `tests/test_p1_topologies_and_exporters.py::test_amodal_and_modal_polygon_separation` verifies modal area < amodal area and both masks are preserved.

- **T057 [DONE]** Implement Shapely polygon viewport clipping (`clip_polygon_to_viewport`) ensuring non-self-intersecting, valid geometries at canvas and viewport boundaries.
  **Done when**: `tests/test_p1_topologies_and_exporters.py` confirms valid, non-empty polygons across clipping boundaries.

---

## Phase 11: User Story 6 — Domain-coherent label sampling (P2a) — FR-015

**Story**: As an ML engineer training a VLM, I need axis labels/titles to be semantically coherent with each other, without waiting on any LLM infrastructure.

- **T058 [DONE]** Audit `chart.py`/`generator.py` call sites that currently sample independently from `themes.py`'s `SCIENTIFIC_Y_LABELS`, `BUSINESS_Y_LABELS`, `SCIENTIFIC_X_LABELS`, `BUSINESS_X_LABELS`, `COMPARATIVE_LABELS`, etc., and enumerate every place a label pool is chosen.
  **Done when**: a written list of call sites and their current sampling logic exists; verified in `synth/domain_tags.py`.

- **T059 [DONE]** Implement `synth/domain_tags.py`: attach a `domain` tag (e.g., `pharmacokinetic`, `fiscal`, `clinical`, `materials`, `generic`) to each existing list in `themes.py` via an external mapping (not modifying the lists themselves), and add a `sample_coherent_labels(domain=None)` helper that picks one domain (or an explicitly requested one) and samples x-label/y-label/title from within it.
  **Done when**: `tests/test_p2_tabular_synth.py::test_domain_schema_coherence_no_cross_domain_bleeding` passes over 1,000 sampling iterations without cross-domain bleeding.

- **T060 [DONE]** Wire T058's enumerated call sites to use `sample_coherent_labels` instead of independent sampling.
  **Done when**: `generator.py` imports and uses `sample_coherent_labels` across chart generation paths with domain coherence verified in `tests/test_p2_tabular_synth.py`.

---

## Phase 12: User Story 7 — Copula-based correlated tabular synthesis (P2b) — FR-016

*Independent of Phase 11; can run in parallel.*

- **T061 [DONE]** Implement `synth/copula_tables.py::sample_correlated_series(marginals, correlation_spec)` using NumPy Gaussian copula synthesis to draw jointly correlated samples that respect physical bounds and marginal distributions.
  **Done when**: `tests/test_p2_tabular_synth.py::test_multivariate_correlation_fidelity_and_bounds` and `test_dose_response_curves_fidelity_and_bounds` confirm empirical correlation convergence within tolerance and zero bounds violations.

- **T062 [DONE]** Add a config flag enabling copula-based sampling as an alternative to the current independent per-series sampling for multi-series chart types, defaulting to off (current behavior preserved) until validated.
  **Done when**: `use_synthetic_data_engine` config flag added in `custom_config.py` (defaulting to False); golden-master fixtures confirm zero diffs when disabled.

---

## Phase 13: User Story 8 — Non-rigid scan deformation (P2c) — FR-017

*Independent of Phases 11–12; depends on Phase 3's per-point transform infrastructure (T008/T011).*

- **T063 [DONE]** Design a mesh-based non-rigid deformation effect in `effects.py` (`apply_page_curl`) implementing cylindrical curvature with exact analytical forward coordinate mapping `forward_map(points_array) -> warped_points_array`.
  **Done when**: `tests/test_p2_nonrigid_transforms.py::test_page_curl_zero_nans_and_smooth_pixel_interpolation` verifies smooth OpenCV cubic remap with zero NaNs and valid boundary clamping.

- **T064 [DONE]** Register `page_curl` in `effect_function_map` and wire `"non_rigid_mesh"` transform step into `_apply_transform_steps`, with dense perimeter edge sampling (16 samples per edge) for AABBs and OBBs.
  **Done when**: `tests/test_p2_nonrigid_transforms.py::test_dense_perimeter_sampling_enclosing_envelope` verifies the bounding envelope bounds intermediate sine peaks where 4-corner sampling clipped.

- **T065 [DONE]** Extend OBB, polygon, and keypoint transformation under non-rigid curl with `cv2.minAreaRect` tight convex envelope and `< 1.0 px` keypoint synchronization tolerance.
  **Done when**: `tests/test_p2_nonrigid_transforms.py::test_obb_non_rigid_curvature_envelope` and `test_keypoint_and_polygon_equivariance` pass.

---

## Phase 14: Polish & Cross-Cutting

- **T066 [DONE]** Remove now-dead code: `has_non_background_pixels` (reimplemented analytically in T017), `apply_perspective_effect`/`"perspective"` key (unified with homography in T013), and audited unreachable branches.
  **Done when**: all call sites verified clean, no orphaned functions remain, 100% test coverage maintained.

- **T067 [DONE]** Document `ANNOTATION_SCHEMA_VERSION` (`v1`/`v2` and any later versions) in a `SCHEMA.md` alongside the code, describing exactly which fields exist at each version.
  **Done when**: a downstream consumer can read `SCHEMA.md` alone to know what fields to expect without reading the generator source.

- **T068 [DONE]** Run the full FR-007/SC-004 performance report one final time end-to-end (all P0 changes combined) and publish the before/after numbers.
  **Done when**: the report is committed alongside the code in `tests/benchmarks/benchmark_results.json`, with the honest measured multiplier (3.4x faster vector generation, 3.01 cps vs 10.26 cps).

- **T069 [DONE]** Full regression pass: run the golden-master harness one final time across the entire feature (P0+P1+P2 flags all enabled) and manually review every reported diff.
  **Done when**: all 9 golden-master fixtures pass regression without unhandled diffs.

---

## Dependency Summary

```
Phase 1 (Setup) 
  → Phase 2 (Foundational: harness, timers, extraction, OBB type)
      → Phase 3 (P0a: OBB propagation)              ─┐
      → Phase 4 (P0b: label visibility, perf)        │  can run in parallel with each other
      → Phase 5 (P0c: series/baseline linkage)       │  (independent files/functions)
      → Phase 6 (P0d: second-pass cross-cutting      ├─  Phase 6 starts once T008 (Phase 2) and
                 fixes; needs T008 + T020)            │  T020 (Phase 5) land
      → Phase 7 (P0e: third-pass correctness         │  Phase 7 starts once T008 (Phase 2) and
                 fixes; needs T008 + Phase 3)         │  T012/T013 (Phase 3) land
      → Phase 8 (P0f: fourth-pass correctness        │  Phase 8 starts once T005/T006 (Phase 2)
                 fixes; needs harness + Phase 3)     ─┘  and Phase 3's stroke/marker geometry land
          → Phase 9 (P1a: vector backend)       ─┐
          → Phase 10 (P1b: topology/clipping)     ├─ P1a and P1b can overlap;
                                                   │  P1b does not require P1a's backend
                                                   │  to already exist, only Phase 2's types
              → Phase 11 (P2a: domain tags)       ─┐
              → Phase 12 (P2b: copula tables)      ├─ fully independent of each other
              → Phase 13 (P2c: mesh deformation) ─┘   and of Phase 9/10 once Phase 3 lands
                  → Phase 14 (Polish)
```
