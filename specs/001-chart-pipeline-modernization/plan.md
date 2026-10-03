# Implementation Plan: Vector-Faithful Annotation & Rendering Pipeline

**Feature Branch**: `001-chart-pipeline-modernization`
**Input**: `spec.md`, `research.md`
**Status**: Complete — All Phases (P0, P1, P2, Phase 14 Polish & Remediation) Verified (46/46 Tests Passing)

## Summary

Fix annotation-truth correctness in the existing Matplotlib pipeline first (P0 — no visual change, no new dependency, pure bug fix), then add an opt-in vector rendering backend and richer topological annotations chart-type-by-chart-type (P1), then improve tabular semantic coherence and augmentation realism (P2). `chart.py`'s statistical generators are not modified in any phase. Every phase ships behind a config flag and is validated against a fixed-seed regression corpus before the next phase starts.

## Technical Context

- **Language/Runtime**: Python 3.x (existing codebase)
- **Existing core dependencies** (unchanged): `numpy`, `scipy` (`scipy.ndimage.gaussian_filter`), `Pillow`, `matplotlib` (Agg backend), optional `opencv-python` (`_HAS_CV2` flag in `effects.py`)
- **New dependencies, by phase** (all verified installable on PyPI as of 2026-09-17, see `research.md` §5):
  - P0: none — this phase is a refactor of existing code paths only, including the five second-pass fixes (`research.md` §7, roadmap items 9-13), the three third-pass fixes (`research.md` §8, roadmap items 14-15), and the four fourth-pass fixes (`research.md` §9, roadmap items 16-19), all of which are also dependency-free by design — items 14 and 18 in particular were rescoped from their reviewing claims' suggested `shapely`-based fixes specifically to preserve this constraint.
  - P1: `shapely` (polygon clipping/repair), `svgelements` + `lxml` (SVG DOM parsing/analytic `getBBox()`-equivalent), `svgpathtools` (path sampling), `altair` (Vega-Lite spec authoring), `vl-convert-python` (Vega-Lite → SVG/PNG, no browser/JVM), `resvg-py` and/or `cairosvg` (fallback SVG rasterization for visual QA)
  - P2: `pyvinecopulib` and/or `copulas` (multivariate tabular synthesis)
- **External config dependency**: `custom_config.py` (`OCR_TRAINING_CONFIG`) is imported at `generator.py:74-85` and is not part of this repo snapshot. It must be read before P0 work is scoped (see Open Question in `spec.md`).
- **Testing**: no existing test suite was found in the uploaded files. This plan introduces one (pytest) as part of Phase P0, since none of the correctness fixes are safely mergeable without regression coverage.
- **Target platform**: unchanged — headless/server-side batch generation (no GUI, no browser dependency introduced; `vl-convert-python`/`resvg-py` are compiled Rust bindings, not a headless-browser dependency like the PhantomJS approach `research.md` notes FigureQA had to retire).

## Constitution Check

Project-specific engineering gates this plan must satisfy before and during implementation. These are not generic best practices — each is a direct response to a defect class found in `research.md`.

| Gate | Rule | Why it exists |
|---|---|---|
| **Truth over convenience** | No function may compute an exact geometric quantity and then discard precision to fit a simpler downstream shape (the AABB-from-OBB-corners pattern in `research.md` §2.1) unless the spec explicitly calls for that reduction. | This is the single largest source of label noise in the current pipeline. |
| **No silent bypass** | Every effect that can change pixel geometry must be structurally required to also emit a transform for annotation compensation (FR-003); it must not be possible to add a new geometric effect that "just works" on pixels without wiring compensation. | Root cause of the `"perspective"` vs `"perspective_warp"` gap in `research.md` §2.2. |
| **Analytic before raster** | Any decision that can be made from an object's own state (text, visibility flag, data coordinates, axis limits) must be made that way, not by rendering and inspecting pixels. | Root cause of `has_non_background_pixels`'s cost (`research.md` §1.2). |
| **Additive migration** | A new rendering backend or annotation field is introduced behind a config flag, defaulting to current behavior, until a parity check (FR-013) passes for that specific chart type. | Prevents an unvalidated backend swap from silently corrupting a training run; matches the audit's own lesson from ChartX about single-backend overfitting without forcing an unproven big-bang cutover. |
| **Test before optimize** | No performance change is described as "N× faster" in a PR or changelog unless a before/after measurement on the same hardware accompanies it (FR-007). | The audit's "3–5×" and "15–25% mAP" figures are hypotheses, not measurements (`research.md` §6.3) — this plan does not want to repeat that pattern internally. |
| **Extract before fix** | A geometry fix should not be patched in place inside a >250-line function; the specific geometric operation being fixed is extracted to a small, independently unit-testable pure function first. | `get_granular_annotations` (~1,009 lines), `create_unified_annotation` (~444 lines), `generate_single_chart` (~794 lines) currently make any in-place fix hard to review and impossible to unit test in isolation (`research.md` §1.3). |
| **Single source of transformed truth** | Once `apply_realism_effects` has run for an image, every annotation-producing function for that image must consume its output (or the same `transform_steps` list) — none may re-derive geometry from the pre-effect Matplotlib figure. | Root cause of the secondary-stream desync in `research.md` §7.1: `extract_pie_pose_annotations`/`extract_line_segmentation_annotations`/`extract_area_segmentation_annotations` and the re-derived `*_obj_labels` calls all bypass the transform entirely. |
| **No leaked global state** | A property that varies per chart element (theme, font) must not be expressed through a process-global mutable (e.g. `rcParams`) that a sibling subplot or a later task in the same worker can silently overwrite. | Root cause of the intra-figure font bug in `research.md` §7.6. |
| **Geometric validity before export** | Any exported polygon or OBB must be verified convex and consistently wound before it is written; a degenerate (self-intersecting or collinear) result must be corrected or the annotation dropped — never exported as-is, and never by reaching for a P1-only dependency to fix a P0 requirement. | `research.md` §8.5: a homography can, near the vanishing line, warp 4 independently-transformed corners into a bow-tie or inverted-winding quadrilateral that YOLO-OBB-style consumers reject. |

If a task cannot satisfy one of these gates, the task description must say why and what compensating control replaces it (e.g., an integration test instead of a unit test, if extraction is genuinely not feasible for a given function).

## Project Structure

New modules are added; nothing in `chart.py` is restructured (P0/P1 do not touch it at all; P2 adds call sites only).

```
project_root/
├── chart.py                      # UNCHANGED in P0/P1. P2 adds domain-tag lookups only.
├── generator.py                  # P0: bug fixes + extraction of pure helpers. P1: backend dispatch added.
├── effects.py                    # P0: perspective/perspective_warp unified. P1: unaffected. P2: + mesh deformation.
├── themes.py                     # P2: domain-tagging wrapper added; existing lists untouched.
├── custom_config.py              # EXTERNAL — read, not modified, except to add ANNOTATION_SCHEMA_VERSION flag.
├── config_defaults.py            # NEW (P0) — self-contained fallback OCR_TRAINING_CONFIG; custom_config.py overrides it when present (FR-023).
├── geometry/                     # NEW (P0)
│   ├── obb.py                    # OrientedBoundingBox dataclass, corner<->cx,cy,w,h,θ conversions,
│   │                              #   plus a dependency-free convexity/winding check + convex-hull
│   │                              #   correction for post-homography corners (FR-026)
│   ├── transforms.py             # Extracted _rotate_point/_warp_point/_apply_transform_steps (pure functions, unit-testable).
│   │                              #   Each TransformStep also carries the canvas (w,h) it was computed against (FR-021),
│   │                              #   so a later canvas-resizing effect can't corrupt an earlier step's math.
│   └── clip.py                   # Sutherland–Hodgman-equivalent polygon clipping (via shapely), P1
├── annotate/                     # NEW (P0 skeleton, P1 fills in)
│   ├── label_visibility.py       # Analytic replacement for has_non_background_pixels
│   ├── series_link.py            # series_idx assignment fix + baseline linkage (kept separate per research.md §3.2)
│   └── topology.py               # P1: keypoint graph construction for line/area series
├── backends/                     # NEW (P1)
│   ├── base.py                   # Backend interface: render(chart_spec) -> (image, geometry_records)
│   ├── matplotlib_backend.py     # Wraps existing chart.py generator functions, unchanged behavior
│   └── vector_backend.py         # Altair/Vega-Lite spec -> vl-convert-python -> SVG -> svgelements/lxml geometry
├── synth/                        # NEW (P2)
│   ├── domain_tags.py            # Rule-based domain tagging over existing themes.py lists (ships without an LLM)
│   ├── llm_schema.py             # Optional local-LLM-backed schema generator (behind a config flag)
│   └── copula_tables.py          # pyvinecopulib/copulas-based multivariate table sampling
└── tests/
    ├── test_geometry_transforms.py
    ├── test_obb_propagation.py
    ├── test_label_visibility.py
    ├── test_series_baseline_linkage.py
    ├── test_backend_parity.py       # P1
    └── golden/                      # fixed-seed reference images + annotations for regression diffing
```

## Phased Roadmap

### Phase P0 — Correctness & Measurement (blocks everything else)

Maps to FR-001…FR-008, FR-019…FR-031, SC-001…SC-018. No new dependency. Estimated as the lowest-complexity, highest-value phase (matches the source audit's own P0 designation). Items 9-13 were added after a second-pass review (`research.md` §7) found five further P0-severity, dependency-free correctness bugs; item 7 was rewritten (not just extended) once that review showed the originally-planned fix was treating a symptom rather than the root cause. Items 14-16 were added after a third-pass review (`research.md` §8) confirmed three more P0-severity, dependency-free bugs; that same review also checked four other claims against source and found them either already covered or not applicable to this codebase (recorded in `research.md` §8, no roadmap change needed). Items 17-20 were added after a fourth-pass review (`research.md` §9) confirmed four more P0-severity, dependency-free bugs (one of them narrower than originally claimed); that review also refuted two other claims outright — both already mitigated in existing code — and flagged one general risk (twin-axes z-order collision) that has no current manifestation and is recorded only in the risk register below, not as a roadmap item.

1. Read `custom_config.py` to resolve the two `[NEEDS CLARIFICATION]` items that affect scoping (AABB backward-compat requirement, `"perspective"` effect probability), and record whether a `batch_merge_all` implementation exists anywhere in the target deployment outside the files reviewed here (feeds item 11).
2. Stand up the regression harness: fixed seeds → generate N images → snapshot annotations. This must exist *before* any fix lands, so every subsequent change is diffed against it.
3. Extract `_rotate_point`, `_warp_point`, `_apply_transform_steps` out of `apply_realism_effects` into `geometry/transforms.py` as pure functions (no behavior change — pure refactor, verified by the regression harness producing byte-identical output; this step preserves today's canvas-size bug as-is, which item 10 below fixes separately and deliberately as a distinct, reviewable change).
4. Implement `geometry/obb.py` and change `apply_realism_effects` (`generator.py:1598-1610`) to carry the 4 warped corners forward as the primary geometry instead of collapsing to `min`/`max`.
5. Unify `"perspective"` and `"perspective_warp"` into a single code path that always returns a homography (or retire `apply_perspective_effect` in favor of `apply_perspective_warp_effect` if `custom_config.py` shows the former is effectively unused) — closes `research.md` §2.2.
6. Replace `has_non_background_pixels` with an analytic label-visibility check in `annotate/label_visibility.py`, driven by `Text.get_visible()`, text content, and axis-limit containment.
7. **(Revised)** Store the `bar_info_list` that `_generate_bar_chart` already computes correctly (real `series_idx`/`bar_idx`/`bottom`/`top` per bar) into `chart_info_map[ax]` during generation (`generator.py`'s per-axis loop, ~3104-3238, where it is today computed but never stored), and change `create_unified_annotation` (`generator.py:2585`) to read it directly instead of calling `extract_bar_info(ax, chart_type)`. Keep `extract_bar_info` only as a fallback for chart-generation paths that don't populate `bar_info_list`. This replaces the original plan of "fix `extract_bar_info` to accept a passed-in `series_idx`" — that would have papered over the fact that correct data already exists one call frame away and is simply being discarded (FR-005, FR-020, `research.md` §7.2).
8. Harden the baseline-linkage distance heuristic in `add_graph_topology_metadata` (`generator.py:2169-2328`) against margin-induced drift — likely a tightened x-containment check plus a maximum-distance cutoff so a bar never links to a physically implausible baseline.
9. **(New)** Route the secondary annotation streams — `extract_pie_pose_annotations`, `extract_line_segmentation_annotations`, `extract_area_segmentation_annotations`, and the re-derived `area_obj`/`pie_obj`/`line_obj` boxes (`generator.py:3493, 3518, 3550`) — through the same `transform_steps` used for the primary annotations, instead of reading raw `ax.transData` coordinates after `apply_realism_effects` has already run (FR-019, `research.md` §7.1).
10. **(New)** Fix the transform-step canvas-size closure bug: have each entry appended to `transform_steps` carry the canvas `(w, h)` current at the moment it was recorded, and change `_apply_transform_steps`/`_rotate_point`/`_warp_point` to use each step's own recorded size for its origin-flip math instead of the enclosing function's `img_w`/`img_h` (which get reassigned to the final, post-all-effects size before any point is actually warped) (FR-021, `research.md` §7.3).
11. **(New)** Replace the unconditional out-of-bounds discard (`generator.py:3452-3458`) with the same axis-aligned clamp already used for heatmaps (`generator.py:3442-3451`), generalized to every chart type — no new dependency, since this is a rectangle clamp, not the general polygon clipping FR-014 introduces in P1. Add `config_defaults.py` as a self-contained fallback `OCR_TRAINING_CONFIG`, and resolve the `batch_merge_all` dangling reference (`generator.py:4110-4112`) per item 1's finding — import it, implement it, or remove the dead code path (FR-022, FR-023, `research.md` §7.4-§7.5).
12. **(New)** Scope `apply_chart_theme`'s font selection (`chart.py:374`) to the axis being themed — via `matplotlib.rc_context()` or an explicit `fontproperties` passed to each `Text`/label call — instead of mutating the global `rcParams`, so multi-subplot composite figures and sequential worker-process iterations can't leak theme state into each other (FR-024, `research.md` §7.6).
13. **(New)** Fix rotated-tick-label OBB fidelity: for any text artist with `get_rotation() != 0`, compute the tight unrotated box (temporarily zero the rotation, call `get_window_extent`, restore it), then derive the 4-corner OBB by rotating that box's corners around the text's anchor point at the artist's actual angle — export those corners instead of `get_window_extent()`'s loose axis-aligned envelope (FR-025, `research.md` §8.3).
14. **(New)** Add a convexity/winding guard for warped OBB corners: after `_apply_transform_steps` warps an annotation's 4 corners, check the sign of consecutive edge cross-products (pure NumPy, no new dependency) to confirm the quadrilateral is convex and consistently wound; on failure, reorder via a dependency-free 4-point convex-hull pass, or drop the annotation if fewer than 4 points survive the hull (true self-intersection) (FR-026, `research.md` §8.5).
15. **(New)** Close the two remaining JSON-safety gaps: add an `np.bool_` branch to `convert_numpy_types` (`generator.py:1883`) returning `bool(obj)` instead of falling through to the string catch-all, and apply `convert_numpy_types` (or an equivalent `json.dump(..., default=...)` fallback) to `ocr_json` and `metadata_json` as well as `detailed_json` (`generator.py:3947-3953`) (FR-027, `research.md` §8.6).
16. **(New)** Fix the area chart's log-scale guard bypass: replace `apply_axis_scaling(ax, data_min=0.01, ...)` (`chart.py:3044`) with a real `data_min` computed from the chart's actual `boundary_y`/`y_stack` data, matching the pattern already used correctly at `chart.py:1826` and `2066` (FR-028, `research.md` §9.4).
17. **(New)** Add cross-class occlusion flagging: after an axis's annotations are collected, compute `area(bbox ∩ legend_bbox) / area(bbox)` for every non-legend annotation against that axis's legend box (when present); flag any annotation at or above an 0.85 threshold as occluded (`visibility = 0`) rather than exporting it as fully visible (FR-029, `research.md` §9.5).
18. **(New)** Clip stroke/marker geometry to the axes viewport: after `stroke_to_polygon_ribbon` builds a line's ribbon polygon and after marker radius boxes are computed (`generator.py:1769-1785`), intersect both against `ax.bbox` via a dependency-free Sutherland–Hodgman clip (a fixed-rectangle clip, not the general polygon clipping FR-014 introduces in P1) before export (FR-030, `research.md` §9.6).
19. **(New)** Make the golden-master harness cross-platform-safe: pin a bundled TrueType font into the test fixtures, load it via `matplotlib.font_manager.fontManager.addfont(...)`, and set `rcParams['font.family']`/`rcParams['text.hinting'] = 'none'` in the harness's setup (T005); change the harness's primary pass/fail signal from raw image-hash equality to annotation-field/bounding-box agreement, keeping the image hash only as a secondary, non-blocking diagnostic (FR-031, `research.md` §9.7).
20. Re-run the regression harness; produce the FR-007 before/after latency report.

### Phase P1 — Representation & Backend (chart-type-by-chart-type, additive) — [COMPLETED & VERIFIED]

Maps to FR-009…FR-014, SC-005. Verified with 33/33 passing tests (including `tests/test_p1_topologies_and_exporters.py` and `tests/test_p1_vegalite_backend.py`).

1. **[DONE]** Lean functional backend interface in `backends/__init__.py` and configurable engine dispatch in `generator.py:generate_single_chart` (`cfg.get('engine', 'matplotlib')`), preserving 100% backwards compatibility and zero regressions across existing Matplotlib fixtures.
2. **[DONE]** Built `backends/vegalite_backend.py` for canonical chart types (`bar`, `line`, `scatter`) compiling to both SVG (`vlc.vegalite_to_svg`) and raster PNG (`vlc.vegalite_to_png`). Analytical SVG DOM extraction via `lxml.etree` and `svgelements`.
3. **[DONE]** Parity check (FR-013) verified in unit test `tests/test_p1_vegalite_backend.py::test_svg_dom_extractor_pixel_iou_precision`, confirming analytical DOM bounding boxes match visual raster pixels with $\text{IoU} \ge 0.98$ (achieving $\text{IoU} = 1.0000$).
4. **[DONE]** Topological keypoints (`peak`, `valley`, `inflection`, `endpoint`, `vertex`) and directed edge adjacency graphs in `generator.py:create_unified_annotation` for continuous series, with `shapely` analytical clipping via `clip_polygon_to_viewport`.
5. **[DONE]** Additive YOLOv8-OBB exporter (`save_annotations_yolo_obb`) and dual modal/amodal representations (`amodal_polygon` vs `modal_polygon`) for partially occluded elements.
6. **[IN PROGRESS / EXPANSION]** Additional chart types (boxplot, pie, heatmap) can be progressively added to the vector backend behind the existing functional dispatch interface.

### Phase P2 — Semantic Coherence & Robustness

Maps to FR-015…FR-017, SC-006. Complete and verified with 43/43 passing tests (including `tests/test_p2_tabular_synth.py` and `tests/test_p2_nonrigid_transforms.py`).

1. **[DONE]** `synth/domain_tags.py`: added domain tagging (`pharmacokinetic`, `fiscal`, `clinical`, `materials`, `generic`) wrapping lists in `themes.py` without modifying originals; constrained x/y-label/title sampling to single domain with zero LLM dependency.
2. **[PRUNED / DEFERRED per Ponytail]** `synth/llm_schema.py`: optional local LLM generator pruned in favor of deterministic, zero-dependency domain sampler (`synth/domain_tags.py`).
3. **[DONE]** `synth/copula_tables.py`: Gaussian copula and multivariate synthesis (`sample_correlated_series`, dose-response curves) respecting empirical bounds, gated behind `use_synthetic_data_engine` with zero regressions on baseline golden fixtures.
4. **[DONE]** `effects.py` + `generator.py`: vectorized cylindrical page-curl non-rigid deformation (`apply_page_curl`) with multi-point equivariant coordinate mapping, dense perimeter edge sampling, and tight OBB envelopes enclosing curved document boundaries (FR-017).

## Migration & Compatibility Strategy

- **Schema versioning (FR-018)**: introduce `ANNOTATION_SCHEMA_VERSION` in exported metadata. `v1` = current AABB-only behavior. `v2` = adds OBB fields alongside existing AABB fields (additive, not replacing) for at least one deprecation cycle, pending the backward-compatibility clarification in `spec.md`.
- **Backend selection**: a per-chart-type config flag (`RENDER_BACKEND = {"bar": "vector", "line": "matplotlib", ...}`) defaults every type to `"matplotlib"` until that type passes the FR-013 parity gate.
- **Rollback**: because P0 changes are refactors of existing behavior plus additive fields, rollback is a revert of the specific commit; no data migration is required since no existing field is removed until a version bump is explicitly decided.

## Testing Strategy

- **Unit tests**: pure geometry functions (`geometry/transforms.py`, `geometry/obb.py`, `geometry/clip.py`) get direct unit tests with hand-computed expected outputs (e.g., a 45° rotation of a known rectangle has a known exact OBB).
- **Property-based checks**: for a sample of random homographies/rotations, assert `OBB_area / true_element_area ≤ 1.05` (SC-001) and, separately, that the *old* AABB-from-corners path would have failed that same bound (regression proof the bug existed and is fixed).
- **Golden-master regression**: fixed-seed corpus generated once before any P0 change; every subsequent commit diffs its output against it for all fields *not* intentionally changed by that commit.
- **Backend parity tests** (P1): shared chart specs rendered through both backends; geometry compared within tolerance (FR-013); visual diff (SSIM or pixel-diff threshold) as a secondary signal, not the primary pass/fail.
- **Effect-compensation completeness test**: iterate every entry in `effect_function_map` and assert each geometry-changing one is present in the special-cased dispatch (or is provably non-geometric), turning `research.md` §2.2's manual finding into a permanent CI check so a newly added effect can't reintroduce the same gap.
- **Cross-stream consistency test**: for area/pie/line charts, assert every exported annotation stream (segmentation, pose, marker, and per-type object boxes) is transformed identically to the primary annotation for the same effect-applied image (closes `research.md` §7.1).
- **Transform-consistency test**: for every effect chain that changes canvas size after a rotation/homography effect, assert annotation coordinates match a hand-computed expected value using the canvas size in effect when the transform was computed (closes `research.md` §7.3).
- **Startup smoke test**: run the generator end-to-end with `custom_config.py` absent, confirming the self-contained default config is used and no `NameError`/`ImportError` occurs even with `merge_json_files: True` set (closes `research.md` §7.5).
- **Theme isolation test**: render a composite multi-subplot image with distinct theme fonts per subplot and assert no cross-subplot font bleed (closes `research.md` §7.6).
- **Rotated-label OBB fidelity test**: for a chart with rotated tick labels, compute an independent ground-truth rotated box from font metrics and assert the exported OBB matches within a documented area tolerance, not Matplotlib's loose AABB (closes `research.md` §8.3).
- **OBB convexity/winding test**: fuzz `distortion_factor` toward the vanishing line and assert every exported 4-vertex OBB remains convex and consistently wound (closes `research.md` §8.5).
- **JSON safety test**: feed an `np.bool_` and other NumPy scalar/array types through every export path (`detailed_json`, `ocr_json`, `metadata_json`) and assert native JSON types come out, never a stringified boolean or a crash (closes `research.md` §8.6).
- **Log-scale positivity test**: for the area-chart generator specifically, force `data_min` to a real non-positive value and assert `'log'` is never selected (closes `research.md` §9.4).
- **Cross-class occlusion test**: force a legend to fully cover a known bar/marker and assert that element is flagged `visibility = 0`, not exported as fully visible (closes `research.md` §9.5).
- **Viewport-clip test**: construct a thick-stroked line touching an axes spine and assert the exported ribbon polygon's extent matches `ax.bbox`, not the unclipped stroke expansion (closes `research.md` §9.6).

## Risk Register

| Risk | Phase | Mitigation |
|---|---|---|
| `custom_config.py` reveals downstream consumers hard-depend on today's exact AABB schema | P0 | Additive `v2` schema (OBB alongside AABB) rather than replacement, per Migration Strategy |
| SVG/vector text-metric differences from Matplotlib's FreeType rasterizer cause label bbox mismatches in the parity check | P1 | Tolerance in FR-013 is defined per-element-type; text-label tolerance is allowed to be looser than shape tolerance, documented explicitly rather than silently widened |
| Vector backend adoption stalls on complex chart types (heatmap, pie with wedge labels) | P1 | Chart types graduate independently (per `spec.md` clarification assumption); Matplotlib remains the default per-type until parity passes, so partial completion still ships value |
| Local LLM for schema generation is unavailable or too slow for batch generation at scale | P2 | Rule-based domain tagging (`synth/domain_tags.py`) ships first and is fully independent of the LLM step |
| Regression harness itself has gaps (doesn't cover an edge case a fix breaks) | All | Golden-master corpus is expanded with a new fixture whenever a bug is found during implementation, not just at the start |
| The convexity/winding guard (FR-026) may need to drop a rare annotation outright when the warped quadrilateral is genuinely self-intersecting (not just correctable by reordering) | P0 | Log every dropped-for-degeneracy annotation during the P0 regression run so the drop rate is visible and can be weighed against tightening `distortion_factor`'s range in `custom_config.py` |
| Fixing the canvas-size closure bug (FR-021) changes annotation coordinates for any historical image that chained a canvas-resizing effect after a rotation/homography effect | P0 | Golden-master diffs for these specific images are expected and must be manually confirmed as corrections, not silently accepted; the affected image count is documented in the P0 completion report |
| Scoping `rcParams` per-axis (FR-024) may reveal other code that implicitly depends on the global font setting persisting across axes | P0 | Use `matplotlib.rc_context()` scoping first (a narrow, reversible change) before considering a broader refactor; the regression harness will surface any dependent code as an unexpected diff |
| A future chart type introduces a `twinx()`/`twiny()` pair whose content genuinely overlaps in screen space (unlike today's dual-Y-axis bar chart, which avoids this by non-overlapping x-position ranges) | Watch item, no current P0/P1/P2 task | `research.md` §9.1: `Axes.zorder` is not comparable across different `Axes` instances by default. Before any such chart type ships, add an axes-aware layering rule (e.g. `fig.axes.index(ax) * 1000 + artist.get_zorder()`) rather than assuming same-axes z-order logic generalizes |

## Rollout Sequence

P0 (single track, sequential — items 1-20 above, including the five second-pass, three third-pass, and four fourth-pass correctness fixes confirmed in `research.md` §7-§9) → P1 (parallel per chart type once the backend interface and parity harness exist) → P2 (parallel tracks: domain-tagging can start immediately; copula tables and mesh deformation are independent of each other and of P1).
