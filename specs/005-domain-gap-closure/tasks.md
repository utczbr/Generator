# Tasks: Domain-Gap Closure

**Input**: `specs/005-domain-gap-closure/{spec,plan,research}.md`
**Legend**: `[P]` = can run in parallel within its phase; others are sequential. `FR-xxx` = requirement satisfied. Every task has a concrete **Done when**. Numbering continues the global sequence (spec 003 ended at T215); T254–T255 were added late and belong to Phase 1.
**Reproduction**: `REPO=<checkout> python specs/005-domain-gap-closure/research/repro_005.py <experiment>`.

---

## Phase 0: Make the Baseline Trustworthy (blocks everything)

- [x] **T216** Write failing reproducers first — `tests/test_p5_baseline_defects.py` (FR-073…078, FR-074b):
  * N-01: `multi_chart_detection`, one test per primary chart type (`bar, line, area, pie, scatter, box, histogram, heatmap`), 3 images each.
  * N-02: 24-image line/area run → no `class_name == "unknown"`.
  * N-03: forced 2×1 `scenario=multi` composite → annotations in **both** halves; a twin-axis chart annotates its second axis.
  * N-04: forced bar charts with error bars → `error_bar` count > 0.
  * N-06: `_detailed.json` contains `filter_stats`.
  * N-14: line/area charts emit chart-specific class IDs in `labels/*.txt` (not bar IDs).
  * N-15: `apply_legend_variation` with `ncol > 1` creates a multi-column legend with altered width/height extent.
  * N-20: twin-axis chart (`ax.twinx()`) generates zero ghost annotations for the invisible secondary X-axis.
  **Done when**: each test fails on current `main` for exactly the documented reason (`UnboundLocalError: clsmap_obj`, `unknown`, bottom half = 0, `error_bar` = 0, missing key, line using bar IDs, unchanged legend extent, phantom twin-axis boxes).

- [x] **T217** Fix N-01 — define `clsmap_obj` on every path that reads it (`generator.py:~4008-4030`; assigned only when `dataset_format != 'multi_chart_detection'` today) (FR-073).
  **Done when**: T216 N-01 tests pass; a 30-image run per primary chart type in `multi_chart_detection` raises 0 exceptions.

- [x] **T218 [P]** Fix N-02 — pass `CLASS_MAP_LINE_SEG`, `CLASS_MAP_LINE_MARKERS`, `CLASS_MAP_AREA_SEG` instead of `None` at `generator.py:4035,4037,4038` and `4451-4454`; make `detailed._serialize` (`detailed.py:517`) raise in strict mode / warn in lenient on unresolved classes (FR-074).
  **Done when**: T216 N-02 passes; `repro_005.py ink` reports `unknown` = 0%; strict-mode unit test raises on a synthetic unresolved class.

- [x] **T218b** Fix N-14 — resolve line/area class map fallback and stream duplicate IDs (FR-074, OC-8; sequential after T218, same lines of `generator.py`):
  * Key `CHART_CLASS_MAPS` so primary chart types `'line'` and `'area'` resolve to `CLASS_MAP_LINE_OBJ` and `CLASS_MAP_AREA_OBJ` respectively (`generator.py:3872,4029,4446`).
  * Ensure `labels/*.txt` for line/area charts contains true line/area class IDs (e.g. line segment = 1, axis labels = 6), not bar IDs.
  * Harmonize `annotations` and `annotations_obj` stream serialization in `_detailed.json` to prevent duplicated overlapping boxes.
  **Done when**: T216 N-14 passes; `repro_005.py streams` confirms 0 cross-stream duplicate pairs with IoU ≥ 0.8; `labels/*.txt` matches `line_obj_labels/*.txt` class IDs.

- [x] **T219** Fix N-03, N-20, **and** the dedup in one commit (OC-1) (FR-075, FR-076):
  * Move `return annotations` (`generator.py:1413`) out of the `for ax_idx, ax in enumerate(fig.axes)` loop.
  * Prevent N-20 phantom annotations: unconditionally gate X-axis tick labels and title on `if ax.xaxis.get_visible():`, and Y-axis on `if ax.yaxis.get_visible():` across all chart-type branches (`generator.py:535-560` and equivalents).
  * Rewrite the X-label dedup (`generator.py:3952-3965`) to evaluate each axis against **its own** `get_window_extent()` and key duplicates by `(axis, x-center)`.
  * Keep `subplot` attribution correct for twin/aux axes.
  **Done when**: T216 N-03 and N-20 pass; for a 2×1 composite the lower subplot's `axis_labels` count is within ±20% of the upper's (same chart family); twin-axis y2 tick labels are annotated; zero phantom x2 tick boxes are emitted; `repro_005.py multi-axes` shows annotations in both halves.

- [x] **T220 [P]** Fix N-04 — handle `ErrorbarContainer` through `artist.lines` (`barlinecols`, `caplines`) at `generator.py:~1373-1384` instead of `artist.get_window_extent` (FR-077; **OD-4**: fix vs retire).
  **Done when**: T216 N-04 passes and the number of `error_bar` annotations equals the number of bars with error bars (±0 on a 20-image forced run); or, if retired, `docs/SCHEMA.md` records the retirement.

- [x] **T220b [P]** Fix N-15 — fix legend column layout in `apply_legend_variation` (`chart.py:620,622,628`) (FR-074b; dormant, since line/area legends carry ≤ 4 series, so non-blocking for baseline-2 but required before T234b):
  * Pass `ncols=ncol` directly to `ax.legend(...)` (or invoke `legend.set_ncols(ncol)` where supported in matplotlib), replacing `legend._ncol = ncol`.
  **Done when**: T216 N-15 passes; `repro_005.py legend` shows an 8-entry legend expanding width from 66 px to ~149 px under `ncols=2`.

- [x] **T221 [P]** Filter accounting + schema v4.1 (FR-078, FR-108):
  * Replace print-only discards in the size/aspect/viewport/duplicate/overlap filters (`generator.py:4057-4076` and neighbours) with counters `{reason: {class_name: n}}`.
  * Write `filter_stats` into `_detailed.json`; bump `ANNOTATION_SCHEMA_VERSION_V4` → `"v4.1"`, `DATASET_VERSION_V4` → `"4.1.0"`; add a v4.1 section to `docs/SCHEMA.md`.
  **Done when**: T216 N-06 passes; summed `filter_stats` equals (annotations before − after filters) on a 24-image run; schema tests updated and green.

- [x] **T222 [P]** Reproducible environment (FR-079, **OD-5**):
  * Unified `requirements.txt` (core + tools + testing); CI installs it directly.
  * Register a `slow` marker; document matplotlib bound (`<3.11` until goldens verified).
  * `scripts/env_fingerprint.py` → OS, Python, matplotlib, FreeType, font-file hashes.
  **Done when**: a clean clone with `pip install -r requirements.txt` collects the whole suite with 0 import errors; fingerprint prints deterministically.

- [x] **T223 [P]** Triage the 13–14 pre-existing failures (FR-080, research §9.8):
  * `tests/test_m0_goldens.py` (2) → update test to assert legacy catalogs (`legacy_metrics`, `canonical_metrics`, `histogram_y_labels`) strictly for content/order, and assert aggregate counts (`all_metrics`, `all_titles`, `all_pairs`) as lower bounds (`>= 495`, `>= 192`, `>= 96`); resolved fully by T224.
  * Cold startup latency (2): `test_startup_latency.py`, `test_nfr01_import_guard.py` → re-baseline 50 ms budget to 150 ms scaled by measured machine factor to accommodate 6× registry growth (3,197 metrics), while maintaining zero-pydantic import guard.
  * Semantic catalog drift (8): `test_domain_coherence_e2e.py` (4), `test_title_sampling.py` (1, isolate 13 catalog title duplicates), `test_m5_semantic_generalization.py` (1), `test_semantic_catalog.py` (2), `test_semantic_sampling.py` (1) → fix data or `xfail(strict=True, reason=…)` with issue link.
  * Subprocess timing (1 on slow/single-core CPU): `test_m4_pipeline_e2e.py` → adjust timeout window to 10s.
  **Done when**: `pytest tests/ --ignore=tests/regression` has 0 unexplained failures (each remaining one is strict-xfail with a documented reason).

- [x] **T224** Generate and commit **baseline-2** goldens (FR-081, OC-2, OC-8):
  * Create `tests/golden/` directory; add `--update-goldens` switch to `test_m0_goldens.py`.
  * Regenerate `tests/golden/seeded_generation_20_baseline.json` and `registry_catalogs_baseline.json` (JSON only; `*.png` is git-ignored).
  * Embed the T222 environment fingerprint in baseline metadata; allow local non-canonical developer verification while enforcing strict sha256 byte-match in CI container.
  **Done when**: `tests/test_m0_goldens.py` passes from a clean checkout; the files are tracked by git; the changelog entry describing the intentional legacy-output change (R1) is merged.

- [x] **Phase 0 gate**: Passed — SC-002 sweep clean · `unknown` = 0 · `error_bar` > 0 · line/area class IDs aligned · zero ghost twin-axis annotations · all visible axes annotated · suite green/strict-xfail · baseline-2 committed.

---

## Phase 1: Profiles and the Changes That Survived Verification

- [x] **T225** Profile loader (FR-083): `--profile NAME` + `profile` key; overlays in `profiles/`; resolved via `config_loader.load_config` with provenance tag `profile`; ship `legacy.json` (empty) and `domain_gap_v1.json` (filled by later tasks); unknown profile → clear error.
  Record the profile name and a hash of the resolved config in `_detailed.json` metadata and the dataset manifest (FR-084) — omitted under `legacy` to keep it byte-identical.
  **Done when**: `--validate-only --profile domain_gap_v1` prints per-key provenance; running `legacy` reproduces baseline-2 hashes; a `domain_gap_v1` run's `_detailed.json` carries the profile name and config hash.

- [x] **T226** Feature-scoped RNG (FR-082, OC-3): `feature_rng.py: feature_rng(seed, image_idx, name) -> random.Random` using `zlib.crc32`/`hashlib` (never `hash()`).
  **Done when**: a test in two subprocesses with different `PYTHONHASHSEED` yields identical sequences; baseline-2 goldens still pass with the module imported.

- [x] **T227** Config validation + safe defaults (FR-091): reject effect params not in the function signature (**error** in non-legacy profiles, **warning** under `legacy` so existing user configs keep working); reject `perspective.magnitude > 0.15` in non-legacy profiles; change legacy default `perspective.magnitude` 0.5 → 0.08 (`config_defaults.py:398`).
  **Done when**: a typo like `amplitud_ratio` errors under `domain_gap_v1`; baseline-2 hashes unchanged after the default change.

- [x] **T228 [P]** Effects overlay (FR-090, FR-092): fill `profiles/domain_gap_v1.json` with the bounded parameters; add `intensity_range` and randomized `gradient_type` to `apply_uneven_lighting_effect` (`effects.py:285`); scalar `intensity` stays valid.
  **Done when**: overlay values equal FR-090; mean luminance of `uneven_lighting` outputs stays ≥ 195 on a 24-image forced run.

- [x] **T229** Effects-integrity gate (FR-093): `tests/test_p5_effects_integrity.py` (slow) — for each effect enabled in any profile, forced `p=1` over ≥ 16 images, plus one composed case: ≥ 98% annotations preserved, ≤ 1% text boxes with < 5% ink.
  **Done when**: passes for `domain_gap_v1`; a negative control (`perspective 0.5`) fails the gate as in research §4; a second control, `canvas_reframe` with `use_guard: false` and zero slack, fails the off-canvas-OBB check (prototype: 16/400 vs 3/400).

- [x] **T230 [P]** Tick rotation (FR-088): `tick_rotation` config; profile weights provisional; `ha` by angle sign, centered at |90°|; default `rotation_mode`; honor `label_angle` in Matplotlib only under `profile`; document Vega-Lite-only otherwise (`chart.py:427-432`).
  **Done when**: for the reference label, |tick-to-label gap| ≤ 15 px at ±30/45/60°; OBB-vs-ink test passes for each angle × `ha`; `repro_005.py ticks` reproduces the table with the new rule.

- [x] **T231 [P]** Figure geometry + typography scale (FR-086): `figure.size_mode`; aspect set `{4:3, 16:9, 1:1, 3:4, 5:4, 16:10, 3:2}`; width 5.5–9.5 in; DPI set from config; scale fonts, line widths, marker sizes, pads by `clamp(sqrt(W·H/35), 0.75, 1.25)`; single-chart scenario only in this task.
  **Done when**: over 48 images SC-005 holds (0 `tight_layout` warnings, collisions ≤ baseline + 0.10, ≤ 1% empty text) and ≥ 8 distinct aspect ratios appear.

- [x] **T231b [P]** Wire `PUBLICATION_THEMES` into figure sizing and native DPI generation (FR-086):
  * When `figure.size_mode == "publication"`, select from the 9 `PUBLICATION_THEMES` (`nature`, `science`, `cell`, `lancet`, `pnas`, `bmj`, `nanotech_short`, `open_access_poster`, `print_high_contrast`; 3.2–8.0 in, 300–600 DPI). Their fonts (Arial/Helvetica/Times New Roman) MUST resolve through the T234 aliases; without them they render as DejaVu Sans.
  * Direct rendering at target native DPI avoids down/up-sampling artifacts for large target sizes ≥ 1200 px.
  * Adjust typography and line-widths to journal publication norms.
  * **Done when**: forced publication run generates figures adhering to target physical dimensions and DPI metadata without `tight_layout` clipping.

- [x] **T232** `resize` effect (FR-085): render at normal size → resample to a long side drawn from a configurable distribution (default 256–2400, anti-aliased on downscale; for target ≥ 1200 px, prefer native DPI rendering per FR-085/FR-086); add `("scale", sx, sy, new_w, new_h)` to `transform_steps`; compensate AABB, OBB, polygons, keypoints in `apply_realism_effects` (`generator.py:1742ff`).
  **Done when**: a unit test with synthetic boxes/polygons/keypoints scales exactly; over 24 images at scales 0.35–2.0 text boxes stay aligned (≤ 1% empty at s ≥ 0.5); composed with `scan_rotation` + `perspective` the compensation order is correct.

- [x] **T233** Resolution-aware size policy (FR-087, **OD-6**): `annotation_min_side_px` (legacy 8 semantics; profile 4); sub-threshold annotations kept in `_detailed.json` as `ignored: true, ignore_reason`, excluded from YOLO/COCO exports unless requested; counted in `filter_stats`.
  **Done when**: at scale 0.5 no text annotation disappears silently (all either exported or `ignored` with reason); legacy output unchanged (baseline-2 passes).

- [x] **T234 [P]** Bundled fonts (FR-089, **OD-1**): `assets/fonts/` (open-licensed families covering sans, serif, mono; license files; `manifest.json` with sha256); `font_registry.py` registers via `font_manager.addfont` in sorted order, runs the glyph-coverage check, raises on missing font/glyph in non-legacy profiles; ships the alias table (Arial/Helvetica → bundled sans, Times New Roman → bundled serif) that T231b depends on; record chosen family in `_detailed.json`.
  **Done when**: SC-007 holds (≥ 20 distinct rendered faces by `findfont` path, 0 missing glyphs for the manifest character inventory, same choices for the same seed on two machines); total size ≤ 8 MB; `import generator` time does not regress (import guard green).

- [x] **T234b [P]** Granular Legend Extraction & Multi-Series Legend Fix (FR-074b, FR-114, OC-9):
  * Fix legend generation frequency: ensure multi-series bar, scatter, line, and area charts spawn legends at rates representative of publication figures (≥ 60% for multi-series).
  * Implement `extract_legend_elements(legend, ax)` to decompose `Legend` artists into `legend_title` (if present), `legend_marker` (handles), and `legend_label` (texts) with individual bounding boxes, OBBs, and polygon masks.
  * Emit decomposed elements into the dedicated `legend_elements` stream in `_detailed.json` and COCO-O export while preserving the monolithic `legend` box in `labels/*.txt` (per OC-9).
  * **Done when**: empirical test across 50 multi-series charts extracts individual legend entries with no degenerate (< 2 px) boxes, median ink per class and handle type above the calibrated threshold (FR-104; E16 shows raw marker-less line handles are zero-height), and 0 unmapped components; YOLO txt preserves the single monolithic legend box.

- [x] **T234c [P]** Tick Formatters and Numeric Clutter (FR-088):
  * Wire Matplotlib tick formatters (`ScalarFormatter` with scientific notation/powers-of-ten offset, `PercentFormatter`, `LogFormatter`, `FuncFormatter` for currency/units) into axis creation under profile.
  * Annotate scientific multiplier offsets (e.g. `×10⁴`) as `axis_label` or distinct text element with correct extent.
  * **Done when**: scientific notation multipliers and formatted percentage/currency ticks appear in ≥ 25% of profile-generated charts with verified bounding boxes.

- [x] **T254** `canvas_reframe` effect + content guard (FR-109, FR-110, FR-111; needs T226, T227): `apply_canvas_reframe_effect` in `effects.py`; registry entry; guard computed in `apply_realism_effects` (ink box ∪ AABB/OBB/polygon/keypoint geometry of all streams, mapped through current steps, clamped to the canvas, densified plus `minAreaRect` corners after a mesh step); `use_guard` test switch; no key in `DEFAULT_CONFIG`; gate and effect draws from `feature_rng`.
  **Done when**: a unit test shows synthetic boxes/OBBs/polygons/keypoints transform exactly (≤ 1e-6) for pad, crop and mixed margins; over ≥ 24 images with `scan_rotation` + `perspective` + `page_curl` and `tight_px = [0, 0]`, off-canvas OBBs equal the same-seed control (prototype: 3/400 vs 3/400; 47/754 without the `minAreaRect` rule); ≤ 1% empty text boxes; `legacy` hashes unchanged with the module imported (`repro_005.py rng`).

- [x] **T255** Reframe profile overlay, ordering validation, report (FR-111, FR-112, FR-113): add `canvas_reframe` to `profiles/domain_gap_v1.json` after the geometric effects and before `resize` with `p` ≥ 0.8 and the FR-109 provisional parameters; validator rejects an order violation and out-of-range parameters; per-side margin histograms in `dataset_report` (FR-105).
  **Done when**: `repro_005.py reframe` meets SC-011 on ≥ 96 profile images; a mis-ordered overlay fails validation; the TUI lists the effect with its `p` and the docs say margin limits are set through config/`--set`.

- [x] **Phase 1 gate**: Passed — Scenarios 1, 3, 6 and 8 pass · SC-004…SC-007, SC-011, SC-012 met · legend decomposition verified (all 64 test_p5 tests + effects-integrity gate passed, baseline-2 golden hashes byte-identical).

---

## Phase 2: Interoperability and Measurement

- [ ] **T235** Category registry (FR-094, OC-4): generate `categories_v1.json` from `CHART_CLASS_MAPS` and decomposed streams (28 `class_name`s: the 23 existing plus `legend_title`, `legend_label`, `legend_marker`, `source_note`, `panel_label`; `violin` enters with T249; `non_chart` is classification-only) with stable ids and `supercategory`.
  **Done when**: a test asserts every `class_name` in `CHART_CLASS_MAPS` and every v4.0/v4.1 stream class is registered, ids are unique, and the file is append-only versus the committed copy.

- [ ] **T236** COCO exporter (FR-095): `coco_export.py` reading v4.0/v4.1 records (image-frame `xyxy`, no flip code); `images`, `annotations` (`bbox`, `area`, `iscrowd`, modal `segmentation`), `categories`; exclude `unknown`/`ignored` by default.
  **Done when**: output validates against the COCO schema; category ids come only from `categories_v1.json`.

- [ ] **T236b** Classification Mode & Background Distractor Generation (FR-115, OC-4):
  * Implement `--mode classification` (or `generator.py --export-format classification`) producing `train/<chart_type>/<id>.png` plus `classification_manifest.json` (ids from `CLASS_MAP_CLASSIFICATION`, `non_chart` = 8; the manifest's `class_to_idx` is authoritative, FR-115).
  * Add background negative generation (default 8%, configurable 0–10%): non-chart images (tables, text paragraphs, diagrams, blank/ruled pages, photos) labeled as `non_chart` / `background`.
  * Support balanced class splits across the 8 primary chart classes plus `non_chart`.
  * **Done when**: `--mode classification -n 88` outputs 10 images per chart class (80) + 8 background images (9.1%, inside 5–10%) with zero detector annotations and a valid layout; pie and heatmap appear despite weight 0.

- [ ] **T237** COCO-O extension (FR-095): add `obb` (8 points, clockwise) with `obb_derived` where taken from `xyxy`, `keypoints`/`num_keypoints` for pose streams, `attributes`.
  **Done when**: text annotations carry true OBBs (`obb_derived: false`), non-text carry derived ones (`true`); keypoint counts are fixed per category.

- [ ] **T238** Round-trip + validation (FR-096): compare YOLO-txt boxes to COCO boxes; optional `pycocotools` load.
  **Done when**: boxes agree within 1 px on 100 images across all chart types; the file loads in `pycocotools` when installed (test skipped otherwise).

- [ ] **T239** Class-conditioned audit (FR-104): `scripts/audit_annotations.py` + `calibration/ink_ratio_v1.json` (default `max(0.01, 0.5 × p1_class)`, exemptions for `chart`, `line_segment`, `outlier`, `series_keypoints`, …, ring-leakage check); report-only with optional `integrity: low`.
  **Done when**: SC-009 — ≤ 1% flagged on a clean 200-image run and ≥ 90% of text boxes shifted by 8 px flagged.

- [ ] **T240 [P]** Diversity report (FR-105): `scripts/dataset_report.py` with the metrics listed in FR-105.
  **Done when**: it runs on a 500-image profile dataset and emits JSON + a text summary including `filter_stats` totals.

- [ ] **T241** `real_eval_v1` (FR-106, **OD-2**): manifest schema, `scripts/fetch_real_eval.py` (verifies sha256), annotation guide, license policy (CC0/CC-BY-compatible only); set lives outside git.
  **Done when**: ≥ 100 manifest entries with license + checksum; every entry annotated with the registry's class names and with per-side margins (annotation union and ink) recorded; the set is frozen as v1.

- [ ] **T241b** UB PMC (ICPR 2022 CHART-Infographics) Benchmark Bootstrap (FR-117):
  * Fetch and convert the UB PMC (ICPR 2022 CHART-Infographics) test split into the canonical COCO-O evaluation schema; verify each image's license from its PMC id and record annotation redistribution terms.
  * Verify the UB PMC role taxonomy, then map many-to-one into `categories_v1.json` (FR-117); unmapped roles stay `ignored`.
  * **Done when**: ≥ 250 converted candidates with verified licenses and checksums, of which ≥ 50 are frozen into `real_eval_v1` (SC-014; the ~50 paper figures of FR-106) with 0 `unknown` classes.

- [ ] **T242** Evaluation runner (FR-106): `scripts/eval_sim2real.py` — trains a small detector on a synthetic dataset (extras in `requirements-eval.txt`), 3 seeds, reports mAP50, mAP50-95, per-class AP, FP/image on context text.
  **Done when**: end-to-end run on a 1k-image dataset finishes and writes a JSON report with per-seed values and their std.

- [ ] **T243** First measurement + decision record DR-001 (SC-010, FR-107): baseline-2 (`legacy`) vs `domain_gap_v1` vs `domain_gap_v1` without `canvas_reframe` (FR-113f).
  **Done when**: `specs/005-domain-gap-closure/decisions/DR-001.md` states the numbers, the noise, "promoted / not promoted", and whether `canvas_reframe` stays.

**Phase 2 gate**: Scenario 7 · SC-008, SC-009, SC-010, SC-013, SC-014 · DR-001 recorded.

---

## Phase 3: Domain-Gap Closure

- [ ] **T244** Context stream (FR-097): `apply_pdf_document_context_effect` (`effects.py:330`) returns region geometry for caption/body/page-number/header; carried through `transform_steps`; written as `role: "context"`; not exported by default. Wire or remove the dead `multi_chart_detection.caption_probability/body_text_probability/margin_px_range` keys (N-09).
  **Done when**: with the effect forced on, context boxes align with rendered text (ink test) and survive composed geometry effects.

- [ ] **T245** Context-labeling decision (FR-099, **OD-3**): measure FP/image on text outside the chart with context unlabeled vs labeled; write DR-002.
  **Done when**: DR-002 states the result; any new class enters through an append-only registry change.

- [ ] **T246** Asymmetric layouts (FR-100, OC-6): `GridSpec` with merged cells/ratios, inset and shared axes, after T219.
  **Done when**: a property test shows every rendered subplot carries annotations with correct `subplot` indices across all new layouts.

- [ ] **T247** `multi_chart_detection` v4.0 emission (FR-101): write `_detailed.json` with subplot-level records.
  **Done when**: output validates against the v4.1 schema; YOLO txt is unchanged.

- [ ] **T248 [P]** Enable `pie` and `heatmap` in the profile (FR-102a): 200-image audit.
  **Done when**: 0 exceptions, 0 `unknown`, heatmap validation clean; weights set in `domain_gap_v1`.

- [ ] **T249 [P]** Violin (+ optional strip/dot) plots (FR-102b): `PolyCollection` body polygon, quartile/median marks, append-only class names, registry entries, goldens.
  **Done when**: unit tests cover polygon extraction and data mapping; ink-ratio gate passes for the new classes; COCO export includes them.

- [ ] **T250 [P]** Context variants + decals (FR-098): figure-number patterns, page numbers, two-column body, dark slide, browser chrome with scrollbar; watermark/draft stamp/callout arrow/significance bracket decals.
  **Done when**: each variant renders with aligned context regions; decals carry no data-mark class.

- [ ] **T250b [P]** Free-Text Clutter, Panel Labels, and Annotations (FR-116):
  * Inject realistic chart annotations: panel letters (`(a)`, `(b)`, `A`, `B`), in-chart callout notes, source citations (`Source: ...`), statistical significance stars (`***`, `p < 0.01`), and data value labels directly into plot areas.
  * Add `panel_label` and `source_note` to the registry (value text reuses `data_label`) and emit them in `_detailed.json` with appropriate role tagging.
  * Ensure detector suppression or distinct categorization so text clutter does not register as false positive axis titles.
  * **Done when**: ≥ 30% of multi-panel and publication figures carry panel letters and source notes without bounding box overlaps on axis ticks.

- [ ] **T251** Vega-Lite coverage (FR-103): add `area`; explicit `UnsupportedChartType` replacing the `else: # scatter` fallback; engine-aware weights; `engine_mix` via `feature_rng`; evaluate `theme=` in the pinned `vl-convert`.
  **Done when**: a sweep with `engine=vegalite` over supported types raises 0 exceptions; unsupported types error explicitly; the theme finding is written to `research.md`.

- [ ] **T252 [P]** Documentation (FR-108): `docs/SCHEMA.md` v4.1 additions, README profile usage, CHANGELOG, update `docs/README.md` chart-weight table to match defaults.
  **Done when**: docs match the code (a doc-vs-defaults test covers the chart weights).

- [ ] **T253** Promotion review (FR-107, OC-7): decide whether `domain_gap_v1` (or successor) becomes default.
  **Done when**: a decision record cites the measurement; if promoted, baseline-3 goldens are regenerated in a dedicated, announced change.

**Phase 3 gate**: SC-001…SC-014 all green · promotion decision recorded.

---

## Traceability (FR → tasks)

| FR | Tasks | FR | Tasks |
|---|---|---|---|
| 073 | T216, T217 | 096 | T238 |
| 074 | T216, T218, T218b | 097 | T244 |
| 074b| T216, T220b, T234b | 098 | T244, T250 |
| 075 | T216, T219 | 099 | T245 |
| 076 | T219 | 100 | T246 |
| 077 | T216, T220 | 101 | T247 |
| 078 | T216, T221 | 102 | T248, T249 |
| 079 | T222, T224 | 103 | T251 |
| 080 | T223 | 104 | T239 |
| 081 | T224 | 105 | T240 |
| 082 | T226 | 106 | T241, T242 |
| 083 | T225 | 107 | T243, T253 |
| 084 | T225 | 108 | T221, T252 |
| 085 | T231b, T232 | 109 | T254 |
| 086 | T231, T231b | 110 | T254 |
| 087 | T233 | 111 | T254, T255 |
| 088 | T230, T234c | 112 | T227, T255 |
| 089 | T234 | 113 | T229, T243, T255 |
| 090 | T228 | 114 | T234b |
| 091 | T227 | 115 | T236b |
| 092 | T228 | 116 | T250b |
| 093 | T229 | 117 | T241b |
| 094 | T235 | | |
| 095 | T236, T237 | | |

