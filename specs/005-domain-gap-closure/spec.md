# Feature Specification: Domain-Gap Closure for Chart-Detection Training Data

**Feature Branch**: `005-domain-gap-closure`
**Status**: Phase 0 & Phase 1 Complete (Phase 1 Gate Passed)
**Builds on**: specs 001–004 (schema v4.0, transform_steps compensation, declarative manifests)
**Evidence**: `research.md` (claim ledgers `D1-xx` / `D2-xx` / `D3-xx` / `R4-xx`, defects `N-xx`, experiments `E1…E17`)
**Numbering**: FRs continue the global sequence from FR-073; tasks from T216 (see `tasks.md`).

## 1. Problem Statement

A detector trained on this generator's output meets real charts (PDF figures, scans, slides, web captures) that differ from the synthetic ones in **resolution, aspect ratio, typography, framing (margins/crop), scan/print degradation, surrounding page context, plot-type coverage, legend structure, and annotation clutter**. Three external proposals (two documents and a canvas-framing proposal) and a subsequent deep critical audit were verified against the code and experimental runs (`research.md`); the core thesis holds — visual/structural diversity matters more than vocabulary — but:

1. several proposed snippets would **silently corrupt labels or break the golden suite** (flat ink gate, `ha='right'` for negative angles, blind `ttflist` font sampling, `perspective.magnitude` 0.3–0.5, in-place default edits, unconstrained Augraphy pipeline);
2. several premises in earlier proposals are wrong or overstated (layout "collapse", Matplotlib/Vega-Lite 90/10 mix, "dedup wipes labels today");
3. the generator has **pre-existing defects that invalidate the very labels the program would diversify** (N-01…N-07), including line/area primary labels falling back to bar class IDs (N-14), a dormant legend column bug (N-15), a completely dead classification mode (N-16), dead publication themes (N-17), absent legends across bar/scatter charts (N-18), and zero text clutter or tick formatting (N-19);
4. the pipeline includes an upstream chart-type classifier (confirmed by maintainers) that currently receives no dedicated classification export and no open-set negative/background images for false-positive suppression;
5. the baseline is red, so improvements cannot currently be measured or protected.

**This feature therefore delivers, in order: a trustworthy baseline (with corrected class maps and legend layout) → a gated way to ship new behaviour (profiles) → the diversity improvements that survived verification (including granular legends, publication themes, native high-DPI rasterization, and framing) → classification mode & open-set backgrounds → interoperable export → a measurement loop bootstrapping from open benchmarks (UB PMC) that decides what becomes default.**

## 2. Scope

**In scope**: baseline repair (including line/area class maps and `ncols` fix); profile mechanism; figure-size/resolution/typography/tick-angle diversity; high native DPI rendering (for targets ≥ 1200 px); publication themes wiring; canvas reframing (random margins / tight crop); granular per-entry legend extraction and multi-series legend generation; `--mode classification` and open-set negative background images; bounded realism-effect activation; bundled fonts; free-text clutter and tick formatters; COCO/COCO-O export; annotation audit; diversity report; sim-to-real harness with UB PMC bootstrap; document-context regions; asymmetric layouts; extra plot types; Vega-Lite coverage.
**Out of scope**: semantic manifests/vocabulary; training pipelines other than the evaluation harness; Plotly/ggplot2/R backends; copy-paste of data marks (rejected, research §6.9); 3-D charts; unconstrained Augraphy pipelines (rejected, research §9.4).

## 3. Actors and Scenarios

- **Dataset builder** runs `generator.py --profile domain_gap_v1` to produce a diverse detection dataset, or `--mode classification` to train the upstream chart-type classifier.
- **Detector trainer** consumes YOLO / COCO(-O) / v4.0 JSON.
- **Classifier trainer** consumes ImageNet-style directory layouts or classification manifests with negative/background images.
- **Benchmarker** compares against published COCO-format results and a frozen real eval set bootstrapped from UB PMC.
- **Maintainer** must be able to change the generator without silently changing existing datasets.

### Scenario 1 — Legacy users are untouched (P0)
*Given* the `legacy` profile and a fixed seed, *when* the generator runs after this feature ships, *then* labels and `_detailed.json` hashes equal the committed baseline-2 goldens.

### Scenario 2 — Composite figures are fully labeled (P0)
*Given* a 2×1 composite, *when* annotated, *then* both subplots carry annotations with correct `subplot` indices, and no label is lost to X-axis dedup.

### Scenario 3 — A diverse dataset on demand (P1)
*Given* `--profile domain_gap_v1`, *when* 500 images are generated, *then* sizes span ≥ 8 aspect ratios and 256–2400 px long side (with native high DPI for ≥ 1200 px), ≥ 20 distinct rendered font faces appear, rotated tick labels attach to their ticks, multi-series charts carry legends, and the effects-integrity gates (SC-004) hold.

### Scenario 4 — Standard-tool interoperability (P1)
*Given* a generated dataset, *when* exported to COCO, *then* it loads in Detectron2/MMDetection-style readers with stable, collision-free categories (including decomposed legend and clutter elements), and the COCO-O variant additionally carries oriented boxes and keypoints.

### Scenario 5 — Evidence before defaults (P1)
*Given* a candidate profile, *when* the sim-to-real harness runs on the frozen real set (bootstrapped from UB PMC) across 3 seeds, *then* promotion to default happens only if the gain exceeds run-to-run noise.

### Scenario 6 — Framing is no longer a constant (P1)
*Given* `--profile domain_gap_v1`, *when* 500 images are generated, *then* per-side margins vary (SC-011; baseline: constant 4.0% / 5.6%), no crop ever removes annotation geometry, and the `canvas_reframe` ablation arm is reported in DR-001.

### Scenario 7 — Upstream chart classifier dataset generation (P1)
*Given* `--mode classification --num 500`, *when* the generator runs, *then* it emits an ImageNet-style folder structure or classification manifest for the 8 chart types plus 5–10% open-set negative background images (`non_chart`), enabling robust training of the upstream pipeline classifier.

### Scenario 8 — Granular legend decomposition (P1)
*Given* a multi-series chart under `--profile domain_gap_v1`, *when* annotated, *then* `legend_title`, `legend_label`, and `legend_marker` are individually localized with non-degenerate boxes (≥ 2 px per side; ink ratio above the class/handle-type threshold calibrated per FR-104), while maintaining the outer `legend` bounding box for compatibility.

### Edge cases
- Downscaled images where a label falls below the minimum box size → retained as `ignored` with a reason, not silently deleted (FR-087).
- Missing/glyph-poor font in a non-legacy profile → hard error, not silent fallback (FR-089).
- `perspective` or `uneven_lighting` configured outside safe bounds → validation error (FR-091/092).
- COCO export encountering an `unknown` class → export fails loudly (FR-074/095).
- Real-eval image without a redistributable license → excluded from the manifest (FR-106).
- A tight crop would cut ink or annotation geometry (including OBB corners produced under a non-rigid mesh) → the crop box expands to the guard; nothing is lost (FR-110).
- `canvas_reframe` configured before a geometric effect, or with a margin outside its bounds → validation error in non-legacy profiles (FR-111/112).
- High-resolution target (≥ 1200 px) requested → rendered at native DPI, avoiding bilinear blur (FR-085).
- Upstream classifier encounters a document or photo → classified as `non_chart` due to open-set background training data (FR-115).

## 4. Functional Requirements

Priority: **P0** blocks everything; **P1** core value; **P2** valuable, after P1; **P3** optional. `(src: …)` = research trace.

### WS-A — Baseline integrity (Phase 0)
- **FR-073 [P0]**: `multi_chart_detection` MUST NOT raise for any primary chart type; `clsmap_obj` MUST be defined on every path that reads it. *(src: N-01)*
- **FR-074 [P0]**: No v4.0 annotation may carry `class_name == "unknown"`. `area_seg`, `line_seg` and `line_marker` streams MUST resolve through `CLASS_MAP_AREA_SEG/LINE_SEG/LINE_MARKERS`; the serializer MUST raise in strict mode (warn in lenient) when a class cannot be resolved. *(N-02)*
  Additionally, `CHART_CLASS_MAPS` MUST be keyed to match `chart_type_str` for primary types (`line` -> `CLASS_MAP_LINE_OBJ`, `area` -> `CLASS_MAP_AREA_OBJ`), eliminating the silent fallback to `bar` map in `labels/*.txt`. The primary and object streams in `_detailed.json` MUST be harmonized so duplicate conflicting records are eliminated. *(N-14)*
- **FR-074b [P1]**: `apply_legend_variation` (`chart.py:628`) MUST specify `ncols=ncol` when creating the legend with `ax.legend(...)` (or call `legend.set_ncols(ncol)` where supported in matplotlib), replacing the broken `legend._ncol = ncol` no-op. The branch is dormant today (line/area legends carry ≤ 4 series, N-15), so the fix changes no current output and does not gate baseline-2; it MUST land before any profile or FR-114 produces legends with > 6 entries. *(N-15)*
- **FR-075 [P0]**: `get_granular_annotations` MUST annotate **every visible axis**, including twin/auxiliary axes, with correct `subplot` attribution. *(N-03)*
  Crucially, to prevent phantom/ghost bounding boxes on hidden axes (N-20), tick label and axis title extractions MUST verify that the underlying axis is visible: X-axis tick labels and titles MUST check `ax.xaxis.get_visible()`; Y-axis tick labels and titles MUST check `ax.yaxis.get_visible()`. *(N-20)*
- **FR-076 [P0]**: Any X-axis label de-duplication MUST be evaluated **per axis** (not against `axes[0]` geometry) and MUST ship in the same change as FR-075, because FR-075 re-activates the row-blind block at `generator.py:3952-3965`. *(D2-04, N-03)*
- **FR-077 [P0]**: `error_bar` annotations MUST be produced for `ErrorbarContainer` artists via their constituent artists (bar-line collections and caplines); if the maintainers instead retire the class, the retirement MUST be recorded in `docs/SCHEMA.md`. *(N-04)*
- **FR-078 [P0]**: Every annotation discarded by a size/aspect/viewport/duplicate/overlap filter MUST be counted by reason and class and written to `filter_stats` in `_detailed.json` (replacing print-only reporting). *(N-06)*
- **FR-079 [P0]**: The test environment MUST be reproducible: CI installs all test dependencies via unified `requirements.txt`, the matplotlib version is recorded and bounded, and golden fixtures are **JSON-only** with an environment fingerprint (OS, Python, matplotlib, FreeType, font files). *(N-07)*
- **FR-080 [P0]**: Each of the 13–14 pre-existing baseline failures (reconciled in research §9.8) MUST be fixed or marked `xfail(strict=True)` with a reason; none may remain silently red. Stale 50 ms cold-startup latency budgets in `test_startup_latency.py` MUST be re-baselined to reflect the 6× semantic registry expansion (150 ms baseline) scaled by a measured machine-speed factor, preserving the strict zero-pydantic import guard. Registry golden snapshot assertions in `test_m0_goldens.py` MUST assert legacy sub-catalogs strictly for content and order while treating aggregate counts (`all_metrics`, `all_titles`, `all_pairs`) as lower bounds. *(N-07, research §9.8)*
- **FR-081 [P0]**: **baseline-2** goldens MUST be regenerated only after FR-073…FR-078 (including N-14, N-15, and N-20) land and MUST be committed (`tests/golden/*.json`); the `legacy` profile MUST reproduce them byte-identically from then on. Font and FreeType environments MUST be recorded via `scripts/env_fingerprint.py`; local non-canonical developer environments may use `--update-goldens` while exact byte-matching is enforced in CI. *(N-07, research §9.8)*

### WS-B — Profiles and determinism
- **FR-082 [P0]**: New randomized behaviour MUST NOT draw from the global `random`/`np.random` streams when its flag is off. When on, it MUST draw from a **feature-scoped RNG** derived from `(seed, image_index, feature_name)` so enabling or changing one feature does not perturb another's draws. *(N-11)*
- **FR-083 [P0]**: The generator MUST accept `--profile NAME` (and a `profile` config key). Profiles are overlay files under `profiles/` resolved through the existing `config_loader.load_config` layering with provenance; the default is `legacy` (= current defaults, a no-op).
- **FR-084 [P1]**: The profile name and a hash of the resolved config MUST be recorded in `_detailed.json` metadata and the dataset manifest (omitted under `legacy` to preserve byte-identity).

### WS-C — Geometry, resolution, typography
- **FR-085 [P1]**: Target resolution scaling MUST be split by target size:
  (a) for downscaled / low-resolution targets (< 1200 px long side), render at normal size and resample with anti-aliasing (recording a `("scale", sx, sy, new_w, new_h)` transform step and compensating geometry);
  (b) for large targets (≥ 1200 px long side up to 2400 px), render directly at higher native DPI (`dpi = int(target_px / figure_inches)`) to avoid blurry font rasterization and interpolation artifacts. *(D1-01, D1-02, D2-26, R4-06)*
- **FR-086 [P1]**: `figure.size_mode: fixed|randomized|publication`. `randomized` draws aspect ratios from `{4:3, 16:9, 1:1, 3:4, 5:4, 16:10, 3:2}` with width in 5.5–9.5 in and DPI from a configured set; fonts, line widths, marker sizes and pads scale by `clamp(sqrt(W·H/35), 0.75, 1.25)`.
  Additionally, profile `domain_gap_v1` MUST wire the 9 existing `PUBLICATION_THEMES` (`themes.py:300`: `nature`, `science`, `cell`, `lancet`, `pnas`, `bmj`, `nanotech_short`, `open_access_poster`, `print_high_contrast`; 3.2–8.0 in, 300–600 DPI, 6–10 pt) through `size_mode: publication`, closing the scientific publication scale gap. The themes request Arial/Helvetica/Times New Roman, which all render as DejaVu Sans today (D1-04); they MUST resolve through bundled-font aliases (metric-compatible sans/serif, FR-089), otherwise FR-089's hard error applies. *(D1-12, D2-06, N-17)*
- **FR-087 [P1]**: The minimum-box policy MUST be resolution-aware and non-destructive: parameter `annotation_min_side_px` (legacy behaviour = 8 on non-`scatter/box/heatmap` types; profile default 4). Sub-threshold annotations are **kept in `_detailed.json` with `ignored: true, ignore_reason`** and excluded from YOLO/COCO training exports unless requested. *(N-06, E10)*
- **FR-088 [P1]**: `tick_rotation` config `{mode: legacy|profile, angles, weights, p_rotate}`. Under `profile`, positive angles use `ha='right'`, negative `ha='left'`, `|angle|==90` `ha='center'`; `rotation_mode` stays **default** so `get_text_obb_and_bbox` remains valid.
  Additionally, under `profile`, axis tick labels MAY draw from configured tick formatters: percentages (`%`), currency (`$`), metric/engineering scale suffixes (`k`, `M`, `B`), scientific notation, or date formats, replacing uniform raw numbers. *(D1-07, D2-01, D2-18/19, N-19)*
- **FR-089 [P1]**: A bundled font set under `assets/fonts/` (open-licensed, license files included) MUST be registered deterministically with `font_manager.addfont`; each font MUST pass a glyph-coverage check (Latin, digits, `µ μ ± ° α β ² ³ ⁻ × ≥ ≤ Δ`); selection is from the sorted bundled registry; the chosen family is recorded in `_detailed.json`. In non-legacy profiles a missing font or glyph MUST raise (no silent fallback). Mixed families within one chart (title vs labels) are allowed. `ttflist` MUST NOT be sampled. An alias table MUST map the families requested by themes (Arial and Helvetica → a bundled metric-compatible sans; Times New Roman → a bundled metric-compatible serif) so `PUBLICATION_THEMES` resolve to bundled faces instead of DejaVu Sans (N-17, FR-086). *(D1-04, D1-14, N-10)*

### WS-D — Realism effects
- **FR-090 [P1]**: Profile `domain_gap_v1` enables, with bounded parameters: `scan_rotation` p 0.10, ±1.5°; `perspective` p 0.06, magnitude U[0.02, 0.08]; `page_curl` p 0.04, `amplitude_ratio` U[0.010, 0.025]; `clipping` p 0.04, 1–3%; `uneven_lighting` p 0.10, `intensity_range` [0.05, 0.25]; `chromatic_aberration` p 0.05. `grid_occlusion` stays off pending visual review. Augraphy MUST NOT be used as a general dependency; only verified-safe operations (edge shift ≤ 0.5 px) are permitted behind FR-093 gate. *(§4 of research, R4-07)*
- **FR-091 [P1]**: Config validation MUST (a) reject effect parameters not accepted by the effect function (no silent `**kwargs` swallowing) — as an **error in non-legacy profiles and a warning under `legacy`**, so existing user configs keep running — and (b) reject `perspective.magnitude > 0.15` in non-legacy profiles. The legacy default `perspective.magnitude` MUST change from 0.5 to 0.08 — output-neutral because `p = 0`. *(N-08, D2-15)*
- **FR-092 [P1]**: `uneven_lighting` MUST accept `intensity_range` and a randomized `gradient_type`; the scalar `intensity` remains valid.
- **FR-093 [P1]**: Every effect enabled in any profile MUST pass the effects-integrity property test (research §4 method) as a CI gate (slow-marked): forced `p=1`, ≥ 98% annotations preserved, ≤ 1% empty text boxes, plus one composed case.
- **FR-109 [P1]**: A `canvas_reframe` effect (`apply_canvas_reframe_effect`, `effects.py`) MUST return `(new_img, dx, dy)`, be registered in `EFFECT_REGISTRY`, and be dispatched through the existing translate branch (`clipping` / `pdf_document_context`, `generator.py:1835`). Semantics: **crop to the content box, then draw each side's margin independently** — *tight* with probability `p_tight` (U[0, `tight_px`] px; automated figure extraction), else *loose* (U[`margin_frac_range`] × min(W,H); resolution-relative). Fill: the figure background (`p_match_bg`) or a page tone U[245,255]. Provisional defaults `p_tight` 0.4, `tight_px` [0, 6], `margin_frac_range` [0, 0.12], `p_match_bg` 0.5, to be calibrated from the margins recorded in `real_eval_v1` (FR-113e). With the new canvas as box `(bx0, by0, bx1, by1)` in source image coordinates, `dx = −bx0` and `dy = by1 − H` (Matplotlib bottom-left); pad and crop are one formula. The signature MUST be explicit (no `**kwargs`) so FR-091 validation is exact. *(N-12, D3-03, D3-04)*
- **FR-110 [P1]**: **Content guard.** The crop box MUST contain the union of (a) the ink bounding box (border-estimated background, tolerance `ink_tol`) and (b) every annotation geometry in every stream — AABB corners, OBB corners, polygons, non-normalized keypoints — mapped through the transform steps recorded so far, **clamped to the canvas** (some pre-effect geometry already lies > 100 px outside it, N-13). The guard is computed in `apply_realism_effects` (which owns the steps) and passed in as `content_bbox`; the effect never receives annotations. When a `non_rigid_mesh` step precedes, the guard MUST use the densified mapping **and the `minAreaRect` OBB corners** that the final transform produces (E13: without them a zero-slack crop after `page_curl` leaks 47/754 OBBs instead of 5/754). A test-only switch (`use_guard: false`) provides the negative control. *(D3-09, N-13)*
- **FR-111 [P1]**: **Placement and RNG.** `canvas_reframe` MUST NOT be added to `DEFAULT_CONFIG["realism_effects"]` (each key costs one global RNG draw, E12); it is configured only in `profiles/domain_gap_v1.json`, with `p` ≥ 0.8 (`p < 1` leaves ≈ `1 − p` of images at today's constant margins). All its randomness, **including the probability gate**, comes from `feature_rng("canvas_reframe")` (FR-082): feature-scoped effects are gated by the feature RNG and consume nothing from the global stream. *(D3-05, D3-06, N-11)*
- **FR-112 [P1]**: **Ordering and parameters.** In non-legacy profiles `canvas_reframe` MUST follow `scan_rotation`, `perspective`, `page_curl`, `clipping` and `pdf_document_context`, and precede `resize`; a violation is a validation error (E13: reframe-first leaves 8/400 OBBs off-canvas vs 3 for the composed control). Parameters live under `realism_effects.canvas_reframe.params`; probabilities in [0,1], `0 ≤ margin_frac_range ≤ 0.25`, `tight_px ≤ 12`. The effect appears in the TUI's Realism Effects menu automatically (the list is built from the registry) with its `p`; that menu edits only `.p`, so margin limits are set through config / `--set`. *(D3-07)*
- **FR-113 [P1]**: **Acceptance.** (a) FR-093 gate, including a composed case with `page_curl` and zero slack (`tight_px = [0, 0]`); (b) 0 off-canvas OBBs/keypoints *added* relative to the same seeds without the effect (not an absolute rate: the baseline already has 0.75%); (c) framing diversity per SC-011; (d) the `use_guard: false` negative control fails (b), proving the gate can fail; (e) defaults calibrated from `real_eval_v1` margins before any promotion (FR-107); (f) the DR-001 comparison includes a `domain_gap_v1` arm without `canvas_reframe`. *(D3-02, D3-08)*

### WS-E — Export
- **FR-094 [P1]**: A versioned, **append-only** category registry (`categories_v1.json`) MUST map each distinct `class_name` to a stable id with `supercategory` = chart family/stream. New categories added by this feature:
  - Decomposed legend components: `legend_title`, `legend_label`, `legend_marker` (supercategory: `legend_elements`).
  - Text clutter: `source_note`, `panel_label` (supercategory: `text_annotations`). Value text reuses the existing `data_label`; the benchmark name `value_label` is only an alias in the real-eval mapping (FR-117).
  - `non_chart` is **not** a detection category: it is label 8 of the classification label map (FR-115). Background images in detection datasets are images with no annotations.
  - Plot extensions: `violin` (supercategory: `distribution`).
  Class ids from the per-chart-type maps MUST NOT be used as COCO ids. *(D2-09, R4-01, R4-03, R4-05)*
- **FR-095 [P1]**: A COCO exporter MUST consume **v4.0 records** (image-frame `xyxy`; no coordinate-flip code). Two variants: `coco` (bbox, `area`, `iscrowd=0`, modal polygon as `segmentation` when present) and `coco-o` (adds `obb` 8-point clockwise, `obb_derived` flag where derived from `xyxy`, `keypoints`/`num_keypoints` for pose streams, `attributes` {text, visibility, occluded, stream, subplot}). `unknown` categories and `ignored` annotations are excluded by default. *(D2-03, N-02, N-05)*
- **FR-096 [P1]**: Round-trip validation: for generated images, YOLO-txt boxes and COCO boxes agree within 1 px; the file passes a schema check and loads in `pycocotools` when installed.

### WS-F — Document context and decals
- **FR-097 [P2]**: `pdf_document_context` MUST return the geometry of every text region it draws (caption, body, page number, header) as a **context stream** in `_detailed.json` (`role: "context"`), transformed through `transform_steps`; context regions are **not** exported to YOLO/COCO by default. *(D2-11)*
- **FR-098 [P2]**: Context variants: figure-number patterns ("Fig. 1", "Figure 3(a)", "(A)"), page numbers, two-column body, dark slide background with title overlay, browser-chrome with scrollbar; and **decal overlays** (watermark, draft stamp, callout arrow, significance bracket) as the accepted form of copy-paste. Dead keys `multi_chart_detection.caption_probability/body_text_probability/margin_px_range` MUST be wired or removed. *(N-09, D2-14)*
- **FR-099 [P2]**: Whether to introduce labeled `figure_caption`/`body_text` classes MUST be decided from the sim-to-real harness (false positives on text outside the chart, with/without context labels), recorded as a decision record, and enabled only by an append-only registry change. *(D2-11)*

### WS-G — Layouts and chart coverage
- **FR-100 [P2]**: Asymmetric `GridSpec` layouts (merged cells, width/height ratios, inset axes, shared axes) MAY be added **only after** FR-075/076; a property test MUST assert every rendered subplot has annotations.
- **FR-101 [P2]**: `multi_chart_detection` MUST emit v4.0 `_detailed.json` with subplot-level records (currently it writes only `chart_N.txt`). *(N-01)*
- **FR-102 [P2]**: Chart coverage: (a) enable `pie` and `heatmap` in the profile after a 200-image audit shows 0 exceptions and 0 `unknown`; (b) add **violin** (body polygon from `PolyCollection`, quartile/median marks, new append-only class names) and optionally strip/dot plots, each with unit tests and a category-registry entry.
- **FR-103 [P2]**: Vega-Lite: add `area`; replace the silent `else: # scatter` fallback with an explicit `UnsupportedChartType` error and engine-aware chart-type weights; add `engine_mix` (per-image draw via a feature-scoped RNG); evaluate `theme=` support in the pinned `vl-convert` before promising vega themes. *(D2-20/21/22)*

### WS-H — Measurement and quality
- **FR-104 [P1]**: `scripts/audit_annotations.py` MUST implement a **class-conditioned** integrity audit: thresholds derived from a clean reference run (default `max(0.01, 0.5 × p1_class)`), explicit exemptions for container/thin classes (`chart`, `line_segment`, `outlier`, `series_keypoints`, …), and a ring-leakage check. It **reports** (optionally flags `integrity: low`); it never silently deletes. Calibration is stored in `calibration/ink_ratio_v1.json`. *(D1-11, D2-16)*
- **FR-105 [P2]**: `scripts/dataset_report.py` MUST report: size/aspect histograms, class frequency, per-class small/medium/large (COCO area bins), effect frequency, `filter_stats` totals, ignored counts, font-face distribution, tick-angle distribution, annotations/image.
- **FR-106 [P1]**: A sim-to-real harness MUST exist: a **frozen** `real_eval_v1` set (~100 images: ~50 open-access paper figures, ~50 web/slide/scanned) kept **outside git** with a checksummed manifest (source URL, license, sha256, annotation file); only CC0/CC-BY-compatible items; a runner that trains a small detector on synthetic data (optional extra requirements) with **3 seeds** and reports mAP50, mAP50-95, per-class AP and false positives per image on context text. *(D1-13, D2-23)*
- **FR-107 [P1]**: Promotion policy: a profile becomes the default only if its mean mAP50 gain on `real_eval_v1` exceeds one standard deviation across seeds with no per-class regression beyond a configured margin; otherwise the decision record says "not promoted".
- **FR-108 [P1]**: Schema/versioning: additive fields introduced in Phase 0 (`filter_stats`) bump v4.0 → **v4.1** (`versions.py`, `docs/SCHEMA.md`) and are part of baseline-2. Later additive fields (`profile`, `ignored`, context stream, `obb_derived`, `legend_elements`) are **omitted when not applicable** so the legacy profile stays byte-identical.

### WS-I — Legends, Clutter, and Open-Set Classification
- **FR-114 [P1]**: Granular Legend Extraction & Multi-Series Legends:
  (a) Legend generation MUST be added to multi-series `bar` (grouped/stacked), multi-series `scatter`, `box`, and `histogram` charts in the profile via `feature_rng` (probability default 0.70).
  (b) `generator.py` MUST implement a decomposed legend extractor using `ax.get_legend()`, `lg.get_title()`, `lg.get_texts()`, and `lg.legend_handles`, emitting `legend_title`, `legend_label`, and `legend_marker` records in `_detailed.json` under profile control. A marker-less `Line2D` handle has a zero-height window extent (E16: 119/119), so its `legend_marker` box MUST be the line extent padded by half the line width (≥ 2 px per side); boxes below `annotation_min_side_px` follow FR-087 (`ignored`).
  (c) The outer monolithic `legend` box MUST remain present in `annotations` stream for backward compatibility, while granular records populate `legend_elements` stream. *(N-18, R4-01, research §9.2)*
- **FR-115 [P1]**: Classification Mode & Negative Background Generation:
  (a) `--mode classification` / `dataset_format == 'classification'` MUST branch cleanly: emit an ImageNet-style directory tree (`train/<chart_type>/<id>.png`) and a dataset classification manifest (`classification_manifest.json` mapping image path to integer label 0..8). Ids reuse the existing `CLASS_MAP_CLASSIFICATION` (alphabetical: area 0, bar 1, box 2, heatmap 3, histogram 4, line 5, pie 6, scatter 7) with `non_chart` = 8. The manifest is authoritative: `ImageFolder`-style loaders sort folder names and would put `non_chart` at index 6, so consumers MUST take `class_to_idx` from the manifest. Chart classes are sampled uniformly, not by the detection `chart_types` weights (pie and heatmap are 0 there).
  (b) To support the upstream pipeline chart classifier, generation under classification mode MUST include open-set negative / background images (class 8: `non_chart`; default 8%, configurable 0–10%; research §9.6), generating synthetic document pages (text blocks, tables, math formulas, code snippets, whitespace) to suppress false positives on non-chart page content.
  (c) **[P2]** The same background generator SHOULD be able to emit unlabeled background images (empty `labels/*.txt`) into detection datasets at a configurable 0–10% (default 0 under `legacy`); the sourced Ultralytics guidance applies to detection. *(N-16, R4-03, research §9.6)*
- **FR-116 [P2]**: Free-Text Clutter & Panel Annotation:
  Under `profile`, charts MAY stochastically include text clutter: subtitles, source notes/footnotes ("Source: ...", "Data from ..."), R² / regression equation / p-value statistics, and panel letters ("(a)", "(b)", "A", "B") in multi-chart subplots, registered as `source_note`, `panel_label`, and `data_label`. *(N-19, R4-05)*
- **FR-117 [P1]**: Real-Eval Bootstrap from UB PMC:
  `scripts/fetch_real_eval.py` MUST support parsing open-access CC-BY chart figures and ground-truth annotations from the UB PMC benchmark (ICPR 2022 CHART-Infographics extended UB PMC), verifying each image's license from its PMC id (FR-106: CC0/CC-BY-compatible only) and mapping its text roles many-to-one into `categories_v1.json` (`x_title`/`y_title` → `axis_title`, `xlabel`/`ylabel` → `axis_labels`, `value_label` → `data_label`, legend roles → the decomposed legend classes; roles without a generator counterpart, e.g. `other`, `mark_label`, are kept as `ignored` in evaluation rather than forced into a class). The UB PMC role names MUST be verified at T241b; the x/y names above are NVIDIA's. These figures form the ~50 paper figures of `real_eval_v1` (FR-106). *(D1-13, R4-08, research §9.3)*

## 5. Non-Functional Requirements

- **NFR-A Determinism**: same `(seed, profile, environment fingerprint)` ⇒ identical labels and `_detailed.json`. Pixel hashes are not required across OSes (spec 001 FR-031).
- **NFR-B Throughput**: legacy ≥ 90% of baseline-2 images/s (reference ≈ 2.7 img/s single-process, effects off, measured in the research sandbox); `domain_gap_v1` ≤ 2.0× slower on its own size distribution (provisional: native high-DPI rendering is ≈ 2.5× slower per image, E17, and ≈ 30% of a log-uniform 256–2400 px draw is ≥ 1200 px).
- **NFR-C Startup**: no new heavy imports at module import time (existing import-guard test stays green); fonts and optional libraries load lazily.
- **NFR-D Footprint**: bundled fonts ≤ 8 MB total; no image files committed.
- **NFR-E Compatibility**: all new keys are additive; existing YOLO/OBB/pose/seg and v4.0 consumers are unaffected under `legacy`.
- **NFR-F Observability**: every drop, ignore, fallback and validation rejection is counted and reported; non-legacy profiles have no silent fallbacks.

## 6. Key Entities

- **Profile**: named config overlay (`legacy`, `domain_gap_v1`) with provenance and hash.
- **CategoryRegistry**: append-only `class_name → category_id` map used by exporters.
- **FilterStats**: per-image counts of discarded annotations by reason × class.
- **ContextRegion**: non-training text region (caption/body/page number) recorded by document-context effects.
- **IntegrityCalibration**: per-class ink-ratio thresholds derived from a clean run.
- **RealEvalManifest**: checksummed list of real images, licenses and annotation files.
- **ClassificationManifest**: list of images, labels (0..7 chart types + 8 non-chart), and splits for upstream classifier training.

## 7. Success Criteria

- **SC-001**: Under `legacy`, baseline-2 label and `_detailed.json` hashes match (CI).
- **SC-002**: A 500-image sweep over (`detection`, `multi_chart_detection`, `classification`) × (`matplotlib`, `vegalite`) × supported chart types raises **0 exceptions**.
- **SC-003**: `unknown` class share = 0%; `error_bar` > 0 on forced bar charts with error bars; line/area charts emit line/area class IDs in `labels/*.txt`; a 200-composite property test finds ≥ 1 annotation on every visible axis.
- **SC-004**: Effects integrity (FR-093): ≥ 98% annotations preserved and ≤ 1% empty text boxes for each enabled effect and for one composed case.
- **SC-005**: Figure randomization: 0 `tight_layout` warnings, text collisions/img ≤ baseline + 0.10, ≤ 1% empty text boxes; ≥ 8 distinct aspect ratios; long side spanning 256–2400 px.
- **SC-006**: Rotated ticks: |tick-to-label gap| ≤ 15 px at every profile angle on the reference label; OBB-vs-ink test passes for each (angle × `ha`).
- **SC-007**: ≥ 20 distinct rendered font faces in a 500-image profile run; 0 missing glyphs for the manifest character inventory; identical font choices for the same seed on two machines.
- **SC-008**: COCO round-trip within 1 px; no category collisions; 0 `unknown`; loads in `pycocotools` when installed.
- **SC-009**: Audit: ≤ 1% false flags on a clean run and ≥ 90% detection of text boxes deliberately shifted by 8 px (power test).
- **SC-010**: `real_eval_v1` runs end-to-end with 3 seeds; first baseline-2 vs `domain_gap_v1` comparison is published with its decision record.
- **SC-011**: Canvas reframe (≥ 96 profile images): per-side annotation-union margin std ≥ 0.02 at the profile's `p` (baseline ≈ 0.000; prototype 0.020 at p = 0.5, 0.023 at p = 0.8); ≥ 50% of images with a side margin < 1% and ≥ 50% with a side margin > 6%; 100% of annotations preserved; 0 off-canvas OBBs/keypoints added versus the same seeds without the effect (including the composed `page_curl` zero-slack case); `_detailed.json` `image.width/height` equal the PNG size.
- **SC-012**: Granular legend extraction: 0 exceptions across all configurations; every `legend_label` / `legend_marker` box ≥ 2 px per side; median ink ratio per class and handle type (line, marked line, patch, scatter) above the thresholds in `calibration/ink_ratio_v1.json` (FR-104). Reference (E16): labels ≈ 0.32; markers ≈ 0.41 (marked line), ≈ 1.00 (patch), unmeasurable for raw marker-less line handles.
- **SC-013**: Classification mode: `--mode classification` produces valid directory hierarchy and manifest with 5–10% `non_chart` background samples.
- **SC-014**: UB PMC bootstrap: `real_eval_v1` loads ≥ 50 real figures with validated category mapping and 0 `unknown` classes.

## 8. Assumptions and Dependencies

- Spec 001's compensation pipeline (`transform_steps`, per-point mapping) is correct — verified for all currently-disabled geometric effects at bounded parameters (research §4).
- Small-detector evaluation needs an optional extra (`requirements-eval.txt`); it does not affect generator imports.
- The UB PMC dataset (ICPR 2022 CHART-Infographics) states it selected only CC-BY PMC figures; per-image licenses are still verified (FR-117) and annotation redistribution terms are checked before any copy is stored. It is the primary real-eval seed for scientific charts.

## 9. Open Decisions (need an owner before the named task starts)

| ID | Decision | Needed by | Default if unanswered |
|---|---|---|---|
| OD-1 | Commit bundled fonts (≤ 8 MB) vs. fetch with checksums | T234 | Commit (OFL/Apache only) |
| OD-2 | Who annotates `real_eval_v1`, and the license policy | T241 | Bootstrap from UB PMC (CC-BY); maintainer annotates web/scans |
| OD-3 | Labeled caption/body classes | T245 | Not added until measured |
| OD-4 | Fix vs retire `error_bar` | T220 | Fix |
| OD-5 | matplotlib upper bound | T222 | `<3.11` until goldens re-verified |
| OD-6 | Should `ignored` annotations appear in YOLO txt? | T233 | No |
| OD-7 | Classification export format (ImageNet directory layout vs JSON manifest) | T236b | ImageNet directory layout (`train/<class_name>/`) + JSON manifest |
| OD-8 | Granular legend in primary YOLO `labels/*.txt` vs profile opt-in | T234b | Monolithic in primary `labels/*.txt`; granular in `_detailed.json` and COCO |
