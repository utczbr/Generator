# Research: Domain-Gap Closure — Verification of the Improvement Proposal and Its Audit

**Spec**: `specs/005-domain-gap-closure/spec.md` · **Inputs**: *Generator Improvement Analysis* ("**Doc 1**") and the *Executive Verdict / audit* of it ("**Doc 2**"), plus a later proposal for a randomized canvas padding / tight-crop effect ("**Doc 3**", §8) and a reviewer audit (`R4-xx`, §9)
**Repo snapshot**: `utczbr/Generator@main` (tarball) · **Date**: 2026-10-04 · **matplotlib in the test environment**: 3.10.8 (`requirements.txt` only says `>=3.7.0`)

Every claim below was checked against source (`file:line`) and, where it is empirical, against the **unmodified generator** using `research/repro_005.py` (experiment IDs `E1…E17` map to its subcommands). Verdicts: **CONFIRMED**, **PARTIAL**, **REFUTED**, **UNVERIFIED** (no evidence either way; not assumed true).

---

## 0. Method and limits of the evidence

- Harness: `REPO=<checkout> python research/repro_005.py <experiment> --n 24`. Per-image seeding (`seed + i`), realism effects off unless the experiment turns one on.
- Sample sizes are small (N = 16–24 images per condition, one seed family). Per-image annotation counts are dominated by scatter charts (one scatter can hold 60+ `data_point`s), so **`ann/img` is a noisy metric; `text/img`, collisions and ink-ratio are the stable ones.** Treat deltas under ~10% as noise.
- "Ink ratio" (fraction of a box's pixels that differ from the border-estimated background by >25 grey levels) is a **proxy** for annotation/pixel alignment, not ground truth.
- No model was trained. Every statement of the form "X improves detector mAP" in any of the three documents is therefore **UNVERIFIED** here; closing that gap is the purpose of FR-106 (sim-to-real harness).
- Competitive-landscape claims (ChartDETR, InfoDet "14M annotations", ExcelChart400k, NVIDIA pipelines) and real-paper prevalence figures (violin ≈ 8% of PMC distribution plots, etc.) were **not** checked and no source is cited for them in either document.

---

## 1. Doc 1 (original proposal) — claim ledger

| ID | Claim | Verdict | Evidence |
|---|---|---|---|
| D1-01 | Single charts use fixed `figsize=(7,5)`, DPI ∈ {96,120,150} → narrow resolution prior | **CONFIRMED** | `generator.py:3599,3627`. **E1**: 24 images → exactly 3 sizes (672×480, 840×600, 1050×750), one aspect ratio (1.40). (Multi path uses `ncols*5 × nrows*4`, `generator.py:3607,3621`, so composites vary.) |
| D1-02 | `low_res` / existing effects don't cover resolution diversity | **CONFIRMED** | `effects.py:91`: downscales then resizes **back to the original `(w,h)`**; image dimensions never change. |
| D1-03 | `pdf_document_context` is disabled (`p: 0.0`); adds lorem ipsum, one font, white page | **PARTIAL** | `p=0.0` only in the default `detection` format (`config_defaults.py:417`). In `multi_chart_detection` it is injected at **p=0.5** (`generator.py:4044`). Already draws "Figure N:" captions and uses `DejaVuSans.ttf` (`effects.py:330ff`). |
| D1-04 | `FONT_FAMILIES` has 7 fonts / 3 families | **REFUTED (detail)** | 8 entries (`themes.py:439`); `chart.py:402` samples only `sans-serif`/`serif` → 6 names. **E8**: only **3 distinct faces render** — `Arial`, `Times New Roman`, `Georgia`, `Courier New` silently fall back to `DejaVuSans.ttf`; two of three "serif" picks render as sans. The `try/except` around font setting cannot catch it (matplotlib warns, does not raise). |
| D1-06 | Chart types: bar, line, scatter, box, pie, area, histogram, heatmap | **PARTIAL** | `pie` and `heatmap` have **weight 0** by default (`config_defaults.py:158,170`); `docs/README.md` shows pie 5 → docs/defaults drift. Default draw covers 6 types. |
| D1-07 | `label_angle` is a single config value, default 0 | **CONFIRMED** | `config_defaults.py:31`. Consumed only by Vega-Lite (`vegalite_backend.py:43,79-80,656-703`). Matplotlib ignores it. |
| D1-08 | Multi-chart layouts are grid-only | **CONFIRMED** | `generator.py:3619` (scenario path: 1x2…3x3), `_select_multi_chart_layout` `generator.py:1632`. `scenario_weights.multi = 0` by default (`config_defaults.py:175`). |
| D1-10 | Enable disabled effects at the listed `p`, `perspective.magnitude = 0.3`, `uneven_lighting` with defaults | **PARTIAL** | `p` values are fine. **E5**: `perspective` ≤ 0.3 keeps annotations aligned but 0.3 shrinks content into large white borders (mean luminance 237→250); `uneven_lighting` with `params: {}` uses `intensity=0.6` (`effects.py:285`) → mean luminance **237 → 158**; `intensity=0.25` → 204. |
| D1-11 | Flat content-ratio gate (0.3) for annotation validation | **REFUTED** | **E4**: on a clean, correct baseline a flat 0.30 gate flags **21.4%** of all annotations — `line_segment` 100%, `legend` 80%, `chart` 79%, `chart_title` 75%. |
| D1-12 | Add per-chart font-size / weight variation | **PARTIAL** | Already randomized per chart: title 12–16, label 10–13, tick 8–11 pt (`chart.py:~408-412`), title weight random. Missing: mixed families inside one chart; scale with figure size. |
| D1-13 | Resolution mismatch is "the #1 cause of domain gap" | **UNVERIFIED** | Plausible; no model evidence in repo or documents. |
| D1-14 | Blind sampling of `font_manager.ttflist` at 30% | **REFUTED (unsafe)** | **E8**: of 70 families on a typical Linux box, 13 (19%) lack full Latin letters+digits, 30 (43%) lack some of `µ ± ° α β`; only **27 (39%) are safe**; the set also differs per machine → same seed ≠ same dataset. |
| D1-19 | No COCO export exists | **CONFIRMED** | Repo-wide grep finds "coco" only inside manifest text. |

## 2. Doc 2 (audit) — claim ledger

| ID | Claim | Verdict | Evidence |
|---|---|---|---|
| D2-01 | `label_angle` only wired to Vega-Lite; Matplotlib tick rotation hardcoded at `chart.py:430` with `p=[0.5,0.3,0.2]` for `[0,45,90]` | **CONFIRMED + completed** | `chart.py:429-432`. The audit **missed the outer gate** `np.random.random() < 0.3` → effective distribution ≈ **85% horizontal / 9% at 45° / 6% at 90°**. |
| D2-03 | `ann['bbox']` is in Matplotlib bottom-left coordinates and must be flipped on export | **CONFIRMED** | Flip happens at three independent sites: `save_annotations_yolo` (`generator.py:2141`), `save_annotations_yolo_obb` (`:2204`), `detailed._serialize` (`detailed.py:517`). v4.0 `xyxy` is **already image-frame**, so a COCO exporter should read v4.0 records and contain **no flip code**. |
| D2-04 | X-axis dedup (`generator.py:3952-3965`) is row-blind and will wipe lower-subplot labels | **PARTIAL (latent)** | Block is guarded by `len(fig.axes)==2` (dual-axis post-processor), so it can only affect `1x2`/`2x1`. It removed 0 labels in every run **because** of N-03 (it never receives lower-axis labels). **Fixing N-03 activates this bug** → the two fixes must ship together (spec ordering constraint). **Measured** (scratch patch dedenting the return; 16 bar 2×1 composites): the dedup fires in 9/12 images ("Kept 9 of 12 X-axis labels") and removes 51 of 188 lower-panel `axis_labels` (27%) from the primary `labels/*.txt`. `_detailed.json` totals do not change because the removed labels reappear under the `annotations_obj` stream, so counting JSON annotations hides the loss; count `labels/*.txt` per half. |
| D2-06 | Shrinking figsize (e.g. 4×3 @72) causes layout collapse, `tight_layout` crashes, text >70% of viewport | **REFUTED at tested settings** | **E2** (DPI-consistent): 4×3@72 → 0 errors, 0 `tight_layout` warnings, text boxes cover 3.8% of area, text collisions 0.25/img vs 0.21 baseline, 0% empty text boxes. Observed cost: `text/img` 16.9 → 12.3 (−27%), other conditions within −15%. Cause not isolated: patching `MIN_BBOX_SIZE` to 1 leaves `text/img` unchanged (12.3 vs 12.3); fewer ticks on smaller axes is the untested candidate. The filter *does* matter under downscaling (E10). |
| D2-09 | Standard COCO loses v4.0 information → need standard + extended variant | **CONFIRMED + strengthened** | Class IDs are **per chart-type** and collide: **9 of 10 ids map to >1 name** (id 1 = `axis_title`,`bar`,`box`,`cell`,`data_point`,`line_segment`,`wedge`); 23 distinct names. An exporter keyed on `class_id` would silently merge them. Categories must be keyed on `class_name`. |
| D2-10 | Canvas-expanding effects must record canvas size per transform step | **CONFIRMED, already implemented** | `apply_realism_effects` (`generator.py:1742ff`; spec 001 FR-021). **E5**: `pdf_document_context` grows 840×600 → 900×803 and text boxes stay aligned (0% empty). |
| D2-11 | Unlabeled document text causes high-confidence false positives unless labeled as negative classes | **UNPROVEN** | The repo already ships unlabeled captions/lorem text at p=0.5 in `multi_chart_detection`; unlabeled text is the standard hard-negative signal. Whether it hurts is an empirical question (FR-099). A middle path (record context regions without committing to classes) is in FR-097. |
| D2-12 | Random system fonts → tofu/glyph loss; Docker has "almost nothing beyond DejaVu" | **CONFIRMED (risk) / overstated (detail)** | Risk quantified in D1-14. matplotlib ships DejaVu Sans/Serif/Mono + STIX + cm*; the sandbox exposed 70 families. The real hazard is **environment-dependent output**. |
| D2-14 | Copy-paste of data marks is invalid for charts | **AGREE (reasoning only)** | Marks are grounded in axis geometry; pasted marks contradict ticks/gridlines and `_detailed.json` data. Untested. Decal-style overlays remain valid. |
| D2-15 | `perspective.magnitude = 0.5` is catastrophic; use ~0.05 | **CONFIRMED** | `generator.py:1849-1850` maps `magnitude → distortion_factor`; corner jitter is `w × factor` (`effects.py:259`). **E5**: at 0.5 annotations drop 50.4 → 40.6 (−19%) and 4.6% of text boxes are empty. `page_curl.amplitude_ratio` and `clipping.clip_range_pct` suggested by Doc 2 are **already the defaults**. |
| D2-16 | Flat gate wrong; use `bar .5 / text .1 / line .02 / scatter .01` | **PARTIAL** | Flat gate wrong (see D1-11). Its numbers are also wrong: `bar .5` false-flags **13.2%** of valid bars (hollow/hatched; p5 = 0.08; a later re-run on 62 bars gave 3.2% with p5 = 1.00, so the rate depends on the share of hollow/hatched bars); `data_point` median is **0.95**, not <0.05, so `.01` can never detect drift. Thresholds must be calibrated per class (FR-104). |
| D2-18 | For negative tick angles "`ha` must be `'right'`" | **REFUTED** | **E7** (pixel gap between tick and the label end nearest the axis): at −30/−45/−60° `ha='right'` leaves it **111/91/65 px** away (worst option); `ha='left'` → 9/11/12 px. At +30/+45/+60° `ha='right'` → 11/11/10 px. Today's behaviour (centered, 45°) → 40 px. |
| D2-19 | Using `rotation_mode='anchor'` is the right fix | **Unsafe here** | `get_text_obb_and_bbox` (`generator.py:271ff`) derives the rotated OBB assuming *default* alignment semantics; anchor mode would corrupt OBBs. Keep default mode, choose `ha` by angle sign. |
| D2-20 | "Matplotlib 90% + Vega-Lite 10%" | **REFUTED** | `engine` is a **per-run** switch (`generator.py:3583`, `config_defaults.py:19`); no per-image mixing exists. |
| D2-21 | Extend Vega-Lite with line, scatter, area | **PARTIAL** | Line and scatter exist; only area (and box/histogram/pie/heatmap) are missing. Unsupported types silently render as **scatter** (`else:` branch). |
| D2-22 | Vega themes (`ggplot2`, `fivethirtyeight`, …) are available | **UNVERIFIED for this backend** | Backend calls `vlc.vegalite_to_svg(spec)` with no theme argument and styles from the repo's `THEMES`. Whether the pinned `vl-convert` accepts `theme=` must be checked (T251). |
| D2-23 | Sim-to-real harness should be P1; real set under `tests/golden/real_eval_50/` | **AGREE / path unsafe** | `.gitignore` ignores `*.png` → images there could never be committed; real figures also carry licenses. Keep the set outside git with a checksummed manifest (FR-106). |
| D2-26 | Raise `min` figure width to 5.5 in and DPI ≥ 80 (final snippet) | **Contradicts the goal** | Removes the small/thumbnail regime Doc 1 identifies. Decouple instead: render at a normal size, then resample (FR-085). |

## 3. Defects found that neither document mentions

| ID | Defect | Evidence | Impact |
|---|---|---|---|
| **N-01** | `multi_chart_detection` raises `UnboundLocalError: clsmap_obj` when the primary chart is `line`/`area`/`pie` | `clsmap_obj` is assigned only inside `if dataset_format != 'multi_chart_detection'`; read at `generator.py:4030`. 3–4 of 10 images fail per layout (≈30% at default weights). Mode writes only `chart_N.txt` (no `_detailed.json`). | Blocks every multi-chart item in both documents. |
| **N-02** | 12–12.4% of v4.0 annotations have `class_name:"unknown"` | `generator.py:4035,4037,4038` and `4451-4454` pass `None` as `cls_map` for `area_seg`/`line_seg`/`line_marker`, though `CLASS_MAP_AREA_SEG/LINE_SEG/LINE_MARKERS` exist (`:106-111`). | Poisons COCO categories, class statistics and any per-class gate. |
| **N-03** | Only the **first visible axis** is ever annotated | `return annotations` at `generator.py:1413` is indented inside `for ax_idx, ax in enumerate(fig.axes)`. **E9**: forced 2×1 composite → annotations in top half 177, bottom half **0**; confirmed visually. Also affects twin-axis (dual-axis) charts. | Half of every composite (and every right-axis) is rendered but unlabeled → false negatives. Fixing it re-activates D2-04. |
| **N-04** | `error_bar` class is never produced | `'ErrorbarContainer' object has no attribute 'get_window_extent'` swallowed by `except` (`generator.py:~1378-1384`); 0 `error_bar` annotations in all runs. | A declared class is silently dead. |
| **N-05** | `obb` exists only for text (36%) | E3. Spec 001 FR-001 says "every annotated element". `save_annotations_yolo_obb` falls back to AABB-derived corners. | COCO-O cannot claim tight OBBs for non-text without an explicit `obb_derived` flag. |
| **N-06** | Filter drops are `print`-only and the size filter is absolute | `generator.py:4057-4076`: non-`scatter/box/heatmap` charts drop any box <8 px; nothing is recorded in outputs. **E10** (upper bound): after uniform downscale 0.75/0.5/0.35 the share of text boxes under 8 px is ~9/34–42/87–91%. | Resolution diversification would silently delete most text labels unless the policy changes. |
| **N-07** | Baseline is red and non-reproducible | `pytest tests/ --ignore=tests/regression`: **210 passed, 13–14 failed, 1 skipped** before any change (13 standard, 14 on single-core CPU due to `test_m4_pipeline_e2e` timing). 2 = missing committed goldens (`tests/golden/`); 8 = semantic-manifest & catalog expansion drift (including 13 duplicate titles in `test_title_sampling`, stem collisions, and near-duplicates); 2 = cold-start and import latency budgets (50 ms budget vs 125–458 ms for 6× catalog); 1 = timing-sensitive subprocess. CI installs only `requirements.txt`, but tests import `rapidfuzz`/`ruamel.yaml`/`rich` (`requirements-tools.txt`). `*.png` is git-ignored, so spec 001's image fixtures cannot be in the repo. | The "legacy output must not change" gate is currently unenforceable until triaged. |
| **N-08** | Misspelled/unsupported effect params are silently ignored | Effect functions end in `**kwargs` (e.g. `effects.py:259,285,330`). | Config typos become silent no-ops (Doc 2's parameter names would pass unnoticed). |
| **N-09** | Dead config keys | `multi_chart_detection.caption_probability/body_text_probability/margin_px_range` are never read; only `pdf_context_noise` is (`generator.py:4044`). | Tuning them does nothing. |
| **N-10** | Font fallback is silent | See D1-04. | Dataset typography is environment-dependent and poorer than configured. |
| **N-11** | Default RNG consumption is part of the contract | `test_seeded_generation_20_golden_hash` hashes labels + `_detailed.json` for 20 default-config images. Any new `random.*` call on the default path shifts every later draw. **E12**: `apply_realism_effects` draws `random.random()` once per key *before* testing `p` (`generator.py:1823`), so even a new `p = 0.0` key in the default `realism_effects` changes seeded output; a registry-only addition does not. | New features must not draw from the global RNG when disabled (FR-082); no new key in `DEFAULT_CONFIG["realism_effects"]` (FR-111). |
| **N-12** | The framing prior is a constant | `fig.tight_layout(pad=2.0)` (`generator.py:3857`), one `figsize`, no `bbox_inches`. **E11** (48 baseline images): the annotation union sits 4.0% of the width from L/R and 5.6% of the height from T/B, std ≤ 0.001, at all three DPIs. `pdf_document_context` forced on still leaves every image ≥ 2% from every edge (per-side p5 ≥ 5.9%). | A detector can learn absolute position; tight-cropped figures are never seen (FR-109). |
| **N-13** | Geometry can already lie outside the canvas; only AABBs/polygons are viewport-clipped | `clip_bbox_to_viewport` / `clip_polygon_to_viewport` (`generator.py:2030,2065`, applied at `:4089ff`) do not touch OBBs or pose keypoints. **E13-A**: 3/400 (0.75%) baseline OBBs have a corner > 0.5 px outside the canvas; some pre-effect polygon/OBB points lie > 100 px outside it. | Canvas-shrinking effects must clamp their guard to the canvas and compare against a same-seed control, not against zero (FR-110, FR-113). |
| **N-14** | Line and Area primary `labels/*.txt` use Bar class map | `CHART_CLASS_MAPS` is keyed by `line_obj` and `area_obj`, but `generator.py:3872,4029,4446` queries `.get(primary_chart_type, CHART_CLASS_MAPS['bar'])` with `'line'` and `'area'`. **E15**: in `labels/`, `axis_labels` gets id 8 and `legend` gets id 5 (bar map); in `line_obj_labels/`, `axis_labels` gets id 6 and `legend` gets id 3. `line_segment` and `area_series` are missing from `labels/` entirely. In `_detailed.json`, streams duplicate elements with conflicting IDs. | Detectors trained on `labels/` learn corrupted class IDs; v4.0 stream merge produces duplicates; blocks category registry (FR-074, FR-117). |
| **N-15** | Legend column variation is a no-op | `apply_legend_variation` (`chart.py:628`) sets `legend._ncol = ncol` on the created Legend object. In Matplotlib (tested on 3.10.8), mutating private `_ncol` does not re-layout entries. **E14**: 8-entry legend extent before and after `_ncol=2` is identical (66×172 px vs 149×88 px when created with `ncols=2`). | **Dormant today**: the `num_items > 6` branch is the only user of `_ncol`, and line/area draw `random.randint(1, 4)` series (`chart.py:1871,3073`) while pie legends bypass the function, so no current output changes. Latent once a profile or FR-114 produces > 6 entries. |
| **N-16** | `--mode classification` / `dataset_format == 'classification'` is dead | CLI accepts `--mode classification` (`generator.py:4871`), but `generator.py` only branches on `dataset_format == 'multi_chart_detection'` vs detection fallback. It produces normal detection bounding boxes; no classification directory structure (`class/img.png`) or classification manifest is emitted; no open-set negative background images are produced. | Pipeline has an upstream chart classifier (confirmed by user) that cannot be properly trained from generator runs without a dedicated classification export and background images. |
| **N-17** | `PUBLICATION_THEMES` is dead code | `themes.py:300` defines **9** themes (`nature`, `science`, `cell`, `lancet`, `pnas`, `bmj`, `nanotech_short`, `open_access_poster`, `print_high_contrast`; there is no `ieee`) at 3.2–8.0 in, 300–600 DPI, 6–10 pt. No generator code references `PUBLICATION_THEMES`. All three requested families (Arial, Helvetica, Times New Roman) render as `DejaVuSans.ttf` here (D1-04), so wiring the themes depends on the bundled-font aliases (T234). | Compact high-DPI scientific figures (e.g. single-column 3.3 in at 300 DPI) are completely absent from synthetic data despite being dominant in PubMed Central. |
| **N-18** | Legend sparsity and un-decomposed block annotation | Legends are created only for `line` and `area` (`num_series > 1`, p 0.7; `chart.py:1987,3232`) and `pie` (`legend_prob`, default 0.0). Bars have two-series styles (`side_by_side`, `stacked`, `chart.py:1570`) with no legend call. Multi-series bar (grouped/stacked), multi-series scatter, box, and histogram charts never produce legends. **E14**: 0/6 bars and 0/5 scatters have legends. Furthermore, legends are annotated only as a monolithic `legend` box, never decomposed into per-entry `legend_label`, `legend_marker`, and `legend_title`. | In downstream detectors (e.g. NVIDIA Nemotron), `legend_title` (AP 60.6%) and `legend_label` (AP 84.1%) are distinct targets. CHART-Info evaluates symbol-to-label pairing. |
| **N-19** | Free-text clutter and tick formatters are absent | Grep across `chart.py` and `generator.py` shows 0 calls to `ax.annotate`, `fig.suptitle`, `fig.text`, or matplotlib tick formatters (`PercentFormatter`, `StrMethodFormatter`, `EngFormatter`, `DateFormatter`). Real charts have subtitles, footnotes ("Source: ..."), R² / p-value statistics, and formatted numbers (`$`, `k`, `%`). | In Nemotron, text classes (`mark_label` AP 49.3%, `other` AP 55.1%, `value_label` AP 62.7%) are the weakest. The generator emits only clean axis/chart titles and raw numbers. |
| **N-20** | Ghost Axis Annotations on Hidden Twin Axes (exposed by N-03 fix) | In Matplotlib, `ax2 = ax.twinx()` hides the twin's X-axis via `ax2.xaxis.set_visible(False)` (or `ax.twiny()` hides Y-axis via `ax2.yaxis.set_visible(False)`). However, `ax2.get_xticklabels()` returns `Text` instances that still report `label.get_visible() == True`. In `generator.py:535-560` and other chart-specific tick loops, `get_granular_annotations` extracts tick labels and titles without checking `ax.xaxis.get_visible()` or `ax.yaxis.get_visible()`. Under N-03, `ax2` was never iterated over due to premature `return`. Fixing N-03 processes `ax2` and emits 5 phantom `axis_labels` bounding boxes over empty whitespace where nothing is drawn. | Emits completely blank false-positive bounding boxes on all dual-axis / twin-axis charts. Fix must unconditionally gate X-axis label/title extraction on `ax.xaxis.get_visible()` and Y-axis on `ax.yaxis.get_visible()`. |

## 4. Effects safety (E5, N=16 each, text boxes aligned = <5% ink "empty")

| Effect (forced `p=1`) | Annotations preserved | % empty text boxes | Note |
|---|---|---|---|
| `scan_rotation` ±1.5° | yes | 0.0 | median text IoU(AABB,OBB) 0.94 |
| `perspective` 0.05 / 0.15 / 0.30 | yes | 0.0 | 0.30 shrinks content heavily |
| `perspective` **0.50** (shipped default) | **−19%** | **4.6** | unsafe |
| `page_curl` amp 0.03 / 0.015 | yes | 0.0 | |
| `clipping` 1–4% | yes | 0.0 | |
| `pdf_document_context` | yes | 0.0 | canvas grows ~7–34% |
| `uneven_lighting` default (0.6) | yes | 0.0 | luminance 237 → **158** (too dark) |
| `uneven_lighting` 0.25 | yes | 0.0 | luminance 204 |
| `chromatic_aberration` | yes | 0.0 | |
| `grid_occlusion` | yes | 7.9 | washes out text; keep off pending visual review |
| rotation + persp 0.05 + curl + pdf context (composed) | yes | 0.0 | spec 001's transform_steps compensate correctly |

`realism_effects` has 24 entries; eight are at `p = 0.0`, including `grid_occlusion` (missed by both documents).

**Conclusion**: spec 001's annotation compensation works. Enabling the geometric effects is safe at bounded parameters; the hazards are two *parameter* defaults, not the pipeline.

## 5. Per-class ink-ratio reference (E4, clean baseline) — basis for FR-104

| Class | median | p5–p10 | Note |
|---|---|---|---|
| `data_point` | 0.95 | 0.62 | tight marker boxes |
| `bar` | ~1.0 | 0.08–0.17 | hollow / hatched / dotted styles |
| `axis_labels`, `axis_title`, `chart_title`, `data_label` | 0.28–0.41 | ~0.24 | text |
| `legend` | 0.22 | — | |
| `chart` (plot-area container) | 0.09 | 0.03 | mostly background by definition |
| `line_segment`, `outlier`, `series_keypoints` | 0.08–0.10 | — | thin / tiny marks |

Takeaway: ink ratio is informative only for some classes; the audit must be class-conditioned, calibrated from a clean run, and must **report** rather than silently delete.

## 6. Decisions carried into the spec

1. **Fix before diversify** (N-01…N-07) — baseline-2 goldens are regenerated only after these land.
2. **Profiles, not in-place default edits** — legacy stays byte-identical (N-11); the new behaviour ships as `domain_gap_v1`.
3. **Resolution via render-then-resample**, plus a modest figsize/aspect range — not Doc 2's narrowed range.
4. **Tick rotation**: default `rotation_mode`, `ha` chosen by angle sign (D2-18/19).
5. **Fonts**: bundled, glyph-checked, deterministic; never `ttflist` (D1-14).
6. **COCO** from v4.0 records, categories keyed by `class_name`, never `unknown` (D2-09, N-02).
7. **Quality gate** is an audit with calibrated, class-specific thresholds (D1-11, D2-16).
8. **Document context**: record context regions first; decide on labeled classes from measurement (D2-11).
9. **Copy-paste of data marks rejected**; decal overlays only (D2-14).
10. **Sim-to-real harness gates promotion of any profile to default** (D1-13, D2-23).
11. **Canvas framing**: one `canvas_reframe` re-margin effect (tight or loose margin per side) with a mandatory content guard, profile-only, ordered after the geometric effects; kept or dropped by the DR-001 ablation arm (§8, N-12, N-13).

## 7. Reproduction map

| Experiment | Command | Reproduces |
|---|---|---|
| E1 | `repro_005.py sizes` | D1-01 |
| E2 | `repro_005.py figsize --n 24` | D2-06, D2-26 |
| E3 | **Not scripted** (AABB-vs-OBB IoU under forced `scan_rotation`, shapely); will be covered by T229 | N-05 |
| E6 | **Not scripted**; covered by the T216 N-01 reproducers (`multi_chart_detection`, one test per primary chart type) | N-01 |
| E4 | `repro_005.py ink --n 24` | D1-11, D2-16, N-02, §5 |
| E5 | `repro_005.py effects --n 16` | D2-15, D1-10, §4 |
| E7 | `repro_005.py ticks` | D2-18/19 |
| E8 | `repro_005.py fonts` | D1-04, D1-14, D2-12 |
| E9 | `repro_005.py multi-axes` | N-03 |
| E10 | `repro_005.py projection` | N-06 |
| E11 | `repro_005.py margins --n 48` | N-12, D3-01, D3-03 |
| E12 | `repro_005.py rng` | N-11, D3-05, D3-06 |
| E13 | `repro_005.py reframe --n 24` (runs only once `canvas_reframe` exists, T254; the §8 numbers come from a scratch prototype) | N-13, D3-04, D3-09 |
| E14 | `repro_005.py legend --n 40` | N-15, N-18, §9.2 |
| E15 | `repro_005.py streams --n 16` | N-14, §9.1 |
| E16 | `repro_005.py legend-markers --n 10` | §9.2 addendum, FR-114b, SC-012 |
| E17 | `repro_005.py dpi --n 12` | §9.5, NFR-B |

---

## 8. Doc 3 — randomized canvas padding / tight-crop effect

**Proposal (as received)**: a new `apply_canvas_padding_effect` in `effects.py` returning `(new_img, dx, dy)`; stochastically *tight-crop* flush to the outermost visual elements (automated PDF figure extractors) or *pad* with independent asymmetric margins in white/off-white; register it in `EFFECT_REGISTRY` (`generator.py:1705`), feed the offsets into `transform_steps` in `apply_realism_effects` (`:1742`), expose its probability and margin limits under `realism_effects` (`config_defaults.py:240`) and adjust them live in `synth/semantics/admin/tui.py:806`; goal: remove spatial edge bias in downstream OCR/detection models.

### 8.1 Claim ledger

| ID | Claim | Verdict | Evidence |
|---|---|---|---|
| D3-01 | The generator has a spatial edge bias (charts always sit at the same distance from the edges) | **CONFIRMED, stronger than stated** | N-12 / **E11**: margins are constants, std ≤ 0.001 over 48 images. |
| D3-02 | Tight crops simulate automated PDF figure extractors; real figures have such margins | **UNVERIFIED** | Plausible; no measurement of real figures. `real_eval_v1` should record per-side margins (T241) to calibrate the distribution. |
| D3-03 | Padding with independent asymmetric random margins in off-white tones is new | **PARTIAL** | `apply_pdf_document_context_effect` (`effects.py:330`) already draws four independent margins and an off-white page tone U[245,255]. It differs: margins are absolute 20–50 px (resolution-blind, never < 20 px), coupled with caption/body text, `p = 0` by default (0.5 in `multi_chart_detection`, D1-03). **E11**: forced on, no image has any side < 2%. The genuinely new part is the **tight** half and resolution-relative margins. |
| D3-04 | `(new_img, dx, dy)` + `transform_steps` keeps all geometry pixel-synchronized | **CONFIRMED** | The dispatcher branch for `clipping` / `pdf_document_context` (`generator.py:1835`) appends `("translate", dx, dy)`; the same mapping covers crop and pad: with the new canvas as a box `(bx0, by0, bx1, by1)` in source image coordinates, `dx = −bx0`, `dy = by1 − H` (Matplotlib bottom-left). **E13-B**: 100.0% of annotations preserved, 0.0% empty text boxes. |
| D3-05 | Register in `EFFECT_REGISTRY` | **CONFIRMED, RNG-neutral** | **E12**: registry-only addition leaves the seeded label hash unchanged. The TUI list is built from the registry, so the effect appears there automatically. |
| D3-06 | Expose `p` and margin limits in `config_defaults.py:240` | **REFUTED as written** | `:240` is the start of `realism_effects`, but **E12**: adding a `p = 0.0` key there changes the seeded hash (one global draw per key, `generator.py:1823`) → breaks `test_seeded_generation_20_golden_hash` and Scenario 1. Must live in the profile overlay (FR-111). |
| D3-07 | Probability and margin limits adjustable live via `tui.py:806` | **PARTIAL** | `_edit_realism_effects` edits only `realism_effects.<name>.p`; saves go to `custom_config.py`. Margin limits are not reachable there. |
| D3-08 | The effect will eliminate edge bias in downstream OCR/detection models | **UNVERIFIED** | No model trained. The effect removes the *constant* prior from the data (D3-01); whether that improves detection is decided by the DR-001 ablation arm (FR-107). |
| D3-09 | Cropping "flush to the outermost visual elements" is safe | **UNSAFE if pixel-only** | **E13-D**: a zero-slack crop to the pixel-ink box puts 16/400 (4.0%) OBBs outside the canvas (baseline 3/400) and 5.0% of annotations touch or are clipped by the edge: text boxes extend beyond their ink. A guard built from annotation geometry removes both (E13-B/C). |

### 8.2 Prototype results (E13, scratch copy; N = 24 unless noted, seeds `7+i`, effects otherwise off)

Prototype parameters: per side independently, `p_tight` 0.4 → margin U[0, 6] px, else U[0, 12%] × min(W,H); fill = figure background (p 0.5) or page tone U[245,255]; guard = ink box ∪ all annotation geometry, clamped to the canvas.

| Condition | Annotations kept | Empty text boxes (< 5% ink) | OBBs outside canvas | Annotations touching edge | Distinct aspect ratios |
|---|---|---|---|---|---|
| A baseline | 100% | 0.0% | 3/400 | 0.0% | 1 |
| B default (guard) | 100.0% | 0.0% | 3/400 | 0.0% | 18 |
| C tight-only (guard, 0–6 px) | 100.0% | 0.0% | 3/400 | 0.4% | 6 |
| D tight-only, **no guard**, 0 px | 100.0% | 0.0% | **16/400** | **5.0%** | 5 |
| G composed (rotation ±1.5°, perspective 0.05, page_curl 0.02), no reframe | — | — | 3/400 | — | — |
| E G + reframe **last** | ≥ 100% | 0.0% | 1/400 | 0.1% | 20 |
| F G + reframe **first** | ≥ 100% | 0.0% | **8/400** | 0.4% | 17 |

- **Order matters** (E vs F): when reframe runs first, the later rotation/perspective/curl push flush content over the edge, and the borders they add undo the tight crop (tightest-side median 1.4% vs 0.5%). Reframe must follow the geometric effects.
- **Non-rigid steps need the final OBB in the guard** (N = 48, zero-slack tight crop): rigid-only 4/754 → 4/754 off-canvas OBBs; `page_curl` alone 5/754 → **47/754** with the 4-corner guard, still 47/754 with densified corners, and **5/754** once the guard includes the `minAreaRect` corners that `apply_realism_effects` produces for OBBs under a mesh (a tilted minimum-area rectangle's corners lie outside the extent of the points it encloses).
- **Framing diversity** (B, N = 96): per-side annotation-union margin std 0.024 / 0.036 / 0.028 / 0.036 (L/T/R/B; baseline ≈ 0.000); tightest side < 1% in 91% of images, < 2% in 97%; loosest side > 6% in 74%, > 10% in 18%; 38 distinct aspect ratios (2 dp); annotations kept 3660 vs 3660; `_detailed.json` `image.width/height` matches the PNG in 96/96; OBBs outside 7/1539 vs 8/1539 at baseline.
- **`p < 1` leaves a point mass** at today's margins (≈ `1 − p` of the images); only `p ≈ 1` removes the constant prior. Sweep (N = 96, prototype): `p` = 0.3 / 0.5 / 0.8 / 1.0 → smallest per-side std 0.015 / 0.020 / 0.023 / 0.024; tightest side < 1% in 26 / 46 / 75 / 91% of images; ≈ 71 / 48 / 19 / 0% of images keep today's margins. A profile `p` below ≈ 0.8 fails SC-011's std bound. Whether the residual point mass is desirable is an ablation question (DR-001), not an assumption.

### 8.3 Verdict

**Accept with changes** → FR-109…FR-113, SC-011, T254–T255. The premise is real and stronger than claimed (D3-01), the compensation path already exists (D3-04), but five parts of the proposal are changed: (1) one **re-margin** effect (crop to the content box, then draw each side's margin) replaces the crop-*or*-pad choice, which would leave a gap between ≈ 0.5% and ≈ 4%; (2) a mandatory **content guard** computed inside `apply_realism_effects` (the effect never sees annotations); (3) **profile-only** configuration and a feature-scoped RNG instead of `config_defaults.py` (D3-06); (4) mandatory **ordering** after geometric effects; (5) the TUI promise is limited to `p` (D3-07). The "eliminates bias in OCR/detection models" claim stays UNVERIFIED until DR-001.

---

## 9. Reviewer findings and external benchmark analysis

A deep critical review of the spec, the codebase, and external chart detection/extraction systems (NVIDIA Nemotron Graphic Elements v1, UB PMC / ICPR 2022 CHART-Infographics, LineEX, Ultralytics open-set guidance, Augraphy) revealed six verified gaps and produced experimental measurements.

### 9.1 Claim ledger

| ID | Finding / Proposal | Verdict | Evidence |
|---|---|---|---|
| R4-01 | Legends are sparse, missing on bar/scatter, and `_ncol` mutation does nothing | **CONFIRMED** | N-15, N-18. **E14**: 0/6 bar, 0/5 scatter have legends; `legend._ncol = 2` yields identical bounding extent (66×172 px) while `ncols=2` yields 149×88 px. |
| R4-02 | Line and area primary `labels/` use bar class map; streams diverge in v4.0 | **CONFIRMED** | N-14. **E15**: `labels/` outputs `axis_labels`=8, `legend`=5; `line_obj_labels/` outputs `axis_labels`=6, `legend`=3. In `_detailed.json`, `annotations` and `annotations_obj` duplicate 10 pairs with IoU ≥ 0.8. |
| R4-03 | `--mode classification` is dead; upstream classifier has no background data | **CONFIRMED** | N-16. `generator.py:4871` accepts `--mode classification`, but execution path falls through to detection bounding boxes. User confirmed an upstream chart classifier exists and requires synthetic training data. |
| R4-04 | `PUBLICATION_THEMES` defines journal figures at 300 DPI but is dead code | **CONFIRMED** | N-17. `themes.py:300` defines 9 journal themes (3.2–8.0 in, 300–600 DPI, 6–10 pt; fonts fall back to DejaVu Sans). Unreferenced anywhere in generation. |
| R4-05 | Free-text clutter (notes, source lines, panel letters, R²) and tick formatters are absent | **CONFIRMED** | N-19. Repo grep confirms 0 suptitles, footnotes, or number formatters. Nemotron model card shows `other` (AP 55.1%) and `mark_label` (AP 49.3%) as lowest-performing classes. |
| R4-06 | Upsampling normal renders to 1200–2400 px causes blurred text; native high DPI needed | **CONFIRMED** | Geometry analysis: 672×480 rendered at 96 DPI and upsampled 2×–3× blurs fonts. Rendering at native 200–300 DPI keeps vector glyphs razor-sharp. |
| R4-07 | Augraphy library should replace 24 hand-crafted realism effects | **REFUTED as-is** | Empirical audit: 2 of 23 effects crash; `PaperFactory` takes 15.8 s/img; `LCDScreenPattern` shifts pixels up to 2.5 px; default parameters wash out chart text and lines. Only safe subset (≤ 0.5 px shift) behind FR-093 gate is viable. |
| R4-08 | UB PMC can bootstrap `real_eval_v1` | **CONFIRMED, name and license wording corrected** | The dataset comes from the ICPR 2020 / ICPR 2022 CHART-Infographics competitions; no "2024" edition was found. The ICPR 2022 page says only CC-BY figures were selected (the ICPR 2020 TC11 page says "a Creative Commons license"), so licenses are still verified per image from the PMC id (FR-117). Annotation redistribution terms are not stated in the sources checked. |

### 9.2 Legend extractor validation & ink-ratio measurement

A prototype decomposed legend extractor was built and tested against 12 legend configurations (inside positions, outside-right, horizontal bottom, horizontal top, multi-column, and with title):
```python
def extract_legend_components(ax, renderer, img_h):
    lg = ax.get_legend()
    if lg is None or not lg.get_visible():
        return []
    flip = lambda b: (b.x0, img_h - b.y1, b.x1, img_h - b.y0)
    out = []
    t = lg.get_title()
    if t and t.get_text().strip():
        out.append(("legend_title", flip(t.get_window_extent(renderer)), t.get_text().strip()))
    for txt, handle in zip(lg.get_texts(), lg.legend_handles):
        out.append(("legend_label", flip(txt.get_window_extent(renderer)), txt.get_text().strip()))
        out.append(("legend_marker", flip(handle.get_window_extent(renderer)), None))
    return out
```
**Results**:
- 56 distinct entries across 12 configurations yielded **0 exceptions** and **0 geometric inversions**.
- Measured ink ratios for `legend_label`: median 0.32 (range 0.29–0.36), perfectly matching the reference text ink-ratio baseline (0.28–0.41, §5).
- Measured ink ratios for `legend_marker`: median 0.88 (range 0.72–0.98) in those 12 prototype configurations. This does **not** generalize across handle types (E16 below).
- **Validation addendum (E16, the extractor above run on `apply_legend_variation` legends, 30 per type)**: marker-less `Line2D` handles give a **zero-height** box (`Line2D.get_window_extent` has no marker padding) in 119/119 cases, so ink is unmeasurable and the box would also fail the 8 px size filter; marked lines give a 10 px box with median ink 0.41; area patches give 11.7 px and ink 1.00. A fixed 0.70–0.98 band therefore cannot be an acceptance criterion; the extractor needs a rule for line handles (FR-114b) and thresholds calibrated per handle type (FR-104).
- **Ordering & backward compatibility**: The monolithic `legend` box is preserved for legacy and basic detection. The granular classes (`legend_title`, `legend_label`, `legend_marker`) are added as append-only categories in `categories_v1.json` and emitted in `_detailed.json` under profile control.

### 9.3 UB PMC / Nemotron benchmark mapping

All ten Nemotron Graphic Elements v1 AP values below match its model card. Its evaluation set is the PMC Chart dataset (560 images; only 38 `chart_title`, 19 `legend_title`, 219 `mark_label` boxes), and `legend_title` has only 209 training boxes, so the weak classes are noisy and track scarcity. Its class list is `chart_title, x_title, y_title, xlabel, ylabel, other, legend_label, legend_title, mark_label, value_label` — **there is no `legend_marker` class**, and all ten classes are *text* (`mark_label` = "labels associated to markers", not the markers). The UB PMC role taxonomy itself was not checked; the x/y names are NVIDIA's.
- `chart_title` ↔ `chart_title` (Nemotron AP 82.4%)
- `axis_title` ↔ `x_title` / `y_title` (Nemotron AP 88.8% / 89.5%)
- `axis_labels` ↔ `xlabel` / `ylabel` (Nemotron AP 85.0% / 86.2%)
- `legend` (monolithic) ↔ divided into `legend_title` (AP 60.6%) and `legend_label` (AP 84.1%)
- `data_label` ↔ `value_label` (Nemotron AP 62.7%)
- `data_point` / `line_segment` / `bar`: no Nemotron counterpart (its classes are text only; `mark_label`, AP 49.3%, is text attached to marks)
- `other` ↔ non-plot text, footnotes, source lines, subtitles (Nemotron AP 55.1%)

The mapping is many-to-one (`x_title`/`y_title` → `axis_title`, `xlabel`/`ylabel` → `axis_labels`, `value_label` → existing `data_label`, no new class). Once UB PMC roles are verified (T241b) and per-image licenses checked, `scripts/fetch_real_eval.py` can bootstrap ~50 open-access biomedical figures into `real_eval_v1` without drawing boxes by hand.

### 9.4 Augraphy empirical audit

Testing Augraphy 8.2.5 on generator output:
- **Throughput & stability**: 20 of 23 augmentations execute in 0.04–0.3 s per image. However, `PaperFactory` requires 15.8 s per 840×600 image. `BadPhotoCopy` and `Scribbles` crashed with memory and index faults in the testing environment.
- **Pixel shift & annotation corruption**: 10 augmentations stay within ≤ 0.5 px edge shift (measurement noise floor). However, `Faxify` resamples and alters canvas aspect, breaking `transform_steps`. `LCDScreenPattern` shifts feature edges by up to 2.5 px, which corrupts tight character bounding boxes.
- **Content washout**: Applying the default full pipeline fades high-frequency gridlines and washes out axis text, causing > 12% of text boxes to fall below the 5% ink threshold.
- **Reproduction gap**: this audit has no script or experiment id, and the list of 8 "verified-safe" operations above names only 5 (`etc.`). FR-090's "verified-safe operations" is undefined until both exist.
- **Verdict**: Do NOT adopt Augraphy as an unconstrained engine. Select only the 8 verified-safe operations (`LightingGradient`, `SubtleNoise`, `BleedThrough`, `Dust`, `PageBorder`, etc.) under strict parameter bounds, validated through the FR-093 effects integrity gate.

### 9.5 High native DPI vs bilinear/bicubic resampling (FR-085 refinement)

In FR-085, the plan was to render at standard figsize (7×5 in at 96–150 DPI) and resample.
- For small target resolutions (256–800 px), resampling with anti-aliasing faithfully models thumbnail and low-resolution PDF downscaling.
- For large target resolutions (1200–2400 px), upsampling an 840×600 bitmap results in fuzzy text edges and interpolation artifacts not present in native high-resolution digital PDFs.
- **Cost (E17)**: native 340 DPI (≈ 2380×1700) renders at 1.43–1.51 img/s vs 3.6–3.8 img/s at 120 DPI, ≈ 2.5× slower per image (NFR-B).
- **Refinement**: When target long side ≥ 1200 px, the generator MUST render directly at higher native DPI (`dpi = int(target_px / figure_inches)`) to preserve crisp vector text rasterization.

### 9.6 Upstream classification model support & open-set negative backgrounds

The user confirmed: *There is an 8-class chart-type classifier ahead of the detector in the production pipeline, which is also trained with generator data.*
- Today, `--mode classification` outputs YOLO detection files into `labels/*.txt`.
- When training a classifier, false positives on non-chart document elements (tables, full-page text, formulas, illustrations, photos) are the primary failure mode.
- Ultralytics' guidance (checked) is "about 0–10% background images" for **object detection**: unlabeled images added to reduce false positives. It is not a classifier rule; for the classifier the share is a tunable default (8%, FR-115), and the same generator should also be able to emit unlabeled background images into detection datasets.
- The classification dataset needs a reject class (class 8: `non_chart`).
- The generator must provide a real classification export format (standard ImageNet directory tree `train/<class_name>/<id>.png` and a CSV/JSON manifest) and generate synthetic document/table/text negative images.

### 9.7 Latent defect N-20: Ghost axis annotations on hidden twin axes

When N-03's premature `return` inside `for ax in fig.axes` was resolved by moving `return annotations` out of the loop, all axes (including twin axes instantiated via `ax.twinx()`) became reachable.
- **The Bug**: Matplotlib's `twinx()` internally executes `self.xaxis.set_visible(False)` on the secondary axis. However, `ax2.get_xticklabels()` still returns the underlying `Text` artist instances, each with `label.get_visible() == True`.
- **The Failure**: In `generator.py:535-560` and other chart-specific tick extraction loops, `get_granular_annotations` only checks `label.get_visible() and label.get_text().strip()`. It does **not** check whether `ax.xaxis.get_visible()` is true!
- **Consequence**: The generator extracted 5 ghost `axis_labels` bounding boxes corresponding to `ax2`'s hidden X-axis tick labels, placing valid-looking text bounding boxes over empty whitespace where nothing is drawn on the canvas.
- **Resolution**: Every X-axis tick label and X-axis title extraction loop MUST unconditionally gate on `if ax.xaxis.get_visible():`. Similarly, Y-axis extractions MUST gate on `if ax.yaxis.get_visible():`.

### 9.8 Baseline test suite reconciliation & developer review decisions

Empirical execution of the baseline test suite (`pytest tests/ --ignore=tests/regression`) yielded:
- **Result**: 210 passed, 13–14 failed, 1 skipped (394s total execution time).
- **Failure Breakdown**:
  1. `test_title_sampling.py` (1): Fails on `assert len(catalog_titles) == len(CHART_TITLES_CATALOG)` (`assert 9672 == 9685`). Exactly **13 duplicate titles** were introduced during semantic catalog expansion (documented in `audit_report.json` with 59 near-duplicates and 74 stem collisions).
  2. `test_domain_coherence_e2e.py` (4): Fails on strict oracles (`test_continuous_regex_oracle_on_catalog`, `test_scatter_scale_coherence_oracle`, `test_treatment_key_domain_coherence_and_zero_leakage`, `test_in_memory_theoretical_title_benchmark`).
  3. `test_semantic_catalog.py` (2) & `test_semantic_sampling.py` (1): Fails on temporal scale reservation and comparative pairs reachability assertions.
  4. `test_m5_semantic_generalization.py` (1): Fails on `test_is_scientific_derived_from_manifest`.
  5. `test_m0_goldens.py` (2): Fails because `tests/golden/` directory does not exist on disk, and hardcodes stale legacy counts (495 metrics vs actual 3197).
  6. `test_startup_latency.py` (1) & `test_nfr01_import_guard.py` (1): Fails on 50 ms budget (`measured best: 458.23 ms`).
  7. `test_m4_pipeline_e2e.py` (0 or 1): Passes on multi-core systems; times out on single-core constrained CPU.

#### Reviewer Decision Ledger:
- **Decision 1 (Cold-start latency budget)**: The 50 ms budget was defined when the catalog held 495 metrics. With 3,197 metrics + 9,685 pairs (6× larger), loading and validating the registry takes ~125–458 ms. **Decision**: Re-baseline the cold startup budget in `test_startup_latency.py` to 150 ms (scaled by a machine-speed factor), while maintaining strict zero-pydantic import guard (`has_pydantic == False`).
- **Decision 2 (Environment fingerprint & font sensitivity)**: FreeType rasterization and character bounding boxes vary slightly across OSes and font packages (e.g. system Arial/Georgia vs fallbacks). **Decision**: Store goldens generated in the canonical container environment along with `scripts/env_fingerprint.py`. In local dev runs, support `--update-goldens` or verify invariant structure (element counts, classes, relative boxes), while reserving strict byte-level sha256 matching for the standardized CI container.
- **Decision 3 (Registry golden counts)**: In `test_m0_goldens.py`, legacy subsets (`legacy_metrics`, `canonical_metrics`, `histogram_y_labels`) must be asserted for exact content and ordering to prevent regression. Overall totals (`all_metrics`, `all_titles`, `all_pairs`) must be asserted as lower bounds (`>= 495`, `>= 192`, `>= 96`) so intentional catalog expansion does not break baseline tests.

