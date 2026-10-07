# Spec 005 — Domain-Gap Closure

**Status**: Phase 0 & Phase 1 Complete · **Branch**: `005-domain-gap-closure` · **Builds on**: specs 001–004

Verifies external domain analysis documents, user feedback, and empirical audit against the generator code and output, then turns what survived into a rigorous spec-driven program: **fix the baseline defects → gate new behaviour behind profiles → add diversity improvements & granular legends → export to COCO/COCO-O & classification formats → measure against real chart benchmarks → decide what becomes default.**

## Contents

| File | Purpose |
|---|---|
| `research.md` | Claim ledgers (`D1-xx`, `D2-xx`, `D3-xx`, `R4-xx`), verified defects (`N-01…N-19`), effects-safety table, per-class ink reference, benchmark mapping, reproduction map |
| `spec.md` | FR-073…FR-117, FR-074b, NFRs, scenarios 1–8, success criteria SC-001…SC-014, open decisions OD-1…OD-8 |
| `plan.md` | Inherited-constraints check, phases and gates, hard ordering constraints (OC-1…OC-9), risk register (R1…R15), effort ranges |
| `tasks.md` | T216…T255 with concrete **Done when** criteria and an exhaustive FR→task matrix |
| `research/repro_005.py` | Re-runs the measurements: `REPO=<checkout> python research/repro_005.py <sizes\|figsize\|ink\|effects\|ticks\|fonts\|multi-axes\|projection\|legend\|streams\|margins\|rng\|reframe\|legend-markers\|dpi>` (E1–E17; E3 and E6 are not scripted) |

## What the verification and audit changed

- **Baseline defects that invalidate annotations and legacy output**:
  * `multi_chart_detection` crashes on ~30% of images (`UnboundLocalError: clsmap_obj`, N-01).
  * 12% of annotations are `unknown` due to `None` class map fallbacks in line/area streams (N-02).
  * Only the **first axis** of any multi-axis figure is annotated due to premature `return` inside the axes loop (N-03).
  * `error_bar` annotations are completely missing (0 produced, N-04).
  * Baseline test suite is red with uncommitted goldens (N-07).
  * `labels/*.txt` for line and area charts falls back to bar chart class IDs (e.g. axis labels emitted as ID 8 instead of 6, N-14), and `_detailed.json` duplicates cross-stream overlapping annotations.
  * Mutating `legend._ncol` on an instantiated Legend object is a silent no-op (N-15), but dormant today: line/area legends have ≤ 4 series, so it changes no current output.
- **Critical domain-gap findings addressed**:
  * **Granular legend decomposition**: Monolithic legend bounding boxes conflate markers and text labels; `Legend` artists are decomposed into `legend_title`, `legend_label`, and `legend_marker` in the `legend_elements` stream while preserving the monolithic box in primary YOLO txt (OC-9).
  * **Upstream chart classifier alignment**: An 8-class upstream classifier exists; added `--mode classification` flat ImageFolder export and 5–10% negative non-chart background distractors (tables, text blocks, photos) to prevent false-positive chart detections (FR-115).
  * **Resolution & publication themes**: the 9 dead `PUBLICATION_THEMES` are wired to physical journal specifications (fonts need bundled aliases); charts target native DPI rendering for ≥ 1200 px to avoid blur and interpolation artifacts from naive down/up-sampling (FR-085, FR-086).
  * **Framing prior**: baseline margins are constants (4.0% / 5.6%, std ≈ 0). `canvas_reframe` (FR-109…113, WS-D) re-margins each side (tight or loose) behind a content guard, so charts appear flush or loosely framed without losing a label.
  * **Augraphy audit & rejection**: Augraphy produces uncompensated pixel shifts (2.5 px) and heavy pipeline latency (15.8 s); rejected as a blanket dependency; only bounded safe operations passing FR-093 are allowed.
  * **Real-world evaluation**: `real_eval_v1` bootstraps from UB PMC (ICPR 2022 CHART-Infographics) figures after a per-image license check.
- **Proposed snippets that would have hurt**: Flat 0.3 ink gate (flags 21% of correct labels), `ha='right'` for negative tick angles (labels displaced 65–111 px), blind system font sampling (61% broken), uncalibrated `perspective` (0.5 destroys ink), `uneven_lighting` defaults (washes out figures), in-place default edits breaking golden regression tests.

## Phases

0. **[Complete] Baseline integrity** (T216–T224): Fix N-01…N-04, N-14, N-15; test triage; committed baseline-2 goldens.
1. **[Complete] Profiles, diversity & legends** (T225–T234c, T254–T255): Feature-scoped RNG, bounded effects, tick rotation & formatters, publication themes & native DPI, bundled fonts, granular legend extraction, canvas reframe.
2. **[Next] Interop, classification & measurement** (T235–T243): Category registry, COCO / COCO-O exporters, classification ImageFolder export & negative backgrounds, class-aware audit, UB PMC bootstrap (`real_eval_v1`), sim-to-real evaluation runner, DR-001.
3. **Closure** (T244–T253, T250b): Context streams & decals, free-text clutter, asymmetric layouts, pie/heatmap/violin, Vega-Lite, promotion review.
