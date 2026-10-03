# 001 — Chart Pipeline Modernization

Spec-Driven Development artifacts for fixing annotation-truth and rendering-architecture issues in the synthetic chart generator (`chart.py`, `generator.py`, `effects.py`, `themes.py`).

Read in this order:

1. **`research.md`** — Phase-0 findings, in four passes. §1-§4 check every claim from the original engineering audit directly against source, with file:line evidence — confirming some, correcting others, and surfacing one new correctness bug (`§2.2`, the uncompensated `"perspective"` effect) the audit missed. §7 does the same to a second, unattributed set of six claims; all six confirmed (two needed a mechanism correction), folded into P0 as FR-019–FR-024. §8 does the same to a third, unattributed set of seven claims: three confirmed as new (FR-025–FR-027), one confirmed but already mitigated by the §7 fix, and three refuted or found not applicable to this codebase. §9 does the same to a fourth, unattributed set of seven claims: four confirmed (FR-028–FR-031, one narrower than stated), two refuted outright (both already mitigated in existing code), and one — twin-axes z-order collision — a valid general risk with no current manifestation, recorded as forward guidance rather than a fix. §5-§6 list the exact PyPI package versions verified installable for later phases, and the questions that need `custom_config.py` or a profiling run to answer.
2. **`spec.md`** — WHAT and WHY. User scenarios, testable functional requirements (`FR-001`…`FR-018`, plus three P0 addenda — `FR-019`…`FR-024` for the §7 findings, `FR-025`…`FR-027` for the §8 findings, and `FR-028`…`FR-031` for the §9 findings), the new data entities (OBB, Keypoint, SeriesLink, ModalAmodalMask, SemanticDomain, TransformStep), success criteria, explicit out-of-scope, and open `[NEEDS CLARIFICATION]` items.
3. **`plan.md`** — HOW. Technical context, a project-specific "constitution check" (nine non-negotiable engineering gates, each tied to a defect found in `research.md`), new module layout, the P0 → P1 → P2 roadmap, migration/versioning strategy, testing strategy, and a risk register (which also tracks §9.1's twin-axes watch item, with no task attached).
4. **`tasks.md`** — the executable checklist. 69 tasks across 14 phases, each with an exact file/function target and a concrete "Done when" acceptance check. Organized so P0 (pure correctness fixes, no new dependency — now eight user stories, Phases 3-8) is fully sequential and blocking; P1 and P2 tasks are grouped so independent tracks can run in parallel.

## Priority at a glance

| Priority | What it buys | New dependencies |
|---|---|---|
| **P0** | Bounding boxes stop lying after rotation/perspective effects; label-visibility check stops rasterizing pixels to make a decision it could make analytically; bar `series_idx` stops being a hardcoded `0` (via passthrough of data already computed at synthesis time); segmentation/pose/marker annotations stop bypassing realism effects; chained canvas-resizing effects stop corrupting earlier transform steps; out-of-bounds annotations get clamped instead of discarded; the generator runs without `custom_config.py`; per-subplot theming stops leaking across subplots and worker iterations; rotated tick labels get a tight OBB instead of a loose rotated-AABB; warped OBBs are guaranteed convex and correctly wound; every JSON export (not just one) handles NumPy/boolean values correctly; the area chart's log-scale guard uses real data instead of a hardcoded placeholder; legend-covered elements get flagged occluded instead of exported as visible; stroke/marker geometry respects the same axes-boundary clipping Matplotlib applies; the golden-master harness gives the same verdict on any machine | None — pure refactor of existing code |
| **P1** | An additive, parity-checked vector backend (SVG/Vega-Lite) alongside Matplotlib; line/area topology as a typed keypoint graph instead of a ribbon polygon; analytic polygon clipping | `shapely`, `svgelements`, `lxml`, `svgpathtools`, `altair`, `vl-convert-python`, `resvg-py`/`cairosvg` |
| **P2** | Domain-coherent axis-label sampling (no LLM required to start); copula-based correlated multi-series tables; non-rigid scan deformation | `pyvinecopulib`/`copulas` (+ optional local LLM, flagged) |

`chart.py`'s statistical/data-synthesis logic (Hill/Michaelis-Menten curves, AR(1) noise, ALR compositional transforms, spectral matrix seriation, etc.) is out of scope everywhere — it was verified sound in `research.md` §4 and is not touched by any phase.

## Project Execution Status

| Phase / Track | Focus | Status | Tests | Key Deliverables |
|---|---|---|---|---|
| **Phase 1–2 (Harness & Foundational)** | Golden master harness, font determinism, timing baselines | **COMPLETE** | 9/9 pass | `tests/regression/test_golden_master.py`, 8 golden baseline fixtures (clean & augmented) |
| **P0 (Phases 3–8)** | Core geometric compensation, analytical label visibility, series indexing, OBB convexity, transform closure, memory hygiene | **COMPLETE** | 23/23 pass | `generator.py`, `effects.py`, `merge_json.py`, `tests/test_p0_*.py` |
| **P1 (Phase 9 — Vector Backend)** | Lean functional Vega-Lite / SVG engine, analytical DOM extractor, pixel IoU $\ge 0.98$, engine dispatcher | **COMPLETE** | 33/33 pass | `backends/vegalite_backend.py`, `backends/__init__.py`, `generator.py` dispatcher, `tests/test_p1_vegalite_backend.py` |
| **P1 (Phase 10 — Topologies & Exporters)** | Topological keypoints, line adjacency graphs, additive YOLOv8-OBB exporter, modal/amodal masks | **COMPLETE** | 33/33 pass | `save_annotations_yolo_obb`, `tests/test_p1_topologies_and_exporters.py` |
| **P2 (Phases 11–13 — Semantics & Robustness)** | Domain-coherent label sampling, copula tabular synthesis, non-rigid mesh deformation | **COMPLETE** | 43/43 pass | `synth/domain_tags.py`, `synth/copula_tables.py`, `effects.py:apply_page_curl`, `generator.py:dense_perimeter_sampling`, `tests/test_p2_*.py` |
| **Phase 14 & Remediation (Polish & Benchmarking)** | Config fallback, amodal retention, baseline distance cutoff, multiprocessing scaling, benchmark report, schema documentation | **COMPLETE** | 46/46 pass | `config_defaults.py`, `SCHEMA.md`, `tests/benchmarks/benchmark_results.json` |

---

## Before starting P0

`tasks.md` Task T001 requires reading `custom_config.py` (not part of this review) to resolve three open questions in `spec.md`'s Clarifications section, plus one more (d): whether a `batch_merge_all` implementation exists anywhere in the target deployment (`research.md` §7.5). Do this first — it changes the exact scope of three P0 tasks (T013, T020, T030).

