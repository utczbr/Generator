# 002 — Semantic Storage and Domain Registry

Spec-Driven Development artifacts for establishing an authoritative semantic domain registry, resolving axis scale-type mismatches, expanding horizontal line/area lexical diversity with canonical independent variables, eliminating domain-blind fallback titles on ~100% of standard charts, preventing semantic synonym collisions via concept groups, activating all 94 comparative pairs across the pipeline via sector gating, fixing the `is_scientific` serialization bug and twin-axes clobbering, closing tabular PRNG entropy leaks, implementing additive multi-subplot composite dashboard tracking (preserving core taxonomy scalars), and ensuring full pipeline metadata persistence across Matplotlib, Vega-Lite, and `merge_json.py` under Schema Version `v2.1` (`dataset_version = "2.1.0"`).

---

## Schema Versioning

* **Annotation Schema Version**: `v2.1`
* **Dataset Version**: `2.1.0`
* **Shared Module**: `versions.py`

---

## Reading Order

Read the specification documents in this sequence:

1. **`research.md`** — Fourth-pass empirical audit & consensus architecture. Analyzes the true flow of semantic constants across `chart.py`, `generator.py`, `synth/tabular.py`, and `merge_json.py`. Identifies critical serialization drop-points (`detailed_json` save whitelist drop, twin-axes and heatmap colorbar metadata clobbering in `generator.py`, `merge_json.py` OBB dropping), domain override contradictions (dual-axis bar, heatmap), multi-subplot composite overwriting and root taxonomy preservation, comparative pair reachability reality and sector-gated activation, clean line-X typography and domain tagging, and mathematical defect derivations.
2. **`spec.md`** — WHAT and WHY. User scenarios, testable requirements (`FR-031`–`FR-051`), domain routing architecture with configurable subdomain weights, immutable enum-backed data models (`ScaleType` without `ORDINAL`, `AxisRole`, `AxisMetric` with `concept_group`, canonical independent expansion, `ChartTitle`, `ComparativePair`), universal chart rules (role + scale gating for line/area, continuous for scatter, dedicated secondary Y sampling via `METRIC_BY_LABEL`), and independent success criteria (`SC-019`–`SC-026`).
3. **`plan.md`** — HOW. Architectural context, the 7 non-negotiable constitution gates, module layout (`synth/semantics/`, `versions.py`), phased roadmap (Phases 1–6 reordered to ship high-value plumbing first), resilient contextual code landmarks, migration strategy, and risk register.
4. **`tasks.md`** — Executable checklist. 33 concrete tasks (T100–T131, including T109b) across 6 phases with parallelization flags (`[P]`), contextual AST landmarks, and checkable "Done when" acceptance criteria.

---

## Priority at a Glance

| Phase | Focus | What it buys | Key Deliverables |
|---|---|---|---|
| **Phase 1** | Plumbing & Serialization Closure | Creates `versions.py` and immediately imports into `generator.py`, `backends/vegalite_backend.py`, and `merge_json.py` (T100); fixes tabular PRNG seeding in `synth/tabular.py`; adds twin-axes and colorbar guard in `generator.py` preventing metadata clobbering; safely defines `eff_sci` and `eff_dom` before `chart_info_map[ax]` construction (preventing Phase 1 `NameError`) and wires attributes in `create_unified_annotation`; updates `detailed_json` save whitelist and `merge_json.py` to preserve `semantic_domain`, `schema_version`, `dataset_version`, `subplots`, `is_composite`, `composite_chart_types`, `composite_domains`, and `obb`. Updates Vega-Lite after `extract_svg_annotations` and `metadata_json` dictionary creation with `null` fallback. Snapshots baseline vocabulary under a balanced 8-chart mix (sum = 100%). | `versions.py`, `synth/tabular.py`, `generator.py`, `merge_json.py`, `backends/vegalite_backend.py`, `tests/test_metadata_persistence.py`, `tests/test_p2_tabular_synth.py` |
| **Phase 2** | Title Catalog & Fallback Gating | Categorizes all 143 titles from `themes.py:CHART_TITLES` by `is_scientific` and `chart_type`, excluding heatmap titles from Cartesian fallbacks. Pre-indexes into an $O(1)$ lookup table with a $\ge 6$ title top-up. Resolves subplot domain in `generator.py` per subplot with configurable subdomain weights (`cfg.get('scientific_subdomain_weights')`), propagates weights into `theme_config` and `style_config`, migrates universal fallback, and handles domain overrides via `theme_config`. | `synth/semantics/__init__.py`, `synth/semantics/catalog.py`, `synth/semantics/sampler.py`, `generator.py`, `tests/test_title_sampling.py` |
| **Phase 3** | Axis & Comparative Catalog, Canonical Expansion | Annotates the 264 existing unique axis labels with `AxisRole`, `concept_group`, and `domain_tags`, pinning elapsed bench durations to `ScaleType.CONTINUOUS`. Augments `catalog.py` with ~40 canonical independent variables (`Date`, `Year`, `Depth (m)`, etc.) with clean format-free typography and explicit domain tags without touching `themes.py` (lowering line $X$ concentration to $\le 5\times$). Categorizes 94 comparative pairs into biomedical (0–39), engineering (40–54, 70–86), and business (55–69, 87–93). Aligns `DOMAIN_PRESETS` in `synth/tabular.py` via bidirectional alias map (`{"business": "financial", "engineering": "sensor_telemetry"}`). | `synth/semantics/catalog.py`, `synth/tabular.py`, `tests/test_semantic_catalog.py` |
| **Phase 4** | Universal Sampler Engine | Implements role- and scale-constrained sampling across all chart types (banning nominal categories on line/area $X$), dedicated `sample_secondary_y` with stem and `concept_group` exclusion via `METRIC_BY_LABEL`, normalized stem and concept group collision prevention, small-pool fallback, and sector-gated comparative pair sampling. | `synth/semantics/sampler.py`, `tests/test_semantic_sampling.py` |
| **Phase 5** | Call-Site Migration & Composites | Migrates the 17 axis sampling sites in `chart.py` across 7 chart blocks using contextual anchors (sampling standard bar immediately after `style_config['orientation'] = orientation`). Extracts `DUAL_AXIS_PROBABILITY` and `TREATMENT_KEY_PROBABILITY` with lazy boolean evaluation. Re-resolves dual-axis domain immediately at line 1322 before multivariate table generation. Extends `add_treatment_key_xaxis` to business grouped bars. Enriches `chart_info_map[ax]` with `style`, `pattern`, `series_count`, and `series_names` written from `chart.py` (avoiding `series_idx` KeyError). Emits `detailed_metadata["subplots"]`, `"is_composite": True`, root scalar `chart_type` reflecting primary subplot (`axes[0]`), and `"composite_chart_types"` when subplots diverge. Emits Schema Version `v2.1` and Dataset Version `2.1.0`. | `chart.py`, `generator.py`, `merge_json.py`, `backends/vegalite_backend.py`, `SCHEMA.md`, `tests/test_chart_semantic_migration.py` |
| **Phase 6** | Independent Oracle Verification | Verifies scale, role, and concept group rules using an independent continuous-regex oracle across all catalog labels paired with an embedded 50-item blind gold set annotating both `scale_type` and `role`, and an independent domain keyword dictionary for titles/pairs. Verifies 100% reachability of all 94 comparative pairs at the sampler level. Compares post-migration distribution against T106 baseline snapshot and benchmarks title distribution against theoretical uniform baseline ($N = 20,000$, $p_{max} \le 3.5\%$). Runs full test suite. | `tests/test_domain_coherence_e2e.py`, `tests/test_legacy_themes_compatibility.py` |

---

## Project Execution Status

| Phase | Focus | Status | Tests |
|---|---|---|---|
| **Phase 1** | Plumbing & Serialization | Completed | `tests/test_metadata_persistence.py`, `tests/test_p2_tabular_synth.py` |
| **Phase 2** | Title Catalog & Fallback Gating | Completed | `tests/test_title_sampling.py` |
| **Phase 3** | Axis & Comparative Catalog | Completed | `tests/test_semantic_catalog.py` |
| **Phase 4** | Universal Sampler Engine | Completed | `tests/test_semantic_sampling.py` |
| **Phase 5** | Call-Site Migration & Composites | Completed | `tests/test_chart_semantic_migration.py`, `tests/test_legacy_themes_compatibility.py` |
| **Phase 6** | Independent Verification | Completed | `tests/test_domain_coherence_e2e.py` (10/10 pass), Full Suite: 108/108 pass (100%) |


---

## Post-Migration Sunset & Decommissioning Roadmap

Following complete rollout and stabilization of Feature 002, the decommissioning of legacy constants and migration to declarative manifests is specified in:
* **[`../003-legacy-decommissioning-declarative-manifests/`](../003-legacy-decommissioning-declarative-manifests/)** (Schema Version `v3.0` / Dataset Version `3.0.0`).
