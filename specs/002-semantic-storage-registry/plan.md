# Implementation Plan: Semantic Storage & Domain Registry

**Feature Branch**: `002-semantic-storage-registry`  
**Schema Version**: `v2.1` (`dataset_version = "2.1.0"`)  
**Input**: `spec.md`, `research.md`  
**Status**: Complete (Phases 1 through 6 Verified — 108/108 Tests Pass, 100%)  

---

## Summary

This plan addresses the verified semantic incoherence vectors and serialization drop-points in the synthetic chart pipeline. The implementation is sequenced to ship high-value plumbing and serialization fixes first (including a shared `versions.py` and whitelisting `subplots`, `schema_version`, and `dataset_version` to disk and merge outputs), followed by pre-indexed title gating with minimum-pool top-ups, a verified domain routing architecture with configurable subdomain weights, role- and scale-constrained axis sampling augmented with ~40 canonical independent variables in `catalog.py` (with clean typography and explicit domain tags, preventing line $X$ concentration while freezing `themes.py`), semantic concept group (`concept_group`) collision prevention, dedicated secondary Y sampling via `METRIC_BY_LABEL`, activating all 94 comparative pairs via simplified sector gating (extending treatment keys to business grouped bars with reachability verified at sampler level), surgical call-site migration in `chart.py` using resilient contextual code landmarks, multi-subplot composite dashboard tracking (preserving root scalar taxonomies), and independent double-annotated oracle validation under forced chart mixes.

---

## Technical Context

* **Runtime**: Python 3.11+
* **Dependencies**: Standard library (`dataclasses`, `enum`, `typing`, `random`, `re`), `numpy`, `scipy`.
* **Zero External Dependencies**: No new PyPI packages.
* **Testing Framework**: `pytest`.

---

## Constitution Check: 7 Engineering Gates

| Gate | Rule | Rationale |
|---|---|---|
| **1. Fix True Leaks First** | Prioritize title fallback gating (`generator.py:4030` fires on nearly 100% of charts), scale-type mismatches (~76.4% continuous $X$ on bar; 27.3% sci and 70.1% biz scatter plots with $\ge 1$ categorical axis; nominal categories on line/area $X$), treatment key business reachability, and line $X$ lexical diversity. | Directs effort where defect rates are 27%–100%, rather than 0% (cross-pool sci/biz axis mixing is already 0.0% in default code). |
| **2. Role-Based and Scale-Constrained Gating with Canonical Expansion** | Annotate the 264 existing unique axis labels with `AxisRole` (`INDEPENDENT`, `DEPENDENT`, `BIDIRECTIONAL`) and keep measured bench quantities as `ScaleType.CONTINUOUS`. Drop `ORDINAL`. Gate line/area $X$ by both role (`INDEPENDENT`, `BIDIRECTIONAL`) and scale (`CONTINUOUS`, `TEMPORAL`, `DISCRETE_COUNT`). Augment `catalog.py` with ~40 canonical independent variables (`Date`, `Year`, `Depth (m)`, etc.) without parenthetical format hints and with domain tags, without touching `themes.py` (FR-050). Add `concept_group` for $O(1)$ semantic synonym checking. | Keeps bench variables in scatter plots while preventing dependent metrics (`Elastic Modulus`) and nominal categories (`Salesperson`) from appearing on line $X$, lowering line $X$ concentration from $\le 15\times$ to $\le 5\times$ baseline. |
| **3. Domain Routing with Sector Mix Integrity & 100% Pair Reachability** | Respect `cfg['bar_chart_config']['scientific_ratio']` (default 0.60) to determine `is_scientific` first. Subdivide scientific into `biomedical` and `engineering` with weights exposed via `cfg.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})`, and business into `business`. Sampler returns `(x, y, domain_category)`. Extend `add_treatment_key_xaxis` to business grouped bars, achieving 100% reachability across all 94 pairs verified at the sampler level. | Preserves configured sector proportions, provides user control over subdomain ratios without editing code, and connects treatment keys to the chart's physical domain across all 94 pairs. |
| **4. End-to-End Serialization Closure & Composite Tracking** | Every output stage must preserve metadata: `detailed_json` save whitelist (including `is_composite`, `composite_chart_types`, `composite_domains`), twin-axes guard in `generator.py`, `merge_json.py`, `backends/vegalite_backend.py`, and multi-subplot `subplots` array. Top-level scalar `chart_type` and `semantic_domain` strictly preserve core taxonomy values reflecting the primary subplot (`axes[0]`), while composite figures emit `"is_composite": True`, `"composite_chart_types"`, and `"composite_domains"`. Establish a shared `versions.py` and import in Phase 1 to prevent `NameError`. | Closes the drop-points where `semantic_domain`, `subplots`, composite fields, and version fields were never saved to disk, while preventing breakage of downstream visual classifiers. |
| **5. Deterministic PRNG Closure** | `synth/tabular.py` must derive its default RNG at top-level public entry points from the task's seeded `np.random` state (e.g. `rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))` if `rng is None`) and pass that instance down, avoiding repeated independent re-seeding within internal helper functions. | Guarantees dataset determinism run-to-run under a fixed seed, while ensuring successive calls within a run produce distinct series without subplot collisions. |
| **6. Zero Import Regressions & Resilient Contextual Anchoring** | Raw lists and tuples in `themes.py` must remain completely intact. No dynamic PEP 562 proxies or Sequence wrappers. Code edits must target contextual AST landmarks (e.g. `for ax_idx, ax in enumerate(axes):` at line 3917) rather than static line numbers. | Preserves 100% backward compatibility for all existing imports and prevents line drift failures during multi-phase implementation. |
| **7. Independent Double-Annotated Test Oracles** | Verification tests must assert physical properties using an independent continuous-regex detector paired with an embedded 50-item blind gold set for ambiguous semantic boundary labels, and an independent domain keyword dictionary for titles and pairs. Test fixtures force all 8 chart types using explicit `force_dual_axis` and `force_treatment_key` config flags. | Prevents self-fulfilling unit tests that consult the catalog, and catches role distinctions (`Dose` vs `Elastic Modulus`) that regexes cannot see. |

---

## Target Project Layout

```
project_root/
├── versions.py                  # Shared ANNOTATION_SCHEMA_VERSION = "v2.1", DATASET_VERSION = "2.1.0" (project root on sys.path, supports bare "from versions import ...")
├── synth/
│   ├── semantics/
│   │   ├── __init__.py          # Exports sample_axis_pair, sample_secondary_y, sample_chart_title, sample_comparative_pair
│   │   ├── catalog.py           # Annotated AxisMetric (with concept_group, canonical expansion), METRIC_BY_LABEL, ChartTitle, ComparativePair, pre-indexed pools
│   │   └── sampler.py           # Role/scale-constrained, concept_stem & concept_group collision-preventing, chart-aware sampling logic
│   ├── tabular.py               # Closed PRNG entropy leak; DOMAIN_PRESETS aligned via alias mapping
│   └── __init__.py
├── themes.py                    # UNTOUCHED legacy constants (guarantees zero import breakage)
├── chart.py                     # Direct migration of 17 axis sampling sites using contextual anchors, DUAL_AXIS_PROBABILITY, TREATMENT_KEY_PROBABILITY, series info write-back
├── generator.py                 # Subplot domain resolution (configurable weights); title fallback migration; save whitelist (subplots, versions, composite); twin-axes guard
├── merge_json.py                # Preserves semantic_domain, is_scientific, subplots, composite fields, schema_version, dataset_version, obb
├── backends/
│   └── vegalite_backend.py      # syn_table initialized to None; detailed_metadata and metadata.json updated contextually
├── SCHEMA.md                    # Updated v2.1 documentation of semantic metadata, subplots, and composite attributes
└── tests/
    ├── test_metadata_persistence.py    # Phase 1: Validates detailed_metadata in generator, vegalite, merge_json, subplots, composite fields, and versions
    ├── test_p2_tabular_synth.py        # Phase 1: Validates tabular PRNG seeding determinism
    ├── test_title_sampling.py          # Phase 2: Validates pre-indexed title pools, min-6 top-ups, and domain preference
    ├── test_semantic_catalog.py        # Phase 3: Validates scale types, roles, stems, concept_groups, canonical expansion, immutability
    ├── test_semantic_sampling.py       # Phase 4: Validates scale/role rules, secondary Y stem & group exclusion, 100% pair reachability, and determinism
    ├── test_chart_semantic_migration.py # Phase 5: Validates all chart.py call sites with effective domain write-back
    ├── test_legacy_themes_compatibility.py # Phase 6: Validates 100% backward compatibility of themes.py constants
    └── test_domain_coherence_e2e.py    # Phase 6: Independent double-annotated oracle & forced chart fixtures auditing domain coherence
```

---

## Phased Roadmap

### Phase 1: High-Value Plumbing & Serialization Closure
* **T100: Shared Version Constants Creation & Phase 1 Imports** (`versions.py`):
  Create `versions.py` defining `ANNOTATION_SCHEMA_VERSION = "v2.1"` and `DATASET_VERSION = "2.1.0"`. Immediately import these constants into `generator.py`, `backends/vegalite_backend.py`, and `merge_json.py` to prevent Phase 1 `NameError`.
* **T101: Tabular PRNG Seeding Closure** (`synth/tabular.py`):
  Replace the 5 unseeded `np.random.default_rng()` fallback lines with `rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))` when `rng is None` at top-level entry points.
* **T102: Twin-Axes & Heatmap Colorbar Clobber Guard** (`generator.py`):
  In `create_unified_annotation` inside the per-axis metadata extraction loop (`for ax in fig.axes:` landmark line 3305), add `if ax not in chart_info_map: continue` to prevent secondary twin axes (`ax2`) and heatmap colorbars (`chart.py:3270` landmark) from overwriting scalar metadata with `chart_type="unknown"`, `is_scientific=False`, and resetting `dual_axis_info={}`.
* **T103: Detailed JSON Save Whitelist, Phase 1 Isolation & Attribute Wiring** (`generator.py`):
  * Immediately before `chart_info_map[ax]` construction (landmark line 4051), define `eff_sci = theme_config.get('effective_is_scientific', is_scientific)` and `eff_dom = theme_config.get('semantic_domain')`. In Phase 1, `eff_sci` falls back to `is_scientific` and `eff_dom` is `None`, preventing a `NameError` when Phase 1 executes in isolation.
  * In the per-artist dispatch loop (landmark `chart_info_map[ax] = { ... }`), store all 7 attributes: `is_scientific=eff_sci`, `semantic_domain=eff_dom`, `style`, `pattern`, `series_count`, `series_names`, `stacking_mode`.
  * In `create_unified_annotation` (landmark `detailed_metadata = { ... }`), add `detailed_metadata["semantic_domain"] = chart_info.get("semantic_domain")` (while existing reads for `series_count`, `series_names`, `stacking_mode`, `style`, `pattern`, and `is_scientific` now receive populated values from `chart_info`).
  * In `detailed_json` dictionary instantiation (landmark `detailed_json = { ... }`), whitelist `"semantic_domain"`, `"schema_version"`, `"dataset_version"`, `"subplots"`, `"is_composite"`, `"composite_chart_types"`, and `"composite_domains"` (note that `series_count` and `series_names` are already present in `detailed_json`).
* **T104: Merge JSON Preservation & OBB Retention** (`merge_json.py`):
  Preserve `"semantic_domain"`, `"schema_version"`, `"dataset_version"`, `"subplots"`, and composite fields at the top level of `unified` and under `chart_generation_metadata["visual_style"]`, copying directly without fallback default version strings. In `normalize_raw_annotation`, preserve `entry.get("obb")`. Default `is_composite` to `False` on legacy files.
* **T105: Vega-Lite Metadata Parity & Contextual Landmarks** (`backends/vegalite_backend.py`):
  Initialize `syn_table = None` at function start. Directly after the `extract_svg_annotations` landmark, populate `"semantic_domain"` and `"is_scientific"` on `detailed_metadata` (normalizing keys or emitting `null` and `False`). Immediately after the `metadata_json` dictionary instantiation landmark, populate `"schema_version"` and `"dataset_version"` on `metadata_json`.
* **T106: Baseline Snapshot Under Forced 8-Chart Mix**:
  Ensure directory `tests/golden/` is created (`mkdir -p tests/golden`). Snapshot baseline label and title frequency distributions across 1,000 generated charts under a balanced 8-chart mix (`{bar: 15, line: 15, scatter: 15, box: 15, area: 15, histogram: 10, pie: 10, heatmap: 5}`) to `tests/golden/baseline_semantic_distribution.json` by running a generation loop with temporary config overrides before any vocabulary changes.

### Phase 2: Pre-Indexed Title Catalog & Fallback Gating
* **T107: Package Initialization & Title Catalog Definition** (`synth/semantics/__init__.py`, `synth/semantics/catalog.py`):
  * Create `synth/semantics/` package directory and `synth/semantics/__init__.py` exporting `sample_axis_pair`, `sample_secondary_y`, `sample_chart_title`, and `sample_comparative_pair`.
  * Categorize all 143 titles from `themes.py:CHART_TITLES` with `is_scientific: Optional[bool]`, `domain_tags: Tuple[str, ...]`, and `allowed_chart_types: Tuple[str, ...]`.
  * Exclude heatmap-specific titles (~5.6% of titles) from Cartesian chart pools, as heatmaps set their own titles in `chart.py:3362` landmark.
  * Pre-index into `TITLE_INDEX[(is_scientific, chart_type)]`.
  * If any `(is_scientific, chart_type)` pool contains fewer than 6 titles, top up the pool with domain-generic titles matching `is_scientific`.
* **T108: Title Sampler with Domain Guidance** (`synth/semantics/sampler.py`):
  Implement `sample_chart_title(chart_type, is_scientific, domain=None, rng=None)` (defaulting `rng=random`). Preferentially samples domain-matching titles when available with verified $O(1)$ complexity.
* **T109: Subplot Domain Resolution with Configurable Weights & Fallback Migration** (`generator.py`):
  * Inside `generate_chart_with_annotations` in the per-subplot loop (landmark `for ax_idx, ax in enumerate(axes):` at line 3917, NOT `for ax in fig.axes:`): resolve `is_scientific = random.random() < cfg['bar_chart_config']['scientific_ratio']`.
  * Inject `theme_config['scientific_subdomain_weights'] = cfg.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})` and `style_config['scientific_subdomain_weights'] = theme_config['scientific_subdomain_weights']`.
  * Then resolve `semantic_domain`: if `is_scientific`, draw from `("biomedical", "engineering")` with weights `theme_config['scientific_subdomain_weights']`; else set `"business"`. Store both in `theme_config` and `style_config`.
  * In `chart.py` (landmarks 1322, 3197): write back `effective_is_scientific` and `semantic_domain` to `theme_config` (guarded by `if isinstance(theme_config, dict):`). Heatmap maps context configurations to `"biomedical"`, `"business"`, or scientific subdomains.
  * At landmark `ax.set_title(random.choice(CHART_TITLES))`: read effective domain and call `sample_chart_title(chart_type, eff_sci, domain=eff_dom)`.
  * In `chart_info_map[ax]` construction (line 4051): record `is_scientific` and `semantic_domain`.
* **T109b: Title Sampling Unit Tests** (`tests/test_title_sampling.py`):
  Validate pre-indexed title pools, minimum-6 top-ups, domain preference, excluding heatmap titles from Cartesian pools, and $O(1)$ lookup performance.

### Phase 3: Axis & Comparative Catalog Annotation, Canonical Expansion & Concept Groups (COMPLETED)
* **T110: Standard Enums & Dataclasses** (`synth/semantics/catalog.py`):
  Define `ScaleType(str, Enum)` (without `ORDINAL`), `AxisRole(str, Enum)`, and frozen slotted `AxisMetric` (with `concept_group: Optional[str] = None`), `ChartTitle`, and `ComparativePair`. Expose `METRIC_BY_LABEL: Dict[str, AxisMetric]` deduplicating the 5 duplicate scientific labels (`Concentration (μM)`, `Temperature (°C)`, `pH`, `Wavelength (nm)`, `Frequency (Hz)`) to map 178 unique scientific + 86 unique business = 264 unique entries.
* **T111: Hand-Reviewed Classification & Canonical Independent Variable Expansion** (`synth/semantics/catalog.py`):
  * 178 scientific labels: hand-reviewed into continuous dependent metrics (`role=AxisRole.DEPENDENT`) and independent/bidirectional metrics (`role in (INDEPENDENT, BIDIRECTIONAL)` explicitly including `Time (s)`, `Time (min)`, `Time (h)`, `Dose`, `Time Point`, `Age`, `Temperature`, `pH`, `Wavelength`, `Frequency`). Elapsed bench durations (`Time (s)`, `Exposure Time`) are pinned to `ScaleType.CONTINUOUS`.
  * 86 business labels: hand-reviewed into continuous dependent metrics (`role=AxisRole.DEPENDENT`) and independent/bidirectional dimensions (`role in (INDEPENDENT, BIDIRECTIONAL)`). Tag with `domain_tags`.
  * **Canonical Independent Expansion & Clean Typography**: Augment `catalog.py` with ~40 canonical independent variables with clean typography (no parenthetical date format hints: `Date`, `Month`, `Year`, `Quarter`, `Timestamp`, `Cycle Number`, `Epoch`, `Iteration`, `Step`, `Depth (m)`, `Altitude (m)`, `Distance (km)`, `Sample ID`) typed as `ScaleType.CONTINUOUS` or `TEMPORAL` and `AxisRole.INDEPENDENT` or `BIDIRECTIONAL`. Multi-sector metrics include both tags (`domain_tags = ("biomedical", "engineering", "business")`). Computing metrics (`Epoch`, `Iteration`, `Step`) are tagged strictly `("engineering",)`. Preserve `themes.py` strictly frozen (FR-050).
  * Compute `concept_stem` for each metric using `compute_concept_stem(label: str) -> str` (lowercase, strip parenthetical content `r'\(.*?\)'`, strip currency symbols `[$€£¥]`, strip punctuation, collapse whitespace) and assign semantic `concept_group` (e.g. `revenue_metric`, `optical_density`, `duration_metric`).
* **T112: Categorize Comparative Pairs & Bidirectional Preset Normalization** (`synth/semantics/catalog.py`, `synth/tabular.py`):
  * Tag all 94 pairs from `themes.py:COMPARATIVE_LABELS`:
    - Pairs 0–39: `domain_category = "biomedical"`, `is_scientific = True`.
    - Pairs 40–54, 70–86: `domain_category = "engineering"` (computing, systems, physical and materials engineering).
    - Pairs 55–69 and 87–93: `domain_category = "business"`, `is_scientific = False` (activated for business grouped bars via T122).
  * In `synth/tabular.py`, normalize domain keys via bidirectional alias mappings:
    - `DOMAIN_ALIAS_TO_PRESET = {"business": "financial", "engineering": "sensor_telemetry"}` (maps incoming canonical domains to preset keys for lookup).
    - `PRESET_TO_CANONICAL = {"financial": "business", "sensor_telemetry": "engineering", "demographic": "business"}` (maps presets to canonical domains for emitted table schemas).

### Phase 4: Universal Sampler Engine & Scale Rules (COMPLETED)
* **T113: Implement `sample_axis_pair` with Role, Quantitative Scale & Concept Group Gating** (`synth/semantics/sampler.py`):
  * Note: `pie` and `heatmap` chart types are explicitly excluded from `sample_axis_pair` (pie charts use radial wedges without Cartesian axes; heatmaps manage axes via `CONTEXT_CONFIGURATIONS`).
  * `scatter`: enforces quantitative metrics on $X$ and $Y$ (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT, TEMPORAL)`), banning categorical nominal dimensions.
  * `bar` / `box` (vertical): enforces `role in (INDEPENDENT, BIDIRECTIONAL)` for $X$ and quantitative `role in (DEPENDENT, BIDIRECTIONAL)` (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT)`) for $Y$.
  * `bar` / `box` (horizontal): inverts roles so quantitative maps to horizontal $X$ and independent maps to vertical $Y$.
  * `line` / `area`: enforces `role in (INDEPENDENT, BIDIRECTIONAL)` **AND** `scale_type in (CONTINUOUS, TEMPORAL, DISCRETE_COUNT)` for $X$ (banning pure dependent metrics and nominal categories); quantitative `role in (DEPENDENT, BIDIRECTIONAL)` for $Y$.
  * `histogram`: quantitative $X$; $Y$ from `HISTOGRAM_Y_LABELS`.
  * Collision checks: reject candidate if `x.concept_stem == y.concept_stem` OR (`x.concept_group is not None and x.concept_group == y.concept_group`).
  * Small-pool fallback: if filtered candidates pool is empty, fall back gracefully to general independent/continuous catalog metrics.
  * Returns `(x_label, y_label, domain_category)`.
* **T114: Implement Dedicated Secondary Y Sampler** (`synth/semantics/sampler.py`):
  Implement `sample_secondary_y(y1_label, is_scientific, domain_category=None, rng=None) -> str`. Looks up `y1_label` in `METRIC_BY_LABEL`, extracts its `concept_stem` and `concept_group`, filters out candidate metrics sharing that stem OR concept group, and draws a distinct continuous metric from `domain_category`.
* **T115: Implement `sample_comparative_pair` with Sector Gating** (`synth/semantics/sampler.py`):
  Implement `sample_comparative_pair(domain_category=None, rng=None)`. Filters pairs matching `domain_category`: `"biomedical"` draws from pairs 0–39; `"engineering"` draws from pairs 40–54 and 70–86; `"business"` draws from pairs 55–69 and 87–93. Reachability across all 94 pairs is verified at the sampler unit level.

### Phase 5: Surgical Call-Site Migration & Composite Tracking
* **T116: Migrate Dual Y-Axis Bar Chart with Line 1322 Re-Resolution** (`chart.py`):
  * In the dual-axis block **immediately at line 1322** (directly after `is_scientific = True`, BEFORE the `sample_multivariate_table` call at line 1332): if `isinstance(theme_config, dict) and theme_config.get('semantic_domain') == "business"`, re-resolve domain to `"biomedical"` or `"engineering"` using `theme_config.get('scientific_subdomain_weights')` and write back `theme_config['semantic_domain'] = domain_cat` and `theme_config['effective_is_scientific'] = True`.
  * At landmark lines 1406–1408, sample $(X, Y_1)$ via `sample_axis_pair("bar", "vertical", is_scientific=True, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None)` and $Y_2$ via `sample_secondary_y(y1_label, is_scientific=True, domain_category=domain_category)`.
* **T117: Migrate Standard & Grouped Bar Charts with Resilient Contextual Anchor** (`chart.py`):
  In the standard bar path, sample axes **immediately after `style_config['orientation'] = orientation`** (landmark line 1447), before the treatment key check (landmark line 1508):
  `x_label, y_label, domain_category = sample_axis_pair("bar", orientation, is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None)`.
  At landmark `add_treatment_key_xaxis`: pass `domain_category=domain_category`.
  At landmark `ax.set_ylabel`: apply `ax.set_ylabel(y_label); if not has_treatment_axis: ax.set_xlabel(x_label)`.
  Write back `theme_config['semantic_domain'] = domain_category` if `isinstance(theme_config, dict)`.
* **T118: Migrate Line and Area Charts** (`chart.py`):
  At landmarks for line and area axis assignment, replace with `sample_axis_pair("line", "vertical", is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None)` and `sample_axis_pair("area", "vertical", ...)` drawing from the expanded independent pool. Write back `domain_category`.
* **T119: Migrate Scatter Plots** (`chart.py`):
  At landmark for scatter axis assignment, replace with `sample_axis_pair("scatter", "vertical", is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None)`. Write back `domain_category`.
* **T120: Migrate Box Plots** (`chart.py`):
  At landmark for box axis assignment, replace with `sample_axis_pair("box", orientation, is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None)`. Write back `domain_category`.
* **T121: Migrate Histograms** (`chart.py`):
  At landmark for histogram axis assignment, sample $X$ via `sample_axis_pair("histogram", "vertical", is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None)[0]`. Write back `domain_category`.
* **T122: Update `add_treatment_key_xaxis`, Named Constants, Series Info & Business Path Activation** (`chart.py`):
  * Extract module constants `DUAL_AXIS_PROBABILITY = 0.15` and `TREATMENT_KEY_PROBABILITY = 0.30`.
  * Check flags lazily: `style_config['force_dual_axis'] if 'force_dual_axis' in style_config else (random.random() < DUAL_AXIS_PROBABILITY)`.
  * Update `add_treatment_key_xaxis` to accept `domain_category`, calling `sample_comparative_pair(domain_category=domain_category)`.
  * Extend `add_treatment_key_xaxis` to the business grouped bar branch (within the `else: # Standard Styles` branch, specifically under `style == 'side_by_side'` after `ticks_setter(indices, categories)` at landmark line 1594) when `len(bar_info_list) == 4 and orientation == 'vertical' and random.random() < TREATMENT_KEY_PROBABILITY`, activating the 22 business pairs (55–69, 87–93) across the generator.
  * In the 6 `sample_multivariate_table` calls, pass `domain = syn_domain or theme_config.get('semantic_domain') or ('biomedical' if is_scientific else 'business')`. When `use_synthetic and syn_table is not None`, write back `theme_config['semantic_domain'] = PRESET_TO_CANONICAL.get(syn_table.get('domain'), syn_table.get('domain'))`.
  * In `chart.py`, bar plotting branches write back `theme_config['series_count'] = bars_per_group` (or `num_series`, `1`) and `theme_config['series_names'] = series_names`.
* **T123: Enrich `chart_info_map[ax]` and Additive Composite Tracking** (`generator.py`):
  * In `chart_info_map[ax]` construction (landmark line 4051): read `series_count = theme_config.get('series_count', 1)` and `series_names = theme_config.get('series_names', [])`. (Do NOT inspect `b['series_idx']` in `bar_info_list`). Record `style`, `pattern`, `series_count`, `series_names`, `stacking_mode`, `is_scientific`, and `semantic_domain`.
  * In the per-axis loop: build `subplots = []` capturing per-axis metadata for each visible axis in `fig.axes` with `ax in chart_info_map`. Store `detailed_metadata["subplots"] = subplots`.
  * **Additive Composite Subplot Tracking (No Enum Widening of Root Scalars)**:
    - Set `detailed_metadata["is_composite"] = True` if `len(subplots) > 1` else `False`.
    - Top-level scalar `chart_type` and `semantic_domain` ALWAYS reflect the primary subplot (`fig.axes[0]` / `axes[0]`), e.g., `"bar"`, `"line"`. Never `"composite"` or `"mixed"`.
    - If subplot chart types differ across subplots, emit additive list `"composite_chart_types": [s["chart_type"] for s in subplots]`.
    - If subplot domains differ across subplots, emit additive list `"composite_domains": [s["semantic_domain"] for s in subplots]`.
    - In `generator.py:4751`, set `metadata_json["chart_types"] = detailed_json.get("composite_chart_types", [unified_json.get("chart_type", "unknown")])`.
* **T124: Update `SCHEMA.md` and Version Usages**:
  Update `SCHEMA.md` documenting Schema Version `v2.1` (`dataset_version = "2.1.0"`), `semantic_domain`, `subplots`, `is_composite`, `composite_chart_types`, and `composite_domains`.

### Phase 6: Independent Oracle Verification & Benchmarking
* **T125: Unit Tests for Semantic Catalog** (`tests/test_semantic_catalog.py`):
  Validate that all 264 metrics, ~40 canonical independent variables without format hints and with domain tags, and 94 comparative pairs are loaded into frozen slotted dataclasses. Validate role annotations (including `Time (s/min/h)`), stems, concept groups, immutability, and title index pool top-ups ($\ge 6$). Verify deterministic sorting across `PYTHONHASHSEED`.
* **T126: Unit Tests for Sampling Logic & 100% Pair Reachability** (`tests/test_semantic_sampling.py`):
  Validate role and scale rules across all chart types, `sample_secondary_y` stem and group exclusion, normalized and concept group collision avoidance across 10,000 draws, small-pool fallback, and determinism. Validate 100% reachability of all 94 comparative pairs at the sampler unit level across 10,000 draws.
* **T127: Integration Tests for Metadata Persistence** (`tests/test_metadata_persistence.py`):
  Validate that `generator.py`, `merge_json.py`, and `backends/vegalite_backend.py` serialize `semantic_domain`, `is_scientific`, `schema_version = "v2.1"`, `dataset_version = "2.1.0"`, `subplots`, and composite attributes. Validate twin-axes and colorbar clobber guards.
* **T128: Regression Tests for Chart Migration & Legacy Imports** (`tests/test_chart_semantic_migration.py`, `tests/test_legacy_themes_compatibility.py`):
  Validate all 17 call sites in `chart.py` and assert 100% backward compatibility of raw `themes.py` constants.
* **T129: Independent Double-Annotated Oracle Test Suite** (`tests/test_domain_coherence_e2e.py`):
  Implement an augmented continuous-regex oracle across all catalog labels paired with an embedded 50-item blind gold set annotating both `scale_type` and `role` for boundary cases, alongside an independent domain keyword dictionary (`biomedical`, `engineering`, `business`) to audit titles and pairs without consulting `catalog.py`. Test under forced multi-chart mixes across all 8 chart types (using `force_dual_axis` and `force_treatment_key`). Assert zero dependent metrics and zero nominal categories on line/area $X$. Assert zero cross-domain leakage when treatment keys render. Assert zero `concept_group` collisions.
* **T130: Baseline Drift Comparison & In-Memory Theoretical Title Benchmark**:
  * Load `tests/golden/baseline_semantic_distribution.json` (from T106) and assert that post-migration distributions preserve vocabulary coverage $\ge 95\%$ while eliminating title mismatches.
  * Simulate $N = 20,000$ in-memory draws from `sample_chart_title`. Compare against theoretical uniform baseline ($1/143$). Assert maximum title frequency in any cell $\le 5\times$ baseline ($p \le 3.5\%$) under balanced multi-chart mix.
  * Measure axis-label frequency under production mix: maximum line $X$ frequency $\le 5\times (1 / K_{domain})$ baseline.
* **T131: Full Regression Test Suite Execution**:
  Run `pytest tests/` across all P0, P1, and P2 test suites.

---

## Risk Register & Mitigations

| Risk | Impact | Likelihood | Mitigation Strategy |
|---|---|---|---|
| **Default Line-Only Test Blindness** | Tests miss regressions because `config_defaults.py:123` sets 100% line charts. | High | T106 and T129 explicitly force and test balanced multi-chart mixes across all 8 chart types. |
| **Line/Area $X$ Lexical Concentration** | Restricting line $X$ to ~13–20 independent labels in `themes.py` causes up to $15\times$ baseline concentration. | High | T111 expands `catalog.py` with ~40 canonical independent metrics (`Date`, `Year`, `Depth (m)`, etc.) with clean typography and explicit domain tags, lowering maximum frequency to $\le 5\times$ baseline without touching `themes.py`. |
| **Line-Drift Implementation Fragility** | Static line numbers break as multi-phase edits land across `chart.py` and `generator.py`. | High | All tasks use resilient contextual code landmarks (AST landmarks, dict names, loop headers) rather than static line numbers. |
| **Twin Axes & Colorbar Clobbering** | `ax2` and colorbar run last in `for ax in fig.axes:` and reset metadata to `unknown` and `False`. | High (verified) | T102 introduces explicit `if ax not in chart_info_map: continue` guard. |
| **Merged JSON Data Loss & OBB Discard** | `merge_json.py` deletes `_detailed.json`, drops non-whitelisted keys, and discards OBB. | High (verified) | T104 whitelists `semantic_domain`, `schema_version`, `dataset_version`, `subplots`, composite fields, and preserves `obb`. |
| **Multi-Subplot Composite Overwriting & Ambiguity** | Multiple subplots overwrite top-level metadata in last-axis-wins fashion; heterogeneous subplots misrepresent chart type. | High (verified) | T123 introduces `detailed_metadata["subplots"]` array, sets `"is_composite": True`, preserves root scalar taxonomies reflecting `axes[0]`, and emits `"composite_chart_types"`. |
| **Bar Series Count KeyError (`series_idx`)** | 6 of 14 `bar_info_list.append` sites lack `series_idx`, raising `KeyError` at runtime. | High (verified) | T122/T123 writes `series_count` and `series_names` from `chart.py` directly into `theme_config`, eliminating `b['series_idx']` parsing. |
| **Semantic Synonym Collisions** | Stems strip units but miss true synonyms (`Revenue` vs `Sales`, `OD` vs `Absorbance`). | Medium | T110/T113/T114 add `concept_group` to `AxisMetric` and reject same-group pairings with $O(1)$ equality checks. |
| **Tabular Domain Alias Inversion** | Incoming canonical domains (`business`, `engineering`) miss preset lookup if aliases map from legacy to canonical. | High | T112 implements bidirectional mapping: `DOMAIN_ALIAS_TO_PRESET = {"business": "financial", "engineering": "sensor_telemetry"}` and `PRESET_TO_CANONICAL = {"financial": "business", ...}`. |
| **Phase 1 Isolation `NameError: eff_sci`** | Running Phase 1 without Phase 2 causes `NameError` if `eff_sci` is not yet defined in `generator.py`. | High | T103 explicitly defines `eff_sci = theme_config.get('effective_is_scientific', is_scientific)` immediately prior to line 4051. |
| **Subdomain Weight Scope Loss in `chart.py`** | Dual-axis re-resolution in `chart.py` lacks access to `cfg` to read custom weights. | Medium | T109/T116 propagate `scientific_subdomain_weights` into `theme_config` and `style_config`. |

---

## Post-Migration Sunset & Decommissioning Roadmap (Feature 003 / Schema v3.0)

Feature 002 deliberately implements a **Strangler Fig** transition: legacy semantic constants in `themes.py` remain frozen and uncalled, and tabular domain translations use bidirectional aliases.

The final decommissioning phase is formally specified in **Feature 003**:
* **Spec Directory**: [`../003-legacy-decommissioning-declarative-manifests/`](../003-legacy-decommissioning-declarative-manifests/)
* **Implementation Plan**: [`../003-legacy-decommissioning-declarative-manifests/plan.md`](../003-legacy-decommissioning-declarative-manifests/plan.md)
* **Tasks Breakdown**: [`../003-legacy-decommissioning-declarative-manifests/tasks.md`](../003-legacy-decommissioning-declarative-manifests/tasks.md)
* **Sunset Gates**: SG-001 ($\ge 50,000$ v2.1 charts), SG-002 (AST audit zero legacy call sites), SG-003 (downstream consumer sign-off).
* **Decommissioning Actions**:
  1. Purging 758 lines of procedural constants from `themes.py` ($\le 380$ lines remaining).
  2. Transitioning to declarative YAML domain manifests (`synth/semantics/domains/*.yaml`) with Pydantic v2 validation.
  3. Startup loader compilation into frozen slotted dataclasses ($\le 50\,\text{ms}$, verified $O(1)$ runtime access).
  4. Direct tabular domain naming in `synth/tabular.py` (deleting aliases).
  5. Retiring legacy compatibility test shims and emitting Schema Version `v3.0` (`3.0.0`).
