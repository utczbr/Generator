# Feature Specification: Semantic Storage & Domain Registry

**Feature Branch**: `002-semantic-storage-registry`  
**Schema Version**: `v2.1` (`dataset_version = "2.1.0"`)  
**Status**: Complete (Implemented & Verified — 108/108 Tests Pass)  
**Created**: 2026-09-19 (Revised 2026-09-20 following fourth-pass critical review & adopted enhancements)  
**Input**: Empirical codebase audit documented in `research.md`  
**Primary Artifacts Touched**: `versions.py` (new), `synth/semantics/` (new module), `synth/tabular.py`, `chart.py`, `generator.py`, `merge_json.py`, `backends/vegalite_backend.py`, `SCHEMA.md`

---

## Overview

The synthetic chart generation pipeline produces ground-truth images and annotations for visual document understanding, OCR, and multimodal models. While `is_scientific` already enforces zero cross-bleed between scientific and business axis pools in `chart.py` (0.0% cross-pool drawing), deep codebase tracing and empirical audit reveal critical semantic incoherence vectors and serialization drop-points:

1. **Universal Fallback Title Blindness**: `chart.py` only sets titles for heatmaps (`chart.py:3362` landmark) and the synthetic engine (`chart.py:1522` landmark). Therefore, `generator.py:4030` (`ax.set_title(random.choice(CHART_TITLES))`) fires on **nearly 100% of standard charts** (line, bar, scatter, box, pie, histogram, area), selecting uniformly across 143 titles regardless of domain or chart type. While the fallback fires on ~100% of standard charts, the scientific cross-domain mismatch defect rate is ~36.1% (0.60 * 86/143), while the total cross-domain mismatch defect rate across all charts is ~52.0%.
2. **Scale-Type Incoherence Across Chart Types**: In 74.3% of scientific bar chart draws (76.4% of unique labels) and ~54.7% of business bar charts, continuous dependent metrics are assigned to the independent $X$-axis. Furthermore, value axes ($Y$) across bar, line, and area charts suffer unconstrained categorical contamination (14.8% scientific and 45.3% business per draw). In **27.3% of scientific scatter plots and 70.1% of business scatter plots**, at least one categorical dimension (e.g. `Salesperson`, `Cell Line`) is assigned to a continuous axis. In line and area charts, unconstrained continuous sampling allows pure dependent metrics (e.g. `Elastic Modulus (GPa)`) or nominal categories (e.g. `Salesperson`, appearing in 12.6% sci and 34.9% biz) to be assigned to the horizontal axis.
3. **Lexical Diversity Bottleneck & Canonical Expansion**: Restricting line/area $X$ strictly to independent continuous/temporal labels from `themes.py` leaves only ~20 scientific and ~13 business labels. Under the 100% line default configuration (`config_defaults.py:123`), per-label frequency would spike up to $15\times$ baseline. To resolve this without mutating `themes.py` (FR-050), `synth/semantics/catalog.py` is augmented with ~40 canonical independent variables (`Date`, `Month`, `Year`, `Quarter`, `Timestamp`, `Cycle Number`, `Epoch`, `Iteration`, `Step`, `Depth (m)`, `Altitude (m)`, `Distance (km)`, `Sample ID`) with clean typography (no parenthetical date format hints) and explicit domain tags, expanding the eligible pool to 60+ labels and lowering maximum per-label frequency to $\le 5\times$ baseline.
4. **Serialization Drops (`_detailed.json` and `merge_json.py`)**: In `generator.py:4673–4738`, `detailed_json` is built from an explicit dictionary that omits `semantic_domain`, `schema_version`, `dataset_version`, and `subplots`. Furthermore, `merge_json.py` deletes `_detailed.json`, drops keys not explicitly copied to `unified`, and discards oriented bounding box annotations (`entry.get("obb")`).
5. **Twin-Axes & Heatmap Colorbar Metadata Clobbering**: In `generator.py:3305`, `for ax in fig.axes:` iterates over secondary twin axes (`ax2`) and heatmap colorbars (`chart.py:3270`), which are not in `chart_info_map`. Missing a guard, they run last and overwrite `chart_type="unknown"`, `is_scientific=False`, and reset `dual_axis_info={}`.
6. **Local Domain Overrides Contradicting Generator State**: Dual-axis bar charts force `is_scientific = True` locally (`chart.py:1322`), while heatmaps override domain via `context_domain` (`chart.py:3197–3201`). Without an in/out write-back, `generator.py` records mismatched metadata.
7. **PRNG Entropy Leaks in `synth/tabular.py`**: Five unseeded `np.random.default_rng()` fallback lines in `synth/tabular.py` break dataset reproducibility under fixed seeds.
8. **Multi-Subplot Composite Tracking & Additive Schema Parity**: In multi-subplot figures, `for ax in fig.axes:` overwrites top-level scalar metadata on each subplot without emitting a `subplots` record. To prevent breaking downstream visual classifiers that expect standard chart types and domains, root scalar `chart_type` and `semantic_domain` always reflect the primary subplot (`axes[0]`). Heterogeneous subplots are captured cleanly and additively via `is_composite: True`, `composite_chart_types: [...]`, and `composite_domains: [...]`. Series metadata is written back from `chart.py` to `theme_config`, eliminating `KeyError` risks.
9. **Comparative Pairs Reachability and Sector-Gated Routing**: Grouped bar charts calling `add_treatment_key_xaxis` (`chart.py:1509`) draw from 94 comparative pairs. In default code, `add_treatment_key_xaxis` was only called inside `if is_scientific:`, rendering all 22 business pairs (55–69, 87–93) unreachable. By extending `add_treatment_key_xaxis` to the business grouped bar branch (within `else: # Standard Styles`, specifically under `style == 'side_by_side'` after tick setting at landmark `chart.py:1594`), all 94 pairs are activated across the generator (biomedical, engineering, business). Reachability across all 94 pairs is verified at the sampler unit level.

This specification resolves these defects by establishing an authoritative semantic registry, defining a verified domain routing architecture with configurable subdomain weights, implementing role- and scale-constrained sampling (augmented with canonical independent metrics in `catalog.py`), providing a dedicated secondary Y sampler, enforcing $O(1)$ semantic concept group collision avoidance, gating fallback titles with minimum-pool top-ups, activating all 94 comparative pairs via simplified sector gating, whitelisting all metadata through save and merge stages, anchoring code edits contextually, and emitting Schema Version `v2.1` (`dataset_version = "2.1.0"`).

---

## User Scenarios & Acceptance Criteria

### Primary User Story
As an ML engineer training OCR, visual document parsing, and vision-language models, I need synthetic charts to display structurally and physically coherent combinations of axes, titles, and keys across all chart types, with true domain metadata persisted in ground-truth JSONs, while preserving token diversity across the dataset.

### Acceptance Scenarios

1. **Given** a vertical bar or box chart, **When** its axes are assigned, **Then** the independent $X$-axis receives an independent or bidirectional label (`role in (INDEPENDENT, BIDIRECTIONAL)`)—never a pure continuous dependent metric (e.g. `Elastic Modulus (GPa)`, `Net Income ($)`). The value $Y$-axis receives a quantitative metric, eliminating categorical contamination.
2. **Given** a horizontal bar or box plot, **When** its axes are assigned, **Then** functional roles are inverted: the continuous dependent metric is assigned to the horizontal $X$-axis (`role in (DEPENDENT, BIDIRECTIONAL)`) and the categorical/independent dimension to the vertical $Y$-axis (`role in (INDEPENDENT, BIDIRECTIONAL)`).
3. **Given** a scatter plot, **When** its axes are assigned, **Then** both $X$ and $Y$ receive quantitative metrics (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT, TEMPORAL)`)—never nominal identifiers like `Salesperson` or `Cell Line`. Bench-science variables (`Dose`, `Temperature`, `Concentration`, `pH`, `Wavelength`, `Time`) remain eligible.
4. **Given** a line or area chart, **When** its axes are assigned, **Then** $X$ receives an independent or bidirectional quantitative/temporal dimension (`role in (INDEPENDENT, BIDIRECTIONAL)` AND `scale_type in (CONTINUOUS, TEMPORAL, DISCRETE_COUNT)`), drawn from an expanded pool of 60+ independent terms with clean typography (including `Date`, `Year`, `Month`, `Quarter`, `Depth (m)`, `Cycle Number`, `Epoch`, `Iteration`, `Step`), preventing both pure dependent metrics (`Elastic Modulus`) and nominal categories (`Salesperson`) from being plotted on $X$, and keeping per-label frequency $\le 5\times$ baseline. $Y$ receives a dependent or bidirectional continuous metric (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT)`).
5. **Given** a histogram, **When** its axes are assigned, **Then** $X$ receives a continuous metric from the domain, and $Y$ receives a frequency/density term from `HISTOGRAM_Y_LABELS` (51 items).
6. **Given** a dual-axis bar chart, **When** secondary axis $Y_2$ is sampled via `sample_secondary_y(y1_label, is_scientific, domain_category)`, **Then** $Y_2$ belongs to the same domain category as $Y_1$, has a distinct normalized concept stem ($Y_2 \not\approx Y_1$), and does not belong to the same semantic `concept_group` (preventing synonym collisions such as `Revenue` vs `Sales`).
7. **Given** any Cartesian chart, **When** axis labels are selected, **Then** $X$ and $Y$ do not collide in exact text ($X \neq Y$), in normalized concept stem ($X \not\approx Y$), or in semantic concept group (`metric_x.concept_group != metric_y.concept_group`). If filtered candidate pools are small or empty, the sampler falls back gracefully to general independent metrics without raising `IndexError`.
8. **Given** a chart without an explicit title, **When** `generator.py:4030` applies a fallback title, **Then** the title matches both the effective `is_scientific` domain and the `chart_type`. Heatmap-only titles (~5.6% of `CHART_TITLES`) are excluded from Cartesian fallbacks, and pools with $< 6$ titles are topped up from domain-generic titles.
9. **Given** a dual-axis bar chart (`chart.py:1322`) or a structured heatmap (`chart.py:3197`) that overrides the domain, **When** generation completes, **Then** the chart writes back `effective_is_scientific` and `semantic_domain` to `theme_config` (guarded by `if isinstance(theme_config, dict):`), and `generator.py` captures the effective state before title sampling and serialization.
10. **Given** a chart rendered and saved to disk, **When** `<image_id>_detailed.json` is written, **Then** `detailed_metadata` contains `"semantic_domain": "<domain_id>"` (or `null`), `"is_scientific": <bool>`, `"schema_version": "v2.1"`, and `"dataset_version": "2.1.0"`.
11. **Given** a dual-axis bar chart with twin axis `ax2` or a heatmap with colorbar, **When** `generator.py:3305` extracts metadata, **Then** unmapped auxiliary axes are skipped if not in `chart_info_map`, preventing them from clobbering `detailed_metadata` with `is_scientific=False` and `chart_type="unknown"`.
12. **Given** a multi-subplot composite figure, **When** annotations are saved, **Then** `detailed_metadata["subplots"]` records a list of per-axis metadata objects; `"is_composite": True`, top-level scalar `chart_type` and `semantic_domain` reflect the primary subplot (`axes[0]`), and `"composite_chart_types"` lists the individual subplot types.
13. **Given** pipeline execution with `merge_json_files: True`, **When** `<image_id>_unified.json` is generated, **Then** `semantic_domain`, `is_scientific`, `schema_version`, `dataset_version`, `subplots`, and composite attributes are preserved at the top level and under `visual_style`.
14. **Given** a chart generated via the Vega-Lite vector backend (`backends/vegalite_backend.py`), **When** annotations are saved, **Then** `detailed_metadata` and `metadata.json` contain matching `semantic_domain`, `is_scientific`, `schema_version`, and `dataset_version` fields. Non-synthetic Vega-Lite charts emit `null` for `semantic_domain` and boolean `False` for `is_scientific`.
15. **Given** a batch generation task with a fixed random seed, **When** `synth/tabular.py` executes, **Then** all RNG draws are derived deterministically from the task's seeded state, producing identical sequences run-to-run while ensuring successive calls within a run produce distinct series.
16. **Given** grouped bar charts with treatment keys rendered via `add_treatment_key_xaxis` (in scientific or business branches), **When** keys are drawn, **Then** the pair matches the chart's resolved `domain_category`: biomedical charts receive biomedical keys (pairs 0–39), engineering charts receive engineering keys (pairs 40–54, 70–86), and business charts receive business/SaaS keys (pairs 55–69, 87–93). All 94 pairs are 100% active and reachable across the generator.

---

## Requirements

### Domain Architecture
* **FR-031 (Domain Taxonomy & Configurable Subdomain Weights)**:
  * Generator execution MUST respect `cfg['bar_chart_config']['scientific_ratio']` (default 0.60) to determine the top-level sector `is_scientific`.
  * For Cartesian charts:
    - When `is_scientific=True`, `semantic_domain` MUST resolve to `"biomedical"` or `"engineering"` drawn according to `cfg.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})` (allowing users to configure 100% engineering or 100% biomedical generation without editing source code).
    - These weights MUST be propagated into `theme_config['scientific_subdomain_weights']` and `style_config['scientific_subdomain_weights']` so downstream drawing routines in `chart.py` have access to them.
    - When `is_scientific=False`, `semantic_domain` MUST resolve to `"business"`.
  * For synthetic data engine charts (`synth/tabular.py`), allowed domains are `"biomedical"`, `"engineering"`, `"business"`, and `"demographic"`.

### Data Modeling, Canonical Expansion & Semantic Concept Groups
* **FR-032 (Authoritative Annotated Axis Catalog & Canonical Diversity Expansion)**:
  * The system MUST establish `synth/semantics/catalog.py` categorizing all 264 existing unique axis labels from `themes.py` (178 scientific, 86 business) with `ScaleType`, `AxisRole`, `concept_stem`, `concept_group`, and `domain_tags`.
  * The catalog MUST explicitly annotate elapsed bench durations (`Time (s)`, `Time (min)`, `Time (h)`, `Time Point (h)`, `Time Point (min)`, `Exposure Time (s)`, `Culture Duration (hrs)`) with `scale_type=ScaleType.CONTINUOUS` and `role=AxisRole.BIDIRECTIONAL`, reserving `ScaleType.TEMPORAL` strictly for calendrical dates/intervals (`Date`, `Year`, `Month`, `Quarter`). This guarantees durations remain eligible for histograms and line $Y$ value axes.
  * **Canonical Independent Variable Expansion & Clean Typography**: To prevent lexical concentration on line and area horizontal axes under the 100% line default configuration (`config_defaults.py:123`), `catalog.py` MUST be augmented with ~40 canonical independent and temporal metrics with clean typography (no parenthetical date format hints: `Date`, `Month`, `Year`, `Quarter`, `Timestamp`, `Cycle Number`, `Epoch`, `Iteration`, `Step`, `Depth (m)`, `Altitude (m)`, `Distance (km)`, `Sample ID`) typed as `role=AxisRole.INDEPENDENT` or `BIDIRECTIONAL` and `scale_type=ScaleType.CONTINUOUS` or `TEMPORAL`.
    - Cross-sector metrics (`Date`, `Month`, `Year`, `Quarter`, `Timestamp`, `Sample ID`): tagged `domain_tags=("biomedical", "engineering", "business")`.
    - Computing/ML metrics (`Epoch`, `Iteration`, `Step`): tagged strictly `domain_tags=("engineering",)` (never business, preventing terms like `Epoch` from appearing on revenue charts).
    - Spatial dimensions (`Depth (m)`, `Altitude (m)`, `Distance (km)`): tagged `domain_tags=("engineering", "biomedical")`.
    - This expansion MUST NOT modify `themes.py` (preserving FR-050).
  * **Semantic Concept Groups (`concept_group`)**: To prevent semantic synonym collisions ($O(1)$ equality checks), `AxisMetric` MUST define `concept_group: Optional[str] = None` grouping synonymous terms (e.g. `revenue_metric` for `Revenue ($)` and `Sales ($)`, `optical_density` for `Absorbance (OD)` and `Optical Density (600nm)`).
* **FR-033 (Enum-Based Immutability & Scale Typing)**:
  * `ScaleType` MUST be an Enum containing `CONTINUOUS`, `CATEGORICAL`, `DISCRETE_COUNT`, `PERCENTAGE`, `TEMPORAL` (explicitly excluding `ORDINAL`).
  * Measured bench quantities (`Dose`, `Concentration`, `Temperature`, `pH`, `Wavelength`, `Frequency`, `Age`, `Time (s)`) MUST be typed as `ScaleType.CONTINUOUS` and assigned `AxisRole.BIDIRECTIONAL`.
  * `AxisRole` MUST be an Enum containing `INDEPENDENT`, `DEPENDENT`, `BIDIRECTIONAL`.
  * All catalog entities MUST be `@dataclass(slots=True, frozen=True)` with zero mutable fields.
  * The catalog MUST expose `METRIC_BY_LABEL: Dict[str, AxisMetric]` for $O(1)$ metric lookup by label string.
* **FR-034 (Comparative Pairs Categorization & Business Path Activation)**:
  * The 94 pairs in `COMPARATIVE_LABELS` MUST be tagged in `catalog.py` with `domain_category: str`:
    - Pairs 0–39: `domain_category = "biomedical"`, `is_scientific = True`.
    - Pairs 40–54 and 70–86: `domain_category = "engineering"`, `is_scientific = True` (computing, ML, software, physical and materials engineering).
    - Pairs 55–69 and 87–93: `domain_category = "business"`, `is_scientific = False` (business, A/B testing, and SaaS economics).
  * The system MUST extend `add_treatment_key_xaxis` to the business grouped bar path (within the `else: # Standard Styles` branch, specifically under `style == 'side_by_side'` after `ticks_setter(indices, categories)` at landmark line 1594) when `len(bar_info_list) == 4 and orientation == 'vertical' and random.random() < TREATMENT_KEY_PROBABILITY`. This activates the 22 business pairs, making all 94 comparative pairs 100% active and reachable across the pipeline.
* **FR-035 (Pre-Indexed Chart Titles & Minimum Pool Top-Up)**:
  * The 143 titles in `CHART_TITLES` MUST be categorized with `is_scientific: Optional[bool]` and `allowed_chart_types: Tuple[str, ...]`.
  * Heatmap-only titles (`Confusion Matrix`, `Proteomics Heatmap`, etc., ~5.6% of titles) MUST be excluded from the Cartesian fallback index.
  * The catalog MUST pre-index titles by `(is_scientific, chart_type)` for $O(1)$ lookups.
  * If any `(is_scientific, chart_type)` pool contains fewer than 6 titles, the catalog MUST automatically top up the pool with domain-generic titles matching `is_scientific` to prevent low-entropy over-concentration.
  * Title sampling MUST preferentially select titles matching `domain` when available, falling back safely to domain-generic titles matching `is_scientific`.

### Sampling Engine & Rules
* **FR-036 (Universal Axis Sampling API & Contextual Anchoring)**: The system MUST provide `sample_axis_pair(chart_type, orientation, is_scientific, semantic_domain=None, rng=None)` in `synth/semantics/sampler.py` (defaulting `rng=random` when `rng is None`):
  * `scatter`: $X$ and $Y$ MUST have quantitative scale types (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT, TEMPORAL)`), explicitly banning `CATEGORICAL` nominal dimensions (e.g. `Salesperson`, `Cell Line`).
  * `bar` / `box` (vertical): $X$ MUST have `role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)`; $Y$ MUST have `role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)` and `scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT)`.
  * `bar` / `box` (horizontal): Roles inverted: horizontal metric $X$ MUST have `role in (DEPENDENT, BIDIRECTIONAL)` and `scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT)`; vertical grouping $Y$ MUST have `role in (INDEPENDENT, BIDIRECTIONAL)`.
  * `line` / `area`: $X$ MUST have `role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)` **AND** `scale_type in (ScaleType.CONTINUOUS, ScaleType.TEMPORAL, ScaleType.DISCRETE_COUNT)` (banning nominal categories like `Salesperson` and pure dependent metrics like `Elastic Modulus`); $Y$ MUST have `role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)` and `scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT)`.
  * `histogram`: $X$ MUST have `scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT)`; $Y$ is sampled from `HISTOGRAM_Y_LABELS` (51 items).
  * `pie` and `heatmap` chart types are explicitly excluded from `sample_axis_pair`: pie charts use radial wedge geometry without Cartesian axes, and heatmaps manage matrix coordinate labels via `CONTEXT_CONFIGURATIONS`.
  * Normalized and Semantic Collision Check: MUST reject candidate pairs where `x.concept_stem == y.concept_stem` OR (`x.concept_group is not None and x.concept_group == y.concept_group`).
  * **Small-Pool Fallback Rule**: If candidate pool after domain, role, and scale filtering contains fewer than 1 metric, the sampler MUST fall back gracefully to general independent/continuous catalog metrics matching `is_scientific` to guarantee it never raises `IndexError`.
  * Returns a 3-tuple `(x_label, y_label, domain_category)`.
  * MUST be called at **all 17 axis sampling sites** in `chart.py`, passing `semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None`. Each call site MUST write back `theme_config['semantic_domain'] = domain_category` guarded by `if isinstance(theme_config, dict):`.
  * **Dual-Axis Line 1322 Re-Resolution**: On dual-axis bar charts (`chart.py:1322` landmark), because `is_scientific = True` is forced, if `theme_config.get('semantic_domain') == "business"`, the system MUST **immediately at line 1322** (BEFORE the `sample_multivariate_table` call at line 1332) re-resolve `semantic_domain` to `"biomedical"` or `"engineering"` drawing with weights `theme_config.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})` and write back `theme_config['semantic_domain'] = domain_cat` and `theme_config['effective_is_scientific'] = True`. This guarantees synthetic table generation and axis sampling operate on the coherent scientific domain.
  * In `chart.py`, all 6 `sample_multivariate_table` calls (landmarks 1332, 1459, 1523, 1825, 1944, 3058) MUST respect user config precedence: `domain = syn_domain or theme_config.get('semantic_domain') or ('biomedical' if is_scientific else 'business')`. When `use_synthetic and syn_table is not None`, `theme_config['semantic_domain']` MUST be written back from `PRESET_TO_CANONICAL.get(syn_table.get('domain'), syn_table.get('domain'))`.
* **FR-037 (Secondary Axis Sampling)**: The system MUST provide `sample_secondary_y(y1_label, is_scientific, domain_category=None, rng=None) -> str` in `synth/semantics/sampler.py`. It MUST look up `y1_label` in `METRIC_BY_LABEL`, resolve its `concept_stem` and `concept_group`, exclude all candidate metrics sharing that stem OR concept group, and return a distinct continuous metric matching `domain_category`.
* **FR-038 (Normalized Stem Algorithm & Semantic Concept Group Collision Prevention)**: `sample_axis_pair` MUST ensure that $X$ and $Y$ do not share the same `concept_stem` AND do not share the same non-None `concept_group`. The `concept_stem` computation algorithm MUST be implemented as `compute_concept_stem(label: str) -> str`: lowercase the label string, strip parenthetical units and formats via `re.sub(r'\(.*?\)', '', label)`, strip currency symbols (`[$€£¥]`), strip punctuation, and collapse consecutive whitespace.
* **FR-039 (Context-Aware Title Sampling with Domain Gating)**: The system MUST provide `sample_chart_title(chart_type, is_scientific, domain=None, rng=None)`. `generator.py:4030` landmark MUST call this function, passing active `chart_type`, effective `is_scientific`, and `domain=theme_config.get('semantic_domain')`, preferentially selecting titles matching `domain` while falling back to general scientific/business titles.
* **FR-040 (Sector-Gated Comparative Pairs Routing)**: The system MUST provide `sample_comparative_pair(domain_category=None, rng=None)`. In `chart.py:1509` (scientific grouped bar) and `chart.py` within `else: # Standard Styles` under `style == 'side_by_side'` after tick setting (landmark line 1594, business grouped bar), `add_treatment_key_xaxis` MUST accept `domain_category`.
  - When `domain_category == "business"`, it MUST draw from business pairs (55–69, 87–93).
  - When `domain_category == "biomedical"`, it MUST draw from biomedical pairs (0–39).
  - When `domain_category == "engineering"`, it MUST draw from engineering pairs (40–54, 70–86).
  - Reachability across all 94 pairs is verified at the sampler unit level.

### Pipeline Plumbing, Multi-Subplot & Serialization
* **FR-041 (Tabular PRNG Seeding Closure & Bidirectional Preset Aliasing)**:
  * `synth/tabular.py` MUST replace all five unseeded `np.random.default_rng()` fallback lines with `rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))` when `rng is None` at top-level entry points, seamlessly leveraging `generator.py:4781`'s task-level seed without causing identical series across subplots.
  * `synth/tabular.py` MUST support canonical domain lookups via bidirectional alias mappings: `DOMAIN_ALIAS_TO_PRESET: Dict[str, str] = {"business": "financial", "engineering": "sensor_telemetry"}` for resolving presets from canonical inputs in `sample_domain_schema`, and `PRESET_TO_CANONICAL: Dict[str, str] = {"financial": "business", "sensor_telemetry": "engineering", "demographic": "business"}` when returning table schemas and metadata. Tabular `"demographic"` preset maps to business/common during chart axis routing.
* **FR-042 (Subplot Domain Resolution, Bar Sampling Anchor & Write-Back)**:
  * In `generator.py` inside the per-subplot loop `for ax_idx, ax in enumerate(axes):` (landmark line 3917), the system MUST resolve `semantic_domain` per subplot preserving `scientific_ratio`, storing in `theme_config['semantic_domain']` and `theme_config['is_scientific']`.
  * `generator.py` MUST propagate `cfg.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})` into `theme_config['scientific_subdomain_weights']` and `style_config['scientific_subdomain_weights']`.
  * Standard bar chart axis sampling MUST occur **immediately after `style_config['orientation'] = orientation`** (landmark line 1447), before the treatment key check (landmark line 1508), ensuring `orientation` is resolved before calling `sample_axis_pair`.
  * Chart functions in `chart.py` that override domain (dual-axis bar landmark 1322, heatmap landmark 3197) MUST write back `effective_is_scientific` and `semantic_domain` to `theme_config` guarded by `if isinstance(theme_config, dict):`. Heatmap context mapping is formalized via `HEATMAP_CONTEXT_TO_DOMAIN`: `"genomic_expression_heatmap"` and `"pharmacokinetics_heatmap"` map to `"biomedical"`, `"cohort_retention_heatmap"` maps to `"business"`, and `"correlation_matrix"` samples via scientific weights.
  * `generator.py` MUST read this effective state before title sampling and serialization.
* **FR-043 (Twin-Axes & Heatmap Colorbar Clobber Guard)**: In `generator.py` (landmark line 3305), `for ax in fig.axes:` MUST check `if ax not in chart_info_map: continue`, preventing auxiliary twin axes `ax2` AND heatmap colorbar axes (`chart.py:3270` landmark) from overwriting metadata with `chart_type="unknown"`, `is_scientific=False`, and resetting `dual_axis_info={}`.
* **FR-044 (`detailed_json` Save Whitelist Closure & Composite Whitelisting)**: In `generator.py:4673–4738`, `detailed_json` MUST whitelist the following missing fields (note that `series_count`, `series_names`, `style`, `pattern`, and `is_scientific` are already present in `detailed_json`):
  ```python
  "semantic_domain": unified_json.get("semantic_domain"),
  "schema_version": ANNOTATION_SCHEMA_VERSION,
  "dataset_version": DATASET_VERSION,
  "subplots": unified_json.get("subplots", []),
  "is_composite": unified_json.get("is_composite", False),
  "composite_chart_types": unified_json.get("composite_chart_types", []),
  "composite_domains": unified_json.get("composite_domains", []),
  ```
* **FR-045 (Multi-Subplot Composite Tracking, Series Info Write-Back & Attribute Wiring)**:
  * In `generator.py`, immediately before landmark line 4051, define `eff_sci = theme_config.get('effective_is_scientific', is_scientific)` and `eff_dom = theme_config.get('semantic_domain')`. In Phase 1, `eff_sci` safely falls back to `is_scientific` and `eff_dom` is `None`, preventing a `NameError: name 'eff_sci' is not defined` when running Phase 1 in isolation.
  * In `chart.py`, bar plotting branches MUST write back `theme_config['series_count'] = bars_per_group` (or `num_series`, `1`) and `theme_config['series_names'] = series_names` (or generated series names/categories).
  * In `generator.py:4051` landmark, `chart_info_map[ax]` MUST store all 7 fields: `style`, `pattern`, `series_count = theme_config.get('series_count', 1)`, `series_names = theme_config.get('series_names', [])`, `stacking_mode`, `is_scientific=eff_sci`, and `semantic_domain=eff_dom`. (Do NOT inspect `b['series_idx']` from `bar_info_list`).
  * In `generator.py:3471–3478` landmark (`create_unified_annotation`), the system MUST copy `detailed_metadata["semantic_domain"] = chart_info.get("semantic_domain")` (while `series_count`, `series_names`, `stacking_mode`, `style`, `pattern`, and `is_scientific` are already copied from `chart_info` and will now receive populated values from `chart_info_map[ax]`).
  * In `generator.py:3305–3478`, `detailed_metadata["subplots"]` MUST record a list of per-subplot metadata records.
  * **Additive Composite Subplot Tracking (No Enum Widening of Root Scalars)**:
    - When `len(subplots) > 1`, set `detailed_metadata["is_composite"] = True`; when `len(subplots) <= 1`, set `detailed_metadata["is_composite"] = False`.
    - Top-level scalar `chart_type` ALWAYS reflects the primary subplot (`fig.axes[0]` / `axes[0]`), e.g., `"bar"`, `"line"`, `"scatter"`. Root scalar `chart_type` NEVER emits `"composite"`.
    - Top-level scalar `semantic_domain` ALWAYS reflects the primary subplot (`fig.axes[0]`), e.g., `"biomedical"`, `"business"`. Root scalar `semantic_domain` NEVER emits `"mixed"`.
    - If subplot `chart_type`s differ across subplots, emit additive list `"composite_chart_types": [s["chart_type"] for s in subplots]`.
    - If subplot domains differ across subplots, emit additive list `"composite_domains": [s["semantic_domain"] for s in subplots]`.
    - In `generator.py:4751`, set `metadata_json["chart_types"] = detailed_json.get("composite_chart_types", [unified_json.get("chart_type", "unknown")])`.
* **FR-046 (`merge_json.py` Whitelist Preservation & OBB Preservation)**:
  * `merge_json.py` MUST preserve `semantic_domain`, `schema_version`, `dataset_version`, `subplots`, `is_composite`, `composite_chart_types`, and `composite_domains` at the top level of `unified` and under `chart_generation_metadata["visual_style"]`, copying them directly without fallback default version strings. Legacy files default `is_composite` to `False`.
  * `normalize_raw_annotation` MUST preserve `entry.get("obb")` when present, preventing OBB loss for rotated text and elements.
* **FR-047 (Vega-Lite Backend Schema Parity & Contextual Landmarks)**: In `backends/vegalite_backend.py:generate_single_vegalite_chart`:
  * Initialize `syn_table = None` at start of function.
  * Directly after `extract_svg_annotations` (landmark line 683), populate `"semantic_domain"`, `"is_scientific"`, `"schema_version"`, and `"dataset_version"` on `detailed_metadata`. Non-synthetic charts emit `null` for `semantic_domain` and boolean `False` for `is_scientific`.
  * Immediately after `metadata_json` dictionary instantiation (landmark line 730, at line 736), populate `"schema_version"` and `"dataset_version"` on `metadata_json`.
* **FR-048 (Shared Version Constants Module & Phase 1 Import)**: The system MUST establish `versions.py` in Phase 1 defining `ANNOTATION_SCHEMA_VERSION = "v2.1"` and `DATASET_VERSION = "2.1.0"`. `generator.py`, `backends/vegalite_backend.py`, and `merge_json.py` MUST import these constants in Phase 1 to prevent `NameError`.
* **FR-049 (Named Probability Constants & Lazy Test Triggers)**: `chart.py` MUST extract module-level constants `DUAL_AXIS_PROBABILITY: float = 0.15` and `TREATMENT_KEY_PROBABILITY: float = 0.30`, and support `style_config['force_dual_axis'] if 'force_dual_axis' in style_config else (random.random() < DUAL_AXIS_PROBABILITY)` to avoid eager evaluation and random entropy pollution.
* **FR-050 (Legacy Constant Preservation)**: All constants in `themes.py` MUST remain untouched for 100% backward compatibility.
* **FR-051 (Independent Double-Annotated Test Oracle)**: Verification tests MUST validate scale and role rules using an independent continuous-regex oracle across all catalog labels paired with an embedded 50-item blind gold set annotating both `scale_type` and `role` for ambiguous semantic boundary terms, completely detached from `synth/semantics/catalog.py`.

---

## Key Entities

```python
# versions.py
ANNOTATION_SCHEMA_VERSION: str = "v2.1"
DATASET_VERSION: str = "2.1.0"
```

```python
# synth/semantics/catalog.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ScaleType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    DISCRETE_COUNT = "discrete_count"
    PERCENTAGE = "percentage"
    TEMPORAL = "temporal"


class AxisRole(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    BIDIRECTIONAL = "bidirectional"


@dataclass(slots=True, frozen=True)
class AxisMetric:
    label: str
    scale_type: ScaleType
    role: AxisRole
    concept_stem: str
    unit: Optional[str] = None
    concept_group: Optional[str] = None
    is_scientific: bool = False
    domain_tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class ChartTitle:
    title: str
    is_scientific: Optional[bool]
    allowed_chart_types: Tuple[str, ...]
    domain_tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class ComparativePair:
    control: str
    treatment: str
    domain_category: str  # "biomedical", "engineering", "business"
    is_scientific: bool
```

---

## Success Criteria

* **SC-019 (Scatter Scale Validity)**: In an independently audited test run with forced scatter charts, $100\%$ of scatter plots have quantitative metrics (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT, TEMPORAL)`) on both $X$ and $Y$ (verified via independent continuous regex and blind gold set, catching zero nominal categories).
* **SC-020 (Bar & Line Scale/Role Validity & Canonical Expansion)**: In an independently audited test run with forced multi-chart mixes, $100\%$ of vertical bar charts and line charts have independent/bidirectional/temporal $X$ ($0$ pure continuous dependent metric violations such as `Elastic Modulus` on $X$, and $0$ nominal category violations such as `Salesperson` on line $X$). Value axes ($Y$) have zero categorical contamination. Line $X$ draws from an expanded pool of 60+ labels (~20 sci, ~13 biz, ~40 canonical independent variables in `catalog.py` with clean typography and domain tags).
* **SC-021 (Title Gating Validity)**: In $100\%$ of standard charts using fallback titles, the title matches the effective `is_scientific` domain and is permitted for that `chart_type`.
* **SC-022 (Zero Axis Collisions & Semantic Disjointness)**: Across 10,000 sampling draws, $0$ axis pairs exhibit exact ($X == Y$), normalized stem ($X \approx Y$), or semantic concept group (`metric_x.concept_group == metric_y.concept_group`) collisions. Dual-axis $Y_1/Y_2$ pairs sampled via `sample_secondary_y` exhibit zero stem or concept group collisions.
* **SC-023 (Treatment Key 100% Reachability & Coherence)**: In $100\%$ of charts where `add_treatment_key_xaxis` is rendered, the comparative pair matches the chart's resolved `domain_category` ($0\%$ engineering pairs on biomedical charts, $0\%$ clinical pairs on business charts). At the unit/sampler level, across 10,000 draws of `sample_comparative_pair`, all 94 comparative pairs (biomedical, engineering, and business) are 100% reachable.
* **SC-024 (Ground-Truth Persistence Audit)**: $100\%$ of emitted `<image_id>_detailed.json` and `<image_id>_unified.json` files contain a valid boolean `is_scientific`, `"schema_version": "v2.1"`, `"dataset_version": "2.1.0"`, and `subplots`. Authentic non-null `semantic_domain` is present on $100\%$ of Matplotlib charts and synthetic Vega-Lite charts (non-synthetic Vega-Lite emits `null`).
* **SC-025 (Twin-Axes & Composite Subplot Integrity)**: In dual-axis bar charts, `detailed_metadata["chart_type"]` is `"bar"`, not `"unknown"`. In multi-subplot figures, `detailed_metadata["subplots"]` contains records for all visible subplots, sets `"is_composite": True`, top-level scalar `chart_type` reflects `axes[0]`, and `composite_chart_types` lists all subplot types.
* **SC-026 (Dataset Lexical Diversity & Distribution Benchmarks)**:
  * Title distribution over $N = 20,000$ in-memory draws against theoretical uniform baseline ($1/143 \approx 0.00699$): maximum frequency in any cell $\le 5\times$ baseline ($p_{max} \le 3.5\%$) under balanced multi-chart mix, and total vocabulary coverage $\ge 95\%$ of routable catalog terms ($\ge 90\%$ of total 143 titles).
  * Axis-label frequency under production mix: maximum line $X$ frequency $\le 5\times (1 / K_{domain})$ baseline (achieved through canonical independent variable expansion in `catalog.py`).

---

## Post-Migration Sunset & Decommissioning Roadmap

While Feature 002 deliberately retains legacy data structures frozen in `themes.py` ([`FR-050`](#requirements)) and alias translation dictionaries in `synth/tabular.py` to ensure zero breaking changes during rollout, these legacy structures are formally scheduled for retirement in **Feature 003**:
* **Specification Directory**: [`../003-legacy-decommissioning-declarative-manifests/`](../003-legacy-decommissioning-declarative-manifests/)
* **Schema Version**: `v3.0` (`dataset_version = "3.0.0"`)
* **Sunset Gates**: Feature 003 is blocked until $\ge 50,000$ charts are generated under Schema `v2.1` (SG-001), AST verification confirms zero external call sites (SG-002), and downstream consumers sign off on Schema `v2.1` ingestion (SG-003).
* **Decommissioning Scope**:
  - Purging 758 lines of procedural semantic constants from `themes.py` (reducing length from 1,106 to $\le 380$ lines).
  - Promoting declarative YAML domain manifests (`synth/semantics/domains/*.yaml`) validated with Pydantic v2.
  - Compiling manifests into frozen slotted dataclasses in $\le 50\,\text{ms}$ cold startup time ($O(1)$ runtime access).
  - Removing tabular alias translation dictionaries in `synth/tabular.py` for direct domain naming (`"business"`, `"engineering"`).
  - Retiring legacy compatibility test shims (`tests/test_legacy_themes_compatibility.py`).
