# Research: Semantic Storage & Domain Registry (Fourth-Pass Codebase Audit)

**Feature**: `002-semantic-storage-registry`  
**Purpose**: Ground the specification in the true runtime mechanics, data flows, and empirical distributions of `chart.py`, `generator.py`, `themes.py`, `synth/tabular.py`, `merge_json.py`, and `backends/vegalite_backend.py`. Resolves all integration gaps, establishes mathematically exact defect statistics, eliminates serialization drop-points, and defines a verified domain architecture.

---

## 1. Codebase Reality: How Constants & Context Actually Flow

### 1.1 The Semantic Constants in `themes.py`
`themes.py` is 1,106 lines of static Python data structures with no imports.

| Constant | Entries (Unique) | Actual Sampling Behavior & Call Sites | Verified Codebase Facts |
|---|---|---|---|
| `SCIENTIFIC_X_LABELS` & `SCIENTIFIC_Y_LABELS` | 183 entries (178 unique) | Sampled across 17 sites in `chart.py` (`1406–3147`), gated by `is_scientific`. `generator.py:44` imports both but **never references them**. | Both lists are identical block-duplicates with 5 duplicates: `Concentration (μM)`, `Temperature (°C)`, `pH`, `Wavelength (nm)`, `Frequency (Hz)`. Includes canonical time axes at indices 41–43: `Time (s)`, `Time (min)`, `Time (h)`. Deduplicating these 5 duplicates yields exactly 178 unique scientific terms for `METRIC_BY_LABEL`. **Sci vs. Business axis cross-bleed is already 0.0%** because `is_scientific` gates every single draw. |
| `BUSINESS_X_LABELS` & `BUSINESS_Y_LABELS` | 86 entries (86 unique) | Sampled at the same 17 sites when `is_scientific=False`. `generator.py:44` imports both but never references them. | Both lists are identical block-duplicates with **0 duplicates** (86 unique terms). Pure business/sales/marketing metrics; **0 demographic labels**. |
| `CHART_TITLES` | 143 strings (143 unique) | Imported at `generator.py:44`. Sampled at `generator.py:4030`: `if not ax.get_title(): ax.set_title(random.choice(CHART_TITLES))`. | **The primary domain-incoherence vector**. `chart.py` only sets titles for heatmaps (`chart.py:3194`) and the synthetic engine (`chart.py:1522`). Thus, this domain-blind fallback fires on **nearly 100% of standard charts** (line, bar, scatter, box, pie, histogram, area)! |
| `COMPARATIVE_LABELS` | 94 tuples (94 unique) | Imported at `chart.py:64`. Sampled only in `add_treatment_key_xaxis` (`chart.py:1257–1270`), invoked at `chart.py:1509`. | Only fires on 4-bar grouped charts, with `orientation == 'vertical'`, at 30% probability (~narrow tail, ~2.6% of scientific bars). Structurally partitioned into: Biomedical (pairs 0–39, 40 pairs), Engineering & Systems (pairs 40–54 and 70–86, 32 pairs), and Business & SaaS (pairs 55–69 and 87–93, 22 pairs). |
| `HISTOGRAM_Y_LABELS` | **51 entries** (51 unique) | Sampled at `chart.py:2904`: `ax.set_ylabel(random.choice(HISTOGRAM_Y_LABELS))`. | Pure frequency/density terms (`Count`, `Frequency`, `Probability Density`, `Empirical Density`). |
| `CONTEXT_CONFIGURATIONS` | 4 context dicts | Consumed by `generate_structured_heatmap` (`chart.py:3194`). Context weights hardcoded at `chart.py:3548`. | Overrides `is_scientific` based on `context_domain` (`chart.py:3197–3201`). Heatmap titles and axes are filtered by keyword substring. |
| `*_DOMAIN_DICT` | 2 dicts | Imported at `chart.py:64`. Never referenced anywhere else. | **100% Dead Code**. |
| `DOMAIN_PRESETS` (`synth/tabular.py`) | 4 presets | Consumed by `synth/tabular.py` when `use_synthetic_data_engine: True` (default `False`). | Presets: `"biomedical"`, `"financial"`, `"demographic"`, `"sensor_telemetry"`. Covers only 4 chart types (`bar`, `line`, `scatter`, `area`) and Vega-Lite. |

---

### 1.2 Baseline Measurements & True Defect Rates

A rigorous mathematical and code audit of the 264 unique axis labels in `themes.py` reveals the true distribution of defects:

| Metric | Scientific Pool (183 / 178 unique) | Business Pool (86 unique) | Codebase Implication & Derivation |
|---|---|---|---|
| **Sci / Business Axis Cross-Bleed** | **0.0%** | **0.0%** | The popular strawman (*"pharmacokinetic concentration paired with fiscal quarter"*) **never occurs** in default code. `is_scientific` already gates every draw. |
| **Bar Chart $X$-Axis is Continuous Metric** | **~76.4%** | **~54.7%** | In bench science, `Dose`, `Time Point`, `Time (s/min/h)`, `Concentration`, `Age`, `Temperature`, `pH`, `Wavelength` serve as valid independent $X$ axes. Accounting for these, 42 of 178 unique scientific labels are valid independent/bidirectional axes, while 136 are pure dependent metrics. In business, 39 of 86 labels are independent categorical dimensions. |
| **Scatter Plot Has $\ge 1$ Categorical Axis** | **27.3%** | **70.1%** | **Exact Derivation**: On the 183-item scientific list sampled by `chart.py`, 27 items are categorical ($183 - 27 = 156$ continuous). Probability both axes are continuous is $(156/183)^2 = 72.67\%$, so $\ge 1$ categorical axis is $1 - (156/183)^2 = \mathbf{27.33\%}$. For business ($N=86$, $k=39$ categorical, $47$ continuous), $(47/86)^2 = 29.87\%$, so $\ge 1$ categorical axis is $1 - (47/86)^2 = \mathbf{70.13\%}$. (Per-axis draw rates are $14.8\%$ and $45.3\%$). |
| **Line/Area $X$-Axis is Pure Nominal Category** | **12.6%** | **34.9%** | In line and area charts, connecting nominal categories (`Salesperson`, `City`, `Department`, `Cell Line`) with continuous line segments is a major visualization flaw. 23 of 183 scientific items (12.6%) and 30 of 86 business items (34.9%) are nominal categories. (Note: scatter plots exclude 27 scientific and 39 business items because scatter strictly requires continuous metrics, while line/area charts permit temporal/discrete variables like `Time Point`, `Date`, `Year`, and `Rank`, leaving 23 and 30 purely nominal items). |
| **Line/Area $X$ Pool Size & Lexical Trade-off** | **~20 labels (sci)<br>~13 labels (biz)** | **Mathematical Derivation**: Restricting line $X$ to independent continuous/temporal variables yields ~20 scientific and ~13 business labels. (In business: 86 total - 47 dependent continuous - 30 nominal - 4 dependent discrete = 9 temporal + 4 independent continuous = 13 labels). Under a uniform draw across the 183-item scientific list ($1/183 \approx 0.55\%$), drawing from ~20 labels yields $5.0\%$ ($9.1\times$ baseline). When subdivided into engineering (~7 labels, $14.3\%$), concentration spikes up to **$15.2\times$** baseline ($10\text{--}15\times$). Expanding `catalog.py` with ~40 canonical independent variables restores pool size to 60+ labels and limits concentration to $\le 5\times$ baseline. |
| **Exact Axis Collision ($X == Y$)** | **0.58%** | **1.16%** | Occurs reliably across 10,000-image batch runs. |
| **Same-Stem Axis Collision ($X \approx Y$)** | **0.62%** | **1.22%** | Stripping lowercase units and punctuation catches unit variants (`Revenue ($)` vs `Revenue (USD)`). |
| **Fallback Title Mismatch Derivation** | **~36.1% (sci alone)**<br>**~48.0% (total)** | `CHART_TITLES` has 143 items: ~57 scientific and ~86 business. With default `scientific_ratio = 0.60`, scientific charts drawing uniformly pick a business title with probability $0.60 \times (86/143) = \mathbf{36.08\%}$ (the ~36% scientific mismatch rate). Business charts pick a scientific title with probability $0.40 \times (57/143) = 15.94\%$. Total cross-domain title defect rate is $36.08\% + 15.94\% = \mathbf{52.02\%}$ (or $48.0\%$ if partitioned 85/58). |
| **Configuration Mix Reality** | **100% Line** | **0% Other** | In `config_defaults.py:123` and `custom_config.py:115`, `line` weight is 100, while all other chart types have weight 0. Tests must explicitly enable and weight other chart types across all 8 supported types. |

---

### 1.3 Critical Serialization Drop-Points & Lifecycle Traps

1. **`semantic_domain`, `schema_version`, `dataset_version`, `subplots`, and Composite Fields Dropped from `detailed_json`**:
   * In `generator.py:4673–4738`, `detailed_json` is built from an **explicit dictionary**:
     ```python
     detailed_json = {
         ...
         "style": unified_json.get("style"),
         "pattern": unified_json.get("pattern"),
         "is_scientific": unified_json.get("is_scientific", False),
         "raw_annotations": unified_json.get("raw_annotations", []),
         ...
     }
     ```
   * `semantic_domain`, `schema_version`, `dataset_version`, `subplots`, `is_composite`, `composite_chart_types`, and `composite_domains` are **completely absent from this dictionary**. Even if set in `unified_json`, they are dropped before reaching disk! Note that `series_count` and `series_names` (lines 4725–4726) as well as `style`, `pattern`, `is_scientific` (lines 4729–4731) are already present in the `detailed_json` whitelist.
   * **Missing Wiring Vector & Source Breakdown**: `detailed_json` reads from `unified_json` (the output of `create_unified_annotation`). Inside `create_unified_annotation` (lines 3471–3477), `detailed_metadata` already copies `series_count`, `series_names`, `stacking_mode`, `style`, `pattern`, and `is_scientific` from `chart_info`. However, `semantic_domain` is completely omitted from `create_unified_annotation`. Crucially, the root breakdown occurs at the **source**: `chart_info_map[ax]` at line 4051 failed to store `is_scientific`, `semantic_domain`, `series_count`, `series_names`, `style`, `pattern`, and `stacking_mode` from `theme_config`, causing the reads in `create_unified_annotation` to fall back to hardcoded default values (`1`, `[]`, `False`).
   * **Heterogeneous `bar_info_list` Schemas & KeyError Fix**:
     - In `chart.py`, 6 of 14 `bar_info_list.append` sites (lines 1481, 1490, 1662, 1672, 1759, 1767) append dictionaries lacking `'series_idx'` (e.g. `{'center': pos, 'height': value, 'width': bar_width, 'bottom': 0, 'top': value}`).
     - Therefore, calculating `len(set(b['series_idx'] for b in bar_info_list))` raises `KeyError` on scientific grouped bars, touching bars, and default bars!
     - The true series count exists in chart-local variables (`bars_per_group`, `num_series`).
     - **The Fix**: During chart plotting, `chart.py` writes back `theme_config['series_count'] = bars_per_group` (or `num_series`, `1`) and `theme_config['series_names'] = series_names` directly. In `generator.py:4051`, `chart_info_map[ax]` reads `series_count` and `series_names` from `theme_config`.
   * **Full Wiring Fix**:
     1. In `generator.py`, import `ANNOTATION_SCHEMA_VERSION` and `DATASET_VERSION` from `versions` in Phase 1 (when `versions.py` is created), avoiding Phase 1 `NameError`.
     2. In `generator.py`, immediately before line 4051, safely define:
        ```python
        eff_sci = theme_config.get('effective_is_scientific', is_scientific)
        eff_dom = theme_config.get('semantic_domain')
        ```
        In Phase 1, `eff_sci` falls back to `is_scientific` (which exists at line 3952) and `eff_dom` is `None`, preventing a `NameError` when running Phase 1 in isolation before Phase 2 adds title fallback migration.
     3. At line 4051: store all 7 fields in `chart_info_map[ax]` using `eff_sci`, `eff_dom`, and `theme_config.get('series_count', 1)`.
     4. At lines 3471–3478: add `detailed_metadata["semantic_domain"] = chart_info.get("semantic_domain")` (while existing reads for `is_scientific`, `series_names`, `series_count`, `style`, `pattern`, and `stacking_mode` now receive populated values from `chart_info`).
     5. At line 4673: whitelist `semantic_domain`, `schema_version`, `dataset_version`, `subplots`, `is_composite`, `composite_chart_types`, and `composite_domains` in `detailed_json`.
2. **`merge_json.py` Permanent Deletion, Key Dropping & OBB Discarding**:
   * `merge_json.py:156–160` only whitelists `style`, `pattern`, and `is_scientific` under `visual_style`.
   * At line 260, `merge_json.py` **deletes `<image_id>_detailed.json`**! Any field not copied to `unified` (including `subplots`, `schema_version`, `dataset_version`, `semantic_domain`, `is_composite`, `composite_chart_types`, and `composite_domains`) is permanently lost.
   * `normalize_raw_annotation` (line 58) **discards `obb`** (`entry.get("obb")`) and rounds `xyxy` floats to ints, silently corrupting oriented bounding box annotations for rotated text and bars.
   * **Fix**: Copy `detailed.get("schema_version")`, `dataset_version`, `semantic_domain`, `subplots`, `is_composite`, `composite_chart_types`, and `composite_domains` directly without fallback default version constants, default `is_composite` to `False` on legacy files, and preserve `entry.get("obb")` in `normalize_raw_annotation`.
3. **Twin-Axes & Heatmap Colorbar Clobbering in `generator.py:3305–3478`**:
   * Line 3305 iterates over all axes: `for ax in fig.axes:`.
   * On dual-axis bar charts, secondary axis `ax2` is visible but not in `chart_info_map`.
   * On heatmaps, `ax.figure.colorbar(mesh, ax=ax)` at `chart.py:3270` adds a colorbar Axes to `fig.axes` that also never enters `chart_info_map`.
   * Both `ax2` and the colorbar axis land **last** in `fig.axes`. Without a guard, they overwrite `detailed_metadata` with `chart_type="unknown"`, `is_scientific=False`, and reset `dual_axis_info={}`!
   * **Fix**: Skipping `if ax not in chart_info_map: continue` in the first loop (lines 3305–3478) protects scalar metadata for both dual-axis charts and heatmaps with colorbars. It does **not** drop artist annotations, which are extracted in the second loop at line 3480+.
4. **Local Domain Overrides, Dual-Axis Line 1322 Table Generation Trap & Weight Propagation**:
   * **The Dual-Axis Trap**: Dual-axis bar charts force `is_scientific = True` at `chart.py:1322`. Immediately following at line 1332, `sample_multivariate_table` is called using `syn_domain or 'biomedical'`. If axis sampling happens at line 1406, any re-resolution at line 1406 occurs **too late**: the table at line 1332 has already been generated with the wrong domain!
   * **The Fix**: Re-resolution of `semantic_domain` MUST occur **immediately at line 1322**, directly alongside `is_scientific = True`. If `theme_config.get('semantic_domain') == 'business'`, it immediately re-resolves to `"biomedical"` or `"engineering"` using `theme_config.get('scientific_subdomain_weights')` and writes back `theme_config['semantic_domain'] = domain_cat` and `theme_config['effective_is_scientific'] = True` before line 1330.
   * **Weight Propagation Vector**: Because chart generator functions only receive `(ax, theme_config, style_config)` without access to `cfg`, `generator.py` MUST propagate `cfg.get('scientific_subdomain_weights')` into `theme_config['scientific_subdomain_weights']` and `style_config['scientific_subdomain_weights']`.
   * **Lazy Boolean Evaluation**: Avoid `style_config.get('force_dual_axis', random.random() < p)` because Python eagerly evaluates function arguments, burning random entropy even when force flags are supplied. Instead, use:
     `style_config['force_dual_axis'] if 'force_dual_axis' in style_config else (random.random() < DUAL_AXIS_PROBABILITY)`.
   * Heatmaps internally set titles at line 3362 (inside `_generate_heatmap_chart`).
   * **Heatmap Domain Mapping**: In `chart.py:3197`, domain mapping is formalized as:
     ```python
     HEATMAP_CONTEXT_TO_DOMAIN = {
         "genomic_expression_heatmap": "biomedical",
         "pharmacokinetics_heatmap": "biomedical",
         "cohort_retention_heatmap": "business",
         "correlation_matrix": None,  # Dynamically samples from ("biomedical", "engineering") using subdomain weights
     }
     ```
5. **PRNG Determinism in `synth/tabular.py` & Bidirectional Preset Aliasing**:
   * `synth/tabular.py` calls unseeded `np.random.default_rng()` at lines 157, 232, 263, 310, and 392. Deriving `rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))` when `rng is None` at top-level entry points ensures reproducibility without cross-subplot collisions.
   * In `chart.py`, real `sample_multivariate_table` call sites are at lines: **1332, 1459, 1523, 1825, 1944, and 3058**.
   * Precedence MUST respect user config: `domain = syn_domain or theme_config.get('semantic_domain') or ('biomedical' if is_scientific else 'business')`.
   * **Synthetic Data Domain Write-Back**: When `use_synthetic and syn_table is not None`, the recorded metadata domain MUST come from `PRESET_TO_CANONICAL.get(syn_table.get('domain'), syn_table.get('domain'))`, guaranteeing that metadata strictly records the actual synthetic data table generated.
   * **Bidirectional Preset Aliasing**: `DOMAIN_PRESETS` holds legacy keys `{"biomedical", "financial", "demographic", "sensor_telemetry"}`. The rest of the pipeline passes canonical domains `{"business", "engineering"}`. To prevent `KeyError` and stop `sample_domain_schema` from falling back to a random domain, `tabular.py` MUST define:
     - `DOMAIN_ALIAS_TO_PRESET = {"business": "financial", "engineering": "sensor_telemetry"}` for looking up presets from canonical inputs.
     - `PRESET_TO_CANONICAL = {"financial": "business", "sensor_telemetry": "engineering", "demographic": "business"}` for emitting canonical domain strings in output table schemas.
6. **Multi-Subplot Composite Tracking & Additive Schema Parity**:
   * In `generator.py`, domain resolution occurs **inside** the per-subplot loop `for ax_idx, ax in enumerate(axes):` (landmark line 3917, NOT `for ax in fig.axes:` which appears 6 times in the file).
   * On multi-subplot figures, `detailed_metadata["subplots"]` records per-axis metadata for each visible axis in `chart_info_map`.
   * **Preserving Root Scalar Taxonomies**:
     - Setting top-level scalar `chart_type = "composite"` or `semantic_domain = "mixed"` causes severe regressions: `generator.py:4751` sets `metadata_json["chart_types"] = [unified_json.get("chart_type")]`, and `merge_json.py:125` copies it. Emitting `"composite"` clobbers the visual chart type taxonomy for downstream visual classifiers.
     - **The Fix**: Root scalar `chart_type` and `semantic_domain` ALWAYS reflect the primary subplot (`fig.axes[0]` / `axes[0]`), e.g., `"bar"`, `"line"`, `"scatter"`.
     - Multi-subplot figures are tracked cleanly and additively via:
       `"is_composite": True` (when `len(subplots) > 1`),
       `"composite_chart_types": [s["chart_type"] for s in subplots]`,
       `"composite_domains": [s["semantic_domain"] for s in subplots]`.
     - In `generator.py:4751`, set `metadata_json["chart_types"] = detailed_json.get("composite_chart_types", [unified_json.get("chart_type", "unknown")])`.
7. **Bar Chart Call-Site Layout in `chart.py` & Contextual Code Anchoring**:
   * Dual-axis bar logic handles lines 1320–1431 and returns early at line 1430.
   * Standard bar logic begins at line 1432, defining `is_scientific = style_config.get('is_scientific', False)`.
   * `orientation` is not resolved until line 1446: `orientation = 'horizontal' if num_bars > 6 and random.random() < 0.40 else 'vertical'`.
   * Therefore, standard bar axis sampling MUST occur **immediately after `style_config['orientation'] = orientation`** (line 1448 anchor), BEFORE the treatment key check (`if len(bar_info_list) == 4 and ...` at line 1508).
   * **Contextual Anchor Policy**: To eliminate fragility from line drift as Phase 1–5 edits land, all tasks specify contextual AST/code landmarks rather than mutable static line numbers.
8. **Comparative Pairs Architecture, Reachability Reality & Sector Gating**:
   * `add_treatment_key_xaxis` is called at `chart.py:1509`, strictly inside `if is_scientific:`. In the default codebase, the business bar path (`else:` at line 1512) never calls `add_treatment_key_xaxis`, rendering all 22 business pairs (55–69, 87–93) 100% unreachable.
   * **Activation Fix**: Extending `add_treatment_key_xaxis` into the business grouped bar path (within the `else: # Standard Styles` branch, specifically under `style == 'side_by_side'` after `ticks_setter(indices, categories)` at landmark line 1594) when `len(bar_info_list) == 4 and orientation == 'vertical' and random.random() < TREATMENT_KEY_PROBABILITY` activates these 22 business pairs for A/B testing (`Control Group (A)` vs `Variant Group (B)`), marketing campaigns, and SaaS metrics.
   * **Reachability & Probability Reality**:
     - Treatment keys appear only on 4-bar grouped charts with vertical orientation at 30% probability ($P(\text{key} \mid \text{sci bar}) \approx 2.6\%$, $P(\text{key} \mid \text{biz bar}) \approx 2.4\%$). In a balanced 8-chart mix (bar = 15%), treatment keys appear on only 0.38% of images (~38 per 10,000). In line-only default config, 0%.
     - Forcing `< 0.30` alone leaves `len(bar_info_list) == 4` (10%) and vertical orientation (50%) random. Testing all 94 pairs end-to-end would require thousands of forced charts.
     - **The Fix**: 100% reachability across all 94 pairs is tested at the unit/sampler level in `test_semantic_sampling.py` (over 10,000 draws). End-to-end generator tests audit *domain coherence* when keys appear (0% cross-domain leakage).
   * **Simplified Sector Gating**:
     - Sub-domain heuristic matching for physical/materials vs computing within engineering is brittle and unsupported by `AxisMetric` attributes.
     - Sector gating (`biomedical` $\rightarrow$ pairs 0–39; `engineering` $\rightarrow$ pairs 40–54, 70–86; `business` $\rightarrow$ pairs 55–69, 87–93) completely eliminates cross-domain pollution cleanly and reliably.
9. **Inline Probability Literals & Configurable Subdomain Weights**:
   * Line 1320 has hardcoded `< 0.15` (dual-axis) and line 1508 has `< 0.30` (treatment key). Extracted to `DUAL_AXIS_PROBABILITY` and `TREATMENT_KEY_PROBABILITY`.
   * Subdomain resolution in `generator.py:3917` previously hardcoded `[0.70, 0.30]`. Making weights configurable via `cfg.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})` allows users to generate 100% engineering or 100% biomedical datasets via config file without touching source code.
10. **Lexical Diversity Bottleneck, Clean Typography & Explicit Domain Tagging**:
    * Banning nominal categories and pure dependent metrics on line/area $X$ restricts the existing `themes.py` list to only ~20 scientific and ~13 business independent labels. Under 100% line default config, per-label frequency would spike up to $15\times$ baseline.
    * While `themes.py` remains strictly frozen (FR-050) for backward compatibility, `catalog.py` is augmented with ~40 canonical independent metrics (`Date`, `Month`, `Year`, `Quarter`, `Timestamp`, `Cycle Number`, `Epoch`, `Iteration`, `Step`, `Depth (m)`, `Altitude (m)`, `Distance (km)`, `Sample ID`).
    * **Clean Typography (No Format Hints)**: Matplotlib `chart.py` has zero date formatting machinery (`DateFormatter`, `strftime`, `mdates`). Parenthetical hints like `Date (YYYY-MM-DD)` render awkwardly over plain integer ticks. Plain names (`Date`, `Year`, `Month`, `Quarter`) are clean and standard.
    * **Explicit Domain Tagging**: Every canonical variable has explicit `domain_tags`. Specialized computing terms (`Epoch`, `Iteration`, `Step`) are tagged strictly `("engineering",)`, guaranteeing they never land on business or revenue charts.
    * This expands the line/area $X$ pool to 60+ terms, driving maximum per-label frequency down to $\le 5\times (1 / K_{domain})$ baseline under production mix.
11. **Semantic Concept Groups (`concept_group`) & Normalized Stem Algorithm**:
    * `concept_stem` alone only catches orthographic unit variants (`Revenue ($)` vs `Revenue (USD)`), but cannot catch semantic synonyms (`Revenue` vs `Sales`, `Absorbance` vs `Optical Density`).
    * **Normalized Stem Computation Algorithm**: `compute_concept_stem(label: str) -> str` converts the string to lowercase, strips parenthetical units and formats via `re.sub(r'\(.*?\)', '', s)`, strips currency symbols (`$€£¥`), strips punctuation, and collapses consecutive whitespace.
    * Adding `concept_group: Optional[str] = None` to `AxisMetric` clusters synonymous terms (`revenue_metric`, `time_duration`, `optical_density`, `thermal_state`, `concentration_metric`). Sampler rejects axis pairs where `metric_x.concept_group == metric_y.concept_group`, guaranteeing zero semantic synonym collisions with $O(1)$ complexity.
12. **Small-Pool Fallback in `sample_axis_pair`**:
    * If filtered candidate pools after domain, role, and scale filtering are empty on tightly constrained queries, `sample_axis_pair` falls back gracefully to the domain's general independent/continuous catalog pool, preventing `IndexError`.
13. **Vega-Lite Backend Parity & Null Domain Semantics**:
    * `syn_table` is initialized to `None` at function entry in `vegalite_backend.py`.
    * Non-synthetic Vega-Lite charts (fallback mode without synthetic tabular engine) explicitly emit `semantic_domain: null` and boolean `is_scientific: False`.

---

## 2. Refined Architectural Consensus

1. **Universal Domain Resolution & Configurable Propagation**:
   * Respect `cfg['bar_chart_config']['scientific_ratio']` (default 0.60) to determine `is_scientific` first.
   * Subplot domain resolution in `generator.py` (contextual anchor: inside `for ax_idx, ax in enumerate(axes):` at line 3917):
     - Inject `theme_config['scientific_subdomain_weights'] = cfg.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})` and `style_config['scientific_subdomain_weights'] = theme_config['scientific_subdomain_weights']`.
     - If `is_scientific`: draw `semantic_domain` from `("biomedical", "engineering")` with weights `theme_config['scientific_subdomain_weights']`.
     - If not `is_scientific`: set `semantic_domain = "business"`.
   * In `sample_axis_pair(chart_type, orientation, is_scientific, semantic_domain=None, rng=None)`:
     - Returns `(x_label, y_label, domain_category)`.
     - Called at all 17 axis sampling sites in `chart.py`, passing `semantic_domain=theme_config.get('semantic_domain')` guarded by `if isinstance(theme_config, dict):`.
     - Every call site writes back `theme_config['semantic_domain'] = domain_category`.
     - Dual-axis bar charts pass `domain_category` to `sample_secondary_y(y1_label, is_scientific, domain_category=domain_category)`. If `semantic_domain == "business"`, re-resolve using `theme_config.get('scientific_subdomain_weights')`.
     - In synthetic table generation, all 6 `sample_multivariate_table` calls receive `domain=syn_domain or theme_config.get('semantic_domain') or ('biomedical' if is_scientific else 'business')`. Inside `synth/tabular.py`, incoming domain strings are mapped via `DOMAIN_ALIAS_TO_PRESET = {"business": "financial", "engineering": "sensor_telemetry"}` and table metadata emits canonical names via `PRESET_TO_CANONICAL = {"financial": "business", "sensor_telemetry": "engineering", "demographic": "business"}`.
     - When `use_synthetic and syn_table is not None`: write back `theme_config['semantic_domain'] = PRESET_TO_CANONICAL.get(syn_table['domain'], syn_table['domain'])`.
   * In `add_treatment_key_xaxis`:
     - Called in both scientific (line 1509) and business (within the `else: # Standard Styles` branch at landmark line 1594) grouped bar branches.
     - Sector gating: `"biomedical"` draws from pairs 0–39; `"engineering"` from pairs 40–54 and 70–86; `"business"` from pairs 55–69 and 87–93.
     - All 94 pairs are 100% active and reachable (verified at sampler level).
2. **Role-Based and Scale-Constrained Gating with Canonical Expansion**:
   * Measured bench variables kept as `ScaleType.CONTINUOUS` or `TEMPORAL` with `AxisRole.BIDIRECTIONAL`.
   * Augmented `catalog.py` includes ~40 canonical independent variables with clean typography and explicit domain tags.
   * **Scatter**: Quantitative metrics on both axes (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT, TEMPORAL)`), strictly banning nominal dimensions (`CATEGORICAL`).
   * **Vertical Bar / Box**: $X$ requires `role in (INDEPENDENT, BIDIRECTIONAL)`; $Y$ requires quantitative `role in (DEPENDENT, BIDIRECTIONAL)` (`scale_type in (CONTINUOUS, PERCENTAGE, DISCRETE_COUNT)`).
   * **Horizontal Bar / Box**: Roles inverted.
   * **Line / Area**: $X$ requires `role in (INDEPENDENT, BIDIRECTIONAL)` **AND** `scale_type in (CONTINUOUS, TEMPORAL, DISCRETE_COUNT)`; $Y$ requires quantitative `role in (DEPENDENT, BIDIRECTIONAL)`.
3. **Dedicated Secondary Axis Sampler (`sample_secondary_y`)**:
   * Filters candidate continuous dependent/bidirectional metrics from `domain_category` whose `concept_stem != y1_stem` and `concept_group != y1_group`.
4. **Calibrated In-Memory Title Benchmark ($N = 20,000$) & Algorithmic Guarantees**:
   * Pre-index titles by `(is_scientific, chart_type)` for verified $O(1)$ sampling (no wall-clock fragility). Heatmap-only titles (~5.6%) excluded from Cartesian fallbacks.
   * Top up pools with $< 6$ titles from domain-generic titles.
   * Benchmark title distributions over $N = 20,000$ in-memory draws: maximum frequency $\le 5\times$ baseline ($p \le 3.5\%$) under balanced multi-chart mix.
   * Axis-label frequency benchmark measured at production mix: maximum line $X$ frequency $\le 5\times (1 / K_{domain})$ baseline.
5. **Independent Test Oracles & Double-Annotated Gold Set**:
   * Independent continuous regex paired with an **independent 50-item blind gold set** annotating both `scale_type` and `role`.
   * Test fixtures force all 8 chart types using `style_config['force_dual_axis']` and `['force_treatment_key']`.
6. **Shared Version Constants (`versions.py`) & Standardized Sentinels**:
   * Standalone `versions.py` defining `ANNOTATION_SCHEMA_VERSION = "v2.1"` and `DATASET_VERSION = "2.1.0"`.
   * Standardize unassigned `semantic_domain` to `None` (`null` in JSON) across backends.
   * Deterministic tuple construction: all catalog pools constructed via `tuple(sorted(...))` to guarantee determinism across `PYTHONHASHSEED`.
   * Composite dashboard tracking: top-level scalars reflect primary subplot (`axes[0]`), while `"is_composite": True`, `"composite_chart_types"`, and `"composite_domains"` capture multi-subplot structures additively.
