# Tasks: Legacy Decommissioning & Declarative Semantic Manifests

**Input**: `specs/003-legacy-decommissioning-declarative-manifests/spec.md`, `specs/003-legacy-decommissioning-declarative-manifests/plan.md`, `specs/003-legacy-decommissioning-declarative-manifests/research.md`  
**Legend**: `[P]` = can be executed in parallel within phase. Tasks without `[P]` are sequential. Every task includes a concrete **Done when** acceptance criterion.

---

## Phase 1: Sunset Audit & AST Verification Engine (COMPLETED)

- [x] **T199 [P]** Clean up lingering legacy header imports and decouple `sampler.py`:
  * In `generator.py:46`, remove unused legacy imports (`CHART_TITLES`).
  * In `chart.py:64`, remove unused legacy imports.
  * In `synth/semantics/sampler.py:94`, replace `themes.HISTOGRAM_Y_LABELS` with import from `synth.semantics.catalog.HISTOGRAM_Y_LABELS`.  
  **Done when**: `python -c "import generator, chart, synth.semantics.sampler"` executes cleanly, and `git grep "themes\.HISTOGRAM_Y_LABELS"` returns 0 results. (Verified: Clean execution, 0 legacy references).

- [x] **T200 [P]** Create `scripts/audit_legacy_references.py` using Python's standard `ast` module to scan the repository for legacy semantic references:
  * Parse all `.py` files in the repository (excluding `tests/`).
  * Identify any `ImportFrom` node where `node.module == "themes"` importing any of:
    `SCIENTIFIC_X_LABELS`, `SCIENTIFIC_Y_LABELS`, `BUSINESS_X_LABELS`, `BUSINESS_Y_LABELS`, `CHART_TITLES`, `COMPARATIVE_LABELS`, `HISTOGRAM_Y_LABELS`, `SCIENTIFIC_DOMAIN_DICT`, `BUSINESS_DOMAIN_DICT`, `HEATMAP_XLABELS_SCIENTIFIC`, `HEATMAP_YLABELS_SCIENTIFIC`, `HEATMAP_XLABELS_BUSINESS`, `HEATMAP_YLABELS_BUSINESS`, `COLORBAR_TITLES_SCIENTIFIC`, `COLORBAR_TITLES_BUSINESS`, `CONTEXT_CONFIGURATIONS`, `STRUCTURAL_THEMES`.
  * Identify any `Attribute` access where `node.value.id == "themes"` and `node.attr` matches any banned constant name.
  * Print each violation with `filepath:line:column: symbol_name`.
  * Return exit code 0 if 0 violations are found; return exit code 1 if $\ge 1$ violation is found.  
  **Done when**: `python scripts/audit_legacy_references.py` executes cleanly without crashing, accurately flagging synthetic test violations and returning 0 when no violations exist. (Verified: Exit 0 on clean code, Exit 1 with diagnostics on synthetic test).

- [x] **T201** Formally evaluate and verify Sunset Gates SG-001, SG-002, and SG-003:
  * SG-001: Query dataset metadata store to confirm $\ge 50,000$ synthetic chart images and annotations have been successfully generated and stored under Schema Version `v2.1` (`dataset_version = "2.1.0"`).
  * SG-002: Execute `python scripts/audit_legacy_references.py` to verify zero production call sites reference legacy constants.
  * SG-003: Collect formal downstream sign-offs confirming that OCR, layout parsing, and multimodal model training pipelines ingest Schema `v2.1` metadata (`semantic_domain`, `subplots`, `is_composite`).  
  **Done when**: All three Sunset Gates pass and documented sign-off is logged, unblocking Phase 2 execution. (Verified: Audit report documented in `audit_report.md`, Phase 2 unblocked).

---

## Phase 2: Declarative Manifests, Pydantic v2 Schema & Fast Compiler

- [X] **T202 [P]** Implement Pydantic v2 validation schema in `synth/semantics/schema.py` and register dependencies:
  * Update `requirements.txt` to explicitly include `PyYAML>=6.0` and `pydantic>=2.0.0`.
  * Define enums:
    ```python
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
    ```
  * Define models:
    - `MetricDefinition`: `label: str`, `scale_type: ScaleType`, `role: AxisRole`, `concept_stem: str`, `unit: Optional[str] = None`, `concept_group: Optional[str] = None`, `domain_tags: List[str] = Field(default_factory=list)`.
    - `TitleDefinition`: `title: str`, `allowed_chart_types: List[str]`, `domain_tags: List[str] = Field(default_factory=list)`.
    - `PairDefinition`: `control: str`, `treatment: str`, `domain_category: str`, `context_tags: List[str] = Field(default_factory=list)`.
    - `DomainManifest`: `domain_id: str`, `display_name: str`, `is_scientific: bool`, `default_weight: float = 1.0`, `metrics: List[MetricDefinition]`, `titles: List[TitleDefinition]`, `comparative_pairs: List[PairDefinition]`.
  * Add field validators:
    - `concept_stem`: regex `^[a-z0-9_]+$`.
    - Disallow empty strings for `label`, `title`, `control`, `treatment`.  
  **Done when**: `python -c "from synth.semantics.schema import DomainManifest; print(DomainManifest)"` executes without error and properly validates valid and invalid sample payloads.

- [X] **T203 [P]** Author declarative YAML domain manifests under `synth/semantics/domains/`:
  * `biomedical.yaml`: Migrate all clinical, pharmacokinetics, molecular biology, and physiological terms (122 metrics, 42 titles, 40 comparative pairs).
  * `engineering.yaml`: Migrate all materials science, mechanics, control systems, computing, ML, and telemetry terms (136 metrics, 38 titles, 30 comparative pairs).
  * `business.yaml`: Migrate all financial, SaaS, marketing, A/B testing, and operational terms (124 metrics, 48 titles, 24 comparative pairs).
  * `demographic.yaml`: Migrate population, geography, education, and demographic dimensions (22 metrics, 15 titles).
  * `common.yaml`: Migrate canonical independent variables (Date, Year, Month, Step, Depth, etc.), 51 histogram frequency/density metrics, and domain-generic titles.  
  **Done when**: All 5 YAML files exist under `synth/semantics/domains/`, contain valid YAML syntax, and cover 100% of the legacy semantic vocabulary plus canonical expansions.

- [X] **T204** Implement manifest validation test suite in `tests/test_domain_manifests.py`:
  * Load all `.yaml` files in `synth/semantics/domains/`.
  * Validate each file against `DomainManifest` via `model_validate`.
  * Assert zero duplicate labels across the entire semantic catalog.
  * Assert that all comparative pairs have distinct `control` and `treatment` values.
  * Assert that all `concept_group` and `concept_stem` strings adhere to identifier naming conventions.  
  **Done when**: `pytest tests/test_domain_manifests.py` passes 100% with all 5 manifests validated.

- [X] **T205** Implement high-performance startup loader in `synth/semantics/loader.py`:
  * Use `yaml.CSafeLoader` when LibYAML is available, falling back cleanly to `yaml.SafeLoader`.
  * Ensure `loader.py` does NOT import `pydantic` at module top-level to avoid the ~100 ms cold import penalty.
  * Load all manifests under `synth/semantics/domains/`.
  * Compile raw manifest entries into immutable slotted dataclasses:
    ```python
    @dataclass(frozen=True, slots=True)
    class AxisMetric:
        label: str
        scale_type: ScaleType
        role: AxisRole
        concept_stem: str
        unit: Optional[str] = None
        concept_group: Optional[str] = None
        domain_tags: Tuple[str, ...] = ()
        is_scientific: bool = False
    ```
  * Pre-index lookup tables:
    - `METRIC_BY_LABEL: Dict[str, AxisMetric]`
    - `METRICS_BY_DOMAIN: Dict[str, Tuple[AxisMetric, ...]]`
    - `METRICS_BY_ROLE_AND_SCALE: Dict[Tuple[str, AxisRole, ScaleType], Tuple[AxisMetric, ...]]`
    - `TITLES_BY_DOMAIN_AND_CHART: Dict[Tuple[str, str], Tuple[str, ...]]`
    - `PAIRS_BY_DOMAIN: Dict[str, Tuple[ComparativePair, ...]]`
  * Cache the compiled `SemanticRegistry` as a module-level singleton.  
  **Done when**: `from synth.semantics.loader import registry; assert len(registry.metrics_by_label) >= 400` executes instantly on import with zero top-level pydantic imports.

- [X] **T206** Refactor `synth/semantics/catalog.py` to consume the compiled `loader.registry`:
  * Replace static in-memory list definitions in `catalog.py` with references to `loader.registry`.
  * Retain all exported constants and function signatures (`METRIC_BY_LABEL`, `SCIENTIFIC_AXIS_METRICS`, `BUSINESS_AXIS_METRICS`, `CHART_TITLES_BY_TYPE`, `COMPARATIVE_PAIRS_BY_DOMAIN`) to guarantee complete backward compatibility with `synth/semantics/sampler.py`.  
  **Done when**: `pytest tests/test_semantic_catalog.py tests/test_semantic_sampling.py` pass without modifying a single line of `sampler.py`.

- [X] **T207 [P]** Implement cold startup latency benchmark in `tests/test_startup_latency.py`:
  * Launch an isolated Python process using `subprocess.run`.
  * Measure wall-clock time required to execute:
    `python -c "import time; t0 = time.perf_counter(); from synth.semantics.loader import registry; print(f'ELAPSED:{time.perf_counter() - t0}')"`
  * Assert that total elapsed compilation time is strictly $\le 50.0\,\text{ms}$.  
  **Done when**: `pytest tests/test_startup_latency.py` passes and reports cold startup latency $\le 50\,\text{ms}$ (expected $\approx 24.5\,\text{ms}$).

---

## Phase 3: Procedural Purge, Tabular Direct Naming & Schema v3.0 Promotion

- [x] **T208** Purge legacy semantic constants from `themes.py`:
  * Delete Zone 1 (lines 1–318): `SCIENTIFIC_X_LABELS`, `SCIENTIFIC_Y_LABELS`, `BUSINESS_X_LABELS`, `BUSINESS_Y_LABELS`, `CHART_TITLES`, `COMPARATIVE_LABELS`, `HISTOGRAM_Y_LABELS`.
  * Preserve Zone 2 (lines 320–765): `THEMES` (16 styling dicts), `PUBLICATION_THEMES`, `FONT_FAMILIES`.
  * Delete Zone 3 (lines 767–1106): `HEATMAP_*`, `COLORBAR_*`, `SCIENTIFIC_DOMAIN_DICT`, `BUSINESS_DOMAIN_DICT`, `CONTEXT_CONFIGURATIONS`, `STRUCTURAL_THEMES`.
  * Verify remaining file contains strictly visual styling helpers (total length $\le 460$ lines).  
  **Done when**: `wc -l themes.py` $\le 460$, and `hasattr(themes, "SCIENTIFIC_X_LABELS")` is `False`. (Verified: 443 lines, hasattr=False).

- [x] **T209** Clean up synthetic tabular engine in `synth/tabular.py`:
  * In `synth/tabular.py`, rename keys in `DOMAIN_PRESETS`:
    - `"financial"` $\rightarrow$ `"business"`
    - `"sensor_telemetry"` $\rightarrow$ `"engineering"`
  * Delete `DOMAIN_ALIAS_TO_PRESET` and `PRESET_TO_CANONICAL` dictionaries.
  * In `sample_domain_schema(domain, ...)`:
    - Update resolution to direct lookup: `preset = DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS["business"])`.
  * Retain domain multivariate Gaussian copula parameters within `synth/tabular.py`.  
  **Done when**: Calling `sample_domain_schema("business")` and `sample_domain_schema("engineering")` generates correct tables without alias translation dictionaries. (Verified: direct lookups, 5/5 tabular tests pass).

- [x] **T210 [P]** Update `versions.py` and `SCHEMA.md` to Schema `v3.0`:
  * In `versions.py`, update version constants:
    ```python
    ANNOTATION_SCHEMA_VERSION: str = "v3.0"
    DATASET_VERSION: str = "3.0.0"
    ```
  * In `SCHEMA.md`:
    - Update schema specification to `v3.0`.
    - Document the declarative YAML manifest structure and fields.
    - Document that legacy constants in `themes.py` have been purged.  
  **Done when**: `from versions import ANNOTATION_SCHEMA_VERSION, DATASET_VERSION; assert ANNOTATION_SCHEMA_VERSION == "v3.0" and DATASET_VERSION == "3.0.0"` passes, and `SCHEMA.md` documents `v3.0`. (Verified: v3.0 / 3.0.0 exported and documented).

- [x] **T211** Re-run AST audit script to verify complete repository cleanliness:
  * Run `python scripts/audit_legacy_references.py`.
  * Verify exit code 0 and zero warnings.  
  **Done when**: The audit confirms zero imports or attribute accesses of deleted `themes.py` constants across the entire workspace. (Verified: Exit code 0, 0 violations).

---

## Phase 4: Compatibility Retirement & Comprehensive Pipeline Verification

- [x] **T212** Retire legacy compatibility tests:
  * Delete `tests/test_legacy_themes_compatibility.py`.  
  **Done when**: `tests/test_legacy_themes_compatibility.py` no longer exists on disk or in version control. (Verified: file deleted).

- [x] **T212b** Decouple existing test suites from `themes.py`:
  * In `tests/test_title_sampling.py`, replace `from themes import CHART_TITLES` with imports from `synth/semantics/catalog.py`.
  * In `tests/test_semantic_sampling.py`, replace `themes.HISTOGRAM_Y_LABELS` with imports from `synth/semantics/catalog.py`.
  * In `tests/test_semantic_catalog.py`, replace `themes.SCIENTIFIC_X_LABELS` and `themes.BUSINESS_X_LABELS` with canonical catalog assertions.  
  **Done when**: `pytest tests/test_title_sampling.py tests/test_semantic_sampling.py tests/test_semantic_catalog.py` pass without referencing deleted `themes.py` attributes. (Verified: all 35 tests pass cleanly).

- [x] **T213** Run complete test suite regression:
  * Run `pytest tests/ -v`.
  * Verify that all unit and integration tests pass without failures:
    - `test_domain_manifests.py`
    - `test_startup_latency.py`
    - `test_semantic_catalog.py`
    - `test_semantic_sampling.py`
    - All existing chart generation and tabular tests.  
  **Done when**: 100% of test cases pass with zero failures or unexpected errors. (Verified: 103/103 tests pass in 31.03s).

- [x] **T214** Execute end-to-end multi-chart generation validation:
  * Run `python generator.py` to generate 500 images across all 8 chart types with forced multi-chart mixes, dual-axis charts, and treatment keys.
  * Validate all generated `<image_id>_detailed.json` and `<image_id>_unified.json` files:
    - Confirm `"schema_version"` is `"v3.0"`.
    - Confirm `"dataset_version"` is `"3.0.0"`.
    - Confirm `"semantic_domain"` matches the physical domain.
    - Confirm zero unmapped or placeholder labels.  
  **Done when**: 500 charts render and serialize successfully with 100% metadata compliance. (Verified: scripts/verify_e2e_500_charts.py executed 500 charts with 100% metadata compliance).

- [x] **T215 [P]** Wire AST audit and manifest validation into CI workflow:
  * Add `python scripts/audit_legacy_references.py` as a required pre-commit / CI gate.
  * Add `pytest tests/test_domain_manifests.py tests/test_startup_latency.py` to automated test runs.  
  **Done when**: CI configuration file includes the new audit and manifest validation steps. (Verified: .github/workflows/ci.yml created and configured).
