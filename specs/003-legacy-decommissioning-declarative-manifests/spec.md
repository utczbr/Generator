# Feature Specification: Legacy Decommissioning & Declarative Semantic Manifests

**Feature Branch**: `003-legacy-decommissioning-declarative-manifests`  
**Schema Version**: `v3.0` (`dataset_version = "3.0.0"`)  
**Status**: Proposed / Roadmap Complete  
**Predecessor Feature**: `002-semantic-storage-registry` (Schema Version `v2.1`)  
**Primary Artifacts Touched**: `themes.py`, `synth/semantics/domains/*.yaml` (new), `synth/semantics/schema.py` (new), `synth/semantics/loader.py` (new), `synth/semantics/catalog.py`, `synth/tabular.py`, `versions.py`, `SCHEMA.md`, `tests/`

---

## Overview

Feature 002 solved the critical runtime incoherence vectors and serialization drop-points of the synthetic chart pipeline by establishing `synth/semantics/` as the sole runtime authority for axis, title, and comparative pair sampling. To ensure non-breaking rollout, Feature 002 maintained backward compatibility through a **Strangler Fig** approach, leaving legacy semantic data frozen in `themes.py` ([`FR-050`](../002-semantic-storage-registry/spec.md#requirements)) and employing bidirectional alias dictionaries in `synth/tabular.py`.

Feature 003 is the **clean-state decommissioning and declarative promotion feature**. It formalizes the complete retirement of legacy data structures once data stability milestones are met:

1. **Purge Procedural Constants from `themes.py`**: Remove all 264 raw axis labels, 143 chart titles, 94 comparative pairs, 51 histogram labels, dead heatmap/colorbar labels, and dead `*_DOMAIN_DICT` / `CONTEXT_CONFIGURATIONS` / `STRUCTURAL_THEMES` structures by purging Zone 1 (lines 1–318) and Zone 3 (lines 767–1106). This eliminates 658 lines of dead code and procedural data (59.5% reduction), leaving `themes.py` with strictly visual styling themes (`THEMES`, `PUBLICATION_THEMES`, `FONT_FAMILIES`, total length $\le 460$ lines).
2. **Promote Declarative YAML Domain Manifests**: Move domain definitions to modular YAML files (`synth/semantics/domains/*.yaml`), validated at build/test time via Pydantic v2 models and compiled at startup into frozen slotted dataclasses for verified $O(1)$ memory access.
3. **Clean Up Synthetic Tabular Engine**: Rename `DOMAIN_PRESETS` keys in `synth/tabular.py` directly to `"business"` and `"engineering"`, eliminating alias shims while retaining domain copula parameters in the tabular engine.
4. **Decommission Compatibility Test Shims**: Retire legacy compatibility tests, decouple non-legacy tests from legacy constants, and emit Schema Version `v3.0` (`dataset_version = "3.0.0"`).

---

## Sunset Gate Prerequisites

Feature 003 execution is strictly blocked until the following three gates pass:

* **SG-001 (Dataset Stability Milestone)**: The generator pipeline must have successfully rendered and saved at least 50,000 synthetic chart images and annotations under Schema Version `v2.1` (`dataset_version = "2.1.0"`).
* **SG-002 (Zero Legacy Call Sites - AST Verification)**: Running `scripts/audit_legacy_references.py` must verify that **zero** production files in the repository import or reference legacy semantic constants from `themes.py` (`SCIENTIFIC_*`, `BUSINESS_*`, `CHART_TITLES`, `COMPARATIVE_LABELS`, `HISTOGRAM_Y_LABELS`, `*_DOMAIN_DICT`, `HEATMAP_*`, `COLORBAR_*`, `CONTEXT_CONFIGURATIONS`, `STRUCTURAL_THEMES`).
  * *Prerequisite cleanup*: Lingering unused header imports in `generator.py:46` and `chart.py:64` must be removed, and `synth/semantics/sampler.py:94` must be decoupled from `themes.HISTOGRAM_Y_LABELS` before AST verification can pass.
* **SG-003 (Downstream Consumer Readiness)**: Downstream OCR, layout parsing, and multimodal vision-language model training pipelines must confirm production ingestion of Schema Version `v2.1` metadata (`semantic_domain`, `subplots`, `is_composite`).

---

## User Scenarios & Acceptance Criteria

### User Stories

1. **As a domain specialist or research contributor**, I want to add new scientific or commercial vocabulary terms, concept groups, and comparative pairs by editing declarative YAML files, so that I do not have to modify Python source code or risk git merge conflicts in rendering engines.
2. **As a platform maintainer**, I want `themes.py` to contain only visual styling configurations (palettes, fonts, spines, gridlines), so that the module adheres strictly to the single-responsibility principle.
3. **As a synthetic data engineer**, I want `synth/tabular.py` to use canonical domain names (`"business"`, `"engineering"`, `"biomedical"`, `"demographic"`) directly without translation dictionaries or preset aliasing.

### Acceptance Criteria

1. **Given** the repository state after Feature 003, **When** inspecting `themes.py`, **Then** the file contains only the `THEMES`, `PUBLICATION_THEMES`, and `FONT_FAMILIES` definitions and visual styling utilities (length $\le 460$ lines), with zero axis label strings, chart titles, comparative pairs, or dead domain dicts.
2. **Given** domain manifests in `synth/semantics/domains/`, **When** running test suites or CI, **Then** all YAML files strictly pass Pydantic v2 schema validation, catching invalid enums, duplicate labels, or malformed concept groups.
3. **Given** application cold startup, **When** `synth/semantics` loads the domain manifests, **Then** the compiled in-memory graph of frozen slotted dataclasses is instantiated in $\le 50\,\text{ms}$ (using pure `yaml.CSafeLoader` and dataclasses without runtime top-level Pydantic imports), maintaining verified $O(1)$ lookup performance.
4. **Given** execution of `synth/tabular.py:sample_domain_schema`, **When** `domain="business"` or `domain="engineering"` is passed, **Then** the preset is resolved directly from `DOMAIN_PRESETS` without intermediate alias mappings.
5. **Given** any generated `<image_id>_detailed.json` or `<image_id>_unified.json`, **When** inspected on disk, **Then** `"schema_version"` is `"v3.0"` and `"dataset_version"` is `"3.0.0"`.

---

## Requirements

### Sunset Verification & Audit
* **FR-060 (Automated AST Reference Audit)**: The system MUST provide `scripts/audit_legacy_references.py` that parses the Abstract Syntax Tree (AST) of all Python files in the workspace. It MUST return exit code 0 if and only if zero imports or symbol references to `SCIENTIFIC_X_LABELS`, `SCIENTIFIC_Y_LABELS`, `BUSINESS_X_LABELS`, `BUSINESS_Y_LABELS`, `CHART_TITLES`, `COMPARATIVE_LABELS`, `HISTOGRAM_Y_LABELS`, `SCIENTIFIC_DOMAIN_DICT`, `BUSINESS_DOMAIN_DICT`, `HEATMAP_XLABELS_SCIENTIFIC`, `HEATMAP_YLABELS_SCIENTIFIC`, `HEATMAP_XLABELS_BUSINESS`, `HEATMAP_YLABELS_BUSINESS`, `COLORBAR_TITLES_SCIENTIFIC`, `COLORBAR_TITLES_BUSINESS`, `CONTEXT_CONFIGURATIONS`, or `STRUCTURAL_THEMES` exist outside `tests/test_legacy_themes_compatibility.py`.
  * The AST audit MUST inspect both `ImportFrom` nodes and `Attribute` accesses (`themes.<CONSTANT>`).

### Declarative YAML Manifests
* **FR-061 (Modular YAML Domain Manifests)**: The system MUST define domain catalogs in modular YAML files under `synth/semantics/domains/`:
  - `biomedical.yaml`: Clinical, pharmacokinetics, molecular biology, and physiological metrics, titles, and comparative pairs.
  - `engineering.yaml`: Materials science, mechanics, control systems, computing, ML, and telemetry metrics, titles, and pairs.
  - `business.yaml`: Financial, SaaS, marketing, A/B testing, and operational metrics, titles, and pairs.
  - `demographic.yaml`: Population, geography, education, and demographic dimensions and metrics.
  - `common.yaml`: Canonical independent variables (dates, years, steps), frequency/density histogram metrics, and domain-generic chart titles.
* **FR-062 (Pydantic v2 Domain Schema)**: The system MUST define Pydantic v2 models in `synth/semantics/schema.py`:
  - `MetricDefinition`: `label: str`, `scale_type: ScaleType`, `role: AxisRole`, `concept_stem: str`, `unit: Optional[str] = None`, `concept_group: Optional[str] = None`, `domain_tags: List[str] = Field(default_factory=list)`.
  - `TitleDefinition`: `title: str`, `allowed_chart_types: List[str]`, `domain_tags: List[str] = Field(default_factory=list)`.
  - `PairDefinition`: `control: str`, `treatment: str`, `domain_category: str`, `context_tags: List[str] = Field(default_factory=list)`.
  - `DomainManifest`: `domain_id: str`, `display_name: str`, `is_scientific: bool`, `default_weight: float = 1.0`, `metrics: List[MetricDefinition]`, `titles: List[TitleDefinition]`, `comparative_pairs: List[PairDefinition]`.
* **FR-063 (Build-Time Manifest Validation)**: The test suite MUST validate all YAML files against `DomainManifest`. Any missing field, invalid enum string, duplicate label, or malformed concept stem MUST raise a validation error and fail CI.

### Startup Compilation & Memory Layer
* **FR-064 (In-Memory Compilation to Frozen Dataclasses)**: The system MUST provide `synth/semantics/loader.py` that parses the YAML manifests at application startup and compiles them into immutable, slotted dataclasses (`AxisMetric`, `ChartTitle`, `ComparativePair`).
  - `AxisMetric` MUST define `domain_tags: Tuple[str, ...] = ()` (rather than a scalar `domain_category`) to ensure canonical independent metrics (`Date`, `Year`, `Depth`) can be resolved across multiple domains, preserving compatibility with `sampler.py`.
* **FR-065 (Zero-Runtime-Overhead Guarantee)**: Runtime sampling in `synth/semantics/sampler.py` MUST NOT parse YAML files or execute Pydantic validators during rendering loops. All lookups MUST operate on pre-compiled dictionaries and tuples in verified $O(1)$ time.
* **FR-066 (Startup Latency Budget & Dual-Mode Isolation)**: Cold startup compilation of all domain manifests MUST execute in $\le 50\,\text{ms}$ on standard development hardware.
  - `synth/semantics/loader.py` MUST NOT import `pydantic` at module top-level, avoiding the 87–114 ms Pydantic cold import penalty. Pydantic validation is strictly test-time / build-time via `schema.py`.
  - `loader.py` MUST use pure `yaml.CSafeLoader` and built-in slotted dataclasses at runtime.

### Decommissioning & Cleanup
* **FR-067 (Purge Legacy Constants from `themes.py`)**: The system MUST delete lines 1–318 (Zone 1: semantic constants) and lines 767–1106 (Zone 3: dead heatmap lists, domain dicts, context configurations, structural themes) of `themes.py`, purging 658 lines of dead code and procedural data.
* **FR-068 (Single-Responsibility `themes.py`)**: Post-decommissioning `themes.py` MUST contain strictly visual styling themes (`THEMES`), publication themes (`PUBLICATION_THEMES`), font definitions (`FONT_FAMILIES`), spine settings, and gridline styling (Zone 2, lines 320–765). Its total length MUST NOT exceed 460 lines.
* **FR-069 (Canonical Naming in `synth/tabular.py`)**: `synth/tabular.py` MUST directly rename `DOMAIN_PRESETS` keys:
  - `"financial"` $\rightarrow$ `"business"`
  - `"sensor_telemetry"` $\rightarrow$ `"engineering"`
  The alias dictionaries `DOMAIN_ALIAS_TO_PRESET` and `PRESET_TO_CANONICAL` MUST be deleted. Tabular multivariate Gaussian copula parameters remain encapsulated within `synth/tabular.py`.
* **FR-070 (Schema Version 3.0 Emission)**: `versions.py` MUST be updated to `ANNOTATION_SCHEMA_VERSION = "v3.0"` and `DATASET_VERSION = "3.0.0"`. All outputs (`detailed.json`, `unified.json`, Vega-Lite `metadata.json`, and `ocr.json`) MUST emit Schema Version `v3.0`.
* **FR-071 (Retire Compatibility Test Shims & Decouple Existing Tests)**:
  - `tests/test_legacy_themes_compatibility.py` MUST be deleted.
  - `tests/test_title_sampling.py`, `tests/test_semantic_sampling.py`, and `tests/test_semantic_catalog.py` MUST be updated to import canonical symbols from `synth/semantics/catalog.py` instead of deleted `themes.py` constants.
* **FR-072 (Dependency Management)**: `requirements.txt` MUST explicitly specify `PyYAML>=6.0` and `pydantic>=2.0.0` to ensure seamless CI environment reproducibility.

---

## Key Entities & YAML Schema

### Pydantic v2 Schema (`synth/semantics/schema.py`)

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


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


class MetricDefinition(BaseModel):
    label: str
    scale_type: ScaleType
    role: AxisRole
    concept_stem: str
    unit: Optional[str] = None
    concept_group: Optional[str] = None
    domain_tags: List[str] = Field(default_factory=list)


class TitleDefinition(BaseModel):
    title: str
    allowed_chart_types: List[str]
    domain_tags: List[str] = Field(default_factory=list)


class PairDefinition(BaseModel):
    control: str
    treatment: str
    domain_category: str
    context_tags: List[str] = Field(default_factory=list)


class DomainManifest(BaseModel):
    domain_id: str
    display_name: str
    is_scientific: bool
    default_weight: float = 1.0
    metrics: List[MetricDefinition] = Field(default_factory=list)
    titles: List[TitleDefinition] = Field(default_factory=list)
    comparative_pairs: List[PairDefinition] = Field(default_factory=list)
```

### Example Manifest: `synth/semantics/domains/business.yaml`

```yaml
domain_id: business
display_name: "Business, Finance & SaaS Analytics"
is_scientific: false
default_weight: 1.0

metrics:
  - label: "Revenue ($)"
    scale_type: continuous
    role: dependent
    concept_stem: revenue
    concept_group: revenue_metric
    unit: "$"
    domain_tags: ["finance", "saas"]

  - label: "Sales ($)"
    scale_type: continuous
    role: dependent
    concept_stem: sales
    concept_group: revenue_metric
    unit: "$"
    domain_tags: ["sales", "retail"]

  - label: "Fiscal Quarter"
    scale_type: temporal
    role: independent
    concept_stem: fiscal_quarter
    domain_tags: ["accounting", "reporting"]

titles:
  - title: "Quarterly Revenue Breakdown"
    allowed_chart_types: ["bar", "line", "area"]
  - title: "Customer Acquisition Funnel"
    allowed_chart_types: ["bar", "pie"]

comparative_pairs:
  - control: "Control Group (A)"
    treatment: "Variant Group (B)"
    domain_category: business
    context_tags: ["ab_testing"]
```

---

## Success Criteria

* **SC-030 (Legacy Data Elimination)**: `themes.py` contains zero semantic constants or dead dictionaries. File size is reduced by $\ge 58\%$ (total lines $\le 460$, from 1,106 lines to ~450 lines by deleting Zone 1 lines 1–318 and Zone 3 lines 767–1106).
* **SC-031 (Declarative Completeness)**: 100% of the 264 legacy axis metrics, ~40 canonical independent metrics, 143 chart titles, and 94 comparative pairs are defined in YAML manifests under `synth/semantics/domains/` and pass Pydantic v2 validation.
* **SC-032 (Startup Latency Integrity)**: Cold startup compilation from YAML manifests into in-memory frozen dataclasses executes in $\le 50\,\text{ms}$ with zero top-level Pydantic runtime import overhead.
* **SC-033 (Zero Tabular Translation Overhead)**: `synth/tabular.py` resolves domain schemas directly from `DOMAIN_PRESETS` without alias mappings or fallback dictionaries.
* **SC-034 (Schema v3.0 Ground-Truth Emission)**: 100% of emitted annotations carry `"schema_version": "v3.0"` and `"dataset_version": "3.0.0"`.
* **SC-035 (Pipeline Runtime Zero Regressions)**: All 8 chart types render successfully with zero semantic errors, zero axis collisions, and 100% domain coherence across forced multi-chart test runs.
