# 003 — Legacy Decommissioning & Declarative Semantic Manifests

Spec-Driven Development artifacts for the **Post-Migration Sunset & Decommissioning Roadmap**. Retires the legacy compatibility layer introduced in Feature 002, purges procedural semantic data from `themes.py`, transitions domain definitions to modular declarative YAML manifests validated by Pydantic v2, aligns `synth/tabular.py` preset keys to canonical domains, and establishes Schema Version `v3.0` (`dataset_version = "3.0.0"`).

---

## Schema Versioning

* **Annotation Schema Version**: `v3.0`
* **Dataset Version**: `3.0.0`
* **Predecessor Feature**: `002-semantic-storage-registry` (Schema `v2.1`, `dataset_version = "2.1.0"`)

---

## Executive Summary

Feature 002 utilized a **Strangler Fig** architectural pattern: all pipeline rendering functions in `chart.py` and `generator.py` were redirected to `synth/semantics/` while keeping `themes.py` strictly frozen ([`FR-050`](../002-semantic-storage-registry/spec.md#requirements)) to guarantee zero breaking changes.

Feature 003 represents the planned **decommissioning and clean-state promotion phase**:
1. **Purge Legacy Constants from `themes.py`**: Remove all 264 raw axis strings, 143 chart titles, 94 comparative pairs, 51 histogram Y labels, dead heatmap/colorbar labels, and dead `*_DOMAIN_DICT` / `CONTEXT_CONFIGURATIONS` / `STRUCTURAL_THEMES` structures by purging Zone 1 (lines 1–318) and Zone 3 (lines 767–1106). [`themes.py`](../../themes.py) shrinks from 1,106 lines to $\le 460$ lines (59.5% reduction), retaining active styling (`THEMES`, `PUBLICATION_THEMES`, `FONT_FAMILIES`) and fulfilling a single, unambiguous responsibility: visual styling.
2. **Declarative YAML Domain Manifests**: Promote domain specifications to modular human-readable YAML manifests under `synth/semantics/domains/*.yaml`. Validated at build and test time using Pydantic v2 schemas and compiled at application startup into immutable frozen slotted dataclasses for verified $O(1)$ memory access with zero top-level runtime Pydantic imports.
3. **Clean Tabular Alignment**: Remove bidirectional alias shims (`DOMAIN_ALIAS_TO_PRESET` and `PRESET_TO_CANONICAL`) in `synth/tabular.py`, directly standardizing `DOMAIN_PRESETS` keys to `"business"` and `"engineering"`, while retaining domain copula distributions in the tabular module.
4. **Retire Compatibility Shims**: Decommission legacy compatibility test suites (`test_legacy_themes_compatibility.py`) and decouple existing tests once all Sunset Gates are satisfied.

---

## Sunset Gate Prerequisites

Feature 003 execution is **gated** by three objective stability criteria:

| Gate | Requirement | Verification Mechanism |
|---|---|---|
| **SG-001** | **Dataset Stability Milestone** | $\ge 50,000$ synthetic chart images and annotations produced and validated under Schema Version `v2.1`. |
| **SG-002** | **Zero Legacy Call Sites (AST Audit)** | Prerequisite cleanup of lingering header imports (`generator.py:46`, `chart.py:64`, `sampler.py:94`), followed by AST verification confirming **zero references** to `SCIENTIFIC_*`, `BUSINESS_*`, `CHART_TITLES`, `COMPARATIVE_LABELS`, or dead dictionaries outside `test_legacy_themes_compatibility.py`. |
| **SG-003** | **Downstream Pipeline Readiness** | Downstream OCR, document parsing, and vision-language training pipelines confirm full compatibility with Schema `v2.1`/`v3.0` metadata fields (`semantic_domain`, `subplots`, `is_composite`). |

---

## Reading Order

Read the specification documents in this sequence:

1. **`research.md`** — Empirical debt census of the legacy codebase. Measures dead code, duplicate lists, line count reductions, YAML parser benchmarks, and Pydantic v2 compilation overhead.
2. **`spec.md`** — WHAT and WHY. User stories, Sunset Gates (`SG-001`–`SG-003`), functional requirements (`FR-060`–`FR-072`), Pydantic v2 schema definitions, YAML file formats, and success criteria (`SC-030`–`SC-035`).
3. **`plan.md`** — HOW. Decommissioning principles, target layout (clean architecture without compatibility shims), 4-phase implementation roadmap, and risk register.
4. **`tasks.md`** — Executable checklist. 17 concrete tasks (`T199`–`T215`, including `T212b`) covering AST audits, YAML authoring, Pydantic validation, code purging, and testing.

---

## Priority at a Glance

| Phase | Focus | What it buys | Key Deliverables |
|---|---|---|---|
| **Phase 1** | Sunset Audit & AST Verification | Verifies stability gates SG-001 to SG-003. Proves zero production call sites touch legacy `themes.py` constants. | Header import cleanup, `scripts/audit_legacy_references.py`, audit report |
| **Phase 2** | Declarative YAML Manifests & Fast Compiler | Defines YAML schema and Pydantic v2 models. Authorizes domain files (`biomedical.yaml`, `engineering.yaml`, `business.yaml`, `demographic.yaml`, `common.yaml`). Compiles YAML into frozen slotted dataclasses at startup ($\le 50\,\text{ms}$). | `synth/semantics/domains/*.yaml`, `synth/semantics/schema.py`, `synth/semantics/loader.py`, `requirements.txt` |
| **Phase 3** | Legacy Purge & Tabular Preset Alignment | Strips semantic constants from `themes.py` (1,106 -> $\le 460$ lines). Renames `DOMAIN_PRESETS` in `synth/tabular.py` to canonical names; deletes alias shims. Updates version to `v3.0`. | `themes.py`, `synth/tabular.py`, `versions.py`, `SCHEMA.md` |
| **Phase 4** | Verification & Shim Retirement | Retires `test_legacy_themes_compatibility.py`. Decouples existing tests. Validates full test suite and startup performance ($\le 50\,\text{ms}$ YAML compile). | `tests/test_domain_manifests.py`, `tests/test_startup_latency.py`, `tests/` regression suite |

---

## Project Execution Status

| Phase | Focus | Status | Tests |
|---|---|---|---|
| **Phase 1** | Sunset Audit & AST Verification | Gated on Feature 002 completion | `scripts/audit_legacy_references.py` |
| **Phase 2** | Declarative YAML Manifests | Planned | `tests/test_domain_manifests.py`, `tests/test_startup_latency.py` |
| **Phase 3** | Legacy Purge & Tabular Alignment | Planned | `tests/test_semantic_sampling.py`, `tests/test_metadata_persistence.py` |
| **Phase 4** | Verification & Shim Retirement | Planned | Full `pytest tests/` suite |
