"""
tests/test_m6_manifest_admin_engine.py

Comprehensive acceptance test suite for Milestone M6:
Manifest Admin Engine (ADM-01 .. ADM-14).
"""
import io
import json
import os
from pathlib import Path
import pytest
from ruamel.yaml import YAML

from synth.semantics.admin import (
    normalize_label,
    make_dedup_key,
    make_stem,
    ingest_file,
    IngestResult,
    CollisionStatus,
    classify_metric,
    classify_title,
    classify_pair,
    build_plan,
    apply_plan,
    list_backups,
    restore_backup,
    run_audit,
    generate_template,
    FileLock,
    ApplyError,
)
from synth.semantics.enums import AxisRole, ScaleType
from synth.semantics.loader import CARTESIAN_CHART_TYPES, _load_manifest_raw_data, rebuild_cache
from synth.semantics.schema import DomainManifest


# ==============================================================================
# ADM-02 & ADM-03: Normalization & make_stem
# ==============================================================================

def test_normalization_h4_invariants():
    """Verify dedup key collides equivalent unicode while stored label preserves typography (H4, ADM-02)."""
    # 1. Micro sign vs Greek mu
    micro_1 = "Concentration (µM)"  # U+00B5
    micro_2 = "Concentration (μM)"  # U+03BC
    assert make_dedup_key(micro_1) == make_dedup_key(micro_2)
    assert normalize_label(micro_1) == "Concentration (μM)"

    # 2. Superscript preservation
    area_density = "Density (kg/m²)"
    assert normalize_label(area_density) == "Density (kg/m²)"
    assert "m²" in normalize_label(area_density)

    # 3. Non-breaking space and whitespace collapse
    nbsp_label = "Quarterly\u00a0\u00a0Revenue\t($M)"
    assert normalize_label(nbsp_label) == "Quarterly Revenue ($M)"


def test_make_stem_b4_invariants():
    """Verify make_stem produces valid identifier conforming to ^[a-z0-9_]+$ (B4, ADM-03)."""
    test_cases = [
        ("Plasma Concentration (ng/mL)", "plasma_concentration"),
        ("résistance", "resistance"),
        ("δ temperature", "temperature"),
        ("Growth Rate (%)", "growth_rate"),
        ("Core-Rotor / Temp_2", "core_rotor_temp_2"),
        ("(%)", ""),  # Only unit, no concept stem
        ("   ", ""),
    ]
    for raw_label, expected_stem in test_cases:
        actual_stem = make_stem(raw_label)
        assert actual_stem == expected_stem, f"Failed for '{raw_label}': got '{actual_stem}', expected '{expected_stem}'"


# ==============================================================================
# ADM-01 & ADM-04: Ingestion & Enum Handling
# ==============================================================================

def test_ingest_csv_delimiter_and_encoding_sniffing(tmp_path):
    """Verify CSV ingestion handles semicolons, BOM, and header synonyms (ADM-01)."""
    csv_path = tmp_path / "semicolon_domain.csv"
    # Semicolon separated, UTF-8 BOM, synonyms for headers
    csv_content = "\ufeffMetric;Scale;Axis;Units;Stem;Tags\nTurbine Speed;continuous;dependent;rpm;turbine_speed;aerospace\n"
    csv_path.write_bytes(csv_content.encode("utf-8"))

    result = ingest_file(csv_path, target_domain="aerospace")
    assert not result.has_errors
    assert len(result.metrics) == 1
    m = result.metrics[0]
    assert m.label == "Turbine Speed"
    assert m.scale_type == ScaleType.CONTINUOUS
    assert m.role == AxisRole.DEPENDENT
    assert m.unit == "rpm"
    assert m.concept_stem == "turbine_speed"
    assert "aerospace" in m.domain_tags


def test_ingest_ordinal_mapped_to_categorical_with_warning(tmp_path):
    """Verify 'ordinal' ScaleType is mapped to 'categorical' with warning (ADM-04, B5)."""
    csv_path = tmp_path / "ordinal.csv"
    csv_path.write_text("Label,Scale Type,Axis Role\nStage Ranking,ordinal,dependent\n", encoding="utf-8")

    result = ingest_file(csv_path, target_domain="ranking_domain")
    assert not result.has_errors
    assert len(result.metrics) == 1
    assert result.metrics[0].scale_type == ScaleType.CATEGORICAL
    assert len(result.warnings) == 1
    assert "'ordinal' is mapped to 'categorical'" in result.warnings[0].message


def test_ingest_excel_error_cells_flagged(tmp_path):
    """Verify Excel #N/A and formula errors produce line-attributed errors (ADM-01)."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Metrics"
    ws.append(["Label", "Scale Type", "Axis Role"])
    ws.append(["Valid Metric", "continuous", "dependent"])
    ws.append(["#N/A", "continuous", "dependent"])
    ws.append(["Another Metric", "#VALUE!", "dependent"])

    xlsx_path = tmp_path / "errors.xlsx"
    wb.save(xlsx_path)
    wb.close()

    result = ingest_file(xlsx_path, target_domain="test_dom")
    assert result.has_errors
    error_msgs = [e.message for e in result.errors]
    assert any("Excel error cell detected" in m for m in error_msgs)
    assert len(result.metrics) == 1  # Only the valid row parsed


# ==============================================================================
# ADM-06: Classification Invariants (H3 label table)
# ==============================================================================

def test_classification_h3_label_table():
    """Verify legitimate unit variants are classified as UNIT-VARIANT, and typos as SIMILAR (ADM-06, H3)."""
    existing_doc = {
        "domain_id": "lab_domain",
        "metrics": [
            {"label": "Concentration (μM)", "concept_stem": "concentration", "unit": "μM"},
            {"label": "Temperature (°C)", "concept_stem": "temperature", "unit": "°C"},
            {"label": "Revenue ($M)", "concept_stem": "revenue", "unit": "$M"},
            {"label": "Time (h)", "concept_stem": "time", "unit": "h"},
            {"label": "Chamber Pressure (kPa)", "concept_stem": "chamber_pressure", "unit": "kPa"},
        ],
    }

    # 1. Legitimate unit variants -> must be UNIT-VARIANT (NOT SIMILAR)
    unit_variants = [
        ("Concentration (nM)", "concentration", "nM"),
        ("Temperature (K)", "temperature", "K"),
        ("Revenue ($K)", "revenue", "$K"),
        ("Time (s)", "time", "s"),
    ]
    for lbl, stem, unit in unit_variants:
        m = IngestResult(source_file="test").metrics
        from synth.semantics.admin.ingest import IngestedMetric
        metric_cand = IngestedMetric("s", 1, lbl, ScaleType.CONTINUOUS, AxisRole.DEPENDENT, unit, stem, None, ("lab_domain",), ())
        classification = classify_metric(metric_cand, "lab_domain", existing_doc, {})
        assert classification.status == CollisionStatus.UNIT_VARIANT, (
            f"Expected UNIT-VARIANT for '{lbl}', got {classification.status}"
        )

    # 2. Real typo with different stem -> must be SIMILAR
    typo_cand = IngestedMetric("s", 2, "Chamber Presure (kPa)", ScaleType.CONTINUOUS, AxisRole.DEPENDENT, "kPa", "chamber_presure", None, ("lab_domain",), ())
    typo_class = classify_metric(typo_cand, "lab_domain", existing_doc, {})
    assert typo_class.status == CollisionStatus.SIMILAR
    assert typo_class.similarity_score >= 90.0
    assert typo_class.matched_label == "Chamber Pressure (kPa)"


# ==============================================================================
# ADM-07 & ADM-09: Collision Policies & Plan Builder
# ==============================================================================

def test_plan_policies_skip_merge_overwrite_abort():
    """Verify plan behavior across all four collision policies (ADM-07, ADM-09)."""
    raw_manifests = {
        "target": {
            "domain_id": "target",
            "metrics": [
                {"label": "Existing Metric (units)", "scale_type": "continuous", "role": "dependent", "concept_stem": "existing_metric", "domain_tags": ["target"], "pools": ["legacy"]}
            ],
            "titles": [],
            "comparative_pairs": [],
        },
        "other": {
            "domain_id": "other",
            "metrics": [
                {"label": "Shared Metric", "scale_type": "continuous", "role": "independent", "concept_stem": "shared_metric", "domain_tags": ["other"], "pools": ["canonical_independent"]}
            ],
            "titles": [],
            "comparative_pairs": [],
        }
    }

    ingest_res = IngestResult(source_file="test.csv")
    from synth.semantics.admin.ingest import IngestedMetric
    # 1. Exact duplicate
    ingest_res.metrics.append(
        IngestedMetric("s", 1, "Existing Metric (units)", ScaleType.CONTINUOUS, AxisRole.DEPENDENT, "units", "existing_metric", None, ("target", "extra_tag"), ())
    )
    # 2. Shared metric
    ingest_res.metrics.append(
        IngestedMetric("s", 2, "Shared Metric", ScaleType.CONTINUOUS, AxisRole.INDEPENDENT, None, "shared_metric", None, ("target",), ())
    )

    # Test 'skip' policy (default)
    plan_skip = build_plan(ingest_res, "target", policy="skip", raw_manifests=raw_manifests)
    assert plan_skip.metric_actions[0].action == "SKIP"
    assert plan_skip.metric_actions[1].action == "SHARE"
    assert "target" in plan_skip.candidate_manifests["other"]["metrics"][0]["domain_tags"]

    # Test 'merge' policy
    plan_merge = build_plan(ingest_res, "target", policy="merge", raw_manifests=raw_manifests)
    assert plan_merge.metric_actions[0].action == "MERGE"
    assert "extra_tag" in plan_merge.candidate_manifests["target"]["metrics"][0]["domain_tags"]

    # Test 'overwrite' policy
    plan_overwrite = build_plan(ingest_res, "target", policy="overwrite", raw_manifests=raw_manifests)
    assert plan_overwrite.metric_actions[0].action == "OVERWRITE"
    # Pools must be preserved on overwrite (ADM-07)
    assert plan_overwrite.candidate_manifests["target"]["metrics"][0]["pools"] == ["legacy"]

    # Test 'abort' policy
    plan_abort = build_plan(ingest_res, "target", policy="abort", raw_manifests=raw_manifests)
    assert plan_abort.has_abort_condition


# ==============================================================================
# ADM-10: Transactional Apply, Atomic Writes, Rollback & Restore
# ==============================================================================

def test_transactional_apply_and_rollback(tmp_path):
    """Verify atomic write, backup snapshot creation, and automatic rollback on failure (ADM-10)."""
    # Setup test domain manifest
    dom_file = tmp_path / "bio.yaml"
    dom_file.write_text(
        """manifest_version: 2
domain_id: bio
display_name: Bio Domain
is_scientific: true
default_weight: 0.5
default_fallbacks: ["Time (h)", "Dose (mg)"]
capabilities:
  tabular_preset: false
  heatmap_pools: false
metrics:
  - label: Time (h)
    scale_type: continuous
    role: independent
    concept_stem: time
    domain_tags: [bio]
    pools: [canonical_independent]
  - label: Dose (mg)
    scale_type: continuous
    role: dependent
    concept_stem: dose
    domain_tags: [bio]
    pools: [legacy]
titles: []
comparative_pairs: []
""",
        encoding="utf-8",
    )

    ingest_res = IngestResult(source_file="new.csv")
    from synth.semantics.admin.ingest import IngestedMetric
    ingest_res.metrics.append(
        IngestedMetric("s", 1, "New Cell Count", ScaleType.DISCRETE_COUNT, AxisRole.DEPENDENT, None, "new_cell_count", None, ("bio",), ())
    )

    plan = build_plan(ingest_res, "bio", policy="skip", domains_dir=tmp_path)
    apply_summary = apply_plan(plan, domains_dir=tmp_path)
    assert apply_summary["success"] is True
    assert apply_summary["metrics_added"] == 1

    # Verify backup exists
    backups = list_backups(domains_dir=tmp_path)
    assert len(backups) >= 1
    latest_backup = backups[0]

    # Verify disk was updated
    yaml = YAML()
    with open(dom_file, "r", encoding="utf-8") as f:
        updated_data = yaml.load(f)
    assert len(updated_data["metrics"]) == 3
    assert updated_data["metrics"][-1]["label"] == "New Cell Count"

    # Test restore
    restore_success = restore_backup(latest_backup, domains_dir=tmp_path)
    assert restore_success is True

    with open(dom_file, "r", encoding="utf-8") as f:
        restored_data = yaml.load(f)
    # Restored to 2 metrics
    assert len(restored_data["metrics"]) == 2


def test_transactional_apply_fault_injection_rollback(tmp_path):
    """Verify fault injection during candidate validation or compile triggers clean rollback (ADM-10)."""
    dom_file = tmp_path / "fault_dom.yaml"
    initial_content = """manifest_version: 2
domain_id: fault_dom
display_name: Fault Domain
is_scientific: true
default_weight: 0.5
default_fallbacks: ["X", "Y"]
metrics:
  - label: Metric X
    scale_type: continuous
    role: independent
    concept_stem: metric_x
    domain_tags: [fault_dom]
    pools: [canonical_independent]
titles: []
comparative_pairs: []
"""
    dom_file.write_text(initial_content, encoding="utf-8")

    ingest_res = IngestResult(source_file="fault.csv")
    from synth.semantics.admin.ingest import IngestedMetric
    ingest_res.metrics.append(
        IngestedMetric("s", 1, "Valid", ScaleType.CONTINUOUS, AxisRole.DEPENDENT, None, "valid", None, ("fault_dom",), ())
    )

    plan = build_plan(ingest_res, "fault_dom", policy="skip", domains_dir=tmp_path)
    # Inject invalid candidate data that violates schema (e.g. invalid type)
    plan.candidate_manifests["fault_dom"]["default_weight"] = "not_a_float"

    with pytest.raises(ApplyError) as exc:
        apply_plan(plan, domains_dir=tmp_path)
    assert "Pydantic validation failed" in str(exc.value)

    # Verify original file untouched
    assert dom_file.read_text(encoding="utf-8") == initial_content


# ==============================================================================
# ADM-12 & ADM-14: Audit & Template Generator
# ==============================================================================

def test_audit_reports_summary_and_zero_crashes():
    """Verify read-only audit executes cleanly on the active corpus (ADM-12)."""
    report = run_audit()
    assert report.summary_dict()["total_findings"] >= 0
    assert report.error_count == 0  # Clean on production baseline


def test_template_generation_and_reingest(tmp_path):
    """Verify generated Excel domain template re-ingests with zero errors (ADM-14)."""
    tpl_path = tmp_path / "domain_template.xlsx"
    generate_template(tpl_path)
    assert tpl_path.exists()

    result = ingest_file(tpl_path, target_domain="aerospace")
    assert not result.has_errors
    assert len(result.metrics) == 1
    assert len(result.titles) == 1
    assert len(result.pairs) == 1


def test_plan_status_counts_for_titles_and_pairs(tmp_path):
    """Verify build_plan accurately populates status_counts for titles and pairs (ADM-05)."""
    from synth.semantics.admin.ingest import IngestedPair, IngestedTitle, IngestResult
    ingest_res = IngestResult(source_file="test_pairs.csv")
    ingest_res.pairs.append(
        IngestedPair(
            sheet="pairs",
            row=2,
            control="Group A",
            treatment="Group B",
            domain_category="test_dom",
            context_tags=("tag1",),
        )
    )
    ingest_res.titles.append(
        IngestedTitle(
            sheet="titles",
            row=2,
            title="Performance Test Analysis",
            allowed_chart_types=("bar", "line"),
            domain_tags=("test_dom",),
        )
    )

    plan = build_plan(ingest_res, "test_dom", policy="skip", domains_dir=tmp_path)
    assert plan.status_counts["NEW"] == 2
    assert plan.status_counts["DUPLICATE"] == 0


def test_audit_chart_type_coverage_and_export(tmp_path):
    """Verify compute_chart_type_stats and AuditReport.to_dict() chart type coverage serialization."""
    from synth.semantics.admin.audit import compute_chart_type_stats, run_audit
    from synth.semantics.loader import ALL_CHART_TYPES

    stats = compute_chart_type_stats()
    assert "chart_type_totals" in stats
    assert "by_domain" in stats
    for ct in ALL_CHART_TYPES:
        assert ct in stats["chart_type_totals"]
        assert stats["chart_type_totals"][ct] >= 0

    report = run_audit()
    d = report.to_dict()
    assert "chart_type_coverage" in d
    assert "chart_type_totals" in d["summary"]
    for ct in ALL_CHART_TYPES:
        assert ct in d["summary"]["chart_type_totals"]

    # Verify JSON export
    rep_file = tmp_path / "audit_report.json"
    rep_file.write_text(json.dumps(d, indent=2), encoding="utf-8")
    loaded = json.loads(rep_file.read_text(encoding="utf-8"))
    assert loaded["chart_type_coverage"]["chart_type_totals"] == stats["chart_type_totals"]
