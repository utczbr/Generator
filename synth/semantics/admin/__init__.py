"""
synth.semantics.admin package

Build-time and management tools for semantic domain manifests and configurations.
Isolated from runtime execution.
"""
from synth.semantics.admin.normalize import normalize_label, make_dedup_key, make_stem, infer_domain_from_filename
from synth.semantics.admin.ingest import ingest_file, IngestResult, IngestIssue
from synth.semantics.admin.dedupe import (
    CollisionStatus,
    MetricClassification,
    TitleClassification,
    PairClassification,
    classify_metric,
    classify_title,
    classify_pair,
)
from synth.semantics.admin.plan import IngestPlan, build_plan, batch_plans_to_dict, batch_plans_to_json
from synth.semantics.admin.apply import apply_plan, apply_batch_plans, list_backups, restore_backup, FileLock, ApplyError
from synth.semantics.admin.audit import run_audit, AuditReport, AuditFinding, compute_chart_type_stats
from synth.semantics.admin.template import generate_template

__all__ = [
    "normalize_label",
    "make_dedup_key",
    "make_stem",
    "infer_domain_from_filename",
    "ingest_file",
    "IngestResult",
    "IngestIssue",
    "CollisionStatus",
    "MetricClassification",
    "TitleClassification",
    "PairClassification",
    "classify_metric",
    "classify_title",
    "classify_pair",
    "IngestPlan",
    "build_plan",
    "batch_plans_to_dict",
    "batch_plans_to_json",
    "apply_plan",
    "apply_batch_plans",
    "list_backups",
    "restore_backup",
    "FileLock",
    "ApplyError",
    "run_audit",
    "compute_chart_type_stats",
    "AuditReport",
    "AuditFinding",
    "generate_template",
]
