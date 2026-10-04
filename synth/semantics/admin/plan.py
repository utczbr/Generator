"""
synth/semantics/admin/plan.py

Pure, deterministic, serializable ingestion plan builder (ADM-05, ADM-07, ADM-09).
Evaluates ingested records against existing manifests and collision policies:
- skip: Skip same-domain collisions; cross-domain exact matches are SHARED
- merge: Merge tags and chart types for collisions
- overwrite: Overwrite same-domain collisions in-place (preserving list position and pools)
- abort: Abort if any duplicate or similar collision is encountered

Produces IngestPlan containing detailed pre-flight actions and in-memory candidate manifests.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from synth.semantics.admin.dedupe import (
    CollisionStatus,
    MetricClassification,
    TitleClassification,
    PairClassification,
    classify_metric,
    classify_title,
    classify_pair,
)
from synth.semantics.admin.ingest import IngestResult, IngestIssue
from synth.semantics.admin.normalize import make_dedup_key
from synth.semantics.loader import _load_manifest_raw_data, CARTESIAN_CHART_TYPES


@dataclass
class MetricPlanAction:
    action: str  # "APPEND" | "SHARE" | "OVERWRITE" | "MERGE" | "SKIP" | "ERROR"
    status: CollisionStatus
    label: str
    target_domain: str
    row: int
    sheet: str
    message: str
    payload: Dict[str, Any]
    target_manifest: str  # domain_id where the mutation lands


@dataclass
class TitlePlanAction:
    action: str  # "APPEND" | "MERGE" | "SKIP"
    status: CollisionStatus
    title: str
    target_domain: str
    row: int
    sheet: str
    message: str
    payload: Dict[str, Any]


@dataclass
class PairPlanAction:
    action: str  # "APPEND" | "SKIP"
    status: CollisionStatus
    pair: Tuple[str, str]
    target_domain: str
    row: int
    sheet: str
    message: str
    payload: Dict[str, Any]


@dataclass
class IngestPlan:
    source_file: str
    target_domain: str
    policy: str  # "skip" | "merge" | "overwrite" | "abort"
    status_counts: Dict[str, int] = field(default_factory=dict)
    metric_actions: List[MetricPlanAction] = field(default_factory=list)
    title_actions: List[TitlePlanAction] = field(default_factory=list)
    pair_actions: List[PairPlanAction] = field(default_factory=list)
    issues: List[IngestIssue] = field(default_factory=list)
    candidate_manifests: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def has_abort_condition(self) -> bool:
        if any(i.level == "error" for i in self.issues):
            return True
        if self.policy == "abort":
            for a in self.metric_actions:
                if a.status in (CollisionStatus.DUPLICATE, CollisionStatus.SIMILAR):
                    return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "target_domain": self.target_domain,
            "policy": self.policy,
            "status_counts": self.status_counts,
            "metric_actions": [
                {
                    "action": a.action,
                    "status": a.status.value,
                    "label": a.label,
                    "target_domain": a.target_domain,
                    "row": a.row,
                    "sheet": a.sheet,
                    "message": a.message,
                }
                for a in self.metric_actions
            ],
            "title_actions": [
                {
                    "action": a.action,
                    "status": a.status.value,
                    "title": a.title,
                    "target_domain": a.target_domain,
                    "row": a.row,
                    "sheet": a.sheet,
                    "message": a.message,
                }
                for a in self.title_actions
            ],
            "pair_actions": [
                {
                    "action": a.action,
                    "status": a.status.value,
                    "pair": list(a.pair),
                    "target_domain": a.target_domain,
                    "row": a.row,
                    "sheet": a.sheet,
                    "message": a.message,
                }
                for a in self.pair_actions
            ],
            "issues": [
                {
                    "level": i.level,
                    "sheet": i.sheet,
                    "row": i.row,
                    "column": i.column,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def batch_plans_to_dict(folder: str | Path, plans: List[IngestPlan], policy: str) -> Dict[str, Any]:
    """Serialize a collection of IngestPlans from batch directory import."""
    batch_totals: Dict[str, int] = {}
    for p in plans:
        for k, v in p.status_counts.items():
            batch_totals[k] = batch_totals.get(k, 0) + v
    return {
        "mode": "batch",
        "source_folder": str(folder),
        "policy": policy,
        "batch_totals": batch_totals,
        "files_count": len(plans),
        "files": [p.to_dict() for p in plans],
    }


def batch_plans_to_json(folder: str | Path, plans: List[IngestPlan], policy: str, indent: int = 2) -> str:
    """Serialize batch plans dictionary to formatted JSON string."""
    return json.dumps(batch_plans_to_dict(folder, plans, policy), indent=indent)


def build_plan(
    ingest_result: IngestResult,
    target_domain: str,
    policy: str = "skip",
    domains_dir: Optional[Path] = None,
    raw_manifests: Optional[Dict[str, Dict[str, Any]]] = None,
) -> IngestPlan:
    """
    Build a pure, serializable IngestPlan without modifying disk manifests (ADM-09).
    Clones manifests in-memory and produces mutated candidate manifests.
    """
    policy = policy.lower()
    if policy not in ("skip", "merge", "overwrite", "abort"):
        raise ValueError(f"Invalid collision policy '{policy}'. Allowed: skip, merge, overwrite, abort")

    base_manifests = (
        raw_manifests
        if raw_manifests is not None
        else _load_manifest_raw_data(domains_dir=domains_dir)
    )

    # Deep-copy candidate manifests
    candidates: Dict[str, Dict[str, Any]] = json.loads(json.dumps(base_manifests))

    # Ensure target domain exists in candidate set
    if target_domain not in candidates:
        candidates[target_domain] = {
            "manifest_version": 2,
            "domain_id": target_domain,
            "display_name": target_domain.replace("_", " ").title(),
            "is_scientific": False,
            "default_weight": 0.10,
            "default_fallbacks": ["Index", "Value"],
            "capabilities": {"tabular_preset": False, "heatmap_pools": False},
            "metrics": [],
            "titles": [],
            "comparative_pairs": [],
        }

    target_doc = candidates[target_domain]
    other_docs = {k: v for k, v in candidates.items() if k != target_domain}

    plan = IngestPlan(
        source_file=ingest_result.source_file,
        target_domain=target_domain,
        policy=policy,
        issues=list(ingest_result.issues),
    )

    counts: Dict[str, int] = {s.value: 0 for s in CollisionStatus}

    # 1. Process Metrics
    for m in ingest_result.metrics:
        classification = classify_metric(m, target_domain, target_doc, other_docs)
        counts[classification.status.value] += 1

        metric_dict = {
            "label": m.label,
            "scale_type": m.scale_type.value,
            "role": m.role.value,
            "concept_stem": m.concept_stem,
            "domain_tags": list(m.domain_tags),
            "pools": list(m.pools),
        }
        if m.unit:
            metric_dict["unit"] = m.unit
        if m.concept_group:
            metric_dict["concept_group"] = m.concept_group

        # Determine action based on status and policy
        if classification.status == CollisionStatus.NEW or classification.status == CollisionStatus.UNIT_VARIANT:
            action = "APPEND"
            target_doc.setdefault("metrics", []).append(metric_dict)
            action_obj = MetricPlanAction(
                action=action,
                status=classification.status,
                label=m.label,
                target_domain=target_domain,
                row=m.row,
                sheet=m.sheet,
                message=classification.message,
                payload=metric_dict,
                target_manifest=target_domain,
            )
        elif classification.status == CollisionStatus.SHARED:
            # Cross-domain exact match: default policy is SHARE (H7, ADM-07)
            action = "SHARE"
            owner_dom = classification.matched_domain or "common"
            owner_doc = candidates[owner_dom]
            cand_key = make_dedup_key(m.label)
            for existing in owner_doc.get("metrics", []):
                if make_dedup_key(existing.get("label", "")) == cand_key:
                    tags = existing.setdefault("domain_tags", [])
                    if target_domain not in tags:
                        tags.append(target_domain)
                    break
            action_obj = MetricPlanAction(
                action=action,
                status=classification.status,
                label=m.label,
                target_domain=target_domain,
                row=m.row,
                sheet=m.sheet,
                message=f"Added '{target_domain}' to domain_tags of existing metric in '{owner_dom}'",
                payload=metric_dict,
                target_manifest=owner_dom,
            )
        elif classification.status == CollisionStatus.DUPLICATE:
            if policy == "overwrite":
                action = "OVERWRITE"
                cand_key = make_dedup_key(m.label)
                # Overwrite in-place preserving list position and pools (ADM-07)
                for idx, existing in enumerate(target_doc.get("metrics", [])):
                    if make_dedup_key(existing.get("label", "")) == cand_key:
                        existing_pools = existing.get("pools", [])
                        metric_dict["pools"] = existing_pools
                        target_doc["metrics"][idx] = metric_dict
                        break
                msg = f"Overwrote existing '{m.label}' in-place"
            elif policy == "merge":
                action = "MERGE"
                cand_key = make_dedup_key(m.label)
                for existing in target_doc.get("metrics", []):
                    if make_dedup_key(existing.get("label", "")) == cand_key:
                        existing_tags = set(existing.get("domain_tags", []))
                        existing["domain_tags"] = sorted(existing_tags | set(m.domain_tags))
                        break
                msg = f"Merged domain_tags into existing '{m.label}'"
            elif policy == "abort":
                action = "ERROR"
                msg = f"Duplicate metric '{m.label}' violates abort policy"
                plan.issues.append(IngestIssue(level="error", sheet=m.sheet, row=m.row, column="label", message=msg))
            else:  # skip
                action = "SKIP"
                msg = f"Skipped duplicate metric '{m.label}'"

            action_obj = MetricPlanAction(
                action=action,
                status=classification.status,
                label=m.label,
                target_domain=target_domain,
                row=m.row,
                sheet=m.sheet,
                message=msg,
                payload=metric_dict,
                target_manifest=target_domain,
            )
        elif classification.status == CollisionStatus.SIMILAR:
            if policy == "abort":
                action = "ERROR"
                msg = f"Similar metric '{m.label}' matches '{classification.matched_label}' ({classification.similarity_score:.1f}%)"
                plan.issues.append(IngestIssue(level="error", sheet=m.sheet, row=m.row, column="label", message=msg))
            else:
                action = "APPEND"  # Warn but allow append per ADM-06
                target_doc.setdefault("metrics", []).append(metric_dict)
                msg = classification.message

            action_obj = MetricPlanAction(
                action=action,
                status=classification.status,
                label=m.label,
                target_domain=target_domain,
                row=m.row,
                sheet=m.sheet,
                message=msg,
                payload=metric_dict,
                target_manifest=target_domain,
            )
        else:
            action = "SKIP"
            action_obj = MetricPlanAction(
                action=action,
                status=classification.status,
                label=m.label,
                target_domain=target_domain,
                row=m.row,
                sheet=m.sheet,
                message=classification.message,
                payload=metric_dict,
                target_manifest=target_domain,
            )

        plan.metric_actions.append(action_obj)

    # 2. Process Titles
    for t in ingest_result.titles:
        t_class = classify_title(t, target_domain, target_doc, other_docs)
        counts[t_class.status.value] += 1
        title_dict = {
            "title": t.title,
            "allowed_chart_types": list(t.allowed_chart_types),
            "domain_tags": list(t.domain_tags),
        }

        if t_class.status == CollisionStatus.NEW:
            action = "APPEND"
            target_doc.setdefault("titles", []).append(title_dict)
            msg = "New title appended"
        elif t_class.status == CollisionStatus.DUPLICATE:
            if policy == "merge" or policy == "overwrite":
                action = "MERGE"
                # Same title in domain -> set-union of chart types and tags (ADM-05)
                cand_key = make_dedup_key(t.title)
                for existing in target_doc.get("titles", []):
                    if make_dedup_key(existing.get("title", "")) == cand_key:
                        existing["allowed_chart_types"] = sorted(
                            set(existing.get("allowed_chart_types", [])) | set(t.allowed_chart_types)
                        )
                        existing["domain_tags"] = sorted(
                            set(existing.get("domain_tags", [])) | set(t.domain_tags)
                        )
                        break
                msg = "Merged chart types and tags into existing title (set-union)"
            else:
                action = "SKIP"
                msg = f"Skipped duplicate title '{t.title}'"
        else:  # SHARED
            action = "APPEND"
            target_doc.setdefault("titles", []).append(title_dict)
            msg = "Title appended to target domain"

        plan.title_actions.append(
            TitlePlanAction(
                action=action,
                status=t_class.status,
                title=t.title,
                target_domain=target_domain,
                row=t.row,
                sheet=t.sheet,
                message=msg,
                payload=title_dict,
            )
        )

    # 3. Process Comparative Pairs
    for p in ingest_result.pairs:
        p_class = classify_pair(p, target_domain, target_doc, other_docs)
        counts[p_class.status.value] += 1
        pair_dict = {
            "control": p.control,
            "treatment": p.treatment,
            "domain_category": p.domain_category,
            "context_tags": list(p.context_tags),
        }

        if p_class.status == CollisionStatus.NEW:
            action = "APPEND"
            target_doc.setdefault("comparative_pairs", []).append(pair_dict)
            msg = "New comparative pair appended"
        else:
            action = "SKIP"
            msg = f"Skipped duplicate pair ({p.control}, {p.treatment})"

        plan.pair_actions.append(
            PairPlanAction(
                action=action,
                status=p_class.status,
                pair=(p.control, p.treatment),
                target_domain=target_domain,
                row=p.row,
                sheet=p.sheet,
                message=msg,
                payload=pair_dict,
            )
        )

    plan.status_counts = counts
    plan.candidate_manifests = candidates
    return plan
