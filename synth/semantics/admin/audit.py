"""
synth/semantics/admin/audit.py

Read-only audit inspector for domain manifests (ADM-12).
Audits for:
1. Duplicate metric labels across manifests
2. Concept stem collisions within domains
3. Tag-invariant violations (domain_id missing from domain_tags)
4. Positional pool leftovers (missing explicit pools definition)
5. Near-duplicate metric labels (unit-stripped fuzzy ratio >= threshold)
6. Degenerate pools: (domain x Cartesian type x orientation) where x- or y-pool is empty
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import rapidfuzz.fuzz

from synth.semantics.admin.normalize import make_dedup_key, strip_unit_suffix
from synth.semantics.loader import (
    CARTESIAN_CHART_TYPES,
    _load_manifest_raw_data,
    get_domains_dir,
    SemanticRegistry,
)


@dataclass(frozen=True)
class AuditFinding:
    category: str  # "DUPLICATE_LABEL" | "STEM_COLLISION" | "TAG_INVARIANT" | "DEGENERATE_POOL" | "NEAR_DUPLICATE" | "MISSING_POOLS"
    severity: str  # "ERROR" | "WARNING" | "INFO"
    domain: str
    item: str
    message: str


@dataclass
class AuditReport:
    findings: List[AuditFinding] = field(default_factory=list)
    degenerate_pools: List[Tuple[str, str, str, str]] = field(default_factory=list)  # (dom, ctype, orient, axis)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "ERROR" for f in self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "WARNING")

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_findings": len(self.findings),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "degenerate_pools_count": len(self.degenerate_pools),
            "by_category": {
                cat: sum(1 for f in self.findings if f.category == cat)
                for cat in {f.category for f in self.findings}
            },
        }


WHITELISTED_NEAR_DUPLICATES: Set[frozenset] = {
    frozenset({"employment rate", "unemployment rate"}),
    frozenset({"ldl cholesterol", "hdl cholesterol"}),
}


def run_audit(
    domains_dir: Optional[Path] = None,
    fuzzy_threshold: float = 90.0,
    strict: bool = False,
) -> AuditReport:
    """
    Execute complete read-only semantic corpus audit (ADM-12).
    """
    target_dir = domains_dir or get_domains_dir()
    raw = _load_manifest_raw_data(force=True, domains_dir=target_dir)
    report = AuditReport()

    # 1. Duplicate metric labels across manifests
    global_labels: Dict[str, List[Tuple[str, str]]] = {}  # dedup_key -> list of (domain, original_label)
    for dom_id, doc in raw.items():
        for m in doc.get("metrics", []):
            label = m.get("label", "")
            key = make_dedup_key(label)
            global_labels.setdefault(key, []).append((dom_id, label))

    for key, occurrences in global_labels.items():
        if len(occurrences) > 1:
            doms = [o[0] for o in occurrences]
            # If occurrences span multiple manifests, check if shared properly
            labels = [o[1] for o in occurrences]
            report.findings.append(
                AuditFinding(
                    category="DUPLICATE_LABEL",
                    severity="WARNING" if len(set(doms)) > 1 else "ERROR",
                    domain=",".join(sorted(set(doms))),
                    item=labels[0],
                    message=f"Metric label '{labels[0]}' appears in multiple manifests: {doms}",
                )
            )

    # 2. Stem collisions within each domain
    for dom_id, doc in raw.items():
        domain_stems: Dict[str, List[str]] = {}
        for m in doc.get("metrics", []):
            stem = m.get("concept_stem", "")
            domain_stems.setdefault(stem, []).append(m.get("label", ""))

        for stem, labels in domain_stems.items():
            if len(labels) > 1:
                report.findings.append(
                    AuditFinding(
                        category="STEM_COLLISION",
                        severity="INFO",
                        domain=dom_id,
                        item=stem,
                        message=f"Concept stem '{stem}' is shared by {len(labels)} metrics in {dom_id}: {labels}",
                    )
                )

    LEGACY_METRIC_TAG_EXCEPTIONS = {
        "biomedical": {"Pore Volume (cm³/g)", "Voltage (mV)"},
    }

    # 3. Tag-invariant violations
    for dom_id, doc in raw.items():
        if dom_id == "common":
            continue
        metric_exceptions = LEGACY_METRIC_TAG_EXCEPTIONS.get(dom_id, set())
        for m in doc.get("metrics", []):
            tags = m.get("domain_tags", [])
            if dom_id not in tags:
                is_legacy_exemption = m.get("label") in metric_exceptions
                report.findings.append(
                    AuditFinding(
                        category="TAG_INVARIANT",
                        severity="WARNING" if is_legacy_exemption else ("ERROR" if strict else "WARNING"),
                        domain=dom_id,
                        item=m.get("label", ""),
                        message=f"Metric '{m.get('label')}' in {dom_id}.yaml does not include '{dom_id}' in domain_tags: {tags}",
                    )
                )
        for t in doc.get("titles", []):
            tags = t.get("domain_tags", [])
            if dom_id not in tags:
                is_legacy_exemption = (dom_id == "demographic")
                report.findings.append(
                    AuditFinding(
                        category="TAG_INVARIANT",
                        severity="WARNING" if is_legacy_exemption else ("ERROR" if strict else "WARNING"),
                        domain=dom_id,
                        item=t.get("title", ""),
                        message=f"Title '{t.get('title')}' in {dom_id}.yaml does not include '{dom_id}' in domain_tags: {tags}",
                    )
                )

    # 4. Missing pools definition
    for dom_id, doc in raw.items():
        for m in doc.get("metrics", []):
            if "pools" not in m:
                report.findings.append(
                    AuditFinding(
                        category="MISSING_POOLS",
                        severity="WARNING",
                        domain=dom_id,
                        item=m.get("label", ""),
                        message=f"Metric '{m.get('label')}' lacks explicit 'pools:' field",
                    )
                )

    # 5. Near-duplicate metric labels within domains
    for dom_id, doc in raw.items():
        metrics = doc.get("metrics", [])
        for i in range(len(metrics)):
            l1 = metrics[i].get("label", "")
            s1 = strip_unit_suffix(l1).casefold()
            stem1 = metrics[i].get("concept_stem", "")
            for j in range(i + 1, len(metrics)):
                l2 = metrics[j].get("label", "")
                stem2 = metrics[j].get("concept_stem", "")
                # Skip legitimate unit-variants (same stem)
                if stem1 == stem2:
                    continue
                s2 = strip_unit_suffix(l2).casefold()
                if not s1 or not s2:
                    continue
                if frozenset({s1, s2}) in WHITELISTED_NEAR_DUPLICATES:
                    continue
                score = float(rapidfuzz.fuzz.ratio(s1, s2))
                if score >= fuzzy_threshold:
                    report.findings.append(
                        AuditFinding(
                            category="NEAR_DUPLICATE",
                            severity="WARNING",
                            domain=dom_id,
                            item=f"{l1} <-> {l2}",
                            message=f"Near duplicate labels ({score:.1f}% match) with different stems ('{stem1}' vs '{stem2}')",
                        )
                    )

    # 6. Degenerate pools in the registry
    reg = SemanticRegistry.load(raw=raw)
    from synth.semantics.sampler import _build_axis_pools
    axis_pools, _ = _build_axis_pools(reg)

    orientations = ("vertical", "horizontal")
    for dom_id in reg.domains:
        if dom_id == "common":
            continue
        for ctype in CARTESIAN_CHART_TYPES:
            for orient in orientations:
                for axis in ("x", "y"):
                    # Histogram has empty y pool by design
                    if ctype == "histogram" and axis == "y":
                        continue
                    pool = axis_pools.get((dom_id, ctype, orient, axis), ())
                    if not pool:
                        report.degenerate_pools.append((dom_id, ctype, orient, axis))
                        report.findings.append(
                            AuditFinding(
                                category="DEGENERATE_POOL",
                                severity="WARNING",
                                domain=dom_id,
                                item=f"({dom_id}, {ctype}, {orient}, {axis})",
                                message=f"Candidate pool is empty; chart will silently draw from static fallback",
                            )
                        )

    return report
