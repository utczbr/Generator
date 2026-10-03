"""
synth/semantics/admin/dedupe.py

Collision classification and deduplication engine (ADM-06, ADM-07, H3, H4).
Classifies candidate metrics against existing manifest catalogs:
- NEW: Distinct metric not found in target or corpus
- DUPLICATE: Exact dedup key collision within the target domain
- SHARED: Exact dedup key exists in another domain in the corpus
- UNIT_VARIANT: Same concept_stem within target domain with different unit (legitimate)
- SIMILAR: Unit-stripped fuzzy similarity >= threshold (default 90) against target domain
- INVALID: Entry failed structural validation
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import rapidfuzz.fuzz

from synth.semantics.admin.ingest import IngestedMetric, IngestedTitle, IngestedPair
from synth.semantics.admin.normalize import make_dedup_key, strip_unit_suffix

DEFAULT_FUZZY_THRESHOLD = 90.0


class CollisionStatus(str, Enum):
    NEW = "NEW"
    DUPLICATE = "DUPLICATE"
    SHARED = "SHARED"
    UNIT_VARIANT = "UNIT-VARIANT"
    SIMILAR = "SIMILAR"
    INVALID = "INVALID"


@dataclass(frozen=True)
class MetricClassification:
    metric: IngestedMetric
    status: CollisionStatus
    matched_label: Optional[str] = None
    matched_domain: Optional[str] = None
    similarity_score: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class TitleClassification:
    title: IngestedTitle
    status: CollisionStatus
    matched_title: Optional[str] = None
    matched_domain: Optional[str] = None
    message: str = ""


@dataclass(frozen=True)
class PairClassification:
    pair: IngestedPair
    status: CollisionStatus
    matched_pair: Optional[Tuple[str, str]] = None
    matched_domain: Optional[str] = None
    message: str = ""


def classify_metric(
    candidate: IngestedMetric,
    target_domain: str,
    target_manifest: Optional[Dict[str, Any]],
    other_manifests: Dict[str, Dict[str, Any]],
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> MetricClassification:
    """
    Classify a metric row according to ADM-06:
    1. Exact match in target domain -> DUPLICATE
    2. Exact match in another domain -> SHARED
    3. Same stem in target domain with different unit -> UNIT-VARIANT
    4. Unit-stripped fuzzy score >= threshold in target domain -> SIMILAR
    5. Else -> NEW
    """
    cand_key = make_dedup_key(candidate.label)
    cand_stem = candidate.concept_stem
    cand_unit = (candidate.unit or "").strip().lower()
    cand_stripped = strip_unit_suffix(candidate.label).casefold()

    target_metrics = target_manifest.get("metrics", []) if target_manifest else []

    # 1. Check exact collision in target domain
    for m in target_metrics:
        existing_key = make_dedup_key(m.get("label", ""))
        if cand_key == existing_key:
            return MetricClassification(
                metric=candidate,
                status=CollisionStatus.DUPLICATE,
                matched_label=m.get("label"),
                matched_domain=target_domain,
                message=f"Exact match in target domain '{target_domain}'",
            )

    # 2. Check exact collision in other domains (SHARED)
    for other_dom, doc in other_manifests.items():
        if other_dom == target_domain:
            continue
        for m in doc.get("metrics", []):
            existing_key = make_dedup_key(m.get("label", ""))
            if cand_key == existing_key:
                return MetricClassification(
                    metric=candidate,
                    status=CollisionStatus.SHARED,
                    matched_label=m.get("label"),
                    matched_domain=other_dom,
                    message=f"Exact match found in domain '{other_dom}' (can be shared via domain_tags)",
                )

    # 3. Check UNIT-VARIANT in target domain (same concept_stem, different unit)
    for m in target_metrics:
        existing_stem = m.get("concept_stem")
        existing_unit = (m.get("unit") or "").strip().lower()
        if existing_stem == cand_stem:
            # Different unit: legitimate variant
            return MetricClassification(
                metric=candidate,
                status=CollisionStatus.UNIT_VARIANT,
                matched_label=m.get("label"),
                matched_domain=target_domain,
                message=f"Unit variant of existing '{m.get('label')}' (stem '{cand_stem}')",
            )

    # 4. Check SIMILAR (fuzzy comparison on unit-stripped label text)
    best_match_label: Optional[str] = None
    best_score: float = 0.0

    for m in target_metrics:
        existing_label = m.get("label", "")
        existing_stripped = strip_unit_suffix(existing_label).casefold()
        if not existing_stripped or not cand_stripped:
            continue

        score = float(rapidfuzz.fuzz.ratio(cand_stripped, existing_stripped))
        if score >= fuzzy_threshold and score > best_score:
            best_score = score
            best_match_label = existing_label

    if best_match_label is not None and best_score >= fuzzy_threshold:
        return MetricClassification(
            metric=candidate,
            status=CollisionStatus.SIMILAR,
            matched_label=best_match_label,
            matched_domain=target_domain,
            similarity_score=best_score,
            message=f"Near-duplicate ({best_score:.1f}% match) with existing '{best_match_label}'",
        )

    # 5. Clean new metric
    return MetricClassification(
        metric=candidate,
        status=CollisionStatus.NEW,
        message="New unique metric",
    )


def classify_title(
    candidate: IngestedTitle,
    target_domain: str,
    target_manifest: Optional[Dict[str, Any]],
    other_manifests: Dict[str, Dict[str, Any]],
) -> TitleClassification:
    """Classify a chart title: exact match in target domain is DUPLICATE, in other is SHARED, else NEW."""
    cand_key = make_dedup_key(candidate.title)
    target_titles = target_manifest.get("titles", []) if target_manifest else []

    for t in target_titles:
        if make_dedup_key(t.get("title", "")) == cand_key:
            return TitleClassification(
                title=candidate,
                status=CollisionStatus.DUPLICATE,
                matched_title=t.get("title"),
                matched_domain=target_domain,
                message=f"Title already exists in domain '{target_domain}'",
            )

    for other_dom, doc in other_manifests.items():
        if other_dom == target_domain:
            continue
        for t in doc.get("titles", []):
            if make_dedup_key(t.get("title", "")) == cand_key:
                return TitleClassification(
                    title=candidate,
                    status=CollisionStatus.SHARED,
                    matched_title=t.get("title"),
                    matched_domain=other_dom,
                    message=f"Title already exists in domain '{other_dom}'",
                )

    return TitleClassification(
        title=candidate,
        status=CollisionStatus.NEW,
        message="New chart title",
    )


def classify_pair(
    candidate: IngestedPair,
    target_domain: str,
    target_manifest: Optional[Dict[str, Any]],
    other_manifests: Dict[str, Dict[str, Any]],
) -> PairClassification:
    """Classify a comparative pair: unordered de-dup (control, treatment) == (treatment, control)."""
    cand_set = {make_dedup_key(candidate.control), make_dedup_key(candidate.treatment)}
    target_pairs = target_manifest.get("comparative_pairs", []) if target_manifest else []

    for p in target_pairs:
        pair_set = {make_dedup_key(p.get("control", "")), make_dedup_key(p.get("treatment", ""))}
        if cand_set == pair_set:
            return PairClassification(
                pair=candidate,
                status=CollisionStatus.DUPLICATE,
                matched_pair=(p.get("control", ""), p.get("treatment", "")),
                matched_domain=target_domain,
                message=f"Comparative pair already exists in domain '{target_domain}'",
            )

    for other_dom, doc in other_manifests.items():
        if other_dom == target_domain:
            continue
        for p in doc.get("comparative_pairs", []):
            pair_set = {make_dedup_key(p.get("control", "")), make_dedup_key(p.get("treatment", ""))}
            if cand_set == pair_set:
                return PairClassification(
                    pair=candidate,
                    status=CollisionStatus.SHARED,
                    matched_pair=(p.get("control", ""), p.get("treatment", "")),
                    matched_domain=other_dom,
                    message=f"Comparative pair already exists in domain '{other_dom}'",
                )

    return PairClassification(
        pair=candidate,
        status=CollisionStatus.NEW,
        message="New comparative pair",
    )
