"""
synth/semantics/catalog.py

Authoritative semantic catalog for chart titles, domain categorization,
axis metrics, scale types, roles, concept groups, and comparative pairs.
Exposes pre-indexed O(1) lookup tables compiled from declarative YAML manifests
via synth.semantics.loader.registry for 100% backward compatibility.
"""
from typing import Dict, List, Optional, Set, Tuple
import re

from synth.semantics.loader import (
    ScaleType,
    AxisRole,
    AxisMetric,
    ChartTitle,
    ComparativePair,
    ALL_CHART_TYPES,
    CARTESIAN_CHART_TYPES,
    HEATMAP_ONLY_TITLES,
    GENERIC_SCIENTIFIC_BACKUPS,
    GENERIC_BUSINESS_BACKUPS,
    registry,
)


def compute_concept_stem(label: str) -> str:
    """Compute normalized semantic stem by lowercasing, stripping units/currency/punct."""
    s = label.lower()
    s = re.sub(r"\(.*?\)", "", s)   # Strip parenthetical units/annotations
    s = re.sub(r"[$€£¥]", "", s)    # Strip currency symbols
    s = re.sub(r"[^\w\s]", "", s)   # Strip remaining punctuation
    return " ".join(s.split())       # Collapse whitespace


# Authoritative catalogs consuming declarative manifests via loader.registry
CHART_TITLES_CATALOG: Tuple[ChartTitle, ...] = registry.titles_catalog
TITLE_INDEX: Dict[Tuple[bool, str], Tuple[str, ...]] = registry.title_index
DOMAIN_TITLE_INDEX: Dict[Tuple[bool, str, str], Tuple[str, ...]] = registry.domain_title_index

COMPARATIVE_PAIRS_CATALOG: Tuple[ComparativePair, ...] = registry.pairs_catalog
COMPARATIVE_PAIRS_BY_DOMAIN: Dict[str, Tuple[ComparativePair, ...]] = registry.legacy_pairs_by_domain

LEGACY_METRICS_CATALOG: Tuple[AxisMetric, ...] = registry.legacy_metrics
CANONICAL_INDEPENDENT_METRICS: Tuple[AxisMetric, ...] = registry.canonical_metrics
AXIS_METRICS_CATALOG: Tuple[AxisMetric, ...] = registry.axis_metrics_catalog
METRIC_BY_LABEL: Dict[str, AxisMetric] = registry.axis_metric_by_label
LEGACY_METRIC_BY_LABEL: Dict[str, AxisMetric] = registry.legacy_metric_by_label
HISTOGRAM_Y_LABELS: Tuple[str, ...] = registry.histogram_y_labels

# Semantic domain filtering helpers
SCIENTIFIC_AXIS_METRICS: Tuple[AxisMetric, ...] = tuple(m for m in AXIS_METRICS_CATALOG if m.is_scientific)
BUSINESS_AXIS_METRICS: Tuple[AxisMetric, ...] = tuple(m for m in AXIS_METRICS_CATALOG if not m.is_scientific)
CHART_TITLES_BY_TYPE: Dict[Tuple[str, str], Tuple[str, ...]] = registry.titles_by_domain_and_chart
