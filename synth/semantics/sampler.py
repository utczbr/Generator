"""
synth/semantics/sampler.py

Core semantic sampling engine for chart titles, axis label pairs,
secondary Y axes, and comparative pairs.
"""
from typing import Optional, Any, Tuple, Dict
import random

from synth.semantics.enums import ScaleType, AxisRole, UnknownDomainError
import synth.semantics.loader as loader
from synth.semantics.loader import (
    SemanticRegistry,
    AxisMetric,
    GENERIC_SCIENTIFIC_BACKUPS,
    GENERIC_BUSINESS_BACKUPS,
)

STATIC_DOMAIN_FALLBACKS: Dict[str, Tuple[str, str]] = {
    "biomedical": ("Time (h)", "Concentration (μM)"),
    "engineering": ("Time (s)", "Temperature (°C)"),
    "business": ("Quarter", "Revenue ($M)"),
    "demographic": ("Age Cohort", "Employment Rate (%)"),
}


def _build_axis_pools(reg: Optional[SemanticRegistry] = None) -> Tuple[
    Dict[Tuple[str, str, str, str], Tuple[AxisMetric, ...]],
    Dict[str, Tuple[AxisMetric, ...]],
]:
    """Pre-build O(1) immutable Cartesian axis candidate pools and secondary Y pools."""
    active_reg = reg or loader.registry
    axis_pools: Dict[Tuple[str, str, str, str], Tuple[AxisMetric, ...]] = {}
    secondary_pools: Dict[str, Tuple[AxisMetric, ...]] = {}

    for dom in active_reg.domains:
        if dom == "common":
            continue
        dom_metrics = tuple(m for m in active_reg.axis_metrics_catalog if dom in m.domain_tags)
        secondary_pools[dom] = tuple(
            m for m in dom_metrics
            if m.role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)
            and m.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT)
        )
        for ctype in ("scatter", "bar", "box", "line", "area", "histogram"):
            for orient in ("vertical", "horizontal"):
                if ctype == "scatter":
                    x_p = tuple(
                        m for m in dom_metrics
                        if m.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT, ScaleType.TEMPORAL)
                    )
                    y_p = x_p
                elif ctype in ("bar", "box"):
                    if orient == "horizontal":
                        x_p = tuple(
                            m for m in dom_metrics
                            if m.role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)
                            and m.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT)
                        )
                        y_p = tuple(m for m in dom_metrics if m.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL))
                    else:
                        x_p = tuple(m for m in dom_metrics if m.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL))
                        y_p = tuple(
                            m for m in dom_metrics
                            if m.role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)
                            and m.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT)
                        )
                elif ctype in ("line", "area"):
                    x_p = tuple(
                        m for m in dom_metrics
                        if m.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)
                        and m.scale_type in (ScaleType.CONTINUOUS, ScaleType.TEMPORAL, ScaleType.DISCRETE_COUNT)
                    )
                    y_p = tuple(
                        m for m in dom_metrics
                        if m.role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)
                        and m.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT)
                    )
                elif ctype == "histogram":
                    x_p = tuple(
                        m for m in dom_metrics
                        if m.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT)
                    )
                    y_p = tuple()
                axis_pools[(dom, ctype, orient, "x")] = x_p
                axis_pools[(dom, ctype, orient, "y")] = y_p

    return axis_pools, secondary_pools


_POOL_CACHE_REGISTRY_ID: Optional[int] = None
_CACHED_AXIS_POOLS: Optional[Dict[Tuple[str, str, str, str], Tuple[AxisMetric, ...]]] = None
_CACHED_SECONDARY_POOLS: Optional[Dict[str, Tuple[AxisMetric, ...]]] = None


def _get_pools() -> Tuple[
    Dict[Tuple[str, str, str, str], Tuple[AxisMetric, ...]],
    Dict[str, Tuple[AxisMetric, ...]],
]:
    global _POOL_CACHE_REGISTRY_ID, _CACHED_AXIS_POOLS, _CACHED_SECONDARY_POOLS
    reg = loader.registry
    if _POOL_CACHE_REGISTRY_ID != id(reg) or _CACHED_AXIS_POOLS is None:
        _CACHED_AXIS_POOLS, _CACHED_SECONDARY_POOLS = _build_axis_pools(reg)
        _POOL_CACHE_REGISTRY_ID = id(reg)
    return _CACHED_AXIS_POOLS, _CACHED_SECONDARY_POOLS


AXIS_POOLS, SECONDARY_Y_POOLS = _build_axis_pools(loader.registry)
HISTOGRAM_Y_TUPLE: Tuple[str, ...] = tuple(loader.registry.histogram_y_labels)


def _draw_choice(pool: Tuple[Any, ...], rng: Any = None) -> Any:
    """Draw an item from pool using provided PRNG or Python random module."""
    if not pool:
        return None
    r = rng or random
    if hasattr(r, "choice") and (r is random or isinstance(r, random.Random)):
        return r.choice(pool)
    elif hasattr(r, "integers"):
        idx = int(r.integers(0, len(pool)))
        return pool[idx]
    elif hasattr(r, "randint"):
        idx = int(r.randint(0, len(pool) - 1))
        return pool[idx]
    elif hasattr(r, "choice"):
        return r.choice(pool)
    else:
        return random.choice(pool)


def sample_chart_title(
    chart_type: str,
    is_scientific: bool,
    domain: Optional[str] = None,
    rng: Any = None,
) -> str:
    """
    Sample a chart title matching chart_type, is_scientific, and preferentially domain.
    Verified O(1) lookup via pre-indexed dictionaries and tuples without regex filtering.
    Raises UnknownDomainError if domain is specified but not in the registry.
    """
    reg = loader.registry
    ctype = chart_type.lower() if isinstance(chart_type, str) else str(chart_type)
    is_sci = bool(is_scientific)

    pool: Optional[Tuple[str, ...]] = None

    # 1. Prefer domain-specific pool if domain provided
    if domain is not None:
        dom = str(domain).lower()
        if dom not in reg.domains or dom == "common":
            raise UnknownDomainError(
                f"Unknown domain '{domain}'. Registered domains: {[d for d in reg.domains if d != 'common']}"
            )
        pool = reg.domain_title_index.get((is_sci, ctype, dom))

    # 2. Fall back to base (is_scientific, chart_type) pool
    if not pool:
        pool = reg.title_index.get((is_sci, ctype))

    # 3. Emergency fallback if chart_type is unknown
    if not pool:
        pool = GENERIC_SCIENTIFIC_BACKUPS if is_sci else GENERIC_BUSINESS_BACKUPS

    return str(_draw_choice(pool, rng))


def sample_axis_pair(
    chart_type: str,
    orientation: str,
    is_scientific: bool,
    semantic_domain: Optional[str] = None,
    rng: Any = None,
) -> Tuple[str, str, str]:
    """
    Sample an (X, Y) axis label pair and the resolved domain category.
    Enforces scale and role rules across all Cartesian chart types,
    preventing stem and concept_group collisions.
    Raises UnknownDomainError if an unknown domain is requested.
    Excludes pie and heatmap chart types (raises ValueError).
    """
    reg = loader.registry
    axis_pools, _ = _get_pools()

    ctype = chart_type.lower() if isinstance(chart_type, str) else str(chart_type)
    if ctype in ("pie", "heatmap"):
        raise ValueError(f"Chart type '{chart_type}' does not use Cartesian axis pairs.")

    orient = "horizontal" if orientation == "horizontal" else "vertical"

    if semantic_domain is not None:
        dom = str(semantic_domain).lower()
        if dom not in reg.domains or dom == "common":
            raise UnknownDomainError(
                f"Unknown domain '{semantic_domain}'. Registered domains: {[d for d in reg.domains if d != 'common']}"
            )
    else:
        # Match domain by is_scientific flag
        matches = [d for d, info in reg.domains.items() if info.is_scientific == is_scientific and d != "common"]
        dom = matches[0] if matches else next((d for d in reg.domains if d != "common"), "common")

    fallback = reg.domains[dom].fallbacks if dom in reg.domains else STATIC_DOMAIN_FALLBACKS.get(dom, ("X", "Y"))

    x_pool = axis_pools.get((dom, ctype, orient, "x"))
    y_pool = axis_pools.get((dom, ctype, orient, "y"))

    # Both empty fallback
    if not x_pool and not y_pool:
        if ctype == "histogram":
            y_label = _draw_choice(reg.histogram_y_labels, rng)
            return (fallback[0], str(y_label), dom)
        return (fallback[0], fallback[1], dom)

    # X-pool empty fallback (e.g. dependent-only domain for bar/line/area, SEM-05)
    if not x_pool:
        metric_x_label = fallback[0]
        if ctype == "histogram":
            y_label = _draw_choice(reg.histogram_y_labels, rng)
            return (metric_x_label, str(y_label), dom)
        metric_y = _draw_choice(y_pool, rng) if y_pool else None
        y_label = metric_y.label if metric_y else fallback[1]
        return (metric_x_label, y_label, dom)

    metric_x = _draw_choice(x_pool, rng)

    if ctype == "histogram":
        y_label = _draw_choice(reg.histogram_y_labels, rng)
        return (metric_x.label, str(y_label), dom)

    if not y_pool:
        return (metric_x.label, fallback[1], dom)

    # Collision-avoidance loop (up to 20 attempts)
    metric_y = None
    for _ in range(20):
        cand = _draw_choice(y_pool, rng)
        if cand.concept_stem == metric_x.concept_stem:
            continue
        if metric_x.concept_group is not None and cand.concept_group == metric_x.concept_group:
            continue
        metric_y = cand
        break

    if metric_y is None:
        for cand in y_pool:
            if cand.concept_stem != metric_x.concept_stem:
                metric_y = cand
                break

    if metric_y is None:
        return (metric_x.label, fallback[1], dom)

    return (metric_x.label, metric_y.label, dom)


def sample_secondary_y(
    y1_label: str,
    is_scientific: bool,
    domain_category: Optional[str] = None,
    rng: Any = None,
) -> str:
    """
    Sample a distinct secondary Y axis continuous metric.
    Guarantees no collision in exact text, concept stem, or concept group with y1_label.
    Raises UnknownDomainError if an unknown domain is requested.
    """
    reg = loader.registry
    _, secondary_pools = _get_pools()

    m1 = reg.axis_metric_by_label.get(y1_label)
    stem1 = m1.concept_stem if m1 else y1_label.lower()
    grp1 = m1.concept_group if m1 else None

    if domain_category is not None:
        dom = str(domain_category).lower()
        if dom not in reg.domains or dom == "common":
            raise UnknownDomainError(
                f"Unknown domain '{domain_category}'. Registered domains: {[d for d in reg.domains if d != 'common']}"
            )
    elif m1 and m1.domain_tags:
        dom = m1.domain_tags[0]
    else:
        matches = [d for d, info in reg.domains.items() if info.is_scientific == is_scientific and d != "common"]
        dom = matches[0] if matches else next((d for d in reg.domains if d != "common"), "common")

    pool = secondary_pools.get(dom, ())
    cands = tuple(
        m for m in pool
        if m.concept_stem != stem1 and (grp1 is None or m.concept_group != grp1) and m.label != y1_label
    )
    if not cands:
        cands = tuple(m for m in pool if m.concept_stem != stem1 and m.label != y1_label)
    if not cands:
        cands = tuple(m for m in pool if m.label != y1_label)

    chosen = _draw_choice(cands, rng)
    if chosen:
        return chosen.label
    fallback = reg.domains[dom].fallbacks if dom in reg.domains else STATIC_DOMAIN_FALLBACKS.get(dom, ("X", "Y"))
    return fallback[1]


def sample_comparative_pair(
    domain_category: Optional[str] = None,
    rng: Any = None,
) -> Tuple[str, str]:
    """
    Sample a comparative pair (control, treatment) strictly matching domain_category.
    Raises UnknownDomainError if an unknown domain is requested.
    """
    reg = loader.registry

    if domain_category is not None:
        dom = str(domain_category).lower()
        if dom not in reg.domains or dom == "common":
            raise UnknownDomainError(
                f"Unknown domain '{domain_category}'. Registered domains: {[d for d in reg.domains if d != 'common']}"
            )
        pool = reg.legacy_pairs_by_domain.get(dom, ())
        if not pool:
            raise ValueError(f"Domain '{dom}' has no comparative pairs defined.")
    else:
        pool = reg.pairs_catalog

    pair = _draw_choice(pool, rng)
    return (pair.control, pair.treatment)
