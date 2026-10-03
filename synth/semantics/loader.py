"""
synth/semantics/loader.py

High-performance domain manifest loader and SemanticRegistry compiler.
Compiles declarative YAML manifests into immutable slotted dataclasses and pre-indexed O(1) lookup tables.
Enforces strict dual-mode isolation: zero top-level pydantic imports to preserve <= 50 ms cold startup.
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings

from synth.semantics.enums import AxisRole, ScaleType

ALL_CHART_TYPES: Tuple[str, ...] = ("bar", "line", "scatter", "box", "area", "histogram", "pie", "heatmap")
CARTESIAN_CHART_TYPES: Tuple[str, ...] = ("bar", "line", "scatter", "box", "area", "histogram")

HEATMAP_ONLY_TITLES: Set[str] = {
    "Confusion Matrix",
    "Proteomics Heatmap",
    "User Activity Heatmap",
    "Correlation Matrix",
    "Cohort Retention",
    "Clustering Analysis",
    "Cross-Validation Results",
    "Dimensionality Reduction",
}

GENERIC_SCIENTIFIC_BACKUPS: Tuple[str, ...] = (
    "Experimental Results",
    "Comparative Analysis",
    "Statistical Summary",
    "Distribution of Values",
    "Experiment Summary",
    "Time Series",
)

GENERIC_BUSINESS_BACKUPS: Tuple[str, ...] = (
    "Key Metrics Overview",
    "KPI Dashboard",
    "Quarterly Performance",
    "Comparative Analysis",
    "Statistical Summary",
    "Executive Summary Dashboard",
)

STATIC_DOMAIN_FALLBACKS: Dict[str, Tuple[str, str]] = {
    "biomedical": ("Time (h)", "Concentration (μM)"),
    "engineering": ("Time (s)", "Temperature (°C)"),
    "business": ("Quarter", "Revenue ($M)"),
    "demographic": ("Age Cohort", "Employment Rate (%)"),
}

_SCALE_TYPE_MAP: Dict[str, ScaleType] = {s.value: s for s in ScaleType}
_AXIS_ROLE_MAP: Dict[str, AxisRole] = {r.value: r for r in AxisRole}


def get_domains_dir() -> Path:
    override = os.environ.get("SYNTH_DOMAINS_DIR")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent / "domains"


DOMAINS_DIR = get_domains_dir()
CACHE_FILE = DOMAINS_DIR / ".cache.json"


@dataclass(slots=True, frozen=True)
class DomainInfo:
    id: str
    display_name: str
    is_scientific: bool
    default_weight: float
    fallbacks: Tuple[str, str]
    capabilities: Dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AxisMetric:
    label: str
    scale_type: ScaleType
    role: AxisRole
    concept_stem: str
    unit: Optional[str] = None
    concept_group: Optional[str] = None
    domain_tags: Tuple[str, ...] = field(default_factory=tuple)
    is_scientific: bool = False


@dataclass(slots=True, frozen=True)
class ChartTitle:
    title: str
    allowed_chart_types: Tuple[str, ...]
    domain_tags: Tuple[str, ...] = field(default_factory=tuple)
    is_scientific: Optional[bool] = None


@dataclass(slots=True, frozen=True)
class ComparativePair:
    control: str
    treatment: str
    domain_category: str
    context_tags: Tuple[str, ...] = field(default_factory=tuple)
    is_scientific: bool = False


def _compute_cache_fingerprint(yaml_files: List[Path]) -> List[Dict[str, Any]]:
    fp = []
    for yf in sorted(yaml_files, key=lambda p: p.name):
        try:
            st = yf.stat()
            fp.append({"name": yf.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
        except OSError:
            pass
    return fp


def _load_manifest_raw_data(domains_dir: Optional[Path] = None, force: bool = False) -> Dict[str, Any]:
    """
    Load raw dictionary data for all domain manifests.
    Validates cache freshness against fingerprint (names, sizes, mtimes).
    Emits warnings for .yml extensions, enforces unique domain_ids, and performs atomic cache writes.
    """
    target_dir = domains_dir or get_domains_dir()
    cache_file = target_dir / ".cache.json"

    yml_files = list(target_dir.glob("*.yml"))
    for yf in yml_files:
        warnings.warn(f"Manifest '{yf.name}' uses '.yml' extension; standard extension is '.yaml'", UserWarning)

    yaml_files = list(target_dir.glob("*.yaml"))
    if not yaml_files:
        return {}

    current_fp = _compute_cache_fingerprint(yaml_files)

    if not force and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and "fingerprint" in cached and "manifests" in cached:
                if cached["fingerprint"] == current_fp:
                    return cached["manifests"]
        except Exception:
            pass

    # Compile from YAML files using CSafeLoader
    try:
        from yaml import CSafeLoader as FastYamlLoader
    except ImportError:
        from yaml import SafeLoader as FastYamlLoader  # type: ignore
    import yaml

    data: Dict[str, Any] = {}
    seen_domains: Dict[str, str] = {}

    for yf in sorted(yaml_files, key=lambda p: p.name):
        with open(yf, "rb") as f:
            doc = yaml.load(f, Loader=FastYamlLoader)
            if doc and "domain_id" in doc:
                dom_id = doc["domain_id"]
                if dom_id in seen_domains:
                    raise ValueError(
                        f"Duplicate domain_id '{dom_id}' found in '{yf.name}' (already defined in '{seen_domains[dom_id]}')"
                    )
                seen_domains[dom_id] = yf.name
                data[dom_id] = doc

    # Atomic write to cache file
    payload = {
        "fingerprint": current_fp,
        "manifests": data,
    }
    tmp_path = target_dir / f".cache.json.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_file)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return data


@dataclass(slots=True, frozen=True)
class SemanticRegistry:
    domains: Dict[str, DomainInfo]
    metrics_by_label: Dict[str, AxisMetric]
    metrics_by_domain: Dict[str, Tuple[AxisMetric, ...]]
    metrics_by_role_and_scale: Dict[Tuple[str, AxisRole, ScaleType], Tuple[AxisMetric, ...]]
    titles_by_domain_and_chart: Dict[Tuple[str, str], Tuple[str, ...]]
    pairs_by_domain: Dict[str, Tuple[ComparativePair, ...]]
    legacy_pairs_by_domain: Dict[str, Tuple[ComparativePair, ...]]
    title_index: Dict[Tuple[bool, str], Tuple[str, ...]]
    domain_title_index: Dict[Tuple[bool, str, str], Tuple[str, ...]]
    all_metrics: Tuple[AxisMetric, ...]
    all_titles: Tuple[ChartTitle, ...]
    all_pairs: Tuple[ComparativePair, ...]
    legacy_metrics: Tuple[AxisMetric, ...]
    canonical_metrics: Tuple[AxisMetric, ...]
    axis_metrics_catalog: Tuple[AxisMetric, ...]
    axis_metric_by_label: Dict[str, AxisMetric]
    legacy_metric_by_label: Dict[str, AxisMetric]
    histogram_y_labels: Tuple[str, ...]
    titles_catalog: Tuple[ChartTitle, ...]
    pairs_catalog: Tuple[ComparativePair, ...]

    @classmethod
    def load(cls, raw: Optional[Dict[str, Any]] = None, domains_dir: Optional[Path] = None) -> "SemanticRegistry":
        raw_manifests = raw if raw is not None else _load_manifest_raw_data(domains_dir=domains_dir)

        metrics_by_label: Dict[str, AxisMetric] = {}
        metrics_by_domain_lists: Dict[str, List[AxisMetric]] = {}
        role_scale_lists: Dict[Tuple[str, AxisRole, ScaleType], List[AxisMetric]] = {}
        titles_domain_chart_lists: Dict[Tuple[str, str], List[str]] = {}
        pairs_domain_lists: Dict[str, List[ComparativePair]] = {}

        all_metrics_list: List[AxisMetric] = []
        all_titles_list: List[ChartTitle] = []
        all_pairs_list: List[ComparativePair] = []

        legacy_metrics_list: List[AxisMetric] = []
        canonical_metrics_list: List[AxisMetric] = []
        histogram_y_labels_list: List[str] = []

        # Load order: sorted by explicit order if declared, placing 'common' last
        domain_keys = sorted(
            raw_manifests.keys(),
            key=lambda d: (raw_manifests[d].get("order", 50 if d != "common" else 1000), d),
        )

        domains_map: Dict[str, DomainInfo] = {}

        # First pass: parse DomainInfo metadata
        common_metrics_raw = raw_manifests.get("common", {}).get("metrics", [])
        common_ind_labels = [
            m["label"] for m in common_metrics_raw
            if _AXIS_ROLE_MAP.get(m.get("role", "")) in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)
        ]

        for dom_id in domain_keys:
            doc = raw_manifests[dom_id]
            is_sci = bool(doc.get("is_scientific", False))
            def_weight = float(doc.get("default_weight", 1.0))
            caps = dict(doc.get("capabilities") or {"tabular_preset": False, "heatmap_pools": False})

            fallbacks_raw = doc.get("default_fallbacks")
            if fallbacks_raw and len(fallbacks_raw) == 2:
                fallbacks = (str(fallbacks_raw[0]), str(fallbacks_raw[1]))
            elif dom_id in STATIC_DOMAIN_FALLBACKS:
                fallbacks = STATIC_DOMAIN_FALLBACKS[dom_id]
            else:
                # Derive fallbacks per SEM-05
                dom_m_raw = doc.get("metrics", [])
                first_ind = next(
                    (m["label"] for m in dom_m_raw if _AXIS_ROLE_MAP.get(m.get("role", "")) in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)),
                    None,
                )
                if first_ind is None:
                    first_ind = common_ind_labels[0] if common_ind_labels else "Index"
                first_dep = next(
                    (m["label"] for m in dom_m_raw if _AXIS_ROLE_MAP.get(m.get("role", "")) in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL)),
                    None,
                )
                if first_dep is None:
                    first_dep = dom_m_raw[0]["label"] if dom_m_raw else "Value"
                fallbacks = (first_ind, first_dep)

            domains_map[dom_id] = DomainInfo(
                id=dom_id,
                display_name=doc.get("display_name", dom_id.capitalize()),
                is_scientific=is_sci,
                default_weight=def_weight,
                fallbacks=fallbacks,
                capabilities=caps,
            )

        for dom_id in domain_keys:
            doc = raw_manifests[dom_id]
            is_sci = bool(doc.get("is_scientific", False))

            # 1. Metrics
            raw_metrics = doc.get("metrics", [])
            for m_raw in raw_metrics:
                dtags = tuple(m_raw.get("domain_tags", [dom_id]))
                scale = _SCALE_TYPE_MAP[m_raw["scale_type"]]
                role = _AXIS_ROLE_MAP[m_raw["role"]]
                pools = m_raw.get("pools") or []

                metric = AxisMetric(
                    label=m_raw["label"],
                    scale_type=scale,
                    role=role,
                    concept_stem=m_raw["concept_stem"],
                    unit=m_raw.get("unit"),
                    concept_group=m_raw.get("concept_group"),
                    domain_tags=dtags,
                    is_scientific=is_sci,
                )
                metrics_by_label[metric.label] = metric
                all_metrics_list.append(metric)

                # Segment pools explicitly (SEM-03)
                if "canonical_independent" in pools:
                    canonical_metrics_list.append(metric)
                if "histogram_y" in pools:
                    histogram_y_labels_list.append(metric.label)
                if "legacy" in pools:
                    legacy_metrics_list.append(metric)

                for tag in set(dtags) | {dom_id}:
                    metrics_by_domain_lists.setdefault(tag, []).append(metric)
                    role_scale_lists.setdefault((tag, role, scale), []).append(metric)

            # 2. Titles
            raw_titles = doc.get("titles", [])
            for t_raw in raw_titles:
                t_dtags = tuple(t_raw.get("domain_tags", [dom_id]))
                c_types = tuple(t_raw.get("allowed_chart_types", []))

                # Compute is_scientific from domain tags and manifest flags (SEM-02)
                title_is_sci: Optional[bool] = None
                has_sci = False
                has_non_sci = False
                for t in t_dtags:
                    if t in domains_map and t != "common":
                        if domains_map[t].is_scientific:
                            has_sci = True
                        else:
                            has_non_sci = True

                if has_sci and not has_non_sci:
                    title_is_sci = True
                elif has_non_sci and not has_sci:
                    title_is_sci = False

                chart_title = ChartTitle(
                    title=t_raw["title"],
                    allowed_chart_types=c_types,
                    domain_tags=t_dtags,
                    is_scientific=title_is_sci,
                )
                all_titles_list.append(chart_title)

                for tag in set(t_dtags) | {dom_id}:
                    for ctype in c_types:
                        titles_domain_chart_lists.setdefault((tag, ctype), []).append(t_raw["title"])

            # 3. Comparative Pairs
            raw_pairs = doc.get("comparative_pairs", [])
            for p_raw in raw_pairs:
                cat = p_raw.get("domain_category", dom_id)
                pair_is_sci = domains_map[cat].is_scientific if cat in domains_map else is_sci
                pair = ComparativePair(
                    control=p_raw["control"],
                    treatment=p_raw["treatment"],
                    domain_category=cat,
                    context_tags=tuple(p_raw.get("context_tags", [cat])),
                    is_scientific=pair_is_sci,
                )
                all_pairs_list.append(pair)
                pairs_domain_lists.setdefault(cat, []).append(pair)

        # Freeze lookup structures
        metrics_by_domain = {k: tuple(v) for k, v in metrics_by_domain_lists.items()}
        metrics_by_role_and_scale = {k: tuple(v) for k, v in role_scale_lists.items()}
        titles_by_domain_and_chart = {k: tuple(dict.fromkeys(v)) for k, v in titles_domain_chart_lists.items()}
        pairs_by_domain = {k: tuple(v) for k, v in pairs_domain_lists.items()}

        # Build Title Index tables
        base_title_index: Dict[Tuple[bool, str], List[str]] = {}
        domain_title_index_dict: Dict[Tuple[bool, str, str], List[str]] = {}

        for is_sci_flag in (True, False):
            for ctype in ALL_CHART_TYPES:
                base_title_index[(is_sci_flag, ctype)] = []
                for dom in domain_keys:
                    if dom != "common":
                        domain_title_index_dict[(is_sci_flag, ctype, dom)] = []

        for item in all_titles_list:
            for ctype in item.allowed_chart_types:
                for is_sci_flag in (True, False):
                    if item.is_scientific is None or item.is_scientific == is_sci_flag:
                        base_title_index[(is_sci_flag, ctype)].append(item.title)
                        for dom in item.domain_tags:
                            if (is_sci_flag, ctype, dom) in domain_title_index_dict:
                                domain_title_index_dict[(is_sci_flag, ctype, dom)].append(item.title)

        frozen_base_titles: Dict[Tuple[bool, str], Tuple[str, ...]] = {}
        for (is_sci_flag, ctype), pool in base_title_index.items():
            unique_titles = list(dict.fromkeys(pool))
            if len(unique_titles) < 6:
                backups = GENERIC_SCIENTIFIC_BACKUPS if is_sci_flag else GENERIC_BUSINESS_BACKUPS
                for b in backups:
                    if b not in unique_titles and b not in HEATMAP_ONLY_TITLES:
                        unique_titles.append(b)
                    if len(unique_titles) >= 6:
                        break
            frozen_base_titles[(is_sci_flag, ctype)] = tuple(sorted(unique_titles))

        frozen_domain_titles: Dict[Tuple[bool, str, str], Tuple[str, ...]] = {}
        for (is_sci_flag, ctype, dom), pool in domain_title_index_dict.items():
            frozen_domain_titles[(is_sci_flag, ctype, dom)] = tuple(sorted(dict.fromkeys(pool)))

        legacy_metrics = tuple(legacy_metrics_list)
        canonical_metrics = tuple(canonical_metrics_list)
        axis_metrics_catalog = tuple(all_metrics_list)
        axis_metric_by_label = metrics_by_label
        legacy_metric_by_label = {m.label: m for m in legacy_metrics}
        histogram_y_labels = tuple(histogram_y_labels_list)

        pairs_catalog = tuple(all_pairs_list)
        legacy_pairs_by_domain = pairs_by_domain
        titles_catalog = tuple(all_titles_list)

        return cls(
            domains=domains_map,
            metrics_by_label=metrics_by_label,
            metrics_by_domain=metrics_by_domain,
            metrics_by_role_and_scale=metrics_by_role_and_scale,
            titles_by_domain_and_chart=titles_by_domain_and_chart,
            pairs_by_domain=pairs_by_domain,
            legacy_pairs_by_domain=legacy_pairs_by_domain,
            title_index=frozen_base_titles,
            domain_title_index=frozen_domain_titles,
            all_metrics=tuple(all_metrics_list),
            all_titles=tuple(all_titles_list),
            all_pairs=tuple(all_pairs_list),
            legacy_metrics=legacy_metrics,
            canonical_metrics=canonical_metrics,
            axis_metrics_catalog=axis_metrics_catalog,
            axis_metric_by_label=axis_metric_by_label,
            legacy_metric_by_label=legacy_metric_by_label,
            histogram_y_labels=histogram_y_labels,
            titles_catalog=titles_catalog,
            pairs_catalog=pairs_catalog,
        )


def rebuild_cache(force: bool = True, domains_dir: Optional[Path] = None) -> SemanticRegistry:
    """Rebuild manifest cache from YAMLs and reload the global singleton registry."""
    global registry, DOMAINS_DIR, CACHE_FILE
    DOMAINS_DIR = domains_dir or get_domains_dir()
    CACHE_FILE = DOMAINS_DIR / ".cache.json"
    raw = _load_manifest_raw_data(domains_dir=DOMAINS_DIR, force=force)
    registry = SemanticRegistry.load(raw=raw, domains_dir=DOMAINS_DIR)
    return registry


# Cached module singleton
registry: SemanticRegistry = SemanticRegistry.load()
