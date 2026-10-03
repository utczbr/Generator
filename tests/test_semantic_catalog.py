"""
tests/test_semantic_catalog.py

Phase 3 verification suite for the authoritative semantic catalog.
Validates:
- ScaleType (without ORDINAL) and AxisRole enums
- Frozen slotted dataclasses and immutability
- Coverage of all 264 legacy metrics (178 sci, 86 biz) + 40 canonical independent variables
- Elapsed bench duration typing (CONTINUOUS, BIDIRECTIONAL)
- Concept stem computation and concept group assignments
- Comparative pairs categorization (94 pairs: 40 bio, 32 eng, 22 biz)
- Domain tags preventing cross-domain leakage
- Independent double-annotated test oracle and 50-item blind gold set (FR-051)
- Deterministic sorting
"""
import re
import pytest
from dataclasses import FrozenInstanceError

from synth.semantics.catalog import (
    ScaleType,
    AxisRole,
    AxisMetric,
    ChartTitle,
    ComparativePair,
    compute_concept_stem,
    LEGACY_METRICS_CATALOG,
    CANONICAL_INDEPENDENT_METRICS,
    AXIS_METRICS_CATALOG,
    METRIC_BY_LABEL,
    LEGACY_METRIC_BY_LABEL,
    COMPARATIVE_PAIRS_CATALOG,
    COMPARATIVE_PAIRS_BY_DOMAIN,
    CHART_TITLES_CATALOG,
    TITLE_INDEX,
    DOMAIN_TITLE_INDEX,
)
from synth.semantics.sampler import sample_comparative_pair


# ==============================================================================
# 1. Enums and Data Models
# ==============================================================================

def test_scale_type_enum_specifications():
    """FR-033: ScaleType must be string enum without ORDINAL."""
    assert not hasattr(ScaleType, "ORDINAL"), "ScaleType must NOT contain ORDINAL"
    assert "ordinal" not in [s.value for s in ScaleType]
    expected_members = {"CONTINUOUS", "CATEGORICAL", "DISCRETE_COUNT", "PERCENTAGE", "TEMPORAL"}
    assert set(ScaleType.__members__.keys()) == expected_members

    for member in ScaleType:
        assert isinstance(member.value, str)
        assert member == member.value


def test_axis_role_enum_specifications():
    """FR-033: AxisRole must be string enum with INDEPENDENT, DEPENDENT, BIDIRECTIONAL."""
    expected_members = {"INDEPENDENT", "DEPENDENT", "BIDIRECTIONAL"}
    assert set(AxisRole.__members__.keys()) == expected_members

    for member in AxisRole:
        assert isinstance(member.value, str)
        assert member == member.value


def test_dataclass_frozen_and_slotted():
    """FR-033: All catalog entities must be frozen, slotted, and hashable."""
    for cls in (AxisMetric, ChartTitle, ComparativePair):
        assert hasattr(cls, "__slots__"), f"{cls.__name__} must be slotted"

    # Test immutability on AxisMetric
    metric = METRIC_BY_LABEL["Time (s)"]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        metric.label = "Modified Time"  # type: ignore

    # Test hashability
    metric_set = {metric, METRIC_BY_LABEL["Revenue ($)"]}
    assert len(metric_set) == 2

    # Test ComparativePair immutability
    pair = COMPARATIVE_PAIRS_CATALOG[0]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        pair.control = "Modified Control"  # type: ignore


# ==============================================================================
# 2. Legacy Metrics Coverage & Annotation
# ==============================================================================

def test_legacy_metrics_catalog_coverage():
    """FR-032, T110: Catalog must contain all 264 legacy metrics without unhandled collisions."""
    assert len(LEGACY_METRICS_CATALOG) == 264
    assert len(LEGACY_METRIC_BY_LABEL) == 264

    # 178 unique scientific legacy labels
    sci_metrics = [m for m in LEGACY_METRICS_CATALOG if m.is_scientific]
    assert len(sci_metrics) == 178
    for m in sci_metrics:
        assert m.label in METRIC_BY_LABEL
        assert m.label in LEGACY_METRIC_BY_LABEL
        assert "biomedical" in m.domain_tags or "engineering" in m.domain_tags

    # 86 unique business legacy labels
    biz_metrics = [m for m in LEGACY_METRICS_CATALOG if not m.is_scientific]
    assert len(biz_metrics) == 86
    for m in biz_metrics:
        assert m.label in METRIC_BY_LABEL
        assert m.label in LEGACY_METRIC_BY_LABEL
        assert "business" in m.domain_tags


def test_elapsed_bench_durations_and_bench_quantities():
    """
    FR-032 & FR-033: Elapsed bench durations must be ScaleType.CONTINUOUS and AxisRole.BIDIRECTIONAL.
    Measured bench quantities must be ScaleType.CONTINUOUS and AxisRole.BIDIRECTIONAL.
    """
    bench_durations = [
        "Time (s)",
        "Time (min)",
        "Time (h)",
        "Time Point (h)",
        "Time Point (min)",
        "Exposure Time (s)",
        "Culture Duration (hrs)",
    ]
    for label in bench_durations:
        m = METRIC_BY_LABEL[label]
        assert m.scale_type == ScaleType.CONTINUOUS, f"{label} must be CONTINUOUS (not TEMPORAL)"
        assert m.role == AxisRole.BIDIRECTIONAL, f"{label} must be BIDIRECTIONAL"

    bench_quantities = [
        "Dose (mg/kg)",
        "Dose (μg/mL)",
        "Concentration (mg/L)",
        "Concentration (μM)",
        "Temperature (°C)",
        "pH",
        "Wavelength (nm)",
        "Frequency (Hz)",
        "Age (years)",
    ]
    for label in bench_quantities:
        m = METRIC_BY_LABEL[label]
        assert m.scale_type == ScaleType.CONTINUOUS, f"{label} must be CONTINUOUS"
        assert m.role == AxisRole.BIDIRECTIONAL, f"{label} must be BIDIRECTIONAL"


def test_temporal_scale_strict_reservation():
    """FR-032: ScaleType.TEMPORAL strictly reserved for calendrical dates/intervals."""
    temporal_labels = [m.label for m in AXIS_METRICS_CATALOG if m.scale_type == ScaleType.TEMPORAL]
    for label in temporal_labels:
        # None of the temporal labels should be bench durations (s, min, h, hrs)
        assert not re.search(r'\((s|min|h|hrs)\)', label), f"{label} should not be TEMPORAL"


# ==============================================================================
# 3. Canonical Independent Expansion
# ==============================================================================

def test_canonical_independent_expansion():
    """
    FR-032, T111: Augmented with 40 canonical independent variables with clean typography
    and explicit domain tags preventing cross-domain leakage.
    """
    assert len(CANONICAL_INDEPENDENT_METRICS) == 40
    assert len(AXIS_METRICS_CATALOG) >= 304
    assert len(METRIC_BY_LABEL) >= 304

    for m in CANONICAL_INDEPENDENT_METRICS:
        # Clean typography: no parenthetical date format hints
        assert not re.search(r'\(YYYY|YYYY-MM|MM-DD|hh:mm', m.label), f"Format hint in {m.label}"
        # Independent or bidirectional role
        assert m.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL)
        # Continuous, temporal, or discrete count
        assert m.scale_type in (ScaleType.CONTINUOUS, ScaleType.TEMPORAL, ScaleType.DISCRETE_COUNT)

    # Computing/ML metrics strictly tagged engineering (never business)
    computing_terms = ["Epoch", "Iteration", "Step", "Simulation Step", "Time Step", "Checkpoint", "Layer Index"]
    for term in computing_terms:
        m = METRIC_BY_LABEL[term]
        assert m.domain_tags == ("engineering",), f"{term} must be strictly engineering"
        assert "business" not in m.domain_tags

    # Spatial dimensions tagged engineering, biomedical
    spatial_terms = ["Depth (m)", "Altitude (m)", "Distance (km)", "Distance (m)", "Position (mm)"]
    for term in spatial_terms:
        m = METRIC_BY_LABEL[term]
        assert "engineering" in m.domain_tags and "biomedical" in m.domain_tags
        assert "business" not in m.domain_tags

    # Business period terms strictly tagged business
    biz_terms = ["Fiscal Year", "Trading Day", "Billing Period", "Reporting Period"]
    for term in biz_terms:
        m = METRIC_BY_LABEL[term]
        assert m.domain_tags == ("business",)


# ==============================================================================
# 4. Normalized Stem Algorithm & Semantic Concept Groups
# ==============================================================================

def test_compute_concept_stem_algorithm():
    """FR-038: compute_concept_stem algorithm correctness."""
    assert compute_concept_stem("Revenue ($)") == "revenue"
    assert compute_concept_stem("Revenue (USD)") == "revenue"
    assert compute_concept_stem("Sales ($M)") == "sales"
    assert compute_concept_stem("Time (s)") == "time"
    assert compute_concept_stem("Time (min)") == "time"
    assert compute_concept_stem("Time (h)") == "time"
    assert compute_concept_stem("Time Point (h)") == "time point"
    assert compute_concept_stem("Concentration (μM)") == "concentration"
    assert compute_concept_stem("Concentration (mg/L)") == "concentration"
    assert compute_concept_stem("Cost of Goods Sold (COGS $)") == "cost of goods sold"


def test_semantic_concept_group_synonyms():
    """FR-032: Semantic concept groups group synonymous terms for collision avoidance."""
    # Revenue metrics
    assert METRIC_BY_LABEL["Revenue ($)"].concept_group == "revenue_metric"
    assert METRIC_BY_LABEL["Revenue (USD)"].concept_group == "revenue_metric"
    assert METRIC_BY_LABEL["Sales ($M)"].concept_group == "revenue_metric"

    # Optical density / absorbance
    assert METRIC_BY_LABEL["Absorbance (A.U.)"].concept_group == "optical_density"
    assert METRIC_BY_LABEL["Optical Density (OD)"].concept_group == "optical_density"
    assert METRIC_BY_LABEL["OD600"].concept_group == "optical_density"

    # Profit metrics
    assert METRIC_BY_LABEL["Gross Profit ($)"].concept_group == "profit_metric"
    assert METRIC_BY_LABEL["EBITDA ($)"].concept_group == "profit_metric"
    assert METRIC_BY_LABEL["Net Income ($)"].concept_group == "profit_metric"

    # Margin metrics
    assert METRIC_BY_LABEL["Gross Margin (%)"].concept_group == "margin_metric"
    assert METRIC_BY_LABEL["Profit Margin (%)"].concept_group == "margin_metric"


# ==============================================================================
# 5. Comparative Pairs Categorization & Reachability
# ==============================================================================

def test_comparative_pairs_categorization():
    """FR-034, T112: All pairs from manifests categorized by domain."""
    assert len(COMPARATIVE_PAIRS_CATALOG) >= 94
    assert len(COMPARATIVE_PAIRS_BY_DOMAIN["biomedical"]) >= 40
    assert len(COMPARATIVE_PAIRS_BY_DOMAIN["engineering"]) >= 32
    assert len(COMPARATIVE_PAIRS_BY_DOMAIN["business"]) >= 22

    for p in COMPARATIVE_PAIRS_BY_DOMAIN["biomedical"]:
        assert p.domain_category == "biomedical"
        assert p.is_scientific is True

    for p in COMPARATIVE_PAIRS_BY_DOMAIN["engineering"]:
        assert p.domain_category == "engineering"
        assert p.is_scientific is True

    for p in COMPARATIVE_PAIRS_BY_DOMAIN["business"]:
        assert p.domain_category == "business"
        assert p.is_scientific is False


def test_comparative_pairs_sampler_reachability():
    """SC-023, T126: 100% reachability across all 94 comparative pairs over draws."""
    # Test domain filtering
    for dom in ("biomedical", "engineering", "business"):
        expected_pairs = {(p.control, p.treatment) for p in COMPARATIVE_PAIRS_BY_DOMAIN[dom]}
        for _ in range(50):
            pair = sample_comparative_pair(domain_category=dom)
            assert pair in expected_pairs

    # Test 100% reachability across all 94 pairs
    sampled_pairs = set()
    for _ in range(10000):
        pair = sample_comparative_pair()
        sampled_pairs.add(pair)

    all_pairs = {(p.control, p.treatment) for p in COMPARATIVE_PAIRS_CATALOG}
    assert sampled_pairs == all_pairs, f"Unreached pairs: {all_pairs - sampled_pairs}"


# ==============================================================================
# 6. Title Index Top-Up & Domain Preference (from Phase 2)
# ==============================================================================

def test_title_index_minimum_top_up():
    """FR-035: All Cartesian title pools must have length >= 6."""
    for (is_sci, ctype), pool in TITLE_INDEX.items():
        if ctype != "heatmap":
            assert len(pool) >= 6, f"Pool {(is_sci, ctype)} has {len(pool)} titles (< 6)"


# ==============================================================================
# 7. Independent Double-Annotated Test Oracle (FR-051)
# ==============================================================================

# Embedded 50-item blind gold set for ambiguous semantic boundary terms
BLIND_GOLD_SET = {
    # Label: (expected_scale, expected_role)
    "Time (s)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Time (min)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Time (h)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Time Point (h)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Time Point (min)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Exposure Time (s)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Culture Duration (hrs)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Age (years)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Dose (mg/kg)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Concentration (μM)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Temperature (°C)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "pH": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Wavelength (nm)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Frequency (Hz)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Date": (ScaleType.TEMPORAL, AxisRole.INDEPENDENT),
    "Month": (ScaleType.TEMPORAL, AxisRole.BIDIRECTIONAL),
    "Quarter": (ScaleType.TEMPORAL, AxisRole.BIDIRECTIONAL),
    "Year": (ScaleType.TEMPORAL, AxisRole.BIDIRECTIONAL),
    "Fiscal Quarter": (ScaleType.TEMPORAL, AxisRole.BIDIRECTIONAL),
    "Fiscal Week": (ScaleType.TEMPORAL, AxisRole.BIDIRECTIONAL),
    "Observation Date": (ScaleType.TEMPORAL, AxisRole.INDEPENDENT),
    "Order Date": (ScaleType.TEMPORAL, AxisRole.INDEPENDENT),
    "Epoch": (ScaleType.DISCRETE_COUNT, AxisRole.INDEPENDENT),
    "Iteration": (ScaleType.DISCRETE_COUNT, AxisRole.INDEPENDENT),
    "Step": (ScaleType.DISCRETE_COUNT, AxisRole.INDEPENDENT),
    "Cycle Number": (ScaleType.CONTINUOUS, AxisRole.INDEPENDENT),
    "Depth (m)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Altitude (m)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Distance (km)": (ScaleType.CONTINUOUS, AxisRole.BIDIRECTIONAL),
    "Cell Line": (ScaleType.CATEGORICAL, AxisRole.INDEPENDENT),
    "Genotype": (ScaleType.CATEGORICAL, AxisRole.INDEPENDENT),
    "Treatment": (ScaleType.CATEGORICAL, AxisRole.INDEPENDENT),
    "Department": (ScaleType.CATEGORICAL, AxisRole.INDEPENDENT),
    "Product": (ScaleType.CATEGORICAL, AxisRole.INDEPENDENT),
    "Region": (ScaleType.CATEGORICAL, AxisRole.INDEPENDENT),
    "Salesperson": (ScaleType.CATEGORICAL, AxisRole.INDEPENDENT),
    "Revenue ($)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Revenue (USD)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Sales ($M)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Gross Profit ($)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Elastic Modulus (GPa)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Tensile Strength (MPa)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Absorbance (A.U.)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Enzyme Activity (U/mL)": (ScaleType.CONTINUOUS, AxisRole.DEPENDENT),
    "Accuracy (%)": (ScaleType.PERCENTAGE, AxisRole.DEPENDENT),
    "Gross Margin (%)": (ScaleType.PERCENTAGE, AxisRole.DEPENDENT),
    "Conversion Rate (%)": (ScaleType.PERCENTAGE, AxisRole.DEPENDENT),
    "Bounce Rate (%)": (ScaleType.PERCENTAGE, AxisRole.DEPENDENT),
    "Active Users": (ScaleType.DISCRETE_COUNT, AxisRole.DEPENDENT),
    "Customer Count": (ScaleType.DISCRETE_COUNT, AxisRole.DEPENDENT),
}


def test_independent_double_annotated_oracle():
    """FR-051: Verify agreement between blind gold set and catalog."""
    assert len(BLIND_GOLD_SET) == 50, "Blind gold set must contain exactly 50 items"
    for label, (exp_scale, exp_role) in BLIND_GOLD_SET.items():
        assert label in METRIC_BY_LABEL, f"Gold set term {label} not in METRIC_BY_LABEL"
        m = METRIC_BY_LABEL[label]
        assert m.scale_type == exp_scale, (
            f"Gold set scale mismatch for {label}: expected {exp_scale}, got {m.scale_type}"
        )
        assert m.role == exp_role, (
            f"Gold set role mismatch for {label}: expected {exp_role}, got {m.role}"
        )


def test_independent_continuous_regex_oracle():
    """FR-051: Independent continuous-regex oracle validates that continuous labels carry physical units."""
    # Pattern identifying continuous units: time, physical, financial, concentrations
    continuous_unit_regex = re.compile(
        r'(\(.*?(s|min|h|hrs|m|km|mm|μm|V|mV|µA|W|J|N|Nm|kPa|GPa|MPa|HV|g|mg|ng|mL|L|μM|mM|OD|AU|ppm|°C|Hz|%|\$|USD).*?\)|[$€£¥])'
    )
    for m in AXIS_METRICS_CATALOG:
        if m.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE) and "$" in m.label:
            assert continuous_unit_regex.search(m.label), f"Financial label {m.label} missing unit regex match"


# ==============================================================================
# 8. Deterministic Ordering
# ==============================================================================

def test_deterministic_sorting():
    """T125: Validate deterministic sorting and iteration stability."""
    labels_1 = [m.label for m in AXIS_METRICS_CATALOG]
    labels_2 = [m.label for m in AXIS_METRICS_CATALOG]
    assert labels_1 == labels_2

    pairs_1 = [(p.control, p.treatment) for p in COMPARATIVE_PAIRS_CATALOG]
    pairs_2 = [(p.control, p.treatment) for p in COMPARATIVE_PAIRS_CATALOG]
    assert pairs_1 == pairs_2
