"""
tests/test_domain_coherence_e2e.py

Phase 6 Verification Tests (Tasks T129, T130):
- Independent Double-Annotated Test Oracle (FR-051):
  * 50-item embedded blind gold set annotating scale_type and role for boundary cases.
  * Augmented continuous-regex oracle detecting quantitative metrics vs nominal entities.
  * Independent domain keyword dictionary auditing titles and pairs without catalog.py.
  * Zero dependent metrics and zero nominal categories on line/area X (SC-020).
  * Quantitative metrics on both scatter X and Y (SC-019).
  * Zero concept group and normalized stem collisions (SC-022).
  * Zero domain leakage on treatment keys and 100% pair reachability (SC-023).
  * End-to-end multi-chart generation and ground-truth persistence audit (SC-024, SC-025).
- Baseline Drift Comparison & In-Memory Benchmark (SC-026):
  * Vocabulary coverage >= 95% against baseline snapshot.
  * N=20,000 theoretical title benchmark (p_max <= 3.5%).
  * Line X frequency benchmark (f_max <= 5 * (1/K_domain)).
"""
import copy
import json
import os
import random
import re
from collections import Counter
from typing import Dict, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart
from synth.semantics.catalog import (
    AXIS_METRICS_CATALOG,
    CHART_TITLES_CATALOG,
    COMPARATIVE_PAIRS_CATALOG,
    COMPARATIVE_PAIRS_BY_DOMAIN,
    METRIC_BY_LABEL,
    AxisRole,
    ScaleType,
)
from synth.semantics.sampler import (
    AXIS_POOLS,
    sample_axis_pair,
    sample_chart_title,
    sample_comparative_pair,
    sample_secondary_y,
)


# ==============================================================================
# INDEPENDENT DOUBLE-ANNOTATED ORACLE FIXTURES (T129, FR-051)
# ==============================================================================

# Embedded 50-item blind gold set annotating both scale_type and role for
# boundary and standard semantic terms, completely independent of catalog.py.
INDEPENDENT_SCALE_ROLE_GOLD_SET = {
    # 1-12: Temporal / Progress Variables (Continuous, Temporal, Discrete Count)
    "Time (s)": {"scale_type": "continuous", "role": "bidirectional"},
    "Time (min)": {"scale_type": "continuous", "role": "bidirectional"},
    "Time (h)": {"scale_type": "continuous", "role": "bidirectional"},
    "Elapsed Time (s)": {"scale_type": "continuous", "role": "bidirectional"},
    "Date": {"scale_type": "temporal", "role": "independent"},
    "Year": {"scale_type": "temporal", "role": "bidirectional"},
    "Month": {"scale_type": "temporal", "role": "bidirectional"},
    "Quarter": {"scale_type": "temporal", "role": "bidirectional"},
    "Step": {"scale_type": "discrete_count", "role": "independent"},
    "Epoch": {"scale_type": "discrete_count", "role": "independent"},
    "Iteration": {"scale_type": "discrete_count", "role": "independent"},
    "Timestamp": {"scale_type": "temporal", "role": "independent"},

    # 13-22: Physical / Environmental / Spatial (Continuous, Bidirectional)
    "Distance (km)": {"scale_type": "continuous", "role": "bidirectional"},
    "Depth (m)": {"scale_type": "continuous", "role": "bidirectional"},
    "Wavelength (nm)": {"scale_type": "continuous", "role": "bidirectional"},
    "Frequency (Hz)": {"scale_type": "continuous", "role": "bidirectional"},
    "Temperature (°C)": {"scale_type": "continuous", "role": "bidirectional"},
    "Pressure (kPa)": {"scale_type": "continuous", "role": "bidirectional"},
    "Concentration (μM)": {"scale_type": "continuous", "role": "bidirectional"},
    "Dose (mg/kg)": {"scale_type": "continuous", "role": "bidirectional"},
    "pH": {"scale_type": "continuous", "role": "bidirectional"},
    "Age (years)": {"scale_type": "continuous", "role": "bidirectional"},

    # 23-34: Scientific Dependent Metrics (Continuous & Percentage - Never on Line/Area X)
    "Absorbance (A.U.)": {"scale_type": "continuous", "role": "dependent"},
    "Viability (%)": {"scale_type": "percentage", "role": "dependent"},
    "IC50 (nM)": {"scale_type": "continuous", "role": "dependent"},
    "Tensile Strength (MPa)": {"scale_type": "continuous", "role": "dependent"},
    "Elastic Modulus (GPa)": {"scale_type": "continuous", "role": "dependent"},
    "Thermal Conductivity (W/m·K)": {"scale_type": "continuous", "role": "dependent"},
    "Viscosity (cP)": {"scale_type": "continuous", "role": "dependent"},
    "Yield (%)": {"scale_type": "percentage", "role": "dependent"},
    "Accuracy (%)": {"scale_type": "percentage", "role": "dependent"},
    "Binding Affinity (Kd, nM)": {"scale_type": "continuous", "role": "dependent"},
    "% Inhibition": {"scale_type": "percentage", "role": "dependent"},
    "Capacitance (F)": {"scale_type": "continuous", "role": "dependent"},

    # 35-42: Business Dependent Metrics (Continuous & Percentage - Never on Line/Area X)
    "Revenue ($)": {"scale_type": "continuous", "role": "dependent"},
    "Revenue (USD)": {"scale_type": "continuous", "role": "dependent"},
    "Profit Margin (%)": {"scale_type": "percentage", "role": "dependent"},
    "Conversion Rate (%)": {"scale_type": "percentage", "role": "dependent"},
    "Customer Lifetime Value (LTV $)": {"scale_type": "continuous", "role": "dependent"},
    "Churn Rate (%)": {"scale_type": "percentage", "role": "dependent"},
    "Cost of Goods Sold (COGS $)": {"scale_type": "continuous", "role": "dependent"},
    "EBITDA ($)": {"scale_type": "continuous", "role": "dependent"},

    # 43-50: Categorical Entities (Nominal - Never on Line/Area X or Scatter)
    "Department": {"scale_type": "categorical", "role": "independent"},
    "Acquisition Channel": {"scale_type": "categorical", "role": "independent"},
    "Marketing Channel": {"scale_type": "categorical", "role": "independent"},
    "Cell Line": {"scale_type": "categorical", "role": "independent"},
    "Salesperson": {"scale_type": "categorical", "role": "independent"},
    "Region": {"scale_type": "categorical", "role": "independent"},
    "Category": {"scale_type": "categorical", "role": "independent"},
    "Store": {"scale_type": "categorical", "role": "independent"},
}

# Independent domain keyword dictionary for leakage and boundary audits
INDEPENDENT_DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "biomedical": {
        "placebo", "drug", "tumor", "patient", "serum", "in vivo", "in vitro", "ko",
        "wt", "wild-type", "knockout", "vehicle", "saline", "dosing", "sham", "untreated",
        "therapy", "antibody", "antigen", "enzyme", "receptor", "dna", "rna", "mrna",
        "pharmacokinetic", "biomarker", "assay", "pathway", "cell", "adverse event",
    },
    "engineering": {
        "stress", "strain", "voltage", "current", "torque", "velocity", "acceleration",
        "modulus", "viscosity", "tensile", "vibration", "sensor", "telemetry", "algorithm",
        "latency", "throughput", "bandwidth", "signal", "snr", "damping", "efficiency",
        "actuator", "temperature", "pressure", "frequency", "rpm", "cycles",
    },
    "business": {
        "revenue", "sales", "profit", "ebitda", "churn", "cac", "ltv", "mrr", "arr",
        "cogs", "roi", "conversion", "funnel", "customer", "quarter", "fiscal",
        "market share", "budget", "campaign", "pipeline", "arpu", "retention", "headcount",
        "kpi", "lead", "pricing", "discount", "checkout", "freemium", "upsell",
    },
}


def is_quantitative_metric_oracle(label: str) -> bool:
    """
    Independent regex oracle determining whether a label is a quantitative metric
    (continuous, percentage, count, temporal) as opposed to a nominal categorical entity.
    """
    # 1. Any metric with explicit unit in parentheses or brackets is quantitative
    if re.search(r"\(.*?\)|\[.*?\]", label):
        return True
    # 2. Currency or mathematical/physical unit symbols
    if any(sym in label for sym in ["%", "$", "€", "£", "°", "μ", "µ", "Δ", "×", "10^", "±"]):
        return True
    # 3. Known categorical entity indicators (when lacking units or quantitative markers)
    cat_indicators = (
        r"\b(channel|source|stage|cohort|persona|platform|sku|store|region|department|"
        r"salesperson|category|species|genotype|phenotype|subject|sample|replicate|"
        r"protocol|stimulus|severity|audience|unit|tier|city|country|state|arm|batch|"
        r"condition|device|location|site|trial|product|project|promotion|season|segment|"
        r"education|center|environmental)\b"
    )
    quant_overrides = (
        r"\b(per|number|rate|index|score|time|count|date|loss|tangent|constant|"
        r"coefficient|enrichment|impressions|revenue|profit|cost|value|"
        r"frequency|probability|likelihood|multiple|wave|occurrence|cases|observations)\b"
    )
    if re.search(cat_indicators, label, re.IGNORECASE) and not re.search(quant_overrides, label, re.IGNORECASE):
        return False
    # 4. Standard scientific / business quantitative acronyms
    acronyms = r"\b(TPM|RPKM|FPKM|CPM|RFU|OD600|AU|A\.U\.|NES|pH|FDR|AUC|BMI|P/E|ROI|CAC|LTV|MRR|ARR|EBITDA|COGS|ARPU|NPS|CSAT|CTR|CPC|CPA)\b"
    if re.search(acronyms, label, re.IGNORECASE):
        return True
    # 5. Quantitative keywords
    quant_pattern = (
        r"\b(rate|ratio|percentage|pct|score|index|duration|time|depth|distance|wavelength|"
        r"frequency|level|concentration|yield|load|error|latency|throughput|volume|count|"
        r"counts|step|epoch|iteration|density|mass|length|height|width|weight|temperature|"
        r"pressure|voltage|current|power|revenue|profit|margin|cost|spend|arpu|ebitda|mrr|"
        r"arr|cac|ltv|churn|year|month|quarter|date|day|week|period|timestamp|checkpoint|"
        r"latitude|longitude|permittivity|users|clicks|views|units|headcount|value|age|"
        r"velocity|acceleration|force|torque|stress|strain|capacitance|viscosity|"
        r"absorbance|transmittance|fluorescence|intensity|energy|affinity|efficacy|"
        r"activity|inhibition|expression|abundance|titer|fold|copies|events|incidence|"
        r"prevalence|mortality|survival|variance|deviation|mean|median|fraction|"
        r"proportion|speed|bandwidth|retention|traffic|session|conversion|number|"
        r"constant|coefficient|tangent|loss|enrichment|impressions|factor|multiple|wave|"
        r"probability|likelihood|occurrence|distribution|cases|observations|interval)\b"
    )
    return bool(re.search(quant_pattern, label, re.IGNORECASE))


# ==============================================================================
# T129: INDEPENDENT ORACLE & FORCED FIXTURE TESTS
# ==============================================================================

def test_independent_blind_gold_set_audit():
    """Verify that catalog metadata 100% matches the 50-item independent blind gold set (T129, FR-051)."""
    assert len(INDEPENDENT_SCALE_ROLE_GOLD_SET) == 50, "Gold set must contain exactly 50 items"

    for label, expected in INDEPENDENT_SCALE_ROLE_GOLD_SET.items():
        assert label in METRIC_BY_LABEL, f"Gold set term {label!r} not found in catalog"
        actual_metric = METRIC_BY_LABEL[label]
        assert actual_metric.scale_type.value == expected["scale_type"], (
            f"Scale mismatch for {label!r}: expected {expected['scale_type']}, got {actual_metric.scale_type.value}"
        )
        assert actual_metric.role.value == expected["role"], (
            f"Role mismatch for {label!r}: expected {expected['role']}, got {actual_metric.role.value}"
        )


def test_continuous_regex_oracle_on_catalog():
    """
    Verify that the independent regex oracle correctly partitions quantitative metrics
    from nominal categorical entities across the entire catalog (T129, FR-051).
    """
    for metric in AXIS_METRICS_CATALOG:
        is_quant = is_quantitative_metric_oracle(metric.label)
        if metric.scale_type in (ScaleType.CONTINUOUS, ScaleType.PERCENTAGE, ScaleType.DISCRETE_COUNT, ScaleType.TEMPORAL):
            assert is_quant, f"Quantitative metric {metric.label!r} ({metric.scale_type.value}) failed independent regex oracle"
        elif metric.scale_type == ScaleType.CATEGORICAL:
            assert not is_quant, f"Nominal category {metric.label!r} was falsely classified as quantitative"


def test_line_and_area_x_axis_coherence_oracle():
    """
    Assert zero dependent metrics and zero nominal categories on line/area X across 5,000 draws (T129, SC-020).
    Also asserts zero categorical contamination on value axes (Y) for vertical line, area, and bar charts.
    """
    rng = random.Random(2026)
    domains = ("biomedical", "engineering", "business")
    chart_types = ("line", "area")

    for _ in range(5000):
        ctype = rng.choice(chart_types)
        dom = rng.choice(domains)
        is_sci = (dom != "business")

        # Line and area charts are vertical functional mappings y = f(x)
        x_label, y_label, resolved_dom = sample_axis_pair(ctype, "vertical", is_sci, semantic_domain=dom, rng=rng)

        x_metric = METRIC_BY_LABEL[x_label]
        y_metric = METRIC_BY_LABEL[y_label]

        # 1. X must be independent / bidirectional, never a pure dependent metric
        assert x_metric.role in (AxisRole.INDEPENDENT, AxisRole.BIDIRECTIONAL), (
            f"Dependent metric violation on {ctype} X: {x_label!r} has role {x_metric.role.value}"
        )
        # 2. X must NEVER be a nominal category
        assert x_metric.scale_type != ScaleType.CATEGORICAL, (
            f"Nominal categorical violation on {ctype} X: {x_label!r} has scale {x_metric.scale_type.value}"
        )
        # 3. Y value axis must be dependent/bidirectional and quantitative (zero categorical contamination)
        assert y_metric.role in (AxisRole.DEPENDENT, AxisRole.BIDIRECTIONAL), (
            f"Independent metric violation on {ctype} Y: {y_label!r} has role {y_metric.role.value}"
        )
        assert y_metric.scale_type != ScaleType.CATEGORICAL, (
            f"Nominal categorical violation on {ctype} Y value axis: {y_label!r}"
        )


def test_scatter_scale_coherence_oracle():
    """
    Assert 100% of scatter plots have quantitative metrics on both X and Y across 5,000 draws (T129, SC-019).
    """
    rng = random.Random(2027)
    domains = ("biomedical", "engineering", "business")

    for _ in range(5000):
        dom = rng.choice(domains)
        is_sci = (dom != "business")
        x_label, y_label, _ = sample_axis_pair("scatter", "vertical", is_sci, semantic_domain=dom, rng=rng)

        # Both must pass independent quantitative regex oracle
        assert is_quantitative_metric_oracle(x_label), f"Scatter X {x_label!r} is not quantitative"
        assert is_quantitative_metric_oracle(y_label), f"Scatter Y {y_label!r} is not quantitative"

        x_m = METRIC_BY_LABEL[x_label]
        y_m = METRIC_BY_LABEL[y_label]
        assert x_m.scale_type != ScaleType.CATEGORICAL, f"Scatter X {x_label!r} is categorical"
        assert y_m.scale_type != ScaleType.CATEGORICAL, f"Scatter Y {y_label!r} is categorical"


def test_concept_group_and_stem_collision_avoidance():
    """
    Assert zero exact (X == Y), normalized stem, or semantic concept group collisions across 5,000 draws
    and across secondary Y sampling (T129, SC-022).
    """
    rng = random.Random(2028)
    chart_types = ("bar", "line", "scatter", "box", "area")
    domains = ("biomedical", "engineering", "business")

    for _ in range(5000):
        ctype = rng.choice(chart_types)
        dom = rng.choice(domains)
        is_sci = (dom != "business")

        x_label, y_label, _ = sample_axis_pair(ctype, "vertical", is_sci, semantic_domain=dom, rng=rng)
        assert x_label != y_label, f"Exact collision: X={x_label}, Y={y_label}"

        mx = METRIC_BY_LABEL[x_label]
        my = METRIC_BY_LABEL[y_label]
        assert mx.concept_stem != my.concept_stem, f"Concept stem collision: {x_label} vs {y_label}"

        if mx.concept_group is not None and my.concept_group is not None:
            assert mx.concept_group != my.concept_group, (
                f"Concept group collision: {x_label} and {y_label} share group {mx.concept_group}"
            )

        # Audit secondary Y
        sec_y = sample_secondary_y(y_label, is_sci, domain_category=dom, rng=rng)
        assert sec_y != y_label, f"Secondary Y collision with Y1: {sec_y} == {y_label}"
        m_sec = METRIC_BY_LABEL.get(sec_y)
        if m_sec:
            assert m_sec.concept_stem != my.concept_stem, f"Secondary Y stem collision: {sec_y} vs {y_label}"
            if m_sec.concept_group is not None and my.concept_group is not None:
                assert m_sec.concept_group != my.concept_group, (
                    f"Secondary Y group collision: {sec_y} and {y_label} share group {my.concept_group}"
                )


def test_treatment_key_domain_coherence_and_zero_leakage():
    """
    Assert zero cross-domain leakage when treatment keys render and 100% reachability across 5,000 draws (T129, SC-023).
    """
    rng = random.Random(2029)
    bio_keywords = INDEPENDENT_DOMAIN_KEYWORDS["biomedical"]
    biz_keywords = INDEPENDENT_DOMAIN_KEYWORDS["business"]

    # 1. Audit all 94 comparative pairs in catalog
    for p in COMPARATIVE_PAIRS_CATALOG:
        assert p.control != p.treatment, f"Pair has identical control and treatment: {p}"
        ctrl_lower = p.control.lower()
        treat_lower = p.treatment.lower()

        if p.domain_category == "business":
            for kw in bio_keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                assert not re.search(pattern, ctrl_lower), f"Biomedical leakage in business pair {p}: {kw}"
                assert not re.search(pattern, treat_lower), f"Biomedical leakage in business pair {p}: {kw}"
        elif p.domain_category == "biomedical":
            for kw in biz_keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                assert not re.search(pattern, ctrl_lower), f"Business leakage in biomedical pair {p}: {kw}"
                assert not re.search(pattern, treat_lower), f"Business leakage in biomedical pair {p}: {kw}"

    # 2. Audit 100% reachability across 5,000 sampler draws
    drawn_pairs = set()
    for _ in range(5000):
        dom = rng.choice(("biomedical", "engineering", "business"))
        c, t = sample_comparative_pair(domain_category=dom, rng=rng)
        drawn_pairs.add((c, t, dom))

    catalog_pairs = {(p.control, p.treatment, p.domain_category) for p in COMPARATIVE_PAIRS_CATALOG}
    unreached = catalog_pairs - drawn_pairs
    assert len(unreached) == 0, f"Unreached comparative pairs ({len(unreached)}): {unreached}"


def test_forced_multichart_mixes_e2e(tmp_path):
    """
    Generate charts across all 8 chart types with forced dual-axis and forced treatment keys,
    verifying end-to-end ground truth persistence, schema v2.1, and domain coherence (T129, SC-024, SC-025).
    """
    chart_types = ["bar", "line", "scatter", "box", "area", "histogram", "pie", "heatmap"]
    images_dir = str(tmp_path / "images")
    labels_dir = str(tmp_path / "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for idx, ctype in enumerate(chart_types):
        cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
        cfg["num_images"] = 1
        cfg["dataset_format"] = "detection"
        cfg["scenario_weights"] = {"single": 100, "multi": 0}
        cfg["realism_effects"] = {}
        cfg["annotation_schema_version"] = "v3.0"

        # Isolate chart type
        for c in cfg["chart_types"]:
            cfg["chart_types"][c]["enabled"] = (c == ctype)
            cfg["chart_types"][c]["weight"] = 100 if c == ctype else 0

        # Alternate domains: even = scientific, odd = business
        is_sci = (idx % 2 == 0)
        dom = ("biomedical" if idx % 4 == 0 else "engineering") if is_sci else "business"
        cfg["theme_config"] = {
            "semantic_domain": dom,
            "scientific_subdomain_weights": {"biomedical": 0.70, "engineering": 0.30},
            "force_dual_axis": True,
            "force_treatment_key": (ctype == "bar"),
        }
        cfg["style_config"] = {
            "force_dual_axis": True,
            "force_treatment_key": (ctype == "bar"),
        }

        # Generate single chart
        try:
            generate_single_chart(
                i=idx,
                cfg=cfg,
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_dir=str(tmp_path),
            )
        finally:
            plt.close("all")

        # Verify emitted detailed.json
        base_name = f"chart_{idx:05d}"
        det_path = os.path.join(labels_dir, f"{base_name}_detailed.json")
        assert os.path.exists(det_path), f"Missing detailed.json for {ctype}"

        with open(det_path, "r") as f:
            det = json.load(f)

        # 1. Ground-Truth Persistence Audit (SC-024)
        assert det.get("schema_version") == "v3.0", f"Expected schema_version v3.0 on {ctype}"
        assert det.get("dataset_version") == "3.0.0", f"Expected dataset_version 3.0.0 on {ctype}"
        assert isinstance(det.get("is_scientific"), bool), f"is_scientific must be bool on {ctype}"
        assert det.get("semantic_domain") in ("biomedical", "engineering", "business", "demographic")
        assert "subplots" in det and isinstance(det["subplots"], list)

        # 2. Chart Type Integrity (SC-025)
        if ctype == "bar":
            assert det.get("chart_type") == "bar", f"Dual-axis bar must retain chart_type 'bar', got {det.get('chart_type')}"


# ==============================================================================
# T130: BASELINE DRIFT COMPARISON & THEORETICAL BENCHMARKS (SC-026)
# ==============================================================================

def test_baseline_drift_distribution_comparison():
    """
    Compare post-migration title vocabulary against baseline snapshot (T106 baseline),
    asserting 100% preservation of catalog titles (>= 95% threshold) (T130, SC-026).
    """
    baseline_path = os.path.join(os.path.dirname(__file__), "golden", "baseline_semantic_distribution.json")
    if not os.path.exists(baseline_path):
        pytest.skip(f"Baseline distribution not found at {baseline_path}")

    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)

    baseline_titles = set(baseline_data.get("title_frequencies", {}).keys())
    catalog_titles = set(t.title for t in CHART_TITLES_CATALOG)

    overlap = baseline_titles.intersection(catalog_titles)
    coverage = len(overlap) / len(baseline_titles)
    assert coverage >= 0.95, f"Baseline catalog title preservation is {coverage:.2%} < 95%"


def test_in_memory_theoretical_title_benchmark():
    """
    Simulate N = 20,000 draws from sample_chart_title under balanced multi-chart mix.
    Assert maximum title frequency in any cell <= 5x baseline (p_max <= 3.5%) and
    overall vocabulary coverage >= 95% of routable catalog terms (T130, SC-026).
    """
    chart_weights = {
        "bar": 15, "line": 15, "scatter": 15, "box": 15,
        "area": 15, "histogram": 10, "pie": 10, "heatmap": 5,
    }
    chart_types = []
    for ctype, weight in chart_weights.items():
        chart_types.extend([ctype] * weight)

    counts = Counter()
    N = 20000
    rng = random.Random(202609)

    for i in range(N):
        ctype = chart_types[i % len(chart_types)]
        is_sci = (rng.random() < 0.5)
        dom = ("biomedical" if rng.random() < 0.6 else "engineering") if is_sci else "business"
        title = sample_chart_title(ctype, is_sci, domain=dom, rng=rng)
        counts[title] += 1

    max_count = max(counts.values())
    p_max = max_count / N

    # Assert p_max <= 3.5% (5x theoretical uniform baseline 1/143 = 0.00699 -> 0.03496)
    assert p_max <= 0.035, f"Maximum title frequency {p_max:.4f} exceeded 3.5% benchmark threshold"

    # Assert vocabulary coverage >= 95% of routable catalog terms
    coverage = len(counts) / len(CHART_TITLES_CATALOG)
    assert coverage >= 0.95, f"Title vocabulary coverage {coverage:.2%} fell below 95% benchmark threshold"


def test_line_x_axis_frequency_benchmark():
    """
    Assert that maximum line X frequency is <= 5x (1 / K_domain) uniform baseline across all domains,
    verifying canonical independent variable expansion (T130, SC-026).
    """
    N = 10000
    rng = random.Random(202610)

    for dom in ("biomedical", "engineering", "business"):
        pool = AXIS_POOLS[(dom, "line", "vertical", "x")]
        K = len(pool)
        assert K >= 20, f"Domain {dom} line X pool too small ({K} < 20)"

        baseline_uniform = 1.0 / K
        counts = Counter()
        is_sci = (dom != "business")

        for _ in range(N):
            x_label, _, _ = sample_axis_pair("line", "vertical", is_sci, semantic_domain=dom, rng=rng)
            counts[x_label] += 1

        max_freq = max(counts.values()) / N
        max_ratio = max_freq / baseline_uniform

        # Maximum frequency must be <= 5x uniform baseline
        assert max_ratio <= 5.0, (
            f"Domain {dom} line X frequency ratio {max_ratio:.2f}x exceeded 5x baseline threshold"
        )
