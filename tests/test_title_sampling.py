"""
tests/test_title_sampling.py

Unit tests for Phase 2: Pre-Indexed Title Catalog & Fallback Gating.
Validates:
- Pre-indexed immutable title pools
- Minimum-6 top-ups across all chart types
- Exclusion of heatmap-only titles from Cartesian pools
- Domain preference sampling
- O(1) constant-time lookup performance
- Catalog coverage against themes.py:CHART_TITLES
- RNG determinism with standard random and NumPy Generator
"""
import random
import time
import numpy as np
import pytest

from synth.semantics.catalog import (
    CHART_TITLES_CATALOG,
    TITLE_INDEX,
    DOMAIN_TITLE_INDEX,
    HEATMAP_ONLY_TITLES,
    ALL_CHART_TYPES,
    CARTESIAN_CHART_TYPES,
)
from synth.semantics.enums import UnknownDomainError
from synth.semantics.sampler import sample_chart_title


@pytest.mark.xfail(strict=True, reason="13 duplicate titles introduced during semantic catalog expansion (audit_report.json, spec 005 §9.8)")
def test_catalog_matches_themes_chart_titles():
    """Verify all titles from declarative manifests are categorized in CHART_TITLES_CATALOG."""
    assert len(CHART_TITLES_CATALOG) >= 143
    catalog_titles = {item.title for item in CHART_TITLES_CATALOG}
    assert len(catalog_titles) == len(CHART_TITLES_CATALOG), "Titles must be unique"
    for item in CHART_TITLES_CATALOG:
        assert len(item.allowed_chart_types) > 0


def test_cartesian_and_pie_pools_have_min_six_titles():
    """Verify every (is_scientific, chart_type) pool has >= 6 titles."""
    for is_sci in (True, False):
        for ctype in ALL_CHART_TYPES:
            pool = TITLE_INDEX.get((is_sci, ctype))
            assert pool is not None, f"Pool missing for ({is_sci}, {ctype})"
            assert isinstance(pool, tuple), f"Pool must be immutable tuple for ({is_sci}, {ctype})"
            assert len(pool) >= 6, f"Pool ({is_sci}, {ctype}) has only {len(pool)} titles (< 6)"


def test_heatmap_only_titles_excluded_from_cartesian_pools():
    """Verify heatmap-only titles are isolated and excluded from all Cartesian and pie pools."""
    assert len(HEATMAP_ONLY_TITLES) == 8
    for h_title in HEATMAP_ONLY_TITLES:
        # Check base TITLE_INDEX
        for is_sci in (True, False):
            for ctype in ("bar", "line", "scatter", "box", "area", "histogram", "pie"):
                pool = TITLE_INDEX.get((is_sci, ctype), ())
                assert h_title not in pool, (
                    f"Heatmap title '{h_title}' leaked into base pool ({is_sci}, {ctype})"
                )

        # Check DOMAIN_TITLE_INDEX
        for (is_sci, ctype, dom), pool in DOMAIN_TITLE_INDEX.items():
            if ctype in ("bar", "line", "scatter", "box", "area", "histogram", "pie"):
                assert h_title not in pool, (
                    f"Heatmap title '{h_title}' leaked into domain pool ({is_sci}, {ctype}, {dom})"
                )


def test_heatmap_titles_available_for_heatmap():
    """Verify heatmap-only titles are accessible in heatmap pools."""
    heatmap_sci_pool = TITLE_INDEX.get((True, "heatmap"), ())
    heatmap_biz_pool = TITLE_INDEX.get((False, "heatmap"), ())
    combined_heatmap_titles = set(heatmap_sci_pool) | set(heatmap_biz_pool)
    for h_title in HEATMAP_ONLY_TITLES:
        assert h_title in combined_heatmap_titles, (
            f"Heatmap title '{h_title}' should be present in heatmap pools"
        )


def test_sample_chart_title_pie_biomedical():
    """Verify sample_chart_title('pie', True, domain='biomedical') returns valid biomedical title."""
    biomedical_pie_pool = DOMAIN_TITLE_INDEX[(True, "pie", "biomedical")]
    assert len(biomedical_pie_pool) > 0

    for _ in range(100):
        title = sample_chart_title("pie", True, domain="biomedical")
        assert title in biomedical_pie_pool
        assert title not in HEATMAP_ONLY_TITLES
        assert title != "Confusion Matrix"


def test_sample_chart_title_domain_preference():
    """Verify domain preference preferentially selects domain-tagged titles."""
    eng_scatter_pool = DOMAIN_TITLE_INDEX.get((True, "scatter", "engineering"))
    assert eng_scatter_pool is not None and len(eng_scatter_pool) > 0

    sampled = {sample_chart_title("scatter", True, domain="engineering") for _ in range(50)}
    # All sampled titles should belong to the engineering scatter pool
    for t in sampled:
        assert t in eng_scatter_pool


def test_sample_chart_title_fallback_on_unmatched_domain():
    """Verify unknown domain raises UnknownDomainError (SEM-04)."""
    with pytest.raises(UnknownDomainError):
        sample_chart_title("bar", False, domain="unknown_domain_xyz")


def test_constant_time_lookup_performance():
    """Verify O(1) dictionary and tuple access executes 10,000 draws in < 0.2s without regex/string filtering."""
    rng = random.Random(42)
    chart_types = list(ALL_CHART_TYPES)
    domains = ["biomedical", "engineering", "business", None]

    start_time = time.perf_counter()
    for _ in range(10_000):
        ctype = chart_types[_ % len(chart_types)]
        is_sci = (_ % 2 == 0)
        dom = domains[_ % len(domains)]
        _ = sample_chart_title(ctype, is_sci, domain=dom, rng=rng)
    elapsed = time.perf_counter() - start_time

    assert elapsed < 0.20, f"10,000 title samples took {elapsed:.4f}s; expected < 0.20s for O(1) lookup"


def test_rng_determinism():
    """Verify deterministic sampling with fixed seed across standard random and numpy generator."""
    # Standard random.Random
    r1 = random.Random(12345)
    r2 = random.Random(12345)
    seq1 = [sample_chart_title("bar", True, domain="biomedical", rng=r1) for _ in range(20)]
    seq2 = [sample_chart_title("bar", True, domain="biomedical", rng=r2) for _ in range(20)]
    assert seq1 == seq2

    # NumPy default_rng
    np_r1 = np.random.default_rng(54321)
    np_r2 = np.random.default_rng(54321)
    np_seq1 = [sample_chart_title("line", False, domain="business", rng=np_r1) for _ in range(20)]
    np_seq2 = [sample_chart_title("line", False, domain="business", rng=np_r2) for _ in range(20)]
    assert np_seq1 == np_seq2
