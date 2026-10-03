"""
tests/test_m5_semantic_generalization.py

Acceptance and regression test suite for Milestone M5:
Semantic Runtime Generalization (SEM-01 .. SEM-09).
"""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import warnings
import pytest
from pydantic import ValidationError

from synth.semantics.enums import AxisRole, ScaleType, UnknownDomainError
from synth.semantics.loader import (
    ALL_CHART_TYPES,
    CARTESIAN_CHART_TYPES,
    DomainInfo,
    SemanticRegistry,
    _load_manifest_raw_data,
    rebuild_cache,
    registry,
)
from synth.semantics.sampler import (
    sample_axis_pair,
    sample_chart_title,
    sample_comparative_pair,
    sample_secondary_y,
)
from synth.semantics.schema import DomainManifest
from synth.tabular import sample_domain_schema


# ==============================================================================
# SEM-01: Decoupling, AST Lint, and SYNTH_DOMAINS_DIR fixture
# ==============================================================================

def test_ast_lint_no_baseline_domain_tuples():
    """Verify neither loader.py nor sampler.py contains a literal 4-tuple of baseline domains."""
    baseline_domains = {"biomedical", "engineering", "business", "demographic"}

    for module_relpath in ("synth/semantics/loader.py", "synth/semantics/sampler.py"):
        src_path = Path(module_relpath)
        assert src_path.exists(), f"File {src_path} not found"
        tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))

        for node in ast.walk(tree):
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                el_constants = [
                    elt.value
                    for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                # Flag if all 4 baseline domains are in the literal container
                found_domains = set(el_constants) & baseline_domains
                assert found_domains != baseline_domains, (
                    f"Found hardcoded baseline domains container in {module_relpath} line {node.lineno}: {el_constants}"
                )


def test_registry_exposes_domain_info():
    """Verify registry.domains exposes complete DomainInfo instances for all domains."""
    assert hasattr(registry, "domains")
    assert isinstance(registry.domains, dict)
    assert {"biomedical", "engineering", "business", "demographic", "common"}.issubset(
        set(registry.domains.keys())
    )

    for dom_id, info in registry.domains.items():
        assert isinstance(info, DomainInfo)
        assert info.id == dom_id
        assert len(info.display_name) > 0
        assert isinstance(info.is_scientific, bool)
        assert info.default_weight > 0
        assert len(info.fallbacks) == 2
        assert isinstance(info.fallbacks[0], str)
        assert isinstance(info.fallbacks[1], str)
        assert isinstance(info.capabilities, dict)


def test_synth_domains_dir_isolation(tmp_path):
    """
    Verify an isolated fixture domain manifest loaded via SYNTH_DOMAINS_DIR
    yields only that domain's vocabulary for all 6 Cartesian types x 2 orientations.
    """
    fixture_yaml = tmp_path / "aerospace.yaml"
    fixture_yaml.write_text(
        """manifest_version: 2
domain_id: aerospace
display_name: Aerospace Engineering
is_scientific: true
default_weight: 0.15
default_fallbacks: ["Altitude (m)", "True Airspeed (knots)"]
capabilities:
  tabular_preset: false
  heatmap_pools: false
metrics:
  - label: Altitude (m)
    scale_type: continuous
    role: independent
    concept_stem: altitude
    unit: m
    concept_group: position_vertical
    domain_tags: [aerospace]
    pools: [canonical_independent]
  - label: True Airspeed (knots)
    scale_type: continuous
    role: dependent
    concept_stem: airspeed
    unit: knots
    concept_group: velocity_air
    domain_tags: [aerospace]
    pools: [legacy]
  - label: Chamber Pressure (kPa)
    scale_type: continuous
    role: dependent
    concept_stem: chamber_pressure
    unit: kPa
    concept_group: pressure_chamber
    domain_tags: [aerospace]
    pools: [legacy]
titles:
  - title: Flight Envelope Profile
    allowed_chart_types: [bar, line, scatter, box, area, histogram]
    domain_tags: [aerospace]
comparative_pairs:
  - control: Subsonic Wing
    treatment: Transonic Wing
    domain_category: aerospace
    context_tags: [aerospace]
""",
        encoding="utf-8",
    )

    old_env = os.environ.get("SYNTH_DOMAINS_DIR")
    try:
        os.environ["SYNTH_DOMAINS_DIR"] = str(tmp_path)
        isolated_reg = rebuild_cache(force=True, domains_dir=tmp_path)

        assert "aerospace" in isolated_reg.domains
        assert len(isolated_reg.domains) == 1
        assert isolated_reg.domains["aerospace"].is_scientific is True

        cartesian_types = ("scatter", "bar", "box", "line", "area", "histogram")
        orientations = ("vertical", "horizontal")

        # Test sampling strictly uses aerospace vocabulary
        for ctype in cartesian_types:
            for orient in orientations:
                x_lbl, y_lbl, dom = sample_axis_pair(ctype, orient, is_scientific=True)
                assert dom == "aerospace"
                # Labels must come from aerospace manifest
                assert x_lbl in ("Altitude (m)", "True Airspeed (knots)", "Chamber Pressure (kPa)")
                if ctype != "histogram":
                    assert y_lbl in ("Altitude (m)", "True Airspeed (knots)", "Chamber Pressure (kPa)")

        # Title sampling
        t = sample_chart_title("line", is_scientific=True, domain="aerospace")
        assert t == "Flight Envelope Profile"

        # Comparative pair
        c, tr = sample_comparative_pair("aerospace")
        assert (c, tr) == ("Subsonic Wing", "Transonic Wing")
    finally:
        if old_env is not None:
            os.environ["SYNTH_DOMAINS_DIR"] = old_env
        else:
            os.environ.pop("SYNTH_DOMAINS_DIR", None)
        rebuild_cache(force=True)


# ==============================================================================
# SEM-02: Manifest-derived is_scientific
# ==============================================================================

def test_is_scientific_derived_from_manifest():
    """Verify is_scientific on metrics, titles, and pairs derives strictly from manifest flags (SEM-02)."""
    raw = _load_manifest_raw_data()
    for dom_id, doc in raw.items():
        is_sci = bool(doc.get("is_scientific", False))
        for m_raw in doc.get("metrics", []):
            m = registry.axis_metric_by_label[m_raw["label"]]
            assert m.is_scientific is is_sci

    # Title scientific gating
    for t in registry.all_titles:
        has_sci = any(tag in ("biomedical", "engineering") for tag in t.domain_tags)
        has_biz = any(tag in ("business", "demographic") for tag in t.domain_tags)
        if has_sci and not has_biz:
            assert t.is_scientific is True
        elif has_biz and not has_sci:
            assert t.is_scientific is False
        elif set(t.domain_tags) == {"common"}:
            assert t.is_scientific is None

    # Comparative pair scientific gating
    for p in registry.all_pairs:
        expected_sci = registry.domains[p.domain_category].is_scientific if p.domain_category in registry.domains else False
        assert p.is_scientific is expected_sci


# ==============================================================================
# SEM-03: Explicit pool membership (no positional slicing)
# ==============================================================================

def test_explicit_pools_append_metric_safety():
    """Verify appending a new metric without 'histogram_y' in pools leaves HISTOGRAM_Y_LABELS unchanged."""
    raw = _load_manifest_raw_data()
    orig_hist_count = len(registry.histogram_y_labels)

    # Clone common raw data and append a new metric
    mutated_raw = json.loads(json.dumps(raw))
    mutated_raw["common"]["metrics"].append({
        "label": "Appended Test Metric",
        "scale_type": "continuous",
        "role": "independent",
        "concept_stem": "appended_test_metric",
        "unit": "units",
        "domain_tags": ["common"],
        "pools": ["canonical_independent"],
    })

    candidate_reg = SemanticRegistry.load(raw=mutated_raw)
    assert len(candidate_reg.histogram_y_labels) == orig_hist_count
    assert "Appended Test Metric" not in candidate_reg.histogram_y_labels
    assert "Appended Test Metric" in [m.label for m in candidate_reg.canonical_metrics]


# ==============================================================================
# SEM-04: UnknownDomainError across all sampler functions
# ==============================================================================

@pytest.mark.parametrize("func,args", [
    (sample_chart_title, ("bar", True, "unknown_sector")),
    (sample_axis_pair, ("bar", "vertical", True, "unknown_sector")),
    (sample_secondary_y, ("Time (h)", True, "unknown_sector")),
    (sample_comparative_pair, ("unknown_sector",)),
])
def test_unknown_domain_raises_error(func, args):
    """Verify every sampler function raises UnknownDomainError on unrecognized domain."""
    with pytest.raises(UnknownDomainError) as exc_info:
        func(*args)
    assert "unknown_sector" in str(exc_info.value)


# ==============================================================================
# SEM-05: Default fallbacks and dependent-only domain support
# ==============================================================================

def test_dependent_only_domain_returns_valid_cartesian_pair(tmp_path):
    """Verify a domain with only dependent metrics derives independent fallback from common and samples valid pairs."""
    dep_only_yaml = tmp_path / "dep_domain.yaml"
    dep_only_yaml.write_text(
        """manifest_version: 2
domain_id: dep_domain
display_name: Dependent Only Domain
is_scientific: false
default_weight: 0.10
capabilities:
  tabular_preset: false
  heatmap_pools: false
metrics:
  - label: Dependent Metric A
    scale_type: continuous
    role: dependent
    concept_stem: dep_metric_a
    domain_tags: [dep_domain]
    pools: [legacy]
  - label: Dependent Metric B
    scale_type: continuous
    role: dependent
    concept_stem: dep_metric_b
    domain_tags: [dep_domain]
    pools: [legacy]
""",
        encoding="utf-8",
    )

    old_env = os.environ.get("SYNTH_DOMAINS_DIR")
    try:
        os.environ["SYNTH_DOMAINS_DIR"] = str(tmp_path)
        isolated_reg = rebuild_cache(force=True, domains_dir=tmp_path)
        info = isolated_reg.domains["dep_domain"]
        assert info.fallbacks[1] == "Dependent Metric A"

        # Sampling vertical bar: X requires independent metric (falls back), Y uses dependent metric
        x_lbl, y_lbl, dom = sample_axis_pair("bar", "vertical", is_scientific=False, semantic_domain="dep_domain")
        assert dom == "dep_domain"
        assert x_lbl == info.fallbacks[0]
        assert y_lbl in ("Dependent Metric A", "Dependent Metric B")
    finally:
        if old_env is not None:
            os.environ["SYNTH_DOMAINS_DIR"] = old_env
        else:
            os.environ.pop("SYNTH_DOMAINS_DIR", None)
        rebuild_cache(force=True)


# ==============================================================================
# SEM-06: Cache invalidation, atomic write, duplicate ID, and concurrent loading
# ==============================================================================

def test_cache_fingerprint_invalidation_on_delete_or_edit(tmp_path):
    """Verify deleting or altering a manifest invalidates .cache.json."""
    m1 = tmp_path / "d1.yaml"
    m1.write_text("domain_id: d1\ndisplay_name: D1\nmetrics: []\n", encoding="utf-8")
    m2 = tmp_path / "d2.yaml"
    m2.write_text("domain_id: d2\ndisplay_name: D2\nmetrics: []\n", encoding="utf-8")

    # Initial load writes cache
    data1 = _load_manifest_raw_data(domains_dir=tmp_path, force=False)
    assert set(data1.keys()) == {"d1", "d2"}
    cache_file = tmp_path / ".cache.json"
    assert cache_file.exists()

    # Modify m1
    m1.write_text("domain_id: d1\ndisplay_name: D1 Modified\nmetrics: []\n", encoding="utf-8")
    data2 = _load_manifest_raw_data(domains_dir=tmp_path, force=False)
    assert data2["d1"]["display_name"] == "D1 Modified"

    # Delete m2
    m2.unlink()
    data3 = _load_manifest_raw_data(domains_dir=tmp_path, force=False)
    assert set(data3.keys()) == {"d1"}


def test_duplicate_domain_id_error(tmp_path):
    """Verify having two YAML files with the same domain_id raises ValueError."""
    (tmp_path / "file1.yaml").write_text("domain_id: collision\ndisplay_name: One\n", encoding="utf-8")
    (tmp_path / "file2.yaml").write_text("domain_id: collision\ndisplay_name: Two\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        _load_manifest_raw_data(domains_dir=tmp_path, force=True)
    assert "Duplicate domain_id 'collision'" in str(exc.value)


def test_yml_extension_warning(tmp_path):
    """Verify manifests with .yml extension trigger a UserWarning."""
    (tmp_path / "test.yml").write_text("domain_id: yml_dom\ndisplay_name: Yml\n", encoding="utf-8")

    with pytest.warns(UserWarning, match=r"uses '\.yml' extension"):
        _load_manifest_raw_data(domains_dir=tmp_path, force=True)


def test_concurrent_cache_readers(tmp_path):
    """Verify concurrent loaders across multiple threads never raise race condition errors."""
    for i in range(4):
        (tmp_path / f"dom_{i}.yaml").write_text(
            f"domain_id: dom_{i}\ndisplay_name: Dom {i}\nmetrics: []\n",
            encoding="utf-8",
        )

    def load_job():
        return _load_manifest_raw_data(domains_dir=tmp_path, force=False)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_job) for _ in range(16)]
        results = [f.result() for f in futures]

    assert len(results) == 16
    for r in results:
        assert len(r) == 4


# ==============================================================================
# SEM-07: Tabular synthesis and Vegalite integration checks
# ==============================================================================

def test_tabular_unknown_domain_raises():
    """Verify sample_domain_schema raises UnknownDomainError on unsupported domain (SEM-07)."""
    with pytest.raises(UnknownDomainError) as exc_info:
        sample_domain_schema("aerospace")
    assert "aerospace" in str(exc_info.value)
    assert "Supported tabular presets" in str(exc_info.value)


# ==============================================================================
# SEM-08: Pydantic schema validation enforcement
# ==============================================================================

def test_schema_extra_forbid():
    """Verify unexpected fields are forbidden by the schema."""
    with pytest.raises(ValidationError):
        DomainManifest.model_validate({
            "domain_id": "test_dom",
            "display_name": "Test",
            "unsupported_extra_key": 123,
        })


def test_schema_invalid_chart_type():
    """Verify invalid chart types on titles are rejected."""
    with pytest.raises(ValidationError):
        DomainManifest.model_validate({
            "domain_id": "test_dom",
            "display_name": "Test",
            "titles": [
                {
                    "title": "Bad Chart Title",
                    "allowed_chart_types": ["invalid_chart_type"],
                    "domain_tags": ["test_dom"],
                }
            ],
        })


def test_schema_duplicate_metric_label():
    """Verify duplicate metric labels within a manifest produce a located error."""
    with pytest.raises(ValidationError) as exc:
        DomainManifest.model_validate({
            "domain_id": "test_dom",
            "display_name": "Test",
            "metrics": [
                {
                    "label": "Duplicated Metric",
                    "scale_type": "continuous",
                    "role": "dependent",
                    "concept_stem": "stem_1",
                    "domain_tags": ["test_dom"],
                },
                {
                    "label": "Duplicated Metric",
                    "scale_type": "continuous",
                    "role": "dependent",
                    "concept_stem": "stem_2",
                    "domain_tags": ["test_dom"],
                },
            ],
        })
    assert "Duplicate metric label in manifest: 'Duplicated Metric'" in str(exc.value)


def test_schema_invalid_default_fallbacks_length():
    """Verify default_fallbacks with length != 2 is rejected."""
    with pytest.raises(ValidationError):
        DomainManifest.model_validate({
            "domain_id": "test_dom",
            "display_name": "Test",
            "default_fallbacks": ["Only One Item"],
        })
