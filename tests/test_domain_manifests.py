"""
tests/test_domain_manifests.py

Validates all declarative domain manifests against Pydantic v2 schemas:
1. Syntax and Pydantic schema validation.
2. Global uniqueness of metric labels (zero collisions).
3. Comparative pair validity (control != treatment).
4. Identifier naming conventions for concept_stem and concept_group.
5. 100% vocabulary coverage of legacy semantic catalogs.
"""
from pathlib import Path
import re
from typing import Dict, List, Set
import pytest
import yaml

from synth.semantics.schema import DomainManifest
from synth.semantics.catalog import (
    LEGACY_METRICS_CATALOG,
    CANONICAL_INDEPENDENT_METRICS,
    CHART_TITLES_CATALOG,
    COMPARATIVE_PAIRS_CATALOG,
    HISTOGRAM_Y_LABELS,
)

DOMAINS_DIR = Path("synth/semantics/domains")
EXPECTED_DOMAINS = {"biomedical", "engineering", "business", "demographic", "common"}
IDENT_REGEX = re.compile(r"^[a-z0-9_]+$")


@pytest.fixture(scope="module")
def loaded_manifests() -> Dict[str, DomainManifest]:
    manifests: Dict[str, DomainManifest] = {}
    for domain_name in EXPECTED_DOMAINS:
        yaml_file = DOMAINS_DIR / f"{domain_name}.yaml"
        assert yaml_file.exists(), f"Missing expected manifest file: {yaml_file}"
        with open(yaml_file, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        manifest = DomainManifest.model_validate(raw_data)
        manifests[domain_name] = manifest
    return manifests


def test_manifests_exist_and_validate(loaded_manifests: Dict[str, DomainManifest]):
    assert set(loaded_manifests.keys()) == EXPECTED_DOMAINS
    for domain_name, manifest in loaded_manifests.items():
        assert manifest.domain_id == domain_name
        assert len(manifest.display_name) > 0
        assert manifest.default_weight > 0


def test_zero_duplicate_metric_labels(loaded_manifests: Dict[str, DomainManifest]):
    seen_labels: Set[str] = set()
    total_metrics_count = 0
    duplicate_labels: List[str] = []

    for domain_name, manifest in loaded_manifests.items():
        for metric in manifest.metrics:
            total_metrics_count += 1
            if metric.label in seen_labels:
                duplicate_labels.append(f"{metric.label} (in {domain_name})")
            seen_labels.add(metric.label)

    assert not duplicate_labels, f"Duplicate metric labels found: {duplicate_labels}"
    assert len(seen_labels) == total_metrics_count
    assert total_metrics_count >= 400


def test_comparative_pairs_distinct_control_treatment(loaded_manifests: Dict[str, DomainManifest]):
    total_pairs = 0
    for domain_name, manifest in loaded_manifests.items():
        for pair in manifest.comparative_pairs:
            total_pairs += 1
            assert pair.control.strip(), f"Empty control in {domain_name}"
            assert pair.treatment.strip(), f"Empty treatment in {domain_name}"
            assert pair.control != pair.treatment, f"Identical control and treatment: {pair.control}"
    assert total_pairs >= 94


def test_identifier_naming_conventions(loaded_manifests: Dict[str, DomainManifest]):
    for domain_name, manifest in loaded_manifests.items():
        for metric in manifest.metrics:
            assert IDENT_REGEX.match(metric.concept_stem), (
                f"Invalid concept_stem '{metric.concept_stem}' in {domain_name} for '{metric.label}'"
            )
            if metric.concept_group:
                assert IDENT_REGEX.match(metric.concept_group), (
                    f"Invalid concept_group '{metric.concept_group}' in {domain_name} for '{metric.label}'"
                )


def test_vocabulary_coverage(loaded_manifests: Dict[str, DomainManifest]):
    all_metric_labels = {
        metric.label
        for manifest in loaded_manifests.values()
        for metric in manifest.metrics
    }
    all_titles = {
        title.title
        for manifest in loaded_manifests.values()
        for title in manifest.titles
    }
    all_pairs = {
        (pair.control, pair.treatment)
        for manifest in loaded_manifests.values()
        for pair in manifest.comparative_pairs
    }

    # Verify all legacy metrics exist
    for legacy_metric in LEGACY_METRICS_CATALOG:
        assert legacy_metric.label in all_metric_labels, (
            f"Missing legacy metric: {legacy_metric.label}"
        )

    # Verify all canonical independent metrics exist
    for canon_metric in CANONICAL_INDEPENDENT_METRICS:
        assert canon_metric.label in all_metric_labels, (
            f"Missing canonical independent metric: {canon_metric.label}"
        )

    # Verify all histogram y labels exist
    for hist_label in HISTOGRAM_Y_LABELS:
        assert hist_label in all_metric_labels, (
            f"Missing histogram y label: {hist_label}"
        )

    # Verify all legacy chart titles exist
    for title_def in CHART_TITLES_CATALOG:
        assert title_def.title in all_titles, (
            f"Missing legacy chart title: {title_def.title}"
        )

    # Verify all legacy comparative pairs exist
    for pair_def in COMPARATIVE_PAIRS_CATALOG:
        assert (pair_def.control, pair_def.treatment) in all_pairs, (
            f"Missing comparative pair: ({pair_def.control}, {pair_def.treatment})"
        )
