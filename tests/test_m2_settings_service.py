"""
tests/test_m2_settings_service.py

Acceptance tests for Milestone M2 (Settings service):
- CFG-03: Declarative config validation rules (pass and fail cases per rule).
- CFG-04: custom_config.py sentinel block management, diff preservation, idempotency.
- CFG-05: AST-based literal patching of config_defaults.py with rollback and comments preserved.
- list_domains(): Read-only manifest domain discovery.
"""
import copy
import os
from pathlib import Path
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config
from synth.semantics.admin.configspec import (
    Issue,
    list_domains,
    validate,
)
from synth.semantics.admin.configsvc import (
    compute_leaf_diff,
    patch_config_defaults_literal,
    save_overrides_to_custom_config,
)


def test_list_domains():
    """Verify list_domains discovers all base domains and identifies is_scientific."""
    domains = list_domains()
    assert "biomedical" in domains
    assert "engineering" in domains
    assert "business" in domains
    assert "demographic" in domains
    assert "common" in domains

    assert domains["biomedical"]["is_scientific"] is True
    assert domains["engineering"]["is_scientific"] is True
    assert domains["business"]["is_scientific"] is False
    assert domains["demographic"]["is_scientific"] is False


def test_cfg03_validation_pass_case():
    """CFG-03: Default configuration passes validation without errors."""
    issues = validate(DEFAULT_CONFIG)
    errors = [i for i in issues if i.level == "error"]
    assert errors == [], f"Expected 0 errors on defaults, got {errors}"


@pytest.mark.parametrize(
    "mutator,expected_path",
    [
        (lambda c: c.update({"num_images": 0}), "num_images"),
        (lambda c: c.update({"engine": "invalid_engine"}), "engine"),
        (lambda c: c.update({"dataset_format": "unknown_format"}), "dataset_format"),
        (lambda c: c.update({"scientific_ratio": 1.5}), "scientific_ratio"),
        (lambda c: [cc.update({"enabled": False}) for cc in c["chart_types"].values()], "chart_types"),
        (lambda c: c["bar_chart_config"].update({"error_bar_probability": 1.5}), "bar_chart_config.error_bar_probability"),
        (lambda c: c.update({"scientific_subdomain_weights": {"unknown_dom": 1.0}}), "scientific_subdomain_weights.unknown_dom"),
        (lambda c: c.update({"scientific_subdomain_weights": {"business": 1.0}}), "scientific_subdomain_weights.business"),
        (lambda c: c.update({"scientific_subdomain_weights": {"biomedical": 0.0, "engineering": 0.0}}), "scientific_subdomain_weights"),
        (lambda c: c.update({"non_scientific_subdomain_weights": {"biomedical": 1.0}}), "non_scientific_subdomain_weights.biomedical"),
        (lambda c: c.update({"semantic_domain": "invalid_domain"}), "semantic_domain"),
        (lambda c: c["heatmap_validation"].update({"min_cell_coverage": 2.0}), "heatmap_validation.min_cell_coverage"),
        (lambda c: c["realism_effects"].update({"nonexistent_effect": {"p": 0.5}}), "realism_effects.nonexistent_effect"),
        (lambda c: c["realism_effects"]["blur"].update({"p": -0.1}), "realism_effects.blur.p"),
    ],
)
def test_cfg03_validation_fail_cases(mutator, expected_path):
    """CFG-03: Ensure each validation rule fails with an error at the expected path."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    mutator(cfg)
    issues = validate(cfg)
    matching_errors = [i for i in issues if i.level == "error" and i.path == expected_path]
    assert len(matching_errors) >= 1, f"Expected validation error at path '{expected_path}', got {issues}"


def test_cfg04_custom_config_managed_block(tmp_path):
    """CFG-04: Apply overrides -> reload -> diff equals edit set; outside bytes untouched; idempotent."""
    fixture_content = (
        '"""\ncustom_config.py header comment\n"""\n'
        'import copy\n'
        'from config_defaults import OCR_TRAINING_CONFIG as _DEFAULT_CONFIG\n\n'
        'OCR_TRAINING_CONFIG = copy.deepcopy(_DEFAULT_CONFIG)\n\n'
        '# Custom manual note that must not be deleted\n'
    )
    custom_file = tmp_path / "custom_config.py"
    custom_file.write_text(fixture_content, encoding="utf-8")

    overrides = {
        "num_images": 250,
        "chart_types.bar.weight": 45,
    }

    # First write
    saved_text = save_overrides_to_custom_config(overrides, custom_file)
    assert "num_images" in saved_text
    assert "Custom manual note that must not be deleted" in saved_text
    assert '# >>> managed by manage_domains — do not edit >>>' in saved_text
    assert '# <<< managed by manage_domains — do not edit <<<' in saved_text

    # Verify idempotency
    second_saved_text = save_overrides_to_custom_config(overrides, custom_file)
    assert second_saved_text == saved_text

    # Verify reload via python module
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_test", str(custom_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reloaded_cfg = mod.OCR_TRAINING_CONFIG

    diff = compute_leaf_diff(reloaded_cfg, DEFAULT_CONFIG)
    assert diff["num_images"] == 250
    assert diff["chart_types.bar.weight"] == 45
    assert set(diff.keys()) == {"num_images", "chart_types.bar.weight"}


def test_cfg05_ast_patch_config_defaults(tmp_path):
    """CFG-05: Patch literal in config_defaults fixture via AST span with .bak and verification."""
    defaults_file = tmp_path / "config_defaults.py"
    fixture_code = (
        '"""Header docstring with comments."""\n'
        'OCR_TRAINING_CONFIG = {\n'
        '  "num_images": 5000,  # Important count comment\n'
        '  "engine": "matplotlib",\n'
        '  "chart_types": {\n'
        '    "bar": {\n'
        '      "weight": 30\n'
        '    }\n'
        '  }\n'
        '}\n'
    )
    defaults_file.write_text(fixture_code, encoding="utf-8")

    # Patch num_images
    diff = patch_config_defaults_literal("num_images", "1234", defaults_file)
    assert "-  \"num_images\": 5000" in diff
    assert "+  \"num_images\": 1234" in diff

    content = defaults_file.read_text(encoding="utf-8")
    assert '"num_images": 1234,  # Important count comment' in content
    assert defaults_file.with_suffix(".py.bak").exists()

    # Patch nested literal
    diff_nested = patch_config_defaults_literal("chart_types.bar.weight", "99", defaults_file)
    assert "+      \"weight\": 99" in diff_nested
    assert '"weight": 99' in defaults_file.read_text(encoding="utf-8")


def test_cfg05_ast_patch_invalid_rolls_back(tmp_path):
    """CFG-05: A patch resulting in invalid syntax fails verification and rolls back."""
    defaults_file = tmp_path / "config_defaults.py"
    fixture_code = (
        'OCR_TRAINING_CONFIG = {\n'
        '  "num_images": 5000\n'
        '}\n'
    )
    defaults_file.write_text(fixture_code, encoding="utf-8")

    with pytest.raises(ValueError, match="Patch validation failed"):
        patch_config_defaults_literal("num_images", "INVALID SYNTAX +++", defaults_file)

    # Content restored
    assert defaults_file.read_text(encoding="utf-8") == fixture_code
