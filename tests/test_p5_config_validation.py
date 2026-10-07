"""
tests/test_p5_config_validation.py

Verification tests for Phase 1 Task T227: Config Validation & Safe Defaults (FR-091).
Validates:
1. Reject effect params not in function signature:
   - Non-legacy profile (domain_gap_v1): ERROR
   - Legacy profile: WARNING
2. Typo like 'amplitud_ratio' causes error in domain_gap_v1 and exits with code 2 on CLI.
3. Reject perspective.magnitude > 0.15 in non-legacy profiles (ERROR), while allowing <= 0.15.
4. Legacy default perspective.magnitude is 0.08 in config_defaults.py.
"""
import copy
import subprocess
import sys
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config
from synth.semantics.admin.configspec import validate


def test_legacy_default_perspective_magnitude_is_008():
    """Verify legacy default perspective.magnitude was changed from 0.5 to 0.08 (FR-091)."""
    assert DEFAULT_CONFIG["realism_effects"]["perspective"]["params"]["magnitude"] == 0.08


def test_param_typo_errors_under_domain_gap_v1():
    """Verify typo like 'amplitud_ratio' produces validation error under non-legacy profile."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["profile"] = "domain_gap_v1"
    cfg["realism_effects"] = {
        "page_curl": {
            "p": 0.5,
            "params": {
                "amplitud_ratio": 0.03,  # Typo!
            },
        }
    }
    issues = validate(cfg)
    errors = [i for i in issues if i.level == "error" and "amplitud_ratio" in i.path]
    assert len(errors) == 1
    assert "Unknown parameter 'amplitud_ratio'" in errors[0].msg


def test_param_typo_warns_under_legacy():
    """Verify typo like 'amplitud_ratio' produces warning (not error) under legacy profile."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["profile"] = "legacy"
    cfg["realism_effects"] = {
        "page_curl": {
            "p": 0.5,
            "params": {
                "amplitud_ratio": 0.03,  # Typo!
            },
        }
    }
    issues = validate(cfg)
    errors = [i for i in issues if i.level == "error" and "amplitud_ratio" in i.path]
    warnings = [i for i in issues if i.level == "warning" and "amplitud_ratio" in i.path]
    assert len(errors) == 0
    assert len(warnings) == 1
    assert "Unknown parameter 'amplitud_ratio'" in warnings[0].msg


def test_perspective_magnitude_bound_in_non_legacy():
    """Verify perspective.magnitude > 0.15 is rejected in non-legacy profiles."""
    # 0.20 > 0.15 should error in domain_gap_v1
    cfg_bad = copy.deepcopy(DEFAULT_CONFIG)
    cfg_bad["profile"] = "domain_gap_v1"
    cfg_bad["realism_effects"] = {
        "perspective": {
            "p": 0.5,
            "params": {"magnitude": 0.20},
        }
    }
    issues_bad = validate(cfg_bad)
    errors_bad = [i for i in issues_bad if i.level == "error" and "magnitude" in i.path]
    assert len(errors_bad) == 1
    assert "exceeds maximum allowed value 0.15" in errors_bad[0].msg

    # 0.08 <= 0.15 should pass without error
    cfg_good = copy.deepcopy(DEFAULT_CONFIG)
    cfg_good["profile"] = "domain_gap_v1"
    cfg_good["realism_effects"] = {
        "perspective": {
            "p": 0.5,
            "params": {"magnitude": 0.08},
        }
    }
    issues_good = validate(cfg_good)
    errors_good = [i for i in issues_good if i.level == "error" and "magnitude" in i.path]
    assert len(errors_good) == 0


def test_cli_typo_exits_code_2_under_domain_gap_v1():
    """Verify generator.py CLI exits code 2 on typo under --profile domain_gap_v1."""
    cmd = [
        sys.executable,
        "generator.py",
        "--validate-only",
        "--profile",
        "domain_gap_v1",
        "--no-custom",
        "--set",
        "realism_effects.page_curl.params.amplitud_ratio=0.03",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "amplitud_ratio" in res.stderr
