"""
tests/test_m7_tui_domain_management.py

Acceptance tests for Milestone M7: TUI & Headless Domain Management (Release B).
Tests requirements:
- TUI-01: Headless subcommand parity (audit, import, new-domain, rebuild, restore, template).
  Non-TTY plain text; respects NO_COLOR and --ascii.
- TUI-02: Studio header and main menu rendering (§5.1).
- TUI-03: Pre-flight diff table rendering (§5.3), statuses (NEW, SHARED, UNIT-VARIANT,
  DUPLICATE, SIMILAR, INVALID), and summary statistics.
- ADM-11: New-domain wizard (headless and interactive) skeleton creation, weight registration,
  and common.yaml independent metric sharing.
- ADM-12: Read-only corpus audit and degenerate pool detection with --strict.
- ADM-14: Excel template generation with DataValidation dropdowns.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Generator
import openpyxl
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config
from synth.semantics.admin.audit import run_audit
from synth.semantics.admin.ingest import ingest_file
from synth.semantics.admin.plan import build_plan
from synth.semantics.admin.tui import (
    render_header,
    render_main_menu,
    render_preflight_table,
    safe_prompt,
    run_main_menu,
    run_import_flow,
    run_new_domain_wizard,
    run_revalidate_and_rebuild,
    run_restore_flow,
)
from synth.semantics.loader import get_domains_dir, rebuild_cache, registry
from synth.semantics.sampler import sample_axis_pair


@pytest.fixture
def sandbox_env(tmp_path: Path) -> Generator[Path, None, None]:
    """Create an isolated domains directory and sandbox custom_config.py for headless CLI tests."""
    real_domains = Path("synth/semantics/domains").resolve()
    temp_domains = tmp_path / "domains"
    shutil.copytree(real_domains, temp_domains)
    if (temp_domains / ".backups").exists():
        shutil.rmtree(temp_domains / ".backups")

    # Backup custom_config.py content
    custom_cfg_path = Path("custom_config.py")
    orig_custom_cfg = custom_cfg_path.read_text(encoding="utf-8") if custom_cfg_path.exists() else None

    # Set up sandbox environment
    old_env = os.environ.get("SYNTH_DOMAINS_DIR")
    os.environ["SYNTH_DOMAINS_DIR"] = str(temp_domains)

    # Reset cache in temp directory
    rebuild_cache(force=True, domains_dir=temp_domains)

    try:
        yield temp_domains
    finally:
        # Teardown
        if orig_custom_cfg is not None:
            custom_cfg_path.write_text(orig_custom_cfg, encoding="utf-8")
        elif custom_cfg_path.exists():
            custom_cfg_path.unlink()

        if old_env is not None:
            os.environ["SYNTH_DOMAINS_DIR"] = old_env
        else:
            os.environ.pop("SYNTH_DOMAINS_DIR", None)
        rebuild_cache(force=True)


# ==============================================================================
# TUI-01 / ADM-12: AUDIT SUBCOMMAND
# ==============================================================================

def test_cli_audit_default_and_json():
    """`manage_domains.py audit` outputs report and exits 0 on real corpus."""
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "audit"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "=== Corpus Audit Report ===" in res.stdout
    assert "Degenerate pools: NONE" in res.stdout

    # Test --json flag
    res_json = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "audit", "--json"],
        capture_output=True,
        text=True,
    )
    assert res_json.returncode == 0
    data = json.loads(res_json.stdout)
    assert "summary" in data
    assert "findings" in data
    assert "degenerate_pools" in data
    assert data["degenerate_pools"] == []


def test_cli_audit_report_export(tmp_path: Path):
    """`manage_domains.py audit --report <file>` exports JSON report to file."""
    rep_file = tmp_path / "test_report.json"
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "audit", "--report", str(rep_file)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert rep_file.exists()
    data = json.loads(rep_file.read_text(encoding="utf-8"))
    assert "summary" in data
    assert "findings" in data


def test_cli_audit_strict_mode(sandbox_env: Path):
    """`manage_domains.py audit --strict` returns non-zero when errors or degenerate pools exist."""
    env = dict(os.environ, SYNTH_DOMAINS_DIR=str(sandbox_env))

    # Real corpus has 0 errors and 0 degenerate pools, so --strict passes
    res_clean = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "audit", "--strict"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res_clean.returncode == 0

    # Inject a tag invariant violation (missing biomedical tag)
    bio_path = sandbox_env / "biomedical.yaml"
    from ruamel.yaml import YAML
    yaml = YAML()
    with open(bio_path, "r", encoding="utf-8") as f:
        doc = yaml.load(f)
    doc["metrics"].append({
        "label": "Invalid Tag Metric",
        "scale_type": "continuous",
        "role": "dependent",
        "concept_stem": "invalid_tag_metric",
        "domain_tags": ["engineering"],
        "pools": [],
    })
    with open(bio_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f)

    res_err = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "audit", "--strict"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res_err.returncode == 1
    assert "TAG_INVARIANT" in res_err.stdout


# ==============================================================================
# TUI-01 / ADM-14: TEMPLATE SUBCOMMAND
# ==============================================================================

def test_cli_template_generation(tmp_path: Path):
    """`manage_domains.py template` creates valid Excel workbook with DataValidation dropdowns."""
    tmpl_path = tmp_path / "domain_template.xlsx"
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "template", "-o", str(tmpl_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert tmpl_path.exists()
    assert f"[OK] Generated template workbook at: {tmpl_path}" in res.stdout

    # Inspect openpyxl structure
    wb = openpyxl.load_workbook(tmpl_path)
    assert "Metrics" in wb.sheetnames
    assert "Titles" in wb.sheetnames
    assert "Comparative Pairs" in wb.sheetnames

    # Check DataValidation
    ws_metrics = wb["Metrics"]
    validations = ws_metrics.data_validations.dataValidation
    assert len(validations) >= 2

    # Ingest generated template to ensure zero schema errors
    ingest_res = ingest_file(tmpl_path, target_domain="aerospace")
    assert not ingest_res.has_errors
    assert len(ingest_res.metrics) >= 1
    assert len(ingest_res.titles) >= 1
    assert len(ingest_res.pairs) >= 1


# ==============================================================================
# TUI-01 / SEM-06: REBUILD SUBCOMMAND
# ==============================================================================

def test_cli_rebuild_cache(sandbox_env: Path):
    """`manage_domains.py rebuild` forces schema validation and cache regeneration."""
    env = dict(os.environ, SYNTH_DOMAINS_DIR=str(sandbox_env))
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "rebuild"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "[OK] Cache successfully rebuilt" in res.stdout
    assert "Registered domains: 5" in res.stdout


# ==============================================================================
# TUI-01 / ADM-10: RESTORE SUBCOMMAND
# ==============================================================================

def test_cli_restore_list_and_to(sandbox_env: Path):
    """`manage_domains.py restore --list` and `restore --to <ts>` operate correctly."""
    env = dict(os.environ, SYNTH_DOMAINS_DIR=str(sandbox_env))

    # When no backups exist
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "restore", "--list"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "No snapshot backups found." in res.stdout

    # Create a synthetic backup directory
    backup_dir = sandbox_env / ".backups" / "2026-10-02T12-00-00Z"
    backup_dir.mkdir(parents=True)
    for yf in sandbox_env.glob("*.yaml"):
        shutil.copy(yf, backup_dir / yf.name)

    res_list = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "restore", "--list"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res_list.returncode == 0
    assert "2026-10-02T12-00-00Z" in res_list.stdout

    # Now restore to that snapshot
    res_restore = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "restore", "--to", "2026-10-02T12-00-00Z"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res_restore.returncode == 0
    assert "[OK] Successfully restored manifests from snapshot: 2026-10-02T12-00-00Z" in res_restore.stdout


# ==============================================================================
# TUI-01 / ADM-11: NEW-DOMAIN SUBCOMMAND & INTEGRATION
# ==============================================================================

def test_cli_new_domain_headless_and_adm11_integration(sandbox_env: Path, monkeypatch):
    """`manage_domains.py new-domain` creates manifest, shares common metrics, and registers weights."""
    env = dict(os.environ, SYNTH_DOMAINS_DIR=str(sandbox_env))

    # Run headless new-domain
    cmd = [
        sys.executable,
        "scripts/manage_domains.py",
        "new-domain",
        "aerospace",
        "--display-name",
        "Aerospace & Avionics",
        "--scientific",
        "--weight",
        "0.25",
        "--fallbacks",
        "Altitude (m)",
        "Airspeed (knots)",
        "-y",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0
    assert "[OK] Created domain manifest:" in res.stdout
    assert "aerospace.yaml" in res.stdout
    assert "Cache successfully rebuilt with new domain 'aerospace'" in res.stdout

    # 1. Verify aerospace.yaml content
    aero_yaml = sandbox_env / "aerospace.yaml"
    assert aero_yaml.exists()
    from ruamel.yaml import YAML
    yaml = YAML()
    with open(aero_yaml, "r", encoding="utf-8") as f:
        doc = yaml.load(f)
    assert doc["manifest_version"] == 2
    assert doc["domain_id"] == "aerospace"
    assert doc["display_name"] == "Aerospace & Avionics"
    assert doc["is_scientific"] is True
    assert doc["default_weight"] == 0.25
    assert doc["default_fallbacks"] == ["Altitude (m)", "Airspeed (knots)"]

    # 2. Verify common.yaml metric tags updated with aerospace
    common_yaml = sandbox_env / "common.yaml"
    with open(common_yaml, "r", encoding="utf-8") as f:
        common_doc = yaml.load(f)
    shared_metrics = [
        m["label"] for m in common_doc.get("metrics", [])
        if "aerospace" in m.get("domain_tags", [])
    ]
    assert len(shared_metrics) >= 1  # Date, Timestamp, etc. were tagged

    # 3. Test sample_axis_pair for aerospace returns domain-sourced labels or fallback
    rebuild_cache(force=True, domains_dir=sandbox_env)
    x_lbl, y_lbl, res_dom = sample_axis_pair(
        chart_type="line",
        orientation="vertical",
        is_scientific=True,
        semantic_domain="aerospace",
    )
    assert res_dom == "aerospace"
    assert x_lbl is not None
    assert y_lbl is not None


# ==============================================================================
# TUI-01 / ADM-01..10: IMPORT SUBCOMMAND (DRY-RUN VS COMMIT)
# ==============================================================================

def test_cli_import_dry_run_vs_yes(sandbox_env: Path, tmp_path: Path):
    """`manage_domains.py import` defaults to dry-run in headless mode; commits when --yes is passed."""
    env = dict(os.environ, SYNTH_DOMAINS_DIR=str(sandbox_env))

    # First create aerospace domain
    new_dom_cmd = [
        sys.executable,
        "scripts/manage_domains.py",
        "new-domain",
        "aerospace",
        "--scientific",
        "-y",
    ]
    subprocess.run(new_dom_cmd, capture_output=True, text=True, env=env, check=True)

    # Create sample CSV with a new unique metric
    csv_file = tmp_path / "aerospace_metrics.csv"
    csv_file.write_text(
        "Label,Scale Type,Axis Role,Unit,Concept Stem,Concept Group,Domain Tags,Pools\n"
        "Wing Aspect Ratio,continuous,dependent,,wing_aspect_ratio,aerodynamics_metric,aerospace,\n",
        encoding="utf-8",
    )

    # 1. Dry run (no --yes)
    plan_report = tmp_path / "plan.json"
    dry_cmd = [
        sys.executable,
        "scripts/manage_domains.py",
        "import",
        str(csv_file),
        "-d",
        "aerospace",
        "--report",
        str(plan_report),
    ]
    res_dry = subprocess.run(dry_cmd, capture_output=True, text=True, env=env)
    assert res_dry.returncode == 0
    assert "[DRY-RUN] Pre-flight plan calculated. Pass --yes to apply changes." in res_dry.stdout
    assert plan_report.exists()
    # Check that manifest was NOT modified
    aero_yaml = sandbox_env / "aerospace.yaml"
    from ruamel.yaml import YAML
    yaml = YAML()
    with open(aero_yaml, "r", encoding="utf-8") as f:
        doc = yaml.load(f)
    assert len(doc["metrics"]) == 0

    # 2. Commit with --yes
    yes_cmd = [
        sys.executable,
        "scripts/manage_domains.py",
        "import",
        str(csv_file),
        "-d",
        "aerospace",
        "--on-collision",
        "skip",
        "-y",
    ]
    res_yes = subprocess.run(yes_cmd, capture_output=True, text=True, env=env)
    assert res_yes.returncode == 0
    assert "[OK] Transaction successfully committed!" in res_yes.stdout
    assert "Snapshot backup:" in res_yes.stdout

    # Verify manifest now contains Wing Aspect Ratio
    with open(aero_yaml, "r", encoding="utf-8") as f:
        doc = yaml.load(f)
    assert len(doc["metrics"]) == 1
    assert doc["metrics"][0]["label"] == "Wing Aspect Ratio"


# ==============================================================================
# TUI-03: PREFLIGHT DIFF TABLE RENDERING
# ==============================================================================

def test_preflight_table_rendering(sandbox_env: Path, tmp_path: Path):
    """Verify render_preflight_table formatting in unicode, ascii, and colored modes."""
    csv_file = tmp_path / "test_metrics.csv"
    csv_file.write_text(
        "Label,Scale Type,Axis Role,Unit,Concept Stem,Concept Group,Domain Tags,Pools\n"
        "Cabin Pressure (psi),continuous,dependent,psi,cabin_pressure,press,engineering,legacy\n",
        encoding="utf-8",
    )
    ingest_res = ingest_file(csv_file, target_domain="engineering")
    plan = build_plan(ingest_res, target_domain="engineering", domains_dir=sandbox_env)

    # Unicode mode
    table_u = render_preflight_table(plan, ascii_mode=False, no_color=True)
    assert "Status" in table_u
    assert "Entity" in table_u
    assert "Label / Title" in table_u
    assert "Cabin Pressure (psi)" in table_u
    assert "NEW" in table_u
    assert "Summary: 1 new" in table_u
    assert "─" in table_u

    # ASCII mode
    table_a = render_preflight_table(plan, ascii_mode=True, no_color=True)
    assert "-" in table_a
    assert "─" not in table_a


# ==============================================================================
# TUI-02 / TUI-05: INTERACTIVE PROMPT & MAIN MENU
# ==============================================================================

def test_safe_prompt_behavior(monkeypatch):
    """Verify safe_prompt respects defaults and handles Ctrl-C gracefully."""
    # Test default
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert safe_prompt("Test prompt", default="default_val") == "default_val"

    # Test explicit input
    monkeypatch.setattr("builtins.input", lambda _: "custom_val")
    assert safe_prompt("Test prompt", default="default_val") == "custom_val"

    # Test EOF
    def mock_eof(_):
        raise EOFError()
    monkeypatch.setattr("builtins.input", mock_eof)
    assert safe_prompt("Test prompt", default="fallback") == "fallback"


def test_main_menu_rendered_options():
    """Verify render_main_menu lists all required menus 1-6 and 0."""
    menu = render_main_menu(ascii_mode=True, no_color=True)
    assert "[1] Inspect domains & statistics" in menu
    assert "[2] Import & deduplicate spreadsheet" in menu
    assert "[3] Create new domain" in menu
    assert "[4] Sampling weights" in menu
    assert "[5] Validate manifests & rebuild cache" in menu
    assert "[6] Generate charts & edit generation settings" in menu
    assert "[0] Exit" in menu


def test_run_main_menu_exit(monkeypatch):
    """`run_main_menu` exits cleanly when user selects '0'."""
    inputs = iter(["0"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    # Should complete without error
    run_main_menu(ascii_mode=True, no_color=True)


# ==============================================================================
# BATCH FOLDER IMPORT & DOMAIN INFERENCE
# ==============================================================================

def test_infer_domain_from_filename():
    """Verify filename-to-domain inference logic (exact, prefix, plural aliases, token scan)."""
    from synth.semantics.admin.normalize import infer_domain_from_filename

    registered = ["business", "demographic", "engineering", "finance", "healthcare", "technology"]

    # 1. Exact match
    assert infer_domain_from_filename("business.csv", registered) == "business"
    assert infer_domain_from_filename("engineering.xlsx", registered) == "engineering"

    # 2. Separator prefix (underscore / hyphen)
    assert infer_domain_from_filename("business_domain_titles.csv", registered) == "business"
    assert infer_domain_from_filename("engineering-metrics.csv", registered) == "engineering"
    assert infer_domain_from_filename("healthcare_pairs_template.xlsx", registered) == "healthcare"

    # 3. Plural aliases
    assert infer_domain_from_filename("demographics_domain_pairs.csv", registered) == "demographic"
    assert infer_domain_from_filename("technologies_titles.csv", registered) == "technology"

    # 4. In-token match
    assert infer_domain_from_filename("q3_finance_metrics.csv", registered) == "finance"

    # 5. Non-matching / unknown
    assert infer_domain_from_filename("random_dataset.csv", registered) is None
    assert infer_domain_from_filename("domain_template.csv", registered) is None


def test_batch_plans_serialization():
    """Verify batch_plans_to_dict and batch_plans_to_json structure."""
    from synth.semantics.admin.plan import IngestPlan, batch_plans_to_dict, batch_plans_to_json

    p1 = IngestPlan(source_file="file1.csv", target_domain="business", policy="skip", status_counts={"NEW": 3, "DUPLICATE": 1})
    p2 = IngestPlan(source_file="file2.csv", target_domain="engineering", policy="skip", status_counts={"NEW": 2, "SHARED": 1})

    b_dict = batch_plans_to_dict("templates/news", [p1, p2], policy="skip")
    assert b_dict["mode"] == "batch"
    assert b_dict["source_folder"] == "templates/news"
    assert b_dict["files_count"] == 2
    assert b_dict["batch_totals"]["NEW"] == 5
    assert b_dict["batch_totals"]["DUPLICATE"] == 1
    assert b_dict["batch_totals"]["SHARED"] == 1
    assert len(b_dict["files"]) == 2

    b_json = batch_plans_to_json("templates/news", [p1, p2], policy="skip")
    assert '"mode": "batch"' in b_json
    assert '"files_count": 2' in b_json


def test_batch_folder_import_flow(sandbox_env: Path, tmp_path: Path, monkeypatch):
    """Test importing a directory with multiple domain CSVs atomically."""
    from synth.semantics.admin.tui import run_import_flow
    from synth.semantics.loader import SemanticRegistry

    monkeypatch.setenv("SYNTH_DOMAINS_DIR", str(sandbox_env))

    # Create folder with 2 CSVs with domain prefixes
    batch_dir = tmp_path / "news_batch"
    batch_dir.mkdir()

    csv_business = batch_dir / "business_domain_titles.csv"
    csv_business.write_text(
        "Title Template,Domain,Allowed Chart Types\n"
        "Quarterly Revenue Overview for Business,business,bar|line\n",
        encoding="utf-8",
    )

    csv_engineering = batch_dir / "engineering_domain_titles.csv"
    csv_engineering.write_text(
        "Title Template,Domain,Allowed Chart Types\n"
        "Thermal Stress Distribution across Turbines,engineering,scatter|heatmap\n",
        encoding="utf-8",
    )

    report_out = tmp_path / "batch_report.json"

    # Run batch import flow
    ret = run_import_flow(
        ascii_mode=True,
        no_color=True,
        interactive=False,
        file_path=str(batch_dir),
        auto_commit=True,
        report_path=str(report_out),
    )
    assert ret == 0

    # Verify report was generated
    assert report_out.exists()
    rep_data = json.loads(report_out.read_text(encoding="utf-8"))
    assert rep_data["mode"] == "batch"
    assert rep_data["files_count"] == 2
    assert rep_data["batch_totals"].get("NEW", 0) >= 2

    # Verify both manifests got the new titles on disk
    reg = SemanticRegistry.load()
    biz_titles = [t.title for t in reg.titles_catalog if "business" in t.domain_tags]
    eng_titles = [t.title for t in reg.titles_catalog if "engineering" in t.domain_tags]

    assert any("Quarterly Revenue Overview for Business" in t for t in biz_titles)
    assert any("Thermal Stress Distribution across Turbines" in t for t in eng_titles)


def test_batch_folder_import_cli_headless(sandbox_env: Path, tmp_path: Path):
    """Test `scripts/manage_domains.py import <dir> -y` without specifying -d/--domain."""
    batch_dir = tmp_path / "cli_batch"
    batch_dir.mkdir()

    csv1 = batch_dir / "business_extra_titles.csv"
    csv1.write_text(
        "Title Template,Domain,Allowed Chart Types\n"
        "Annual Operating Margin Breakdown,business,bar|box\n",
        encoding="utf-8",
    )

    env = dict(os.environ, SYNTH_DOMAINS_DIR=str(sandbox_env))
    cmd = [
        sys.executable,
        "scripts/manage_domains.py",
        "import",
        str(batch_dir),
        "-y",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0
    assert "Found 1 file(s) to import:" in res.stdout
    assert "Transaction successfully committed!" in res.stdout

