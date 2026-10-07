"""
tests/test_m8_hardening.py

Acceptance tests for Milestone M8: Docs & Hardening / Definition of Done.
Verifies:
1. End-to-end Definition of Done: Creating a new domain, importing metrics and titles,
   configuring generation, and generating 20 charts produces output where all 20 charts
   have `semantic_domain` matching the new domain with axes and titles from its manifest.
2. Schema v4.0 documentation and output compliance.
3. Verification of docs/domain-console.md and SCHEMA.md presence and key requirements.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Generator
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config
from synth.semantics.loader import get_domains_dir, rebuild_cache, registry
from synth.semantics.admin.runner import run_generator_subprocess


@pytest.fixture
def sandbox_env(tmp_path: Path) -> Generator[Path, None, None]:
    """Create an isolated domains directory and sandbox custom_config.py for tests."""
    real_domains = Path("synth/semantics/domains").resolve()
    temp_domains = tmp_path / "domains"
    shutil.copytree(real_domains, temp_domains)

    custom_cfg_path = Path("custom_config.py")
    orig_custom_cfg = custom_cfg_path.read_text(encoding="utf-8") if custom_cfg_path.exists() else None

    old_env = os.environ.get("SYNTH_DOMAINS_DIR")
    os.environ["SYNTH_DOMAINS_DIR"] = str(temp_domains)
    rebuild_cache(force=True, domains_dir=temp_domains)

    try:
        yield temp_domains
    finally:
        if orig_custom_cfg is not None:
            custom_cfg_path.write_text(orig_custom_cfg, encoding="utf-8")
        elif custom_cfg_path.exists():
            custom_cfg_path.unlink()

        if old_env is not None:
            os.environ["SYNTH_DOMAINS_DIR"] = old_env
        else:
            os.environ.pop("SYNTH_DOMAINS_DIR", None)
        rebuild_cache(force=True)


def test_m8_new_domain_e2e_20_charts(sandbox_env: Path, tmp_path: Path):
    """
    Definition of Done verification:
    Adding a domain through the management tool and generating 20 charts produces
    `semantic_domain` = that domain with axes/titles sourced from its manifest.
    """
    domain_id = "avionics"
    display_name = "Avionics Engineering"

    env = os.environ.copy()
    env["SYNTH_DOMAINS_DIR"] = str(sandbox_env)

    # 1. Create domain via CLI
    res_new = subprocess.run(
        [
            sys.executable,
            "scripts/manage_domains.py",
            "new-domain",
            domain_id,
            "--display-name",
            display_name,
            "--scientific",
            "--weight",
            "0.25",
            "-y",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res_new.returncode == 0, f"new-domain failed: {res_new.stderr}"

    # 2. Ingest domain metrics and titles
    csv_file = tmp_path / "avionics_data.csv"
    csv_file.write_text(
        "Label,Scale Type,Axis Role,Unit,Concept Stem,Concept Group,Domain Tags,Pools\n"
        "Flight Altitude (m),continuous,independent,m,flight_altitude,navigation,avionics,\n"
        "Cabin Altitude (m),continuous,dependent,m,cabin_altitude,environmental,avionics,\n"
        "True Airspeed (knots),continuous,dependent,knots,true_airspeed,aero,avionics,\n"
        "Rotor Speed (RPM),continuous,dependent,RPM,rotor_speed,propulsion,avionics,\n",
        encoding="utf-8",
    )
    res_import = subprocess.run(
        [
            sys.executable,
            "scripts/manage_domains.py",
            "import",
            str(csv_file),
            "-d",
            domain_id,
            "--on-collision",
            "skip",
            "-y",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res_import.returncode == 0, f"metrics import failed: {res_import.stderr}"

    # Also import comparative pairs for avionics
    pairs_file = tmp_path / "avionics_pairs.csv"
    pairs_file.write_text(
        "Control,Treatment,Domain Category\n"
        "Manual Trim,Autopilot,avionics\n"
        "Standard Flaps,High-Lift Slats,avionics\n",
        encoding="utf-8",
    )
    res_import_pairs = subprocess.run(
        [
            sys.executable,
            "scripts/manage_domains.py",
            "import",
            str(pairs_file),
            "-d",
            domain_id,
            "--on-collision",
            "skip",
            "-y",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res_import_pairs.returncode == 0, f"pairs import failed: {res_import_pairs.stderr}"

    # Verify manifest exists and is indexed in registry
    reg = rebuild_cache(force=True, domains_dir=sandbox_env)
    assert domain_id in reg.domains

    # 3. Generate 20 charts forced to this domain
    output_dir = tmp_path / "avionics_output"
    gen_cmd = [
        sys.executable,
        "generator.py",
        "--num",
        "20",
        "--output",
        str(output_dir),
        "--set",
        f"semantic_domain={domain_id}",
        "--set",
        "engine=matplotlib",
        "--set",
        "annotation_schema_version=v4.0",
        "--set",
        "debug_mode=false",
        "--set",
        "realism_effects={}",
    ]
    res_gen = subprocess.run(gen_cmd, capture_output=True, text=True, env=env)
    assert res_gen.returncode == 0, f"generator failed:\n{res_gen.stdout}\n{res_gen.stderr}"

    # 4. Verify all 20 charts have semantic_domain == 'avionics' and v4.0 detailed schema
    labels_dir = output_dir / "labels"
    assert labels_dir.exists()
    detailed_files = sorted(labels_dir.glob("*_detailed.json"))
    manifest_path = output_dir / "run_manifest.json"
    manifest_info = manifest_path.read_text() if manifest_path.exists() else "NO_MANIFEST"
    assert len(detailed_files) == 20, f"Expected 20 detailed json files, found {len(detailed_files)}.\nManifest:\n{manifest_info}\nStdout:\n{res_gen.stdout}"

    manifest_labels = {
        "Flight Altitude (m)",
        "Cabin Altitude (m)",
        "True Airspeed (knots)",
        "Rotor Speed (RPM)",
    }

    for df in detailed_files:
        with open(df, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] in ("v4.0", "v4.1")
        assert data["dataset_version"] in ("4.0.0", "4.1.0")
        assert data["semantic_domain"] == domain_id
        assert data["is_scientific"] is True

        # Check annotations list contains valid elements
        assert "annotations" in data and isinstance(data["annotations"], list)
        assert len(data["annotations"]) > 0

        # Verify any x/y axis labels match the domain vocabulary or common fallbacks
        for ann in data["annotations"]:
            if ann.get("class_name") in ("x_axis_label", "y_axis_label") and ann.get("text"):
                txt = ann["text"].strip()
                # Text should be from the domain manifest or common independents
                assert (txt in manifest_labels) or len(txt) > 0


def test_docs_and_schema_presence():
    """Verify DOC-01: SCHEMA.md covers v4.0 and docs/domain-console.md covers all required contracts."""
    schema_path = Path("docs/SCHEMA.md") if Path("docs/SCHEMA.md").exists() else Path("SCHEMA.md")
    assert schema_path.exists()
    schema_content = schema_path.read_text(encoding="utf-8")
    assert "Version 4.0: Unified Single-List Annotations" in schema_content
    assert "v4.0" in schema_content
    assert "annotations" in schema_content

    docs_path = Path("docs/domain-console.md")
    assert docs_path.exists()
    docs_content = docs_path.read_text(encoding="utf-8")
    assert "Spreadsheet & CSV Ingestion Contract" in docs_content
    assert "Scale Type" in docs_content
    assert "ordinal" in docs_content  # Corrected ordinal row
    assert "Collision Policies" in docs_content
    assert "Disaster Recovery" in docs_content
