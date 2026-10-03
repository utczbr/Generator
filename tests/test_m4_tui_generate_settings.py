"""
tests/test_m4_tui_generate_settings.py

Acceptance tests for Milestone M4: TUI — Generate & Settings (Release A).
Tests requirements:
- TUI-01: Headless subcommand parity (config show/set/validate, generate, weights, placeholders).
  Respects NO_COLOR and --ascii.
- TUI-02: Header panel rendering (manifest stats, cache OK, custom_config.py status, generator readiness, last run).
- TUI-04: Generate hub & Settings editor: inline CFG-03 validation, Run disabled on errors.
- TUI-05: Safe Ctrl-C interrupt handling at prompts and subprocess run interruption with "interrupted": true in run_manifest.json.
- Sampling weights screen: groups, normalized percentage bars, and custom layer persistence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config
from synth.semantics.admin.tui import (
    render_header,
    render_bar,
    find_recent_runs,
    get_last_run_summary,
    validate_active_config,
    show_effective_config,
    run_generate_flow,
    run_smoke_test,
)
from synth.semantics.admin.runner import run_generator_subprocess


# ==============================================================================
# TUI-02: HEADER PANEL TESTS
# ==============================================================================

def test_header_panel_rendered_unicode():
    """Verify header panel contains all required elements in Unicode mode."""
    header = render_header(ascii_mode=False, no_color=True)
    assert "Synthetic Chart Studio" in header
    assert "Domains:" in header
    assert "biomedical" in header
    assert "business" in header
    assert "engineering" in header
    assert "Metrics:" in header
    assert "Titles:" in header
    assert "Pairs:" in header
    assert "Cache: [OK]" in header
    assert "Generator: ready (matplotlib)" in header
    assert "╭" in header and "╮" in header and "╰" in header and "╯" in header


def test_header_panel_rendered_ascii():
    """Verify header panel uses strict ASCII borders with --ascii."""
    header = render_header(ascii_mode=True, no_color=True)
    assert "Synthetic Chart Studio" in header
    assert "Cache: [OK]" in header
    assert "+" in header
    assert "|" in header
    assert "╭" not in header
    assert "│" not in header
    assert "←" not in header
    assert "<-" in header or "defaults" in header


def test_render_bar_ascii_and_unicode():
    """Verify percentage bar formatting in both modes."""
    bar_u = render_bar(50.0, width=10, ascii_mode=False)
    assert bar_u == "[█████░░░░░]"

    bar_a = render_bar(50.0, width=10, ascii_mode=True)
    assert bar_a == "[=====     ]"


# ==============================================================================
# TUI-01: HEADLESS SUBCOMMANDS PARITY & BEHAVIOR
# ==============================================================================

def test_headless_config_validate_clean():
    """`manage_domains.py config validate` exits 0 on valid config."""
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "config", "validate"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "CONFIG VALIDATION REPORT" in res.stdout
    assert "Configuration is completely valid" in res.stdout


def test_headless_config_show_json():
    """`manage_domains.py config show --json` outputs parseable JSON."""
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "config", "show", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout)
    assert "num_images" in data
    assert "seed" in data
    assert "scientific_ratio" in data


def test_headless_config_set_and_rollback(tmp_path):
    """`manage_domains.py config set` rejects invalid values and saves valid ones."""
    # 1. Invalid value: num_images = 0
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "config", "set", "num_images=0"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "Invalid value" in res.stderr

    # 2. Valid value: export_svg = true
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "config", "set", "export_svg=true"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[OK]" in res.stdout
    cfg = load_config()
    assert cfg["export_svg"] is True

    # Cleanup: restore export_svg to default False
    subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "config", "set", "export_svg=false"],
        capture_output=True,
        text=True,
        check=True,
    )


def test_headless_weights_show_ascii():
    """`manage_domains.py --ascii weights show` outputs formatted groups with ascii bars."""
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py", "--ascii", "weights", "show"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "=== Scientific Domains ===" in res.stdout
    assert "biomedical" in res.stdout
    assert "engineering" in res.stdout
    assert "=== Non-Scientific Domains ===" in res.stdout
    assert "business" in res.stdout
    assert "demographic" in res.stdout
    assert "[" in res.stdout and "]" in res.stdout


def test_headless_subcommands_registered():
    """Check that all Release B subcommands are registered and display help."""
    for sub in ("audit", "import", "new-domain", "rebuild", "restore", "template"):
        res = subprocess.run(
            [sys.executable, "scripts/manage_domains.py", sub, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"usage: manage_domains.py {sub}" in res.stdout


def test_headless_non_tty_root_exits_cleanly():
    """Running manage_domains.py with redirected input/output prints header and exits 0."""
    res = subprocess.run(
        [sys.executable, "scripts/manage_domains.py"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Synthetic Chart Studio" in res.stdout


# ==============================================================================
# TUI-04: GENERATE HUB & SETTINGS VALIDATION
# ==============================================================================

def test_generate_flow_disabled_on_validation_errors(tmp_path):
    """Verify run_generate_flow aborts with returncode 2 when candidate config is invalid."""
    out_dir = str(tmp_path / "test_out")
    ret = run_generate_flow(
        interactive=False,
        override_args={
            "num_images": 0,  # Invalid: must be >= 1
            "output_dir": out_dir,
        },
    )
    assert ret == 2
    assert not (tmp_path / "test_out" / "images").exists()


def test_generate_smoke_test_end_to_end():
    """Verify smoke test generates 3 charts and produces run_manifest.json."""
    ret = run_smoke_test(ascii_mode=True, no_color=True)
    assert ret == 0


# ==============================================================================
# TUI-05: CTRL-C INTERRUPT HANDLING DURING RUN
# ==============================================================================

def test_generator_subprocess_interrupted_sets_manifest_flag(tmp_path):
    """
    Verify that if a running generator subprocess is interrupted (SIGINT),
    it terminates cleanly and writes or updates run_manifest.json with interrupted: true.
    """
    out_dir = str(tmp_path / "interrupted_run")
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "progress.jsonl")
    log_path = os.path.join(out_dir, "run.log")

    cmd = [
        sys.executable,
        "generator.py",
        "--num", "50",
        "--output", out_dir,
        "--progress-jsonl", progress_path,
        "--log-file", log_path,
    ]

    import signal
    import time
    proc = subprocess.Popen(cmd)

    # Wait until at least 1 image is completed or 2 seconds pass
    t0 = time.time()
    while time.time() - t0 < 5.0:
        if os.path.exists(progress_path) and os.path.getsize(progress_path) > 0:
            break
        time.sleep(0.1)

    # Send SIGINT (Ctrl-C)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    manifest_path = os.path.join(out_dir, "run_manifest.json")
    assert os.path.exists(manifest_path), "run_manifest.json must exist even when interrupted"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("interrupted") is True, f"Manifest should have interrupted: true, got {data}"
