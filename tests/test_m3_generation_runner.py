"""
tests/test_m3_generation_runner.py

Acceptance tests for Milestone M3 (Generation runner):
- GEN-02: Flags (--no-custom, --config, --set, --progress-jsonl, --log-file, --validate-only, --start-index).
- GEN-03: Subprocess exit codes (0 ok, 1 strict failure, 2 invalid config, 3 output-dir conflict).
- GEN-04: run_manifest.json schema, hashes, and determinism check across two runs.
- GEN-07: Subprocess runner integration (run_generator_subprocess) with progress events and log capture.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from synth.semantics.admin.runner import (
    run_generator_subprocess,
    tail_file,
)


def test_gen02_validate_only_exit_codes(tmp_path):
    """GEN-02: --validate-only exits 0 on valid config and 2 on invalid config."""
    # Valid run
    res_ok = subprocess.run(
        [sys.executable, "generator.py", "--validate-only", "--no-custom"],
        capture_output=True,
        text=True,
    )
    assert res_ok.returncode == 0
    assert "Configuration valid" in res_ok.stdout

    # Invalid run via --set
    res_err = subprocess.run(
        [sys.executable, "generator.py", "--validate-only", "--no-custom", "--set", "num_images=0"],
        capture_output=True,
        text=True,
    )
    assert res_err.returncode == 2
    assert "[CONFIG ERROR] num_images" in res_err.stderr


def test_gen03_exit_code_3_output_dir_conflict(tmp_path):
    """GEN-03: Exit code 3 when output dir has existing files and --overwrite is omitted."""
    out_dir = str(tmp_path / "conflict_test")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    # Put a pre-existing label file
    Path(os.path.join(labels_dir, "chart_00000.txt")).write_text("dummy", encoding="utf-8")

    # Run without --overwrite -> exits 3
    res_conflict = subprocess.run(
        [sys.executable, "generator.py", "--num", "1", "--output", out_dir, "--no-custom"],
        capture_output=True,
        text=True,
    )
    assert res_conflict.returncode == 3
    assert "already contains generated files" in res_conflict.stderr

    # Run with --overwrite -> succeeds
    res_overwrite = subprocess.run(
        [
            sys.executable,
            "generator.py",
            "--num",
            "1",
            "--output",
            out_dir,
            "--no-custom",
            "--overwrite",
            "--set",
            "use_parallel=false",
        ],
        capture_output=True,
        text=True,
    )
    assert res_overwrite.returncode == 0


def test_gen04_run_manifest_and_determinism(tmp_path):
    """GEN-04: run_manifest.json created; two runs with same inputs have identical config & dataset hashes."""
    run1_dir = str(tmp_path / "run1")
    run2_dir = str(tmp_path / "run2")

    cmd1 = [
        sys.executable,
        "generator.py",
        "--num",
        "2",
        "--output",
        run1_dir,
        "--no-custom",
        "--set",
        "seed=123",
        "--set",
        "use_parallel=false",
    ]
    cmd2 = [
        sys.executable,
        "generator.py",
        "--num",
        "2",
        "--output",
        run2_dir,
        "--no-custom",
        "--set",
        "seed=123",
        "--set",
        "use_parallel=false",
    ]

    res1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
    res2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)

    manifest1_path = os.path.join(run1_dir, "run_manifest.json")
    manifest2_path = os.path.join(run2_dir, "run_manifest.json")

    assert os.path.exists(manifest1_path)
    assert os.path.exists(manifest2_path)

    with open(manifest1_path, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(manifest2_path, "r", encoding="utf-8") as f:
        m2 = json.load(f)

    # Required fields in manifest
    for req_field in (
        "schema_version",
        "dataset_version",
        "seed",
        "config_hash",
        "dataset_content_hash",
        "manifest_registry_fingerprint",
        "counts",
        "resolved_config",
    ):
        assert req_field in m1, f"Missing field in manifest: {req_field}"

    assert m1["counts"]["successful"] == 2
    assert m1["counts"]["failed"] == 0

    # Determinism assertion
    assert m1["config_hash"] == m2["config_hash"]
    assert m1["dataset_content_hash"] == m2["dataset_content_hash"]
    assert m1["manifest_registry_fingerprint"] == m2["manifest_registry_fingerprint"]


def test_gen07_subprocess_runner_integration(tmp_path):
    """GEN-07: run_generator_subprocess runs 3 images, captures logs and JSONL progress."""
    out_dir = str(tmp_path / "subproc_run")

    events = []
    result = run_generator_subprocess(
        args_list=["--num", "3", "--no-custom", "--set", "use_parallel=false", "--set", "seed=42"],
        output_dir=out_dir,
        progress_callback=lambda evt: events.append(evt),
    )

    assert result.returncode == 0
    assert os.path.exists(result.log_path)
    assert os.path.exists(result.manifest_path)
    assert result.manifest_data is not None
    assert result.manifest_data["counts"]["successful"] == 3

    # Progress events captured
    assert len(events) >= 3
    for evt in events:
        assert evt.get("event") == "image_done"
        assert "index" in evt
        assert "ok" in evt
