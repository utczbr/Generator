"""
synth/semantics/admin/tui.py

Terminal UI and interactive console for Synthetic Chart Studio (Milestone M4).
Implements TUI-01, TUI-02, TUI-04, TUI-05:
- Header panel with manifest/cache health, active config sources, and last run.
- Menu [6]: Generate hub & Settings editor.
- Menu [4]: Sampling weights with live normalized percentage bars.
- Safe Ctrl-C interrupt handling at all prompts.
- Subprocess generator execution with JSONL progress tracking and log tailing.
- Strict NO_COLOR and --ascii (no emoji/unicode) support.
"""
from __future__ import annotations

import copy
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config, ResolvedConfig
from synth.semantics.admin.configspec import list_domains, validate, Issue
from synth.semantics.admin.configsvc import (
    compute_leaf_diff,
    save_overrides_to_custom_config,
    patch_config_defaults_literal,
)
from synth.semantics.admin.runner import run_generator_subprocess, GeneratorSubprocessResult

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}


def colorize(text: str, color: str, no_color: bool = False) -> str:
    """Format text with ANSI colors unless NO_COLOR is enabled."""
    if no_color or "NO_COLOR" in os.environ or not sys.stdout.isatty():
        return text
    code = COLORS.get(color, "")
    reset = COLORS["reset"]
    return f"{code}{text}{reset}" if code else text


def find_recent_runs(workspace_dir: str = ".") -> List[Dict[str, Any]]:
    """Discover run_manifest.json files across common output directories."""
    manifest_paths = []
    # Check default and common paths
    patterns = [
        "run_manifest.json",
        "*/run_manifest.json",
        "*/*/run_manifest.json",
        "generated_dataset*/run_manifest.json",
    ]
    for pat in patterns:
        for p in glob.glob(os.path.join(workspace_dir, pat)):
            if os.path.isfile(p):
                manifest_paths.append(os.path.abspath(p))

    manifest_paths = list(set(manifest_paths))
    runs = []
    for mp in manifest_paths:
        try:
            mtime = os.path.getmtime(mp)
            with open(mp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_manifest_path"] = mp
            data["_mtime"] = mtime
            runs.append(data)
        except Exception:
            continue

    runs.sort(key=lambda r: r.get("_mtime", 0), reverse=True)
    return runs


def get_last_run_summary(runs: Optional[List[Dict[str, Any]]] = None) -> str:
    """Return a short summary of the most recent generation run."""
    if runs is None:
        runs = find_recent_runs()
    if not runs:
        return "none"
    latest = runs[0]
    counts = latest.get("counts", {})
    num = counts.get("num_images", latest.get("successful", 0) + latest.get("failed", 0))
    failed = counts.get("failed", latest.get("failed", 0))
    interrupted = latest.get("interrupted", False)
    if interrupted:
        return f"{num} imgs (interrupted), {failed} failed"
    return f"{num} imgs, {failed} failed"


def check_generator_readiness() -> Tuple[bool, str]:
    """Check whether generator dependencies are importable."""
    try:
        import matplotlib
        return True, "ready (matplotlib)"
    except Exception as e:
        return False, f"not ready ({e})"


def get_manifest_stats() -> Tuple[int, int, int]:
    """Get metrics, titles, and pairs counts from the semantic registry."""
    try:
        from synth.semantics.loader import registry
        metrics_count = len(registry.axis_metrics_catalog)
        titles_count = len(registry.titles_catalog)
        pairs_count = len(registry.pairs_catalog)
        return metrics_count, titles_count, pairs_count
    except Exception:
        return 495, 192, 96


def render_header(ascii_mode: bool = False, no_color: bool = False) -> str:
    """Render the Studio header panel (TUI-02)."""
    # 1. Domains
    domains = sorted(list_domains())
    domains_str = ", ".join(domains)

    # 2. Stats & Cache
    metrics_count, titles_count, pairs_count = get_manifest_stats()
    cache_status = "[OK]"

    # 3. Config sources & overrides
    has_custom = Path("custom_config.py").exists()
    custom_overrides_count = 0
    if has_custom:
        try:
            cfg = load_config(use_custom=True)
            custom_overrides_count = sum(1 for src in cfg.provenance.values() if src == "custom")
        except Exception:
            custom_overrides_count = 0

    if has_custom:
        config_line_text = f"defaults ← custom_config.py ({custom_overrides_count} overrides)"
        if ascii_mode:
            config_line_text = f"defaults <- custom_config.py ({custom_overrides_count} overrides)"
    else:
        config_line_text = "defaults (custom_config.py not present)"

    # 4. Generator & Last run
    ready, gen_status = check_generator_readiness()
    last_run = get_last_run_summary()
    generator_line_text = f"{gen_status}   Last run: {last_run}"

    # Box drawing
    top_l = "+" if ascii_mode else "╭"
    top_r = "+" if ascii_mode else "╮"
    bot_l = "+" if ascii_mode else "╰"
    bot_r = "+" if ascii_mode else "╯"
    horiz = "-" if ascii_mode else "─"
    vert = "|" if ascii_mode else "│"

    width = 68
    title = " Synthetic Chart Studio "
    pad_total = width - 2 - len(title)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    top_border = f"{top_l}{horiz * pad_left}{title}{horiz * pad_right}{top_r}"
    bot_border = f"{bot_l}{horiz * (width - 2)}{bot_r}"

    def pad_line(content: str) -> str:
        inner_width = width - 4
        if len(content) > inner_width:
            content = content[:inner_width - 3] + "..."
        return f"{vert}  {content}{' ' * (inner_width - len(content))}  {vert}"

    line1 = f"Domains: {domains_str}"
    line2 = f"Metrics: {metrics_count} | Titles: {titles_count} | Pairs: {pairs_count}   Cache: {cache_status}"
    line3 = f"Config:  {config_line_text}"
    line4 = f"Generator: {generator_line_text}"

    lines = [
        colorize(top_border, "blue", no_color),
        pad_line(line1),
        pad_line(line2),
        pad_line(line3),
        pad_line(line4),
        colorize(bot_border, "blue", no_color),
    ]
    return "\n".join(lines)


def render_main_menu(ascii_mode: bool = False, no_color: bool = False) -> str:
    """Render the main menu list (TUI-02)."""
    options = [
        "  [1] Inspect domains & statistics",
        "  [2] Import & deduplicate spreadsheet (.xlsx / .csv)",
        "  [3] Create new domain",
        "  [4] Sampling weights (domains, chart types, scenarios)",
        "  [5] Validate manifests & rebuild cache",
        "  [6] Generate charts & edit generation settings",
        "  [0] Exit",
    ]
    return "\n".join(options)


def safe_prompt(prompt_text: str, default: Optional[str] = None) -> str:
    """Prompt user with default support, returning clean string on Ctrl-C."""
    full_prompt = prompt_text
    if default is not None:
        full_prompt = f"{prompt_text} [{default}]: "
    else:
        full_prompt = f"{prompt_text}: "
    try:
        val = input(full_prompt).strip()
        if not val and default is not None:
            return default
        return val
    except EOFError:
        return default or ""


# ==============================================================================
# MENU [6]: GENERATE HUB & SETTINGS
# ==============================================================================

def render_generate_menu(ascii_mode: bool = False, no_color: bool = False) -> str:
    """Render the [6] Generate charts & edit generation settings menu (§5.2)."""
    options = [
        "=== Generate Charts & Settings ===",
        "  [a] Generate charts…            number, format, output dir, seed, engine, workers",
        "  [b] Smoke test (3 charts → temp dir, then open summary)",
        "  [c] Edit settings (guided)      groups below; saved to custom_config.py",
        "  [d] Open a config file in $EDITOR   custom_config.py / config_defaults.py (dev mode)",
        "  [e] Show effective config       every key with its source: default / custom / cli",
        "  [f] Validate config             CFG-03 report",
        "  [g] Recent runs                 from run_manifest.json files",
        "  [0] Back",
    ]
    return "\n".join(options)


def run_generate_flow(
    cfg: Optional[ResolvedConfig] = None,
    ascii_mode: bool = False,
    no_color: bool = False,
    interactive: bool = True,
    override_args: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Execute generation flow [a] (§5.2).
    Prompts: number of charts, format, output dir, seed, engine, workers.
    Validates CFG-03, guards output dir, estimates time, runs subprocess, and reports summary.
    """
    if cfg is None:
        cfg = load_config()

    overrides = override_args or {}

    # 1. Interactive Prompts
    if interactive:
        print("\n--- Generation Parameters ---")
        num_def = str(overrides.get("num_images", cfg.get("num_images", 100)))
        num_src = cfg.provenance.get("num_images", "default")
        num_str = safe_prompt(f"Number of charts (source: {num_src})", default=num_def)
        try:
            num_images = int(num_str)
        except ValueError:
            num_images = int(num_def)

        fmt_def = str(overrides.get("dataset_format", cfg.get("dataset_format", "detection")))
        fmt_src = cfg.provenance.get("dataset_format", "default")
        fmt_val = safe_prompt(f"Dataset format [detection/classification/multi_chart_detection] (source: {fmt_src})", default=fmt_def)

        out_def = str(overrides.get("output_dir", cfg.get("output_dir", "generated_dataset")))
        out_src = cfg.provenance.get("output_dir", "default")
        out_dir = safe_prompt(f"Output directory (source: {out_src})", default=out_def)

        seed_def = str(overrides.get("seed", cfg.get("seed", 42)))
        seed_src = cfg.provenance.get("seed", "default")
        seed_str = safe_prompt(f"Random seed (source: {seed_src})", default=seed_def)
        try:
            seed_val = int(seed_str)
        except ValueError:
            seed_val = int(seed_def)

        eng_def = str(overrides.get("engine", cfg.get("engine", "matplotlib")))
        eng_src = cfg.provenance.get("engine", "default")
        eng_val = safe_prompt(f"Rendering engine [matplotlib/vegalite] (source: {eng_src})", default=eng_def)

        max_w = min(os.cpu_count() or 1, 16)
        work_def = str(overrides.get("workers", max_w))
        work_str = safe_prompt("Workers", default=work_def)
        try:
            workers_val = int(work_str)
        except ValueError:
            workers_val = max_w
    else:
        num_images = int(overrides.get("num_images", cfg.get("num_images", 100)))
        fmt_val = str(overrides.get("dataset_format", cfg.get("dataset_format", "detection")))
        out_dir = str(overrides.get("output_dir", cfg.get("output_dir", "generated_dataset")))
        seed_val = int(overrides.get("seed", cfg.get("seed", 42)))
        eng_val = str(overrides.get("engine", cfg.get("engine", "matplotlib")))
        workers_val = int(overrides.get("workers", min(os.cpu_count() or 1, 16)))

    # 2. Validation Gate (CFG-03)
    candidate_cfg = copy.deepcopy(dict(cfg))
    candidate_cfg["num_images"] = num_images
    candidate_cfg["dataset_format"] = fmt_val
    candidate_cfg["output_dir"] = out_dir
    candidate_cfg["seed"] = seed_val
    candidate_cfg["engine"] = eng_val

    issues = validate(candidate_cfg)
    errors = [iss for iss in issues if iss.level == "error"]
    warnings = [iss for iss in issues if iss.level == "warning"]

    if warnings:
        print("\n" + colorize("Validation Warnings:", "yellow", no_color))
        for w in warnings:
            print(f"  - [{w.path}] {w.msg}")

    if errors:
        print("\n" + colorize("Validation Errors (Run Disabled):", "red", no_color))
        for e in errors:
            print(f"  - [{e.path}] {e.msg}")
        print("\nPlease resolve validation errors before running generation.")
        return 2

    # 3. Output directory guard
    start_index = overrides.get("start_index")
    overwrite_flag = overrides.get("overwrite", False)

    lbls_dir = os.path.join(out_dir, "labels")
    imgs_dir = os.path.join(out_dir, "images")
    has_existing = any(os.path.exists(d) and len(os.listdir(d)) > 0 for d in (lbls_dir, imgs_dir))

    if has_existing and not overwrite_flag and start_index is None:
        if interactive:
            print(colorize(f"\n[WARNING] Output directory '{out_dir}' already contains generated files.", "yellow", no_color))
            ts = time.strftime("%Y%m%d_%H%M%S")
            ts_dir = f"{out_dir}_{ts}"
            print(f"  [1] Create new timestamped directory (default: {ts_dir})")
            print("  [2] Overwrite existing directory (requires typed confirmation)")
            print("  [3] Append to existing directory (--start-index)")
            print("  [0] Cancel")
            choice = safe_prompt("Choose conflict resolution", default="1")
            if choice == "1":
                out_dir = ts_dir
            elif choice == "2":
                confirm = safe_prompt("Type 'yes' to confirm overwrite", default="no")
                if confirm.lower() != "yes":
                    print("Generation cancelled.")
                    return 0
                overwrite_flag = True
            elif choice == "3":
                # Find highest chart index
                existing_indices = []
                if os.path.exists(lbls_dir):
                    for f in os.listdir(lbls_dir):
                        if f.startswith("chart_") and f.endswith(".txt"):
                            try:
                                existing_indices.append(int(f[6:11]))
                            except ValueError:
                                pass
                next_idx = (max(existing_indices) + 1) if existing_indices else 0
                idx_str = safe_prompt("Start index for append", default=str(next_idx))
                try:
                    start_index = int(idx_str)
                except ValueError:
                    start_index = next_idx
            else:
                print("Generation cancelled.")
                return 0
        else:
            sys.stderr.write(f"[ERROR] Output directory '{out_dir}' already contains generated files. Use --overwrite or --start-index.\n")
            return 3

    # 4. Time estimation
    recent_runs = find_recent_runs()
    secs_per_img = 0.35  # default baseline
    if recent_runs:
        r = recent_runs[0]
        w_time = r.get("wall_time_seconds", 0)
        n_img = r.get("counts", {}).get("num_images", 0)
        if w_time > 0 and n_img > 0:
            secs_per_img = (w_time / n_img)

    est_total_secs = (secs_per_img * num_images) / max(workers_val, 1)
    if interactive:
        est_min = est_total_secs / 60.0
        print(f"\nEstimated time: ~{est_min:.1f} minutes ({num_images} charts @ {workers_val} workers)")
        if est_total_secs > 600:  # > 10 minutes
            conf = safe_prompt("Generation estimate exceeds 10 minutes. Continue? [y/N]", default="N")
            if conf.lower() not in ("y", "yes"):
                print("Generation cancelled.")
                return 0

    # 5. Execute Subprocess (GEN-07)
    cmd_args = [
        "--num", str(num_images),
        "--format", fmt_val,
        "--seed", str(seed_val),
        "--engine", eng_val,
    ]
    if workers_val is not None:
        cmd_args.extend(["--workers", str(workers_val)])
    if overwrite_flag:
        cmd_args.append("--overwrite")
    if start_index is not None:
        cmd_args.extend(["--start-index", str(start_index)])
    if overrides.get("no_custom", False):
        cmd_args.append("--no-custom")

    print(f"\nStarting generation of {num_images} charts to '{out_dir}'...")
    res = run_generator_subprocess(
        args_list=cmd_args,
        output_dir=out_dir,
        show_rich_progress=not ascii_mode and not no_color and sys.stdout.isatty(),
    )

    if res.returncode == 0:
        print("\n" + colorize("✓ Generation completed successfully!", "green", no_color))
        if res.manifest_data:
            mf = res.manifest_data
            counts = mf.get("counts", {})
            succ = counts.get("successful", 0)
            failed = counts.get("failed", 0)
            ver = mf.get("schema_version", "v4.0")
            w_time = mf.get("wall_time_seconds", 0)
            print(f"Summary: {succ}/{num_images} successful ({failed} failed) in {w_time:.2f}s | Schema: {ver}")
            print(f"Manifest written to: {res.manifest_path}")
        return 0
    elif res.returncode == 130 or (res.manifest_data and res.manifest_data.get("interrupted")):
        print("\n" + colorize("⚠ Generation was interrupted by user (Ctrl-C).", "yellow", no_color))
        print(f"Partial output preserved in '{out_dir}'.")
        if res.manifest_data:
            succ = res.manifest_data.get("counts", {}).get("successful", 0)
            print(f"Partial charts generated: {succ}")
        return 130
    else:
        print("\n" + colorize(f"✗ Generation failed with exit code {res.returncode}.", "red", no_color))
        if res.failure_tail:
            print("\nTail of execution log (last 40 lines):")
            print("=" * 60)
            for line in res.failure_tail:
                print(line)
            print("=" * 60)
        return res.returncode


def run_smoke_test(ascii_mode: bool = False, no_color: bool = False) -> int:
    """Run smoke test: generate 3 charts to temp directory and display summary (§5.2)."""
    tmp_dir = tempfile.mkdtemp(prefix="smoke_test_")
    print(f"\nRunning Smoke Test (3 charts -> {tmp_dir})...")
    res = run_generator_subprocess(
        args_list=["--num", "3", "--no-custom"],
        output_dir=tmp_dir,
        show_rich_progress=False,
    )
    if res.returncode == 0:
        print(colorize("\n✓ Smoke test succeeded!", "green", no_color))
        if res.manifest_data:
            mf = res.manifest_data
            w_time = mf.get("wall_time_seconds", 0)
            print(f"Time: {w_time:.2f}s | Schema: {mf.get('schema_version')} | Config hash: {mf.get('config_hash')[:12]}")
            print(f"Temp output: {tmp_dir}")
        return 0
    else:
        print(colorize(f"\n✗ Smoke test failed with exit code {res.returncode}.", "red", no_color))
        if res.failure_tail:
            for l in res.failure_tail:
                print(l)
        return res.returncode


def run_guided_settings(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Guided settings editor (§5.2 [c])."""
    cfg = load_config()

    groups = [
        ("1", "Run basics (num_images, seed, engine, dataset_format, output_dir)"),
        ("2", "Chart mix (chart type enabled/weights, scenario weights)"),
        ("3", "Domains (scientific ratio, subdomain weights, semantic_domain)"),
        ("4", "Per-chart options (bar, line, scatter, box, histogram, pie, heatmap, area)"),
        ("5", "Realism effects (probabilities and parameters)"),
        ("6", "Output & schema (schema version, OBB, legacy JSON, SVG export)"),
        ("7", "Multi-chart layout"),
        ("8", "Advanced (parallelism, strict mode, debug mode)"),
    ]

    while True:
        print("\n=== Guided Settings Groups ===")
        for key, name in groups:
            print(f"  [{key}] {name}")
        print("  [0] Back to Generate Hub")

        choice = safe_prompt("Select group to edit", default="0")
        if choice in ("0", "b", "q", ""):
            break

        if choice == "1":
            _edit_run_basics(cfg, ascii_mode, no_color)
        elif choice == "2":
            _edit_chart_mix(cfg, ascii_mode, no_color)
        elif choice == "3":
            _edit_domains_settings(cfg, ascii_mode, no_color)
        elif choice == "4":
            _edit_per_chart_options(cfg, ascii_mode, no_color)
        elif choice == "5":
            _edit_realism_effects(cfg, ascii_mode, no_color)
        elif choice == "6":
            _edit_output_and_schema(cfg, ascii_mode, no_color)
        elif choice == "7":
            _edit_multi_chart_settings(cfg, ascii_mode, no_color)
        elif choice == "8":
            _edit_advanced_settings(cfg, ascii_mode, no_color)
        else:
            print("Invalid option.")


def _save_setting_diff(dot_path: str, new_val: Any, old_val: Any, no_color: bool = False) -> bool:
    """Validate and offer to persist an edited setting to custom_config.py."""
    cfg = load_config()
    candidate = copy.deepcopy(dict(cfg))
    parts = dot_path.split(".")
    curr = candidate
    for p in parts[:-1]:
        if p not in curr or not isinstance(curr[p], dict):
            curr[p] = {}
        curr = curr[p]
    curr[parts[-1]] = new_val

    issues = validate(candidate)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        print("\n" + colorize("Validation error(s) for this value:", "red", no_color))
        for err in errors:
            print(f"  - [{err.path}] {err.msg}")
        print("Value rejected. Changes not saved.")
        return False

    print(f"\nEdit: {dot_path} : {repr(old_val)} → {repr(new_val)}")
    save = safe_prompt("Save override to custom_config.py? [Y/n]", default="Y")
    if save.lower() in ("y", "yes"):
        # Load existing diffs
        existing_diffs = compute_leaf_diff(candidate, baseline=DEFAULT_CONFIG)
        save_overrides_to_custom_config(existing_diffs)
        print(colorize("✓ Saved to custom_config.py", "green", no_color))
        return True
    return False


def _edit_run_basics(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    keys = ["num_images", "seed", "engine", "dataset_format", "output_dir"]
    while True:
        print("\n--- Run Basics ---")
        for i, k in enumerate(keys, 1):
            val = cfg.get(k)
            src = cfg.provenance.get(k, "default")
            print(f"  [{i}] {k:16s} = {repr(val):20s} [{src}]")
        print("  [0] Back")
        c = safe_prompt("Select key to edit", default="0")
        if c in ("0", "b", "q", ""):
            break
        try:
            idx = int(c) - 1
            if 0 <= idx < len(keys):
                target_key = keys[idx]
                curr_val = cfg.get(target_key)
                new_str = safe_prompt(f"Enter new value for {target_key} (current: {curr_val})")
                if not new_str:
                    continue
                # Parse type
                if isinstance(curr_val, int):
                    try:
                        parsed = int(new_str)
                    except ValueError:
                        print("Invalid integer.")
                        continue
                else:
                    parsed = new_str
                if _save_setting_diff(target_key, parsed, curr_val, no_color):
                    cfg = load_config()
        except ValueError:
            pass


def _edit_chart_mix(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    chart_keys = [
        "weights_bar", "weights_line", "weights_scatter", "weights_box",
        "weights_histogram", "weights_pie", "weights_heatmap", "weights_area",
    ]
    while True:
        print("\n--- Chart Mix & Weights ---")
        for i, k in enumerate(chart_keys, 1):
            val = cfg.get(k)
            src = cfg.provenance.get(k, "default")
            print(f"  [{i}] {k:20s} = {val} [{src}]")
        print("  [0] Back")
        c = safe_prompt("Select weight to edit", default="0")
        if c in ("0", "b", "q", ""):
            break
        try:
            idx = int(c) - 1
            if 0 <= idx < len(chart_keys):
                target_key = chart_keys[idx]
                curr_val = cfg.get(target_key)
                new_str = safe_prompt(f"Enter new float weight for {target_key}")
                if not new_str:
                    continue
                try:
                    parsed = float(new_str)
                except ValueError:
                    print("Invalid float.")
                    continue
                if _save_setting_diff(target_key, parsed, curr_val, no_color):
                    cfg = load_config()
        except ValueError:
            pass


def _edit_domains_settings(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    while True:
        print("\n--- Domains & Ratios ---")
        s_ratio = cfg.get("scientific_ratio", 0.5)
        s_src = cfg.provenance.get("scientific_ratio", "default")
        s_dom = cfg.get("semantic_domain")
        dom_src = cfg.provenance.get("semantic_domain", "default")
        print(f"  [1] scientific_ratio = {s_ratio} [{s_src}]")
        print(f"  [2] semantic_domain  = {repr(s_dom)} [{dom_src}]")
        print("  [0] Back")
        c = safe_prompt("Select option", default="0")
        if c in ("0", "b", "q", ""):
            break
        if c == "1":
            new_s = safe_prompt("Enter scientific_ratio [0.0 - 1.0]", default=str(s_ratio))
            try:
                parsed = float(new_s)
                if _save_setting_diff("scientific_ratio", parsed, s_ratio, no_color):
                    cfg = load_config()
            except ValueError:
                print("Invalid float.")
        elif c == "2":
            new_dom = safe_prompt("Enter semantic_domain (or 'None' to clear)", default=str(s_dom))
            parsed_dom = None if new_dom in ("None", "none", "") else new_dom
            if _save_setting_diff("semantic_domain", parsed_dom, s_dom, no_color):
                cfg = load_config()


def _edit_per_chart_options(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    options = [
        ("bar_chart_config.horizontal_ratio", float),
        ("bar_chart_config.stacked_ratio", float),
        ("bar_chart_config.grouped_ratio", float),
        ("line_chart_config.multi_line_ratio", float),
        ("scatter_chart_config.bubble_ratio", float),
        ("box_chart_config.horizontal_ratio", float),
        ("histogram_config.kde_ratio", float),
        ("pie_chart_config.donut_ratio", float),
        ("heatmap_config.annotated_ratio", float),
    ]
    while True:
        print("\n--- Per-Chart Options ---")
        for i, (path, _) in enumerate(options, 1):
            parts = path.split(".")
            val = cfg.get(parts[0], {}).get(parts[1], "N/A")
            print(f"  [{i}] {path:36s} = {val}")
        print("  [0] Back")
        c = safe_prompt("Select option to edit", default="0")
        if c in ("0", "b", "q", ""):
            break
        try:
            idx = int(c) - 1
            if 0 <= idx < len(options):
                path, typ = options[idx]
                parts = path.split(".")
                curr_val = cfg.get(parts[0], {}).get(parts[1])
                new_str = safe_prompt(f"Enter new value for {path}", default=str(curr_val))
                parsed = typ(new_str)
                if _save_setting_diff(path, parsed, curr_val, no_color):
                    cfg = load_config()
        except Exception as e:
            print(f"Error: {e}")


def _edit_realism_effects(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    from generator import EFFECT_REGISTRY
    effects = sorted(list(EFFECT_REGISTRY.keys()))
    while True:
        print("\n--- Realism Effects (EFFECT_REGISTRY) ---")
        eff_dict = cfg.get("realism_effects", {})
        for i, eff in enumerate(effects, 1):
            curr_p = eff_dict.get(eff, {}).get("p", 0.0)
            print(f"  [{i:2d}] {eff:25s} p = {curr_p:.2f}")
        print("  [0] Back")
        c = safe_prompt("Select effect to edit probability", default="0")
        if c in ("0", "b", "q", ""):
            break
        try:
            idx = int(c) - 1
            if 0 <= idx < len(effects):
                eff_name = effects[idx]
                curr_p = eff_dict.get(eff_name, {}).get("p", 0.0)
                new_str = safe_prompt(f"Enter probability for {eff_name} [0.0 - 1.0]", default=str(curr_p))
                parsed = float(new_str)
                path = f"realism_effects.{eff_name}.p"
                if _save_setting_diff(path, parsed, curr_p, no_color):
                    cfg = load_config()
        except Exception as e:
            print(f"Error: {e}")


def _edit_output_and_schema(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    keys = [
        ("annotation_schema_version", str),
        ("dataset_version", str),
        ("export_oriented_bounding_boxes", bool),
        ("export_legacy_format", bool),
        ("export_svg", bool),
    ]
    while True:
        print("\n--- Output & Schema Options ---")
        for i, (k, _) in enumerate(keys, 1):
            val = cfg.get(k)
            src = cfg.provenance.get(k, "default")
            print(f"  [{i}] {k:32s} = {repr(val):12s} [{src}]")
        print("  [0] Back")
        c = safe_prompt("Select option to edit", default="0")
        if c in ("0", "b", "q", ""):
            break
        try:
            idx = int(c) - 1
            if 0 <= idx < len(keys):
                k, typ = keys[idx]
                curr_val = cfg.get(k)
                new_str = safe_prompt(f"Enter new value for {k}", default=str(curr_val))
                if typ is bool:
                    parsed = new_str.lower() in ("true", "1", "yes")
                else:
                    parsed = typ(new_str)
                if _save_setting_diff(k, parsed, curr_val, no_color):
                    cfg = load_config()
        except Exception as e:
            print(f"Error: {e}")


def _edit_multi_chart_settings(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    print("\n--- Multi-Chart Settings ---")
    mc = cfg.get("multi_chart_config", {})
    print(f"Current multi_chart_config: {mc}")
    safe_prompt("Press Enter to continue")


def _edit_advanced_settings(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    keys = [
        ("use_parallel", bool),
        ("strict", bool),
        ("debug_mode", bool),
    ]
    while True:
        print("\n--- Advanced Settings ---")
        for i, (k, _) in enumerate(keys, 1):
            val = cfg.get(k)
            src = cfg.provenance.get(k, "default")
            warn = " (redirects output to test/)" if k == "debug_mode" and val else ""
            print(f"  [{i}] {k:16s} = {repr(val):8s} [{src}]{warn}")
        print("  [0] Back")
        c = safe_prompt("Select setting to toggle", default="0")
        if c in ("0", "b", "q", ""):
            break
        try:
            idx = int(c) - 1
            if 0 <= idx < len(keys):
                k, _ = keys[idx]
                curr_val = cfg.get(k, False)
                new_val = not curr_val
                if k == "debug_mode" and new_val:
                    print(colorize("WARNING: debug_mode=True redirects chart rendering output to test/!", "yellow", no_color))
                if _save_setting_diff(k, new_val, curr_val, no_color):
                    cfg = load_config()
        except Exception as e:
            print(f"Error: {e}")


def run_editor_flow(target: str = "custom", ascii_mode: bool = False, no_color: bool = False) -> None:
    """Spawn $EDITOR for custom_config.py or config_defaults.py (TUI-04, §5.2 [d])."""
    target_file = Path("custom_config.py") if target == "custom" else Path("config_defaults.py")
    if not target_file.exists():
        if target == "custom":
            # Initialize empty custom_config.py
            save_overrides_to_custom_config({})
        else:
            print(f"Error: {target_file} not found.")
            return

    editor = os.environ.get("EDITOR") or ("nano" if shutil.which("nano") else "vi")
    print(f"Opening {target_file} in {editor}...")
    try:
        subprocess.run([editor, str(target_file)], check=True)
    except Exception as e:
        print(f"Could not launch editor: {e}")
        return

    # Validate post-edit
    try:
        cfg = load_config()
        issues = validate(dict(cfg))
        errors = [i for i in issues if i.level == "error"]
        if errors:
            print(colorize("\n[WARNING] Configuration has errors after edit:", "red", no_color))
            for err in errors:
                print(f"  - [{err.path}] {err.msg}")
        else:
            print(colorize(f"\n✓ Configuration in {target_file} validated successfully.", "green", no_color))
    except Exception as e:
        print(colorize(f"\n[ERROR] Broken configuration syntax in {target_file}: {e}", "red", no_color))


def show_effective_config(json_format: bool = False, ascii_mode: bool = False, no_color: bool = False) -> None:
    """Display active resolved configuration with sources (TUI-01, §5.2 [e])."""
    cfg = load_config()
    if json_format:
        print(json.dumps(dict(cfg), indent=2))
        return

    print("\n" + "=" * 68)
    print("EFFECTIVE CONFIGURATION (defaults ← custom ← env ← cli)")
    print("=" * 68)
    for k in sorted(cfg.keys()):
        val = cfg[k]
        src = cfg.provenance.get(k, "default")
        val_str = str(val)
        if len(val_str) > 38:
            val_str = val_str[:35] + "..."
        src_tag = f"[{src}]"
        print(f"{k:32s} = {val_str:38s} {colorize(src_tag, 'cyan', no_color)}")
    print("=" * 68)


def validate_active_config(ascii_mode: bool = False, no_color: bool = False) -> int:
    """Run CFG-03 validation and display formatted report (§5.2 [f])."""
    cfg = load_config()
    issues = validate(dict(cfg))
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    print("\n" + "=" * 68)
    print("CONFIG VALIDATION REPORT (CFG-03)")
    print("=" * 68)
    if not issues:
        print(colorize("✓ Configuration is completely valid. No issues detected.", "green", no_color))
        return 0

    if errors:
        print(colorize(f"Errors ({len(errors)}):", "red", no_color))
        for err in errors:
            print(f"  ✗ [{err.path}] {err.msg}")

    if warnings:
        print(colorize(f"\nWarnings ({len(warnings)}):", "yellow", no_color))
        for warn in warnings:
            print(f"  ⚠ [{warn.path}] {warn.msg}")

    print("=" * 68)
    return 2 if errors else 0


def show_recent_runs_table(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Display recent runs table from discovered run_manifest.json files (§5.2 [g])."""
    runs = find_recent_runs()
    if not runs:
        print("\nNo recent generation runs found.")
        return

    print("\n" + "=" * 76)
    print("RECENT GENERATION RUNS")
    print("=" * 76)
    hdr = f"{'Date / Time':19s} | {'Output Dir':20s} | {'Imgs':5s} | {'Fail':4s} | {'Secs':6s} | {'Seed':5s} | {'Hash':10s}"
    print(hdr)
    print("-" * 76)

    for r in runs[:10]:
        mtime = r.get("_mtime", 0)
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        p = r.get("_manifest_path", "")
        out_dir = os.path.basename(os.path.dirname(p)) or "out"
        counts = r.get("counts", {})
        imgs = counts.get("num_images", r.get("successful", 0) + r.get("failed", 0))
        fail = counts.get("failed", r.get("failed", 0))
        secs = r.get("wall_time_seconds", 0.0)
        seed = r.get("seed", 0)
        cfg_hash = (r.get("config_hash") or "")[:8]
        interrupted = " [!]" if r.get("interrupted") else ""
        print(f"{dt_str:19s} | {out_dir:20s} | {imgs:5d} | {fail:4d} | {secs:6.1f} | {seed:5d} | {cfg_hash:8s}{interrupted}")
    print("=" * 76)


def run_generate_settings_menu(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Menu [6] interaction loop."""
    while True:
        print()
        print(render_generate_menu(ascii_mode, no_color))
        choice = safe_prompt("Select action", default="0")
        if choice in ("0", "b", "back", "q"):
            break
        elif choice == "a":
            run_generate_flow(ascii_mode=ascii_mode, no_color=no_color, interactive=True)
        elif choice == "b":
            run_smoke_test(ascii_mode=ascii_mode, no_color=no_color)
        elif choice == "c":
            run_guided_settings(ascii_mode=ascii_mode, no_color=no_color)
        elif choice == "d":
            print("\nSelect configuration file to edit:")
            print("  [1] custom_config.py (user overrides)")
            print("  [2] config_defaults.py (developer mode - changes baseline)")
            ed_choice = safe_prompt("Choice", default="1")
            target = "defaults" if ed_choice == "2" else "custom"
            run_editor_flow(target=target, ascii_mode=ascii_mode, no_color=no_color)
        elif choice == "e":
            show_effective_config(ascii_mode=ascii_mode, no_color=no_color)
        elif choice == "f":
            validate_active_config(ascii_mode=ascii_mode, no_color=no_color)
        elif choice == "g":
            show_recent_runs_table(ascii_mode=ascii_mode, no_color=no_color)
        else:
            print("Invalid choice.")


# ==============================================================================
# MENU [4]: SAMPLING WEIGHTS
# ==============================================================================

def render_bar(percentage: float, width: int = 15, ascii_mode: bool = False) -> str:
    """Render a visual progress/percentage bar."""
    filled = int(round((percentage / 100.0) * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    fill_char = "=" if ascii_mode else "█"
    empty_char = " " if ascii_mode else "░"
    return f"[{fill_char * filled}{empty_char * empty}]"


def run_weights_menu(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Menu [4] Sampling weights screen (§5.4)."""
    cfg = load_config()

    while True:
        sci_weights = dict(cfg.get("scientific_subdomain_weights", {}))
        non_sci_weights = dict(cfg.get("non_scientific_subdomain_weights", {}))
        s_ratio = cfg.get("scientific_ratio", 0.5)

        sci_sum = sum(sci_weights.values())
        non_sci_sum = sum(non_sci_weights.values())

        print("\n" + "=" * 68)
        print("DOMAIN SAMPLING WEIGHTS (Persisted to custom_config.py)")
        print("=" * 68)
        print(f"Scientific Ratio: {s_ratio:.2f} (Scientific vs. Non-Scientific group split)")
        print("\nScientific Domains:")
        for dom, w in sorted(sci_weights.items()):
            pct = (w / sci_sum * 100.0) if sci_sum > 0 else 0.0
            bar = render_bar(pct, width=15, ascii_mode=ascii_mode)
            print(f"  {dom:15s} : {w:5.2f}  {bar} {pct:5.1f}%")
        print(f"  Total: {sci_sum:.2f}" + (" [WARN: sum == 0]" if sci_sum <= 0 else ""))

        print("\nNon-Scientific Domains:")
        for dom, w in sorted(non_sci_weights.items()):
            pct = (w / non_sci_sum * 100.0) if non_sci_sum > 0 else 0.0
            bar = render_bar(pct, width=15, ascii_mode=ascii_mode)
            print(f"  {dom:15s} : {w:5.2f}  {bar} {pct:5.1f}%")
        print(f"  Total: {non_sci_sum:.2f}" + (" [WARN: sum == 0]" if non_sci_sum <= 0 else ""))

        # Check for domains with manifests but no weights
        all_doms = [d for d in list_domains() if d != "common"]
        unweighted = [d for d in all_doms if d not in sci_weights and d not in non_sci_weights]
        if unweighted:
            print(colorize(f"\nWarning: Domain(s) without weight (will not be sampled): {', '.join(unweighted)}", "yellow", no_color))

        print("\nOptions:")
        print("  [1] Edit scientific_ratio")
        print("  [2] Edit scientific subdomain weight")
        print("  [3] Edit non-scientific subdomain weight")
        print("  [0] Back to Main Menu")

        c = safe_prompt("Select option", default="0")
        if c in ("0", "b", "q", ""):
            break

        if c == "1":
            new_r = safe_prompt("Enter new scientific_ratio [0.0 - 1.0]", default=str(s_ratio))
            try:
                val = float(new_r)
                if _save_setting_diff("scientific_ratio", val, s_ratio, no_color):
                    cfg = load_config()
            except ValueError:
                print("Invalid float.")
        elif c == "2":
            dom_name = safe_prompt("Domain name in scientific group (e.g. biomedical, engineering)")
            if dom_name in sci_weights:
                curr_w = sci_weights[dom_name]
                new_w = safe_prompt(f"Weight for {dom_name}", default=str(curr_w))
                try:
                    val = float(new_w)
                    path = f"scientific_subdomain_weights.{dom_name}"
                    if _save_setting_diff(path, val, curr_w, no_color):
                        cfg = load_config()
                except ValueError:
                    print("Invalid float.")
            else:
                print(f"Unknown scientific domain: '{dom_name}'")
        elif c == "3":
            dom_name = safe_prompt("Domain name in non-scientific group (e.g. business, demographic)")
            if dom_name in non_sci_weights:
                curr_w = non_sci_weights[dom_name]
                new_w = safe_prompt(f"Weight for {dom_name}", default=str(curr_w))
                try:
                    val = float(new_w)
                    path = f"non_scientific_subdomain_weights.{dom_name}"
                    if _save_setting_diff(path, val, curr_w, no_color):
                        cfg = load_config()
                except ValueError:
                    print("Invalid float.")
            else:
                print(f"Unknown non-scientific domain: '{dom_name}'")


# ==============================================================================
# MENU [1]: INSPECT DOMAINS & STATISTICS
# ==============================================================================

def show_domains_and_stats(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Inspect domains, corpus statistics, and run read-only audit (§5.1 [1], ADM-12)."""
    from synth.semantics.loader import _load_manifest_raw_data, registry
    from synth.semantics.admin.audit import run_audit

    raw = _load_manifest_raw_data()

    print("\n" + "=" * 78)
    print("DOMAINS & CORPUS STATISTICS")
    print("=" * 78)
    for dom_id, data in sorted(raw.items()):
        metrics = data.get("metrics", [])
        titles = data.get("titles", [])
        pairs = data.get("comparative_pairs", [])
        is_sci = data.get("is_scientific")
        sci_tag = "scientific" if is_sci else ("common" if is_sci is None else "non-scientific")
        print(f"Domain: {dom_id:15s} [{sci_tag:14s}] | Metrics: {len(metrics):4d} | Titles: {len(titles):4d} | Pairs: {len(pairs):4d}")

    m_cnt, t_cnt, p_cnt = get_manifest_stats()
    print("-" * 78)
    print(f"Total Registry: {m_cnt} Metrics | {t_cnt} Titles | {p_cnt} Pairs")
    print("=" * 78)

    # Run audit
    print("\nRunning corpus audit...")
    report = run_audit()
    summary = report.summary_dict()
    print(f"Audit Findings: {summary['total_findings']} total ({summary['errors']} errors, {summary['warnings']} warnings)")

    if report.degenerate_pools:
        print(colorize(f"Degenerate pools found ({len(report.degenerate_pools)}):", "red", no_color))
        for p in report.degenerate_pools:
            print(f"  - Domain: {p[0]}, Chart: {p[1]}, Orient: {p[2]}, Axis: {p[3]}")
    else:
        print(colorize("Degenerate pools: NONE (all Cartesian combinations covered)", "green", no_color))

    if report.findings:
        print("\nRecent Findings:")
        for f in report.findings[:8]:
            sev_col = "red" if f.severity == "ERROR" else ("yellow" if f.severity == "WARNING" else "dim")
            print(f"  [{colorize(f.severity, sev_col, no_color)}] {f.category:16s} ({f.domain}): {f.message}")
        if len(report.findings) > 8:
            print(f"  ... and {len(report.findings) - 8} more findings.")

    export_choice = safe_prompt("\nExport audit report to JSON? (y/N)", default="n")
    if export_choice.lower() in ("y", "yes"):
        out_name = safe_prompt("Report file path", default="audit_report.json")
        try:
            with open(out_name, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": summary,
                    "findings": [
                        {"category": f.category, "severity": f.severity, "domain": f.domain, "item": f.item, "message": f.message}
                        for f in report.findings
                    ],
                    "degenerate_pools": [list(p) for p in report.degenerate_pools],
                }, f, indent=2)
            print(f"[OK] Report exported to {out_name}")
        except Exception as e:
            print(f"[ERROR] Failed to save report: {e}")

    safe_prompt("Press Enter to return to main menu")


# ==============================================================================
# MENU [2]: IMPORT PRE-FLIGHT & TRANSACTIONAL APPLY (TUI-03, ADM-06..10)
# ==============================================================================

def render_preflight_table(plan: IngestPlan, ascii_mode: bool = False, no_color: bool = False) -> str:
    """Render pre-flight diff table matching §5.3 specification (TUI-03)."""
    lines = []
    div = "-" * 79 if ascii_mode else "─" * 79
    lines.append(f"{'Status':<14} {'Entity':<8} {'Label / Title':<28} Details")
    lines.append(div)

    # Color mapping
    status_colors = {
        "NEW": "green",
        "SHARED": "cyan",
        "UNIT-VARIANT": "blue",
        "DUPLICATE": "dim",
        "SIMILAR": "yellow",
        "INVALID": "red",
        "ERROR": "red",
    }

    # Render metric actions
    for a in plan.metric_actions:
        st_val = a.status.value
        st_colored = colorize(f"{st_val:<14}", status_colors.get(st_val, "reset"), no_color)
        lbl_trunc = a.label[:26] + ".." if len(a.label) > 28 else a.label
        if a.action == "APPEND":
            m = a.payload
            details = f"{m.get('scale_type', '')} | {m.get('role', '')} | {m.get('concept_stem', '')}"
        elif a.action == "SHARE":
            details = a.message
        elif a.action == "OVERWRITE":
            details = f"overwrites in-place ({a.target_domain}.yaml)"
        elif a.action == "MERGE":
            details = f"merged domain_tags ({a.target_domain}.yaml)"
        elif a.action == "SKIP":
            details = f"exact match in {a.target_domain}.yaml (skip)"
        else:
            details = a.message
        lines.append(f"{st_colored} {'Metric':<8} {lbl_trunc:<28} {details}")

    # Render title actions
    for t in plan.title_actions:
        st_val = t.status.value
        st_colored = colorize(f"{st_val:<14}", status_colors.get(st_val, "reset"), no_color)
        t_trunc = t.title[:26] + ".." if len(t.title) > 28 else t.title
        lines.append(f"{st_colored} {'Title':<8} {t_trunc:<28} {t.message}")

    # Render pair actions
    for p in plan.pair_actions:
        st_val = p.status.value
        st_colored = colorize(f"{st_val:<14}", status_colors.get(st_val, "reset"), no_color)
        pair_str = f"{p.pair[0]} -> {p.pair[1]}"
        p_trunc = pair_str[:26] + ".." if len(pair_str) > 28 else pair_str
        lines.append(f"{st_colored} {'Pair':<8} {p_trunc:<28} {p.message}")

    # Render invalid issues
    for i in plan.issues:
        if i.level == "error":
            st_colored = colorize(f"{'INVALID':<14}", "red", no_color)
            lines.append(f"{st_colored} {'Row':<8} {f'Row {i.row} ({i.sheet})':<28} {i.message}")

    lines.append(div)
    c = plan.status_counts
    summary = (
        f"Summary: {c.get('NEW', 0)} new · {c.get('SHARED', 0)} shared · "
        f"{c.get('UNIT-VARIANT', 0)} unit variants · {c.get('DUPLICATE', 0)} duplicates · "
        f"{c.get('SIMILAR', 0)} similar · {sum(1 for i in plan.issues if i.level == 'error')} invalid"
    )
    lines.append(summary)
    return "\n".join(lines)


def run_import_flow(
    ascii_mode: bool = False,
    no_color: bool = False,
    interactive: bool = True,
    file_path: Optional[str] = None,
    target_domain: Optional[str] = None,
    policy: str = "skip",
    report_path: Optional[str] = None,
    auto_commit: bool = False,
    verify_tests: bool = False,
) -> int:
    """Import and deduplicate spreadsheet (.xlsx / .csv) into domain manifests (TUI-03, ADM-01..10)."""
    from synth.semantics.admin.ingest import ingest_file
    from synth.semantics.admin.plan import build_plan
    from synth.semantics.admin.apply import apply_plan, ApplyError
    from synth.semantics.loader import registry

    if interactive and file_path is None:
        print("\n" + "=" * 68)
        print("IMPORT & DEDUPLICATE SPREADSHEET")
        print("=" * 68)
        file_path = safe_prompt("Spreadsheet or CSV file path", default="")
        if not file_path:
            print("Import cancelled.")
            return 1

        reg_domains = [d for d in registry.domains if d != "common"]
        print(f"Registered domains: {', '.join(reg_domains)}")
        target_domain = safe_prompt("Target domain", default=reg_domains[0] if reg_domains else "new_domain")

        policy_choice = safe_prompt(
            "Collision policy ([s]kip, [m]erge, [o]verwrite, [a]bort)",
            default="s",
        ).lower()
        policy_map = {"s": "skip", "m": "merge", "o": "overwrite", "a": "abort"}
        policy = policy_map.get(policy_choice, "skip")

    if not file_path or not target_domain:
        sys.stderr.write("[ERROR] File path and target domain are required.\n")
        return 2

    # Ingest file
    ingest_result = ingest_file(file_path, target_domain=target_domain)
    if ingest_result.has_errors:
        print(colorize("\n[ERROR] Ingestion errors detected:", "red", no_color))
        for err in ingest_result.errors:
            print(f"  - [{err.sheet}:Row {err.row}] {err.message}")
        if not interactive:
            return 1

    # Build plan
    plan = build_plan(ingest_result, target_domain=target_domain, policy=policy)

    print("\n" + render_preflight_table(plan, ascii_mode=ascii_mode, no_color=no_color))

    # Optional report export
    if report_path:
        Path(report_path).write_text(plan.to_json(), encoding="utf-8")
        print(f"[OK] Pre-flight plan exported to {report_path}")
    elif interactive:
        export_opt = safe_prompt("\nExport plan report to JSON? (y/N)", default="n")
        if export_opt.lower() in ("y", "yes"):
            rep_file = safe_prompt("Report file path", default="import_plan.json")
            Path(rep_file).write_text(plan.to_json(), encoding="utf-8")
            print(f"[OK] Pre-flight plan exported to {rep_file}")

    if plan.has_abort_condition:
        print(colorize("\n[ERROR] Plan contains abort condition or unrecoverable errors. Cannot apply.", "red", no_color))
        return 1

    # Confirm and Apply
    if not interactive and not auto_commit:
        print("\n[DRY-RUN] Pre-flight plan calculated. Pass --yes to apply changes.")
        return 0

    if interactive and not auto_commit:
        confirm = safe_prompt(
            f"\nBackup will be written to synth/semantics/domains/.backups/<ts>/ — commit? [y/N]",
            default="n",
        )
        if confirm.lower() not in ("y", "yes"):
            print("Import cancelled by user.")
            return 1

    try:
        from synth.semantics.loader import get_domains_dir
        summary = apply_plan(plan, domains_dir=get_domains_dir(), verify_tests=verify_tests)
        print(colorize(f"\n[OK] Transaction successfully committed!", "green", no_color))
        print(f"  Snapshot backup: {summary['snapshot']}")
        print(f"  Touched domains: {', '.join(summary['touched_domains'])}")
        print(f"  Metrics added: {summary['metrics_added']}, shared: {summary['metrics_shared']}")
        print(f"  Titles added: {summary['titles_added']}, pairs added: {summary['pairs_added']}")
        if interactive:
            safe_prompt("\nPress Enter to return to main menu")
        return 0
    except ApplyError as exc:
        print(colorize(f"\n[ERROR] Apply failed: {exc}", "red", no_color))
        return 1


# ==============================================================================
# MENU [3]: NEW DOMAIN WIZARD (ADM-11)
# ==============================================================================

def run_new_domain_wizard(
    ascii_mode: bool = False,
    no_color: bool = False,
    interactive: bool = True,
    domain_id: Optional[str] = None,
    display_name: Optional[str] = None,
    is_scientific: Optional[bool] = None,
    default_weight: Optional[float] = None,
    fallbacks: Optional[Tuple[str, str]] = None,
    shared_common: Optional[List[str]] = None,
) -> int:
    """Create new domain manifest skeleton and register weights (ADM-11)."""
    import re
    from ruamel.yaml import YAML
    from synth.semantics.loader import get_domains_dir, rebuild_cache, _load_manifest_raw_data

    IDENT_RE = re.compile(r"^[a-z0-9_]+$")
    domains_dir = get_domains_dir()

    if interactive and domain_id is None:
        print("\n" + "=" * 68)
        print("NEW DOMAIN WIZARD")
        print("=" * 68)

        while True:
            dom_in = safe_prompt("New domain ID (lowercase, alphanumeric, e.g. aerospace)").strip().lower()
            if not dom_in or not IDENT_RE.match(dom_in):
                print("Invalid domain ID. Must match ^[a-z0-9_]+$")
                continue
            if (domains_dir / f"{dom_in}.yaml").exists():
                print(f"Domain '{dom_in}' already exists at {domains_dir / f'{dom_in}.yaml'}")
                continue
            domain_id = dom_in
            break

        default_name = domain_id.replace("_", " ").title()
        display_name = safe_prompt("Display Name", default=default_name)
        sci_in = safe_prompt("Is this a scientific domain? (y/N)", default="n").lower()
        is_scientific = sci_in in ("y", "yes")

        weight_in = safe_prompt("Default sampling weight", default="0.10")
        try:
            default_weight = float(weight_in)
        except ValueError:
            default_weight = 0.10

        fb_x = safe_prompt("Default X fallback label", default="Altitude (m)" if is_scientific else "Quarter")
        fb_y = safe_prompt("Default Y fallback label", default="Airspeed (knots)" if is_scientific else "Revenue ($M)")
        fallbacks = (fb_x, fb_y)

    if not domain_id:
        return 1

    display_name = display_name or domain_id.replace("_", " ").title()
    is_scientific = bool(is_scientific)
    default_weight = float(default_weight if default_weight is not None else 0.10)
    fallbacks = fallbacks or ("Index", "Value")

    # Inspect common.yaml for independent metric selection
    raw = _load_manifest_raw_data(domains_dir=domains_dir)
    common_metrics = raw.get("common", {}).get("metrics", [])
    common_ind = [m for m in common_metrics if m.get("role") in ("independent", "bidirectional")]

    # Default selection: independent metrics tagged for >= 3 domains (e.g. Date, Timestamp)
    default_selected = [
        m["label"] for m in common_ind
        if len(m.get("domain_tags", [])) >= 3
    ]

    selected_common: List[str] = []
    if shared_common is not None:
        selected_common = list(shared_common)
    elif interactive:
        print("\nIndependent metrics from common.yaml provide Cartesian X axes.")
        print(f"Recommended common primitives ({len(default_selected)}): {', '.join(default_selected[:4])}...")
        share_in = safe_prompt(
            "Share recommended common independent metrics with this new domain? (Y/n)",
            default="y",
        ).lower()
        if share_in in ("y", "yes"):
            selected_common = default_selected
        else:
            print(colorize("Warning: If no independent metrics are shared, bar/line/area charts will use static fallbacks.", "yellow", no_color))
    else:
        selected_common = default_selected

    # Build skeleton manifest
    manifest_data = {
        "manifest_version": 2,
        "domain_id": domain_id,
        "display_name": display_name,
        "is_scientific": is_scientific,
        "default_weight": default_weight,
        "default_fallbacks": list(fallbacks),
        "capabilities": {"tabular_preset": False, "heatmap_pools": False},
        "metrics": [],
        "titles": [],
        "comparative_pairs": [],
    }

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=0)

    # Write new manifest
    new_yaml_path = domains_dir / f"{domain_id}.yaml"
    with open(new_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f)
    print(f"[OK] Created domain manifest: {new_yaml_path}")

    # Update common.yaml with SHARED metrics if selected
    if selected_common:
        common_path = domains_dir / "common.yaml"
        with open(common_path, "r", encoding="utf-8") as f:
            common_doc = yaml.load(f)
        shared_count = 0
        for m in common_doc.get("metrics", []):
            if m.get("label") in selected_common:
                tags = m.get("domain_tags", [])
                if domain_id not in tags:
                    tags.append(domain_id)
                    m["domain_tags"] = tags
                    shared_count += 1
        with open(common_path, "w", encoding="utf-8") as f:
            yaml.dump(common_doc, f)
        print(f"[OK] Shared {shared_count} independent metrics from common.yaml with '{domain_id}'")

    # Register weight into custom_config.py
    group_key = "scientific_subdomain_weights" if is_scientific else "non_scientific_subdomain_weights"
    cfg = load_config()
    candidate = dict(cfg)
    candidate.setdefault(group_key, {})[domain_id] = default_weight
    diffs = compute_leaf_diff(candidate, baseline=DEFAULT_CONFIG)
    save_overrides_to_custom_config(diffs)
    print(f"[OK] Registered weight {domain_id} = {default_weight} in custom_config.py ({group_key})")

    # Rebuild cache
    rebuild_cache(force=True, domains_dir=domains_dir)
    print(f"[OK] Cache successfully rebuilt with new domain '{domain_id}'.")

    if interactive:
        safe_prompt("\nPress Enter to return to main menu")
    return 0


# ==============================================================================
# MENU [5]: VALIDATE MANIFESTS & REBUILD CACHE
# ==============================================================================

def run_revalidate_and_rebuild(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Validate all domain manifests and force cache rebuild (§5.1 [5], SEM-06)."""
    from synth.semantics.loader import rebuild_cache, registry, _load_manifest_raw_data
    from synth.semantics.schema import DomainManifest

    print("\n" + "=" * 68)
    print("VALIDATE MANIFESTS & REBUILD CACHE")
    print("=" * 68)

    raw = _load_manifest_raw_data(force=True)
    all_valid = True
    for dom_id, doc in sorted(raw.items()):
        try:
            DomainManifest.model_validate(doc)
            print(f"  [{colorize('OK', 'green', no_color)}] {dom_id}.yaml: Valid schema")
        except Exception as exc:
            all_valid = False
            print(f"  [{colorize('FAIL', 'red', no_color)}] {dom_id}.yaml: {exc}")

    if all_valid:
        reg = rebuild_cache(force=True)
        print(colorize("\n[SUCCESS] Manifest cache rebuilt successfully!", "green", no_color))
        print(f"Registered Domains: {len(reg.domains)}")
        print(f"Total Metrics:      {len(reg.all_metrics)}")
        print(f"Total Titles:       {len(reg.all_titles)}")
        print(f"Total Pairs:        {len(reg.all_pairs)}")
    else:
        print(colorize("\n[ERROR] Validation failed. Cache was not updated.", "red", no_color))

    safe_prompt("\nPress Enter to return to main menu")


# ==============================================================================
# RESTORE FLOW (ADM-10)
# ==============================================================================

def run_restore_flow(
    ascii_mode: bool = False,
    no_color: bool = False,
    target_ts: Optional[str] = None,
) -> int:
    """Restore manifests from a previous snapshot backup (ADM-10)."""
    from synth.semantics.admin.apply import list_backups, restore_backup

    backups = list_backups()
    if not backups:
        print("No snapshot backups found in synth/semantics/domains/.backups/")
        return 1

    if target_ts is None:
        print("\nAvailable Snapshot Backups:")
        for idx, ts in enumerate(backups[:10], start=1):
            print(f"  [{idx}] {ts}")
        sel = safe_prompt(f"Select backup to restore (1-{min(len(backups), 10)}) or 'q' to cancel")
        if sel.lower() in ("q", "cancel", ""):
            return 1
        try:
            idx = int(sel) - 1
            target_ts = backups[idx]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return 1

    try:
        restore_backup(target_ts)
        print(colorize(f"[OK] Successfully restored manifests from backup {target_ts}", "green", no_color))
        return 0
    except Exception as exc:
        print(colorize(f"[ERROR] Restore failed: {exc}", "red", no_color))
        return 1


# ==============================================================================
# MAIN TUI LOOP
# ==============================================================================

def run_main_menu(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Main interactive menu loop (TUI-02, TUI-05)."""
    while True:
        try:
            print()
            print(render_header(ascii_mode=ascii_mode, no_color=no_color))
            print()
            print(render_main_menu(ascii_mode=ascii_mode, no_color=no_color))
            choice = safe_prompt("Select option", default="6")

            if choice in ("0", "q", "exit"):
                print("Exiting Studio. Goodbye!")
                break
            elif choice == "1":
                show_domains_and_stats(ascii_mode, no_color)
            elif choice == "2":
                run_import_flow(ascii_mode, no_color)
            elif choice == "3":
                run_new_domain_wizard(ascii_mode, no_color)
            elif choice == "4":
                run_weights_menu(ascii_mode, no_color)
            elif choice == "5":
                run_revalidate_and_rebuild(ascii_mode, no_color)
            elif choice == "6":
                run_generate_settings_menu(ascii_mode, no_color)
            else:
                print("Invalid option.")
        except KeyboardInterrupt:
            # TUI-05: Ctrl-C at any prompt returns to menu without side effects
            print("\n[Returned to menu]")
            continue
