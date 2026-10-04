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
import re
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


ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(text: str) -> int:
    """Return visible string length, stripping ANSI escape sequences."""
    return len(ANSI_STRIP_RE.sub("", text))


def render_panel(
    title: str = "",
    lines: Optional[List[str]] = None,
    width: int = 70,
    border_color: str = "blue",
    ascii_mode: bool = False,
    no_color: bool = False,
) -> str:
    """Wrap lines in a uniform rounded card with a centered title badge."""
    top_l = "+" if ascii_mode else "╭"
    top_r = "+" if ascii_mode else "╮"
    bot_l = "+" if ascii_mode else "╰"
    bot_r = "+" if ascii_mode else "╯"
    horiz = "-" if ascii_mode else "─"
    vert = "|" if ascii_mode else "│"

    if title:
        t_clean = f" {title.strip()} "
        pad_total = max(0, width - 2 - visible_len(t_clean))
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        top_border = f"{top_l}{horiz * pad_l}{t_clean}{horiz * pad_r}{top_r}"
    else:
        top_border = f"{top_l}{horiz * (width - 2)}{top_r}"

    bot_border = f"{bot_l}{horiz * (width - 2)}{bot_r}"
    inner_w = width - 4

    out = [colorize(top_border, border_color, no_color)]
    if lines:
        for line in lines:
            vlen = visible_len(line)
            if vlen > inner_w:
                display_line = line
            else:
                display_line = f"{line}{' ' * (inner_w - vlen)}"
            out.append(f"{vert}  {display_line}  {vert}")
    out.append(colorize(bot_border, border_color, no_color))
    return "\n".join(out)


def render_badge(text: str, badge_type: str = "dim", no_color: bool = False) -> str:
    """Format semantic status badge [TEXT] with ANSI color."""
    b_map = {
        "ok": "green",
        "valid": "green",
        "green": "green",
        "new": "green",
        "ready": "green",
        "warn": "yellow",
        "warning": "yellow",
        "similar": "yellow",
        "fail": "red",
        "error": "red",
        "invalid": "red",
        "abort": "red",
        "shared": "cyan",
        "info": "cyan",
        "unit-variant": "blue",
        "blue": "blue",
        "dim": "dim",
        "dup": "dim",
        "skip": "dim",
    }
    col = b_map.get(badge_type.lower(), "dim")
    return colorize(f"[{text}]", col, no_color)


def render_ratio_bar(val: float, width: int = 12, ascii_mode: bool = False) -> str:
    """Render a compact percentage meter for 0.0–1.0 floats."""
    pct = max(0.0, min(100.0, float(val) * 100.0))
    bar = render_bar(pct, width=width, ascii_mode=ascii_mode)
    return f"{bar} {pct:4.0f}%"


def render_segmented_bar(
    segments: List[Tuple[str, int, str]],
    width: int = 24,
    ascii_mode: bool = False,
    no_color: bool = False,
) -> str:
    """
    Render proportional multi-segment bar for status distributions.
    segments: List of (label, count, color)
    """
    total = sum(c for _, c, _ in segments)
    if total <= 0:
        empty_char = " " if ascii_mode else "░"
        return f"[{empty_char * width}]"

    ascii_chars = ["#", "=", "+", "-", ".", ":"]
    out_chars = []
    allocated = 0

    for idx, (lbl, count, color) in enumerate(segments):
        if count <= 0:
            continue
        seg_len = int(round((count / total) * width))
        if idx == len(segments) - 1:
            seg_len = max(0, width - allocated)
        else:
            seg_len = min(seg_len, width - allocated)
        allocated += seg_len

        char = ascii_chars[idx % len(ascii_chars)] if ascii_mode else "█"
        block = char * seg_len
        out_chars.append(colorize(block, color, no_color))

    if allocated < width:
        rem = width - allocated
        char = " " if ascii_mode else "░"
        out_chars.append(char * rem)

    return f"[{''.join(out_chars)}]"


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
        "  [a] Generate charts…            number, format, output dir, seed, engine, workers",
        "  [b] Smoke test (3 charts → temp dir, then open summary)",
        "  [c] Edit settings (guided)      groups below; saved to custom_config.py",
        "  [d] Open a config file in $EDITOR   custom_config.py / config_defaults.py (dev mode)",
        "  [e] Show effective config       every key with its source: default / custom / cli",
        "  [f] Validate config             CFG-03 report",
        "  [g] Recent runs                 from run_manifest.json files",
        "  [0] Back",
    ]
    return render_panel("Generate Charts & Settings Hub", options, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color)


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
        preview_lines = [
            f"Target Charts  : {num_images} images  ·  Format: {fmt_val}",
            f"Output Dir     : {out_dir}",
            f"Engine & Seed  : {eng_val}  ·  Seed: {seed_val}",
            f"Parallelism    : {workers_val} workers",
            f"Time Estimate  : ~{est_min:.1f} minutes",
        ]
        print("\n" + render_panel("Generation Launch Preview", preview_lines, width=70, border_color="green", ascii_mode=ascii_mode, no_color=no_color))
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
        if res.manifest_data:
            mf = res.manifest_data
            counts = mf.get("counts", {})
            succ = counts.get("successful", 0)
            failed = counts.get("failed", 0)
            ver = mf.get("schema_version", "v4.0")
            w_time = mf.get("wall_time_seconds", 0.0)
            summary_lines = [
                f"Status     : {render_badge('COMPLETED', 'ok', no_color)}",
                f"Generated  : {succ}/{num_images} successful ({failed} failed)",
                f"Wall Time  : {w_time:.2f} seconds ({w_time/max(succ, 1):.2f}s/img)",
                f"Schema     : {ver}",
                f"Manifest   : {res.manifest_path}",
            ]
            print("\n" + render_panel("Generation Run Completed", summary_lines, width=70, border_color="green", ascii_mode=ascii_mode, no_color=no_color))
        else:
            print("\n" + colorize("✓ Generation completed successfully!", "green", no_color))
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
        menu_lines = [f"  [{key}] {name}" for key, name in groups]
        menu_lines.append("  [0] Back to Generate Hub")
        print("\n" + render_panel("Guided Configuration Editor", menu_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

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
        lines = []
        for i, k in enumerate(keys, 1):
            val = cfg.get(k)
            src = cfg.provenance.get(k, "default")
            lines.append(f"  [{i}] {k:16s} = {repr(val):18s} [{src}]")
        lines.append("  [0] Back")
        print("\n" + render_panel("Run Basics Configuration", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
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
    _edit_chart_type_weights(cfg, ascii_mode, no_color)


def _edit_domains_settings(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    while True:
        s_ratio = float(cfg.get("scientific_ratio", 0.5))
        s_src = cfg.provenance.get("scientific_ratio", "default")
        s_dom = cfg.get("semantic_domain")
        dom_src = cfg.provenance.get("semantic_domain", "default")
        s_meter = render_ratio_bar(s_ratio, width=10, ascii_mode=ascii_mode)
        lines = [
            f"  [1] scientific_ratio = {s_ratio:.2f}  {s_meter} [{s_src}]",
            f"  [2] semantic_domain  = {repr(s_dom):20s} [{dom_src}]",
            "  [0] Back",
        ]
        print("\n" + render_panel("Domains & Split Ratios", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
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
        lines = []
        for i, (path, _) in enumerate(options, 1):
            parts = path.split(".")
            val = cfg.get(parts[0], {}).get(parts[1], 0.0)
            meter = render_ratio_bar(val, width=10, ascii_mode=ascii_mode) if isinstance(val, (int, float)) else str(val)
            lines.append(f"  [{i}] {path:34s} = {val:.2f} {meter}")
        lines.append("  [0] Back")
        print("\n" + render_panel("Per-Chart Feature Ratios", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
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
        eff_dict = cfg.get("realism_effects", {})
        lines = []
        for i, eff in enumerate(effects, 1):
            curr_p = float(eff_dict.get(eff, {}).get("p", 0.0))
            meter = render_ratio_bar(curr_p, width=10, ascii_mode=ascii_mode)
            lines.append(f"  [{i:2d}] {eff:22s} p = {curr_p:.2f} {meter}")
        lines.append("  [0] Back")
        print("\n" + render_panel("Realism Effects Probabilities", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
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
        lines = []
        for i, (k, typ) in enumerate(keys, 1):
            val = cfg.get(k)
            src = cfg.provenance.get(k, "default")
            if isinstance(val, bool):
                val_str = render_badge("TRUE", "ok", no_color) if val else render_badge("FALSE", "dim", no_color)
            else:
                val_str = f"'{val}'"
            lines.append(f"  [{i}] {k:32s} = {val_str:14s} [{src}]")
        lines.append("  [0] Back")
        print("\n" + render_panel("Output & Schema Configuration", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
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
    mc = cfg.get("multi_chart_config", {})
    lines = [
        f"Min / Max Sub-Charts : {mc.get('min_charts', 2)} .. {mc.get('max_charts', 4)}",
        f"Title Probability    : {mc.get('title_probability', 0.85):.2f}",
        f"Grid Padding         : {mc.get('padding', 0.05):.2f}",
        colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
        "Layout Weights:",
    ]
    lw = mc.get("layout_weights", {})
    tot_lw = sum(lw.values()) if lw else 1
    for lk, lv in sorted(lw.items()):
        pct = (lv / tot_lw * 100.0) if tot_lw > 0 else 0.0
        bar = render_bar(pct, width=10, ascii_mode=ascii_mode)
        lines.append(f"  {lk:14s} : {lv:5.2f} {bar} {pct:4.0f}%")
    print("\n" + render_panel("Multi-Chart Layout Configuration", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
    safe_prompt("Press Enter to continue")


def _edit_advanced_settings(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> None:
    keys = [
        ("use_parallel", bool),
        ("strict", bool),
        ("debug_mode", bool),
    ]
    while True:
        lines = []
        for i, (k, _) in enumerate(keys, 1):
            val = cfg.get(k, False)
            src = cfg.provenance.get(k, "default")
            badge = render_badge("ENABLED", "ok", no_color) if val else render_badge("DISABLED", "dim", no_color)
            warn = " (redirects output to test/)" if k == "debug_mode" and val else ""
            lines.append(f"  [{i}] {k:16s} = {badge:14s} [{src}]{warn}")
        lines.append("  [0] Back")
        print("\n" + render_panel("Advanced Engine Settings", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
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

    lines = []
    for k in sorted(cfg.keys()):
        val = cfg[k]
        src = cfg.provenance.get(k, "default")
        val_str = str(val)
        if len(val_str) > 34:
            val_str = val_str[:31] + "..."
        src_tag = f"[{src}]"
        lines.append(f"{k:26s} = {val_str:24s} {colorize(src_tag, 'cyan', no_color)}")

    print("\n" + render_panel("Effective Configuration (defaults ← custom)", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))


def validate_active_config(ascii_mode: bool = False, no_color: bool = False) -> int:
    """Run CFG-03 validation and display formatted report (§5.2 [f])."""
    cfg = load_config()
    issues = validate(dict(cfg))
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    lines = []
    if not issues:
        lines.append(colorize("✓ Configuration is completely valid. No issues detected.", "green", no_color))
        print("\n" + render_panel("CONFIG VALIDATION REPORT (CFG-03)", lines, width=70, border_color="green", ascii_mode=ascii_mode, no_color=no_color))
        return 0

    if errors:
        lines.append(colorize(f"Errors ({len(errors)}):", "red", no_color))
        for err in errors:
            lines.append(f"  ✗ [{err.path}] {err.msg}")

    if warnings:
        if errors:
            lines.append("")
        lines.append(colorize(f"Warnings ({len(warnings)}):", "yellow", no_color))
        for warn in warnings:
            lines.append(f"  ⚠ [{warn.path}] {warn.msg}")

    border_col = "red" if errors else "yellow"
    print("\n" + render_panel("CONFIG VALIDATION REPORT (CFG-03)", lines, width=70, border_color=border_col, ascii_mode=ascii_mode, no_color=no_color))
    return 2 if errors else 0


def show_recent_runs_table(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Display recent runs table from discovered run_manifest.json files (§5.2 [g])."""
    runs = find_recent_runs()
    if not runs:
        print("\nNo recent generation runs found.")
        return

    hdr = f"{'Date / Time':19s} │ {'Output Dir':14s} │ {'Imgs':5s} │ {'Fail':4s} │ {'Secs':6s} │ {'Seed':5s}"
    lines = [
        hdr,
        colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
    ]

    for r in runs[:10]:
        mtime = r.get("_mtime", 0)
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        p = r.get("_manifest_path", "")
        out_dir = os.path.basename(os.path.dirname(p)) or "out"
        if len(out_dir) > 14:
            out_dir = out_dir[:12] + ".."
        counts = r.get("counts", {})
        imgs = counts.get("num_images", r.get("successful", 0) + r.get("failed", 0))
        fail = counts.get("failed", r.get("failed", 0))
        secs = r.get("wall_time_seconds", 0.0)
        seed = r.get("seed", 0)
        interrupted = " [!]" if r.get("interrupted") else ""
        lines.append(f"{dt_str:19s} │ {out_dir:14s} │ {imgs:5d} │ {fail:4d} │ {secs:6.1f} │ {seed:5d}{interrupted}")

    print("\n" + render_panel("Recent Generation Runs", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))


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


def _edit_domain_weights(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> ResolvedConfig:
    """Submenu for domain sampling weights (§5.4)."""
    while True:
        sci_weights = dict(cfg.get("scientific_subdomain_weights", {}))
        non_sci_weights = dict(cfg.get("non_scientific_subdomain_weights", {}))
        s_ratio = cfg.get("scientific_ratio", 0.5)

        sci_sum = sum(sci_weights.values())
        non_sci_sum = sum(non_sci_weights.values())

        lines = [
            f"Scientific Ratio: {s_ratio:.2f} (Scientific vs. Non-Scientific group split)",
            colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
            colorize("Scientific Domains:", "cyan", no_color),
        ]
        for dom, w in sorted(sci_weights.items()):
            pct = (w / sci_sum * 100.0) if sci_sum > 0 else 0.0
            bar = render_bar(pct, width=15, ascii_mode=ascii_mode)
            lines.append(f"  {dom:15s} : {w:5.2f}  {bar} {pct:5.1f}%")
        lines.append(f"  Total: {sci_sum:.2f}" + (" [WARN: sum == 0]" if sci_sum <= 0 else ""))

        lines.append("")
        lines.append(colorize("Non-Scientific Domains:", "cyan", no_color))
        for dom, w in sorted(non_sci_weights.items()):
            pct = (w / non_sci_sum * 100.0) if non_sci_sum > 0 else 0.0
            bar = render_bar(pct, width=15, ascii_mode=ascii_mode)
            lines.append(f"  {dom:15s} : {w:5.2f}  {bar} {pct:5.1f}%")
        lines.append(f"  Total: {non_sci_sum:.2f}" + (" [WARN: sum == 0]" if non_sci_sum <= 0 else ""))

        # Check for domains with manifests but no weights
        all_doms = [d for d in list_domains() if d != "common"]
        unweighted = [d for d in all_doms if d not in sci_weights and d not in non_sci_weights]
        if unweighted:
            lines.append("")
            lines.append(colorize(f"Warning: Domain(s) without weight: {', '.join(unweighted)}", "yellow", no_color))

        print("\n" + render_panel("Domain Sampling Weights", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
        print("Options:")
        print("  [1] Edit scientific_ratio")
        print("  [2] Edit scientific subdomain weight")
        print("  [3] Edit non-scientific subdomain weight")
        print("  [0] Back")

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

    return cfg


def _edit_chart_type_weights(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> ResolvedConfig:
    """Submenu for chart type weights and enabled mix with visual percentage bars."""
    chart_types_list = ["bar", "line", "scatter", "box", "histogram", "pie", "heatmap", "area"]

    while True:
        ct_config = cfg.get("chart_types", {})
        enabled_weights = {
            ct: ct_config.get(ct, {}).get("weight", 0.0)
            for ct in chart_types_list
            if ct_config.get(ct, {}).get("enabled", True)
        }
        total_w = sum(enabled_weights.values())

        lines = [
            f"{'Type':12s} {'En':4s} {'Weight':8s} {'Distribution':19s} {'Share':7s} Source",
            colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
        ]

        for ct in chart_types_list:
            info = ct_config.get(ct, {})
            en = info.get("enabled", True)
            w = float(info.get("weight", 0.0))
            en_str = colorize("[✓]", "green", no_color) if en else colorize("[ ]", "dim", no_color)
            pct = (w / total_w * 100.0) if (en and total_w > 0) else 0.0
            bar = render_bar(pct, width=12, ascii_mode=ascii_mode)
            src = cfg.provenance.get(f"chart_types.{ct}.weight", "default")
            lines.append(f"{ct:12s} {en_str:4s} {w:7.2f} {bar} {pct:5.1f}% [{src}]")

        lines.append(colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color))
        active_count = len(enabled_weights)
        total_warn = " [WARN: sum == 0]" if total_w <= 0 else ""
        lines.append(f"Active: {active_count}/8 enabled  ·  Total Weight: {total_w:.2f}{total_warn}")

        print("\n" + render_panel("Chart Type Sampling Weights", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
        print("Options:")
        print("  [1] Edit chart type weight")
        print("  [2] Toggle chart type enabled/disabled")
        print("  [0] Back")

        c = safe_prompt("Select option", default="0")
        if c in ("0", "b", "q", ""):
            break

        if c == "1":
            ct_name = safe_prompt(f"Chart type ({'/'.join(chart_types_list)})").lower().strip()
            if ct_name in chart_types_list:
                curr_w = ct_config.get(ct_name, {}).get("weight", 0.0)
                new_w = safe_prompt(f"Weight for {ct_name}", default=str(curr_w))
                try:
                    val = float(new_w)
                    path = f"chart_types.{ct_name}.weight"
                    if _save_setting_diff(path, val, curr_w, no_color):
                        cfg = load_config()
                except ValueError:
                    print("Invalid float.")
            else:
                print(f"Unknown chart type: '{ct_name}'")
        elif c == "2":
            ct_name = safe_prompt(f"Chart type to toggle ({'/'.join(chart_types_list)})").lower().strip()
            if ct_name in chart_types_list:
                curr_en = ct_config.get(ct_name, {}).get("enabled", True)
                path = f"chart_types.{ct_name}.enabled"
                if _save_setting_diff(path, not curr_en, curr_en, no_color):
                    cfg = load_config()
            else:
                print(f"Unknown chart type: '{ct_name}'")

    return cfg


def _edit_scenario_weights(cfg: ResolvedConfig, ascii_mode: bool, no_color: bool) -> ResolvedConfig:
    """Submenu for single-chart vs multi-chart layout scenario weights."""
    while True:
        scenarios = dict(cfg.get("scenario_weights", {}))
        tot_w = sum(scenarios.values())

        lines = [
            f"{'Scenario':16s} {'Weight':8s} {'Distribution':19s} {'Share':7s} Source",
            colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
        ]

        for sc, w in sorted(scenarios.items()):
            w_float = float(w)
            pct = (w_float / tot_w * 100.0) if tot_w > 0 else 0.0
            bar = render_bar(pct, width=12, ascii_mode=ascii_mode)
            src = cfg.provenance.get(f"scenario_weights.{sc}", "default")
            lines.append(f"{sc:16s} {w_float:7.2f} {bar} {pct:5.1f}% [{src}]")

        lines.append(colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color))
        total_warn = " [WARN: sum == 0]" if tot_w <= 0 else ""
        lines.append(f"Total Weight: {tot_w:.2f}{total_warn}")

        print("\n" + render_panel("Scenario Sampling Weights", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
        print("Options:")
        print("  [1] Edit scenario weight (e.g. single, multi)")
        print("  [0] Back")

        c = safe_prompt("Select option", default="0")
        if c in ("0", "b", "q", ""):
            break

        if c == "1":
            sc_name = safe_prompt("Scenario name (e.g. single, multi)").strip().lower()
            if sc_name in scenarios:
                curr_w = scenarios[sc_name]
                new_w = safe_prompt(f"Weight for '{sc_name}'", default=str(curr_w))
                try:
                    val = float(new_w) if isinstance(curr_w, float) else int(float(new_w))
                    path = f"scenario_weights.{sc_name}"
                    if _save_setting_diff(path, val, curr_w, no_color):
                        cfg = load_config()
                except ValueError:
                    print("Invalid number.")
            else:
                print(f"Unknown scenario: '{sc_name}'")

    return cfg


def run_weights_menu(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Menu [4] Sampling weights screen (§5.4)."""
    while True:
        cfg = load_config()
        lines = [
            "Adjust sampling distributions across domains, chart types, and scenarios.",
            "All modifications are validated and persisted to custom_config.py.",
            "",
            f"  {colorize('[1]', 'cyan', no_color)} Domain weights (scientific & non-scientific groups)",
            f"  {colorize('[2]', 'cyan', no_color)} Chart type weights & enabled mix (bar, line, etc.)",
            f"  {colorize('[3]', 'cyan', no_color)} Scenario weights (single vs. multi-chart layout)",
            f"  {colorize('[0]', 'dim', no_color)} Back to Main Menu",
        ]
        print("\n" + render_panel("Sampling Weights Hub", lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))
        choice = safe_prompt("Select weights group to inspect/edit", default="0")
        if choice in ("0", "b", "q", ""):
            break
        elif choice == "1":
            _edit_domain_weights(cfg, ascii_mode, no_color)
        elif choice == "2":
            _edit_chart_type_weights(cfg, ascii_mode, no_color)
        elif choice == "3":
            _edit_scenario_weights(cfg, ascii_mode, no_color)


# ==============================================================================
# MENU [1]: INSPECT DOMAINS & STATISTICS
# ==============================================================================

def show_domains_and_stats(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Inspect domains, corpus statistics, and run read-only audit (§5.1 [1], ADM-12)."""
    from synth.semantics.loader import _load_manifest_raw_data, registry, ALL_CHART_TYPES
    from synth.semantics.admin.audit import run_audit

    raw = _load_manifest_raw_data()

    stat_lines = [
        f"{'Domain':16s} {'Category':16s} {'Metrics':9s} {'Titles':8s} {'Pairs':8s}",
        colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
    ]
    for dom_id, data in sorted(raw.items()):
        metrics = data.get("metrics", [])
        titles = data.get("titles", [])
        pairs = data.get("comparative_pairs", [])
        is_sci = data.get("is_scientific")
        sci_tag = "scientific" if is_sci else ("common" if is_sci is None else "non-scientific")
        stat_lines.append(f"{dom_id:16s} [{sci_tag:14s}] {len(metrics):7d}  {len(titles):6d}  {len(pairs):6d}")

    m_cnt, t_cnt, p_cnt = get_manifest_stats()
    stat_lines.append(colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color))
    stat_lines.append(f"Total Registry: {m_cnt} Metrics │ {t_cnt} Titles │ {p_cnt} Pairs")

    print("\n" + render_panel("Domains & Corpus Statistics", stat_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

    # Run audit and compute coverage statistics
    print("\nRunning corpus audit & computing coverage...")
    report = run_audit()
    summary = report.summary_dict()

    chart_stats = report.chart_type_stats
    by_dom = chart_stats.get("by_domain", {})
    totals = chart_stats.get("chart_type_totals", {})

    ct_abbrev = {
        "bar": "bar",
        "line": "line",
        "scatter": "scat",
        "box": "box",
        "area": "area",
        "histogram": "hist",
        "pie": "pie",
        "heatmap": "heat",
    }
    header_cols = " ".join(f"{ct_abbrev.get(ct, ct):>4s}" for ct in ALL_CHART_TYPES)
    cov_lines = [
        f"{'Domain':14s}  {header_cols}",
        colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
    ]

    for dom_id, d_info in sorted(by_dom.items()):
        dom_titles = d_info.get("titles", {})
        row_vals = []
        for ct in ALL_CHART_TYPES:
            cnt = dom_titles.get(ct, 0)
            cnt_str = f"{cnt:4d}"
            if dom_id != "common":
                if cnt == 0:
                    cnt_str = colorize(cnt_str, "red", no_color)
                elif cnt < 15:
                    cnt_str = colorize(cnt_str, "yellow", no_color)
                else:
                    cnt_str = colorize(cnt_str, "green", no_color)
            row_vals.append(cnt_str)
        cov_lines.append(f"{dom_id:14s}  " + " ".join(row_vals))

    cov_lines.append(colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color))
    total_vals = [f"{totals.get(ct, 0):4d}" for ct in ALL_CHART_TYPES]
    cov_lines.append(f"{'Total Titles':14s}  " + " ".join(total_vals))

    print("\n" + render_panel("Titles Coverage by Chart Type", cov_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

    # Cartesian Axis Pools Depth Summary
    pool_lines = [
        f"{'Chart Type':12s} {'X Pool (min..max)':20s} {'Y Pool (min..max)':20s}",
        colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
    ]
    for ct in ("bar", "line", "scatter", "box", "area", "histogram"):
        x_pools = [
            d_info["axis_pools_depth"][ct]["vertical"]["x"]
            for d_id, d_info in by_dom.items()
            if d_id != "common" and ct in d_info.get("axis_pools_depth", {})
        ]
        y_pools = [
            d_info["axis_pools_depth"][ct]["vertical"]["y"]
            for d_id, d_info in by_dom.items()
            if d_id != "common" and ct in d_info.get("axis_pools_depth", {})
        ]
        min_x = min(x_pools) if x_pools else 0
        max_x = max(x_pools) if x_pools else 0
        min_y = min(y_pools) if y_pools else 0
        max_y = max(y_pools) if y_pools else 0
        x_str = f"{min_x:2d}..{max_x:2d} metrics"
        y_str = f"{min_y:2d}..{max_y:2d} metrics" if ct != "histogram" else "N/A (histogram)"
        pool_lines.append(f"{ct:12s} {x_str:20s} {y_str:20s}")

    print("\n" + render_panel("Cartesian Metric Pools Depth", pool_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

    audit_lines = [
        f"Findings: {summary['total_findings']} total ({summary['errors']} errors, {summary['warnings']} warnings)",
    ]
    if report.degenerate_pools:
        audit_lines.append(render_badge("FAIL", "fail", no_color) + f" Degenerate pools found ({len(report.degenerate_pools)}):")
        for p in report.degenerate_pools:
            audit_lines.append(f"  - Domain: {p[0]}, Chart: {p[1]}, Orient: {p[2]}, Axis: {p[3]}")
    else:
        audit_lines.append(render_badge("OK", "ok", no_color) + " Degenerate pools: NONE (all Cartesian combinations covered)")

    if report.findings:
        audit_lines.append("")
        audit_lines.append(colorize("Recent Findings:", "dim", no_color))
        for f in report.findings[:6]:
            sev_b = render_badge(f.severity[:4], f.severity, no_color)
            audit_lines.append(f"  {sev_b} {f.category:16s} ({f.domain}): {f.message[:26]}")
        if len(report.findings) > 6:
            audit_lines.append(f"  ... and {len(report.findings) - 6} more findings.")

    print("\n" + render_panel("Corpus Audit & Quality Findings", audit_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

    export_choice = safe_prompt("\nExport audit report to JSON? (y/N)", default="n")
    if export_choice.lower() in ("y", "yes"):
        out_name = safe_prompt("Report file path", default="audit_report.json")
        try:
            with open(out_name, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
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
    inv_count = sum(1 for i in plan.issues if i.level == "error")
    segments = [
        ("NEW", c.get("NEW", 0), "green"),
        ("SHARED", c.get("SHARED", 0), "cyan"),
        ("UNIT-VARIANT", c.get("UNIT-VARIANT", 0), "blue"),
        ("DUPLICATE", c.get("DUPLICATE", 0), "dim"),
        ("SIMILAR", c.get("SIMILAR", 0), "yellow"),
        ("INVALID", inv_count, "red"),
    ]
    seg_bar = render_segmented_bar(segments, width=18, ascii_mode=ascii_mode, no_color=no_color)
    summary = (
        f"Summary: {c.get('NEW', 0)} new · {c.get('SHARED', 0)} shared · "
        f"{c.get('UNIT-VARIANT', 0)} unit variants · {c.get('DUPLICATE', 0)} duplicates · "
        f"{c.get('SIMILAR', 0)} similar · {inv_count} invalid"
    )
    lines.append(f"{summary}  {seg_bar}")
    return "\n".join(lines)



def render_batch_preflight_table(
    folder_name: str,
    plans: List[Any],
    ascii_mode: bool = False,
    no_color: bool = False,
) -> str:
    """Render summary table for a batch folder import across multiple files."""
    lines = []
    div = "-" * 88 if ascii_mode else "─" * 88
    lines.append(f"{'Source File':<32} {'Domain':<13} {'New':<6} {'Shared':<7} {'Dup/Sim':<8} {'Invalid':<8} Status")
    lines.append(div)

    tot_new = 0
    tot_shared = 0
    tot_dup = 0
    tot_invalid = 0

    for plan in plans:
        fname = Path(plan.source_file).name if plan.source_file else "unknown"
        if len(fname) > 30:
            fname = fname[:28] + ".."
        dom = plan.target_domain
        if len(dom) > 12:
            dom = dom[:10] + ".."

        c = plan.status_counts
        n_new = c.get("NEW", 0)
        n_shared = c.get("SHARED", 0)
        n_dup = c.get("DUPLICATE", 0) + c.get("SIMILAR", 0)
        n_inv = sum(1 for i in plan.issues if i.level == "error")

        tot_new += n_new
        tot_shared += n_shared
        tot_dup += n_dup
        tot_invalid += n_inv

        status_text = "ABORT" if plan.has_abort_condition else "READY"
        st_color = "red" if plan.has_abort_condition else "green"
        st_colored = colorize(f"{status_text:<7}", st_color, no_color)

        lines.append(
            f"{fname:<32} {dom:<13} {n_new:<6} {n_shared:<7} {n_dup:<8} {n_inv:<8} {st_colored}"
        )

    lines.append(div)
    summary = (
        f"Batch Total: {len(plans)} files · {tot_new} new items · {tot_shared} shared · "
        f"{tot_dup} duplicates · {tot_invalid} invalid"
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
    """Import and deduplicate spreadsheet (.xlsx / .csv) or folder into domain manifests (TUI-03, ADM-01..10)."""
    from synth.semantics.admin.ingest import ingest_file
    from synth.semantics.admin.plan import build_plan, batch_plans_to_json
    from synth.semantics.admin.apply import apply_plan, apply_batch_plans, ApplyError
    from synth.semantics.admin.normalize import infer_domain_from_filename
    from synth.semantics.loader import registry, get_domains_dir, _load_manifest_raw_data

    reg_domains = list(registry.domains)

    if interactive and file_path is None:
        print("\n" + "=" * 68)
        print("IMPORT & DEDUPLICATE SPREADSHEET")
        print("=" * 68)
        file_path = safe_prompt("Spreadsheet, CSV file, or directory path", default="")
        if not file_path:
            print("Import cancelled.")
            return 1

    if not file_path:
        sys.stderr.write("[ERROR] File or directory path is required.\n")
        return 2

    target_path = Path(file_path).expanduser()
    if not target_path.exists():
        sys.stderr.write(f"[ERROR] Path does not exist: {file_path}\n")
        return 2

    # Check if target is a folder (BATCH MODE)
    if target_path.is_dir():
        all_files = sorted([
            f for f in target_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in (".csv", ".xlsx", ".xls")
            and not f.name.startswith("~")
            and not f.name.startswith(".")
        ])

        if not all_files:
            print(colorize(f"\n[ERROR] No supported spreadsheet files (.csv, .xlsx) found in: {target_path}", "red", no_color))
            return 1

        if interactive:
            policy_choice = safe_prompt(
                "Collision policy ([s]kip, [m]erge, [o]verwrite, [a]bort)",
                default="s",
            ).lower()
            policy_map = {"s": "skip", "m": "merge", "o": "overwrite", "a": "abort"}
            policy = policy_map.get(policy_choice, "skip")

        file_domain_pairs: List[Tuple[Path, str]] = []
        for f in all_files:
            inferred = infer_domain_from_filename(f.name, reg_domains)
            if inferred:
                dom = inferred
            elif target_domain:
                dom = target_domain
            elif interactive:
                print(f"\nCould not infer domain from filename: {f.name}")
                print(f"Registered domains: {', '.join(reg_domains)}")
                dom = safe_prompt(f"Target domain for '{f.name}'", default=reg_domains[0] if reg_domains else "new_domain")
            else:
                sys.stderr.write(f"[ERROR] Could not infer domain for '{f.name}'. Please specify -d/--domain or name files with domain prefix.\n")
                return 2
            file_domain_pairs.append((f, dom))

        print(f"\nFound {len(all_files)} file(s) to import:")
        for f, dom in file_domain_pairs:
            print(f"  • {f.name:<34} -> domain: {dom}")

        # Ingest and build chained plans
        domains_dir = get_domains_dir()
        current_manifests = _load_manifest_raw_data(domains_dir=domains_dir)
        plans: List[Any] = []
        has_ingest_errors = False

        for f, dom in file_domain_pairs:
            ingest_result = ingest_file(str(f), target_domain=dom)
            if ingest_result.has_errors:
                has_ingest_errors = True
                print(colorize(f"\n[ERROR] Ingestion errors in '{f.name}':", "red", no_color))
                for err in ingest_result.errors:
                    print(f"  - [{err.sheet}:Row {err.row}] {err.message}")
                continue

            plan = build_plan(
                ingest_result,
                target_domain=dom,
                policy=policy,
                domains_dir=domains_dir,
                raw_manifests=current_manifests,
            )
            current_manifests = plan.candidate_manifests
            plans.append(plan)

        if has_ingest_errors and not interactive:
            return 1
        if not plans:
            print(colorize("\n[ERROR] No valid plans generated from batch files.", "red", no_color))
            return 1

        print("\n" + render_batch_preflight_table(str(target_path), plans, ascii_mode=ascii_mode, no_color=no_color))

        # Optional interactive drill-down
        if interactive and len(plans) > 0:
            while True:
                inspect_choice = safe_prompt(
                    "\nInspect detailed diff for a file? (enter 1-%d, or press Enter to continue)" % len(plans),
                    default="",
                ).strip()
                if not inspect_choice:
                    break
                try:
                    idx = int(inspect_choice)
                    if 1 <= idx <= len(plans):
                        chosen_plan = plans[idx - 1]
                        print(f"\n--- Detailed Diff: {Path(chosen_plan.source_file).name} ({chosen_plan.target_domain}) ---")
                        print(render_preflight_table(chosen_plan, ascii_mode=ascii_mode, no_color=no_color))
                    else:
                        print(f"Please enter a number between 1 and {len(plans)}.")
                except ValueError:
                    print("Invalid input.")

        # Optional report export
        if report_path:
            Path(report_path).write_text(batch_plans_to_json(target_path, plans, policy=policy), encoding="utf-8")
            print(f"[OK] Batch pre-flight plan exported to {report_path}")
        elif interactive:
            export_opt = safe_prompt("\nExport plan report to JSON? (y/N)", default="n")
            if export_opt.lower() in ("y", "yes"):
                rep_file = safe_prompt("Report file path", default="import_plan.json")
                Path(rep_file).write_text(batch_plans_to_json(target_path, plans, policy=policy), encoding="utf-8")
                print(f"[OK] Batch pre-flight plan exported to {rep_file}")

        if any(p.has_abort_condition for p in plans):
            print(colorize("\n[ERROR] One or more plans contain abort condition or unrecoverable errors. Cannot apply batch.", "red", no_color))
            return 1

        # Confirm and Apply
        if not interactive and not auto_commit:
            print("\n[DRY-RUN] Batch pre-flight plan calculated. Pass --yes to apply changes.")
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
            summary = apply_batch_plans(plans, domains_dir=domains_dir, verify_tests=verify_tests)
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

    # SINGLE FILE MODE
    if target_domain is None:
        inferred = infer_domain_from_filename(target_path.name, reg_domains)
        if inferred:
            target_domain = inferred
        elif interactive:
            print(f"Registered domains: {', '.join(reg_domains)}")
            target_domain = safe_prompt("Target domain", default=reg_domains[0] if reg_domains else "new_domain")
        else:
            sys.stderr.write("[ERROR] Target domain is required.\n")
            return 2

    if interactive:
        policy_choice = safe_prompt(
            "Collision policy ([s]kip, [m]erge, [o]verwrite, [a]bort)",
            default="s",
        ).lower()
        policy_map = {"s": "skip", "m": "merge", "o": "overwrite", "a": "abort"}
        policy = policy_map.get(policy_choice, "skip")

    # Ingest file
    ingest_result = ingest_file(str(target_path), target_domain=target_domain)
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
        step1_lines = [
            "Step 1 of 3: Identity & Classification",
            "Define unique domain identifier, display title, and category.",
        ]
        print("\n" + render_panel("New Domain Wizard", step1_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

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

        step2_lines = [
            "Step 2 of 3: Metric Fallbacks & Common Primitives",
            "Define fallback axis labels and select independent Cartesian primitives.",
        ]
        print("\n" + render_panel("New Domain Wizard", step2_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

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

    # Interactive Step 3: Review Preview Card
    if interactive:
        preview_lines = [
            f"Domain ID       : {domain_id}",
            f"Display Name    : {display_name}",
            f"Classification  : {'Scientific' if is_scientific else 'Non-Scientific'}",
            f"Default Weight  : {default_weight:.2f}",
            f"Fallback X / Y  : {fallbacks[0]} / {fallbacks[1]}",
            f"Common Prims    : {len(selected_common)} shared ({', '.join(selected_common[:3])}{'...' if len(selected_common) > 3 else ''})",
            f"Target Manifest : {domains_dir / f'{domain_id}.yaml'}",
        ]
        print("\n" + render_panel("Manifest Review & Confirmation", preview_lines, width=70, border_color="green", ascii_mode=ascii_mode, no_color=no_color))
        confirm = safe_prompt("Write manifest and rebuild cache? [Y/n]", default="y")
        if confirm.lower() not in ("y", "yes"):
            print("Domain creation cancelled.")
            return 1

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
    shared_count = 0
    if selected_common:
        common_path = domains_dir / "common.yaml"
        with open(common_path, "r", encoding="utf-8") as f:
            common_doc = yaml.load(f)
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
        summary_lines = [
            f"Manifest File  : {new_yaml_path}",
            f"Common Metrics : {shared_count} primitives linked",
            f"Sampling Weight: {default_weight:.2f} ({group_key})",
            f"Cache Status   : Registry successfully updated",
        ]
        print("\n" + render_panel("Domain Successfully Created", summary_lines, width=70, border_color="green", ascii_mode=ascii_mode, no_color=no_color))
        safe_prompt("\nPress Enter to return to main menu")
    return 0


# ==============================================================================
# MENU [5]: VALIDATE MANIFESTS & REBUILD CACHE
# ==============================================================================

def run_revalidate_and_rebuild(ascii_mode: bool = False, no_color: bool = False) -> None:
    """Validate all domain manifests and force cache rebuild (§5.1 [5], SEM-06)."""
    from synth.semantics.loader import rebuild_cache, registry, _load_manifest_raw_data
    from synth.semantics.schema import DomainManifest

    raw = _load_manifest_raw_data(force=True)
    all_valid = True
    val_lines = [
        f"{'Manifest':22s} {'Schema':8s} {'Status':12s} Details",
        colorize("-" * 64 if ascii_mode else "─" * 64, "dim", no_color),
    ]
    for dom_id, doc in sorted(raw.items()):
        fname = f"{dom_id}.yaml"
        try:
            DomainManifest.model_validate(doc)
            st_badge = render_badge("VALID", "valid", no_color)
            m_cnt = len(doc.get("metrics", []))
            t_cnt = len(doc.get("titles", []))
            val_lines.append(f"{fname:22s} {'v2.0':8s} {st_badge:12s} {m_cnt} metrics, {t_cnt} titles")
        except Exception as exc:
            all_valid = False
            st_badge = render_badge("ERROR", "error", no_color)
            val_lines.append(f"{fname:22s} {'v2.0':8s} {st_badge:12s} {str(exc)[:20]}")

    print("\n" + render_panel("Domain Manifest Validation", val_lines, width=70, border_color="blue", ascii_mode=ascii_mode, no_color=no_color))

    if all_valid:
        t0 = time.time()
        reg = rebuild_cache(force=True)
        elapsed = time.time() - t0
        res_lines = [
            f"Status: {render_badge('CACHE REBUILT', 'ok', no_color)} in {elapsed:.2f}s",
            f"Registered Domains: {len(reg.domains)}",
            f"Total Metrics:      {len(reg.all_metrics)}",
            f"Total Titles:       {len(reg.all_titles)}",
            f"Total Pairs:        {len(reg.all_pairs)}",
        ]
        print("\n" + render_panel("Manifest Cache Status", res_lines, width=70, border_color="green", ascii_mode=ascii_mode, no_color=no_color))
    else:
        err_lines = [
            render_badge("ERROR", "error", no_color) + " Validation failed. Cache was not updated.",
            "Please fix the schema errors listed above in domain manifests.",
        ]
        print("\n" + render_panel("Validation Failure", err_lines, width=70, border_color="red", ascii_mode=ascii_mode, no_color=no_color))

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
