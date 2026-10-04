#!/usr/bin/env python3
"""
scripts/manage_domains.py

Synthetic Chart Studio & Domain Management Console (Milestone M4).
Provides both headless subcommands (TUI-01) and interactive terminal UI:
- Headless: audit, import, new-domain, rebuild, restore, config show/set/validate, generate, template, weights
- Interactive: TUI with status header, guided settings, sampling weights, and runner integration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
from config_loader import load_config
from synth.semantics.admin.configspec import validate, Issue
from synth.semantics.admin.configsvc import (
    compute_leaf_diff,
    save_overrides_to_custom_config,
    patch_config_defaults_literal,
)
from synth.semantics.admin.tui import (
    render_header,
    run_main_menu,
    run_generate_flow,
    run_smoke_test,
    show_effective_config,
    validate_active_config,
    run_weights_menu,
    render_bar,
)


def _parse_value(val_str: str) -> Any:
    """Safely parse typed value from CLI string."""
    try:
        return json.loads(val_str)
    except Exception:
        pass
    if val_str.lower() in ("true", "yes"):
        return True
    if val_str.lower() in ("false", "no"):
        return False
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def handle_config_cmd(args: argparse.Namespace) -> int:
    """Handle `config show`, `config set`, and `config validate`."""
    ascii_mode = getattr(args, "ascii", False)
    no_color = getattr(args, "no_color", False) or ("NO_COLOR" in os.environ)

    if args.config_action == "show":
        show_effective_config(json_format=args.json, ascii_mode=ascii_mode, no_color=no_color)
        return 0
    elif args.config_action == "validate":
        return validate_active_config(ascii_mode=ascii_mode, no_color=no_color)
    elif args.config_action == "set":
        assignment = args.assignment
        if "=" not in assignment:
            sys.stderr.write(f"[ERROR] Assignment must be in key=value format (got: '{assignment}')\n")
            return 2
        key_path, val_str = assignment.split("=", 1)
        key_path = key_path.strip()
        val = _parse_value(val_str.strip())

        target_file = getattr(args, "file", "custom")
        if target_file == "defaults":
            try:
                diff = patch_config_defaults_literal(key_path, repr(val))
                print(f"[OK] Patched config_defaults.py for {key_path}")
                print(diff)
                return 0
            except Exception as e:
                sys.stderr.write(f"[ERROR] Failed to patch config_defaults.py: {e}\n")
                return 2
        else:
            cfg = load_config()
            candidate = dict(cfg)
            parts = key_path.split(".")
            curr = candidate
            for p in parts[:-1]:
                if p not in curr or not isinstance(curr[p], dict):
                    curr[p] = {}
                curr = curr[p]
            curr[parts[-1]] = val

            issues = validate(candidate)
            errors = [i for i in issues if i.level == "error"]
            if errors:
                sys.stderr.write(f"[ERROR] Invalid value for '{key_path}':\n")
                for err in errors:
                    sys.stderr.write(f"  - [{err.path}] {err.msg}\n")
                return 2

            diffs = compute_leaf_diff(candidate, baseline=DEFAULT_CONFIG)
            save_overrides_to_custom_config(diffs)
            print(f"[OK] Saved override {key_path} = {repr(val)} to custom_config.py")
            return 0
    else:
        sys.stderr.write("Unknown config action. Choose from: show, set, validate.\n")
        return 2


def handle_generate_cmd(args: argparse.Namespace) -> int:
    """Handle `generate` subcommand."""
    ascii_mode = getattr(args, "ascii", False)
    no_color = getattr(args, "no_color", False) or ("NO_COLOR" in os.environ)

    if getattr(args, "smoke", False):
        return run_smoke_test(ascii_mode=ascii_mode, no_color=no_color)

    override_args: dict[str, Any] = {}
    if args.num is not None:
        override_args["num_images"] = args.num
    if args.format is not None:
        override_args["dataset_format"] = args.format
    if args.output is not None:
        override_args["output_dir"] = args.output
    if args.seed is not None:
        override_args["seed"] = args.seed
    if args.engine is not None:
        override_args["engine"] = args.engine
    if args.workers is not None:
        override_args["workers"] = args.workers
    if args.start_index is not None:
        override_args["start_index"] = args.start_index
    if getattr(args, "overwrite", False):
        override_args["overwrite"] = True
    if getattr(args, "no_custom", False):
        override_args["no_custom"] = True

    is_tty = sys.stdout.isatty() and not getattr(args, "non_interactive", False)
    return run_generate_flow(
        ascii_mode=ascii_mode,
        no_color=no_color,
        interactive=is_tty,
        override_args=override_args,
    )


def handle_weights_cmd(args: argparse.Namespace) -> int:
    """Handle `weights show` and `weights set`."""
    ascii_mode = getattr(args, "ascii", False)
    no_color = getattr(args, "no_color", False) or ("NO_COLOR" in os.environ)

    cfg = load_config()
    if args.weights_action == "show":
        sci = dict(cfg.get("scientific_subdomain_weights", {}))
        non_sci = dict(cfg.get("non_scientific_subdomain_weights", {}))
        sci_sum = sum(sci.values())
        non_sci_sum = sum(non_sci.values())

        print("=== Scientific Domains ===")
        for d, w in sorted(sci.items()):
            pct = (w / sci_sum * 100.0) if sci_sum > 0 else 0.0
            bar = render_bar(pct, width=15, ascii_mode=ascii_mode)
            print(f"  {d:15s} : {w:5.2f} {bar} {pct:5.1f}%")
        print("=== Non-Scientific Domains ===")
        for d, w in sorted(non_sci.items()):
            pct = (w / non_sci_sum * 100.0) if non_sci_sum > 0 else 0.0
            bar = render_bar(pct, width=15, ascii_mode=ascii_mode)
            print(f"  {d:15s} : {w:5.2f} {bar} {pct:5.1f}%")
        return 0
    elif args.weights_action == "set":
        assignment = args.assignment
        if "=" not in assignment:
            sys.stderr.write(f"[ERROR] Assignment must be in domain=weight format (got: '{assignment}')\n")
            return 2
        dom, val_str = assignment.split("=", 1)
        dom = dom.strip()
        try:
            w_val = float(val_str.strip())
        except ValueError:
            sys.stderr.write(f"[ERROR] Invalid weight value: '{val_str}'\n")
            return 2

        sci = dict(cfg.get("scientific_subdomain_weights", {}))
        non_sci = dict(cfg.get("non_scientific_subdomain_weights", {}))
        if dom in sci:
            key_path = f"scientific_subdomain_weights.{dom}"
        elif dom in non_sci:
            key_path = f"non_scientific_subdomain_weights.{dom}"
        else:
            sys.stderr.write(f"[ERROR] Unknown domain '{dom}'. Valid domains: {list(sci.keys()) + list(non_sci.keys())}\n")
            return 2

        candidate = dict(cfg)
        parts = key_path.split(".")
        candidate[parts[0]][parts[1]] = w_val
        diffs = compute_leaf_diff(candidate, baseline=DEFAULT_CONFIG)
        save_overrides_to_custom_config(diffs)
        print(f"[OK] Set {key_path} = {w_val} in custom_config.py")
        return 0
    return 0


def handle_audit_cmd(args: argparse.Namespace) -> int:
    """Handle `audit` subcommand (ADM-12)."""
    from synth.semantics.admin.audit import run_audit

    strict = getattr(args, "strict", False)
    threshold = getattr(args, "threshold", 90.0)

    report = run_audit(fuzzy_threshold=threshold, strict=strict)
    summary = report.summary_dict()

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("=== Corpus Audit Report ===")
        print(f"Total findings: {summary['total_findings']} ({summary['errors']} errors, {summary['warnings']} warnings)")
        if report.degenerate_pools:
            print(f"Degenerate pools ({len(report.degenerate_pools)}):")
            for p in report.degenerate_pools:
                print(f"  - Domain: {p[0]}, Chart: {p[1]}, Orient: {p[2]}, Axis: {p[3]}")
        else:
            print("Degenerate pools: NONE")

        chart_totals = summary.get("chart_type_totals", {})
        if chart_totals:
            totals_str = ", ".join(f"{k}={v}" for k, v in chart_totals.items())
            print(f"Chart type totals: {totals_str}")

        if report.findings:
            print("\nFindings:")
            for f in report.findings:
                print(f"  [{f.severity:7s}] {f.category:16s} ({f.domain}): {f.message}")

    if getattr(args, "report", None):
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[OK] Audit report exported to {args.report}")

    if strict and (report.has_errors or len(report.degenerate_pools) > 0):
        return 1
    return 0


def handle_import_cmd(args: argparse.Namespace) -> int:
    """Handle `import` subcommand (ADM-01..10, TUI-03)."""
    from synth.semantics.admin.tui import run_import_flow

    ascii_mode = getattr(args, "ascii", False)
    no_color = getattr(args, "no_color", False) or ("NO_COLOR" in os.environ)
    is_tty = sys.stdin.isatty() and sys.stdout.isatty() and not getattr(args, "yes", False)

    return run_import_flow(
        ascii_mode=ascii_mode,
        no_color=no_color,
        interactive=is_tty,
        file_path=args.file,
        target_domain=args.domain,
        policy=args.on_collision,
        report_path=args.report,
        auto_commit=getattr(args, "yes", False),
        verify_tests=getattr(args, "verify", False),
    )


def handle_new_domain_cmd(args: argparse.Namespace) -> int:
    """Handle `new-domain` subcommand (ADM-11)."""
    from synth.semantics.admin.tui import run_new_domain_wizard

    ascii_mode = getattr(args, "ascii", False)
    no_color = getattr(args, "no_color", False) or ("NO_COLOR" in os.environ)
    is_tty = sys.stdin.isatty() and sys.stdout.isatty() and not getattr(args, "yes", False)

    shared = [] if getattr(args, "no_shared_common", False) else getattr(args, "shared_common", None)
    fallbacks = tuple(args.fallbacks) if getattr(args, "fallbacks", None) else None

    return run_new_domain_wizard(
        ascii_mode=ascii_mode,
        no_color=no_color,
        interactive=is_tty and (args.domain is None),
        domain_id=args.domain,
        display_name=args.display_name,
        is_scientific=args.scientific,
        default_weight=args.weight,
        fallbacks=fallbacks,
        shared_common=shared,
    )


def handle_rebuild_cmd(args: argparse.Namespace) -> int:
    """Handle `rebuild` subcommand (SEM-06)."""
    from synth.semantics.loader import rebuild_cache, _load_manifest_raw_data
    from synth.semantics.schema import DomainManifest

    raw = _load_manifest_raw_data(force=True)
    all_valid = True
    for dom_id, doc in sorted(raw.items()):
        try:
            DomainManifest.model_validate(doc)
            print(f"  [OK] {dom_id}.yaml: Valid schema")
        except Exception as exc:
            all_valid = False
            print(f"  [FAIL] {dom_id}.yaml: {exc}")

    if not all_valid:
        sys.stderr.write("[ERROR] Manifest schema validation failed. Cache not rebuilt.\n")
        return 1

    reg = rebuild_cache(force=True)
    print(f"[OK] Cache successfully rebuilt. Registered domains: {len(reg.domains)}, metrics: {len(reg.all_metrics)}")
    return 0


def handle_restore_cmd(args: argparse.Namespace) -> int:
    """Handle `restore` subcommand (ADM-10)."""
    from synth.semantics.admin.apply import list_backups, restore_backup
    from synth.semantics.admin.tui import run_restore_flow

    backups = list_backups()
    if getattr(args, "list", False) or (not getattr(args, "to", None) and not sys.stdin.isatty()):
        if not backups:
            print("No snapshot backups found.")
            return 0
        print("Available snapshot backups:")
        for b in backups:
            print(f"  - {b}")
        return 0

    if getattr(args, "to", None):
        try:
            restore_backup(args.to)
            print(f"[OK] Successfully restored manifests from snapshot: {args.to}")
            return 0
        except Exception as exc:
            sys.stderr.write(f"[ERROR] Restore failed: {exc}\n")
            return 1

    # Interactive restore
    return run_restore_flow(
        ascii_mode=getattr(args, "ascii", False),
        no_color=getattr(args, "no_color", False) or ("NO_COLOR" in os.environ),
    )


def handle_template_cmd(args: argparse.Namespace) -> int:
    """Handle `template` subcommand (ADM-14)."""
    from synth.semantics.admin.template import generate_template

    try:
        out_path = generate_template(args.output)
        print(f"[OK] Generated template workbook at: {out_path}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Failed to generate template: {exc}\n")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build root CLI parser with all headless subcommands (TUI-01)."""
    parser = argparse.ArgumentParser(
        prog="manage_domains.py",
        description="Synthetic Chart Studio & Domain Management Console",
    )
    parser.add_argument("--ascii", action="store_true", help="Plain ASCII output (no Unicode box drawing or emojis)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # config show / set / validate
    cfg_parser = subparsers.add_parser("config", help="Manage and validate configuration")
    cfg_sub = cfg_parser.add_subparsers(dest="config_action", required=True)

    cfg_show = cfg_sub.add_parser("show", help="Show effective configuration")
    cfg_show.add_argument("--json", action="store_true", help="Output JSON format")

    cfg_set = cfg_sub.add_parser("set", help="Set configuration override (key=value)")
    cfg_set.add_argument("assignment", help="Key assignment, e.g. num_images=50 or scientific_ratio=0.6")
    cfg_set.add_argument("--file", choices=["custom", "defaults"], default="custom", help="Target configuration file")

    cfg_sub.add_parser("validate", help="Validate active configuration (CFG-03)")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Run chart generation in subprocess")
    gen_parser.add_argument("-n", "--num", type=int, help="Number of charts to generate")
    gen_parser.add_argument("-f", "--format", choices=["detection", "classification", "multi_chart_detection"], help="Dataset format")
    gen_parser.add_argument("-o", "--output", help="Output directory")
    gen_parser.add_argument("-s", "--seed", type=int, help="Random seed")
    gen_parser.add_argument("-e", "--engine", choices=["matplotlib", "vegalite"], help="Rendering engine")
    gen_parser.add_argument("-w", "--workers", type=int, help="Number of worker processes")
    gen_parser.add_argument("--start-index", type=int, help="Start image index for appending")
    gen_parser.add_argument("--smoke", action="store_true", help="Run quick 3-chart smoke test to temp directory")
    gen_parser.add_argument("--overwrite", action="store_true", help="Overwrite non-empty output directory")
    gen_parser.add_argument("--no-custom", action="store_true", help="Ignore custom_config.py")
    gen_parser.add_argument("--non-interactive", action="store_true", help="Do not prompt for inputs")

    # weights
    w_parser = subparsers.add_parser("weights", help="Inspect and adjust domain sampling weights")
    w_sub = w_parser.add_subparsers(dest="weights_action", required=True)
    w_sub.add_parser("show", help="Display domain weights and normalized percentages")
    w_set = w_sub.add_parser("set", help="Set domain weight (domain=weight)")
    w_set.add_argument("assignment", help="Domain assignment, e.g. biomedical=0.75")

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit corpus manifests and check for degenerate pools (ADM-12)")
    audit_parser.add_argument("--strict", action="store_true", help="Fail with non-zero exit code if findings or degenerate pools exist")
    audit_parser.add_argument("--threshold", type=float, default=90.0, help="Fuzzy match threshold for near-duplicate labels (default: 90.0)")
    audit_parser.add_argument("--json", action="store_true", help="Output audit report in JSON format")
    audit_parser.add_argument("--report", type=str, help="Save audit report to JSON file")

    # import
    import_parser = subparsers.add_parser("import", help="Import & deduplicate spreadsheet into domain manifest (ADM-01..10)")
    import_parser.add_argument("file", help="Path to spreadsheet (.xlsx, .csv, or folder)")
    import_parser.add_argument("-d", "--domain", required=False, default=None, help="Target domain ID (e.g. aerospace). Optional if inferred from filename.")
    import_parser.add_argument("--on-collision", choices=["skip", "merge", "overwrite", "abort"], default="skip", help="Collision policy (default: skip)")
    import_parser.add_argument("--report", type=str, help="Export pre-flight diff plan to JSON file")
    import_parser.add_argument("-y", "--yes", action="store_true", help="Commit changes (skip confirmation prompt / dry-run)")
    import_parser.add_argument("--verify", action="store_true", help="Run domain manifest pytest suite after import")

    # new-domain
    new_dom_parser = subparsers.add_parser("new-domain", help="Create new domain manifest skeleton (ADM-11)")
    new_dom_parser.add_argument("domain", nargs="?", help="Domain ID (lowercase, alphanumeric, e.g. aerospace)")
    new_dom_parser.add_argument("--display-name", help="Display name for UI/reports")
    new_dom_parser.add_argument("--scientific", action="store_true", default=None, help="Tag as scientific domain")
    new_dom_parser.add_argument("--non-scientific", dest="scientific", action="store_false", help="Tag as non-scientific domain")
    new_dom_parser.add_argument("--weight", type=float, default=0.10, help="Default sampling weight (default: 0.10)")
    new_dom_parser.add_argument("--fallbacks", nargs=2, metavar=("X_FALLBACK", "Y_FALLBACK"), help="Default fallback labels for X and Y axes")
    new_dom_parser.add_argument("--shared-common", nargs="*", help="Metric labels to share from common.yaml")
    new_dom_parser.add_argument("--no-shared-common", action="store_true", help="Do not share any common metrics")
    new_dom_parser.add_argument("-y", "--yes", action="store_true", help="Non-interactive mode")

    # rebuild
    rebuild_parser = subparsers.add_parser("rebuild", help="Force manifest validation and cache rebuild (SEM-06)")
    rebuild_parser.add_argument("--force", action="store_true", default=True, help="Force rebuild even if fingerprints match")

    # restore
    restore_parser = subparsers.add_parser("restore", help="Restore manifest snapshot from backup (ADM-10)")
    restore_parser.add_argument("--list", action="store_true", help="List available snapshot backups")
    restore_parser.add_argument("--to", help="Timestamp of backup snapshot to restore (e.g. 2026-10-02T12-00-00Z)")

    # template
    tmpl_parser = subparsers.add_parser("template", help="Generate spreadsheet template for authoring domain manifests (ADM-14)")
    tmpl_parser.add_argument("-o", "--output", default="domain_template.xlsx", help="Output file path (default: domain_template.xlsx)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ascii_mode = args.ascii
    no_color = args.no_color or ("NO_COLOR" in os.environ)

    if args.subcommand == "config":
        sys.exit(handle_config_cmd(args))
    elif args.subcommand == "generate":
        sys.exit(handle_generate_cmd(args))
    elif args.subcommand == "weights":
        sys.exit(handle_weights_cmd(args))
    elif args.subcommand == "audit":
        sys.exit(handle_audit_cmd(args))
    elif args.subcommand == "import":
        sys.exit(handle_import_cmd(args))
    elif args.subcommand == "new-domain":
        sys.exit(handle_new_domain_cmd(args))
    elif args.subcommand == "rebuild":
        sys.exit(handle_rebuild_cmd(args))
    elif args.subcommand == "restore":
        sys.exit(handle_restore_cmd(args))
    elif args.subcommand == "template":
        sys.exit(handle_template_cmd(args))
    else:
        # No subcommand passed: Launch TUI or Non-TTY Header
        is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
        if is_interactive:
            run_main_menu(ascii_mode=ascii_mode, no_color=no_color)
        else:
            # Non-TTY => no prompts, plain text summary
            print(render_header(ascii_mode=ascii_mode, no_color=no_color))
            sys.exit(0)


if __name__ == "__main__":
    main()
