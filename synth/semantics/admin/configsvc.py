"""
synth/semantics/admin/configsvc.py

Service layer for managing configuration overrides and developer-mode defaults patching.
- CFG-04: Persists user overrides to custom_config.py between sentinels as leaf assignments.
- CFG-05: Safely patches literals in config_defaults.py using AST spans with diff, .bak, and verification.
"""
from __future__ import annotations

import ast
import copy
import difflib
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG

SENTINEL_START = "# >>> managed by manage_domains — do not edit >>>"
SENTINEL_END = "# <<< managed by manage_domains — do not edit <<<"
SENTINEL_PATTERN = re.compile(
    r"# >>> managed by manage_domains — do not edit >>>.*?(?:# <<< managed by manage_domains — do not edit <<<|# <<<)",
    re.DOTALL,
)


def compute_leaf_diff(
    modified: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Compute dot-separated leaf differences between modified and baseline dicts."""
    if baseline is None:
        baseline = DEFAULT_CONFIG

    diffs: Dict[str, Any] = {}
    for k, v in modified.items():
        path = f"{prefix}.{k}" if prefix else k
        if k not in baseline:
            diffs[path] = v
        elif isinstance(v, dict) and isinstance(baseline.get(k), dict):
            diffs.update(compute_leaf_diff(v, baseline[k], prefix=path))
        elif v != baseline[k]:
            diffs[path] = v
    return diffs


def format_leaf_assignment(dot_path: str, value: Any) -> str:
    """Format a dot path into an OCR_TRAINING_CONFIG assignment line."""
    parts = dot_path.split(".")
    accessors = "".join(f"[{json_or_repr_key(p)}]" for p in parts)
    return f"OCR_TRAINING_CONFIG{accessors} = {repr(value)}"


def json_or_repr_key(key: str) -> str:
    """Format dict key access safely."""
    return f'"{key}"'


def save_overrides_to_custom_config(
    overrides: Dict[str, Any],
    custom_config_path: Path = Path("custom_config.py"),
) -> str:
    """
    Save leaf overrides to custom_config.py between sentinels (CFG-04).
    Leaves everything outside the sentinels completely untouched.
    """
    if custom_config_path.exists():
        content = custom_config_path.read_text(encoding="utf-8")
    else:
        content = (
            '"""\ncustom_config.py\n\n'
            'User-specific overrides for OCR training and synthetic dataset generation.\n'
            '"""\nimport copy\n'
            'from config_defaults import OCR_TRAINING_CONFIG as _DEFAULT_CONFIG\n\n'
            'OCR_TRAINING_CONFIG = copy.deepcopy(_DEFAULT_CONFIG)\n'
        )

    lines = [SENTINEL_START]
    for path, val in sorted(overrides.items()):
        lines.append(format_leaf_assignment(path, val))
    lines.append(SENTINEL_END)
    block = "\n".join(lines)

    if SENTINEL_PATTERN.search(content):
        new_content = SENTINEL_PATTERN.sub(block, content)
    else:
        # Append to end of file with newline
        prefix = content.rstrip() + "\n\n" if content.strip() else ""
        new_content = prefix + block + "\n"

    custom_config_path.write_text(new_content, encoding="utf-8")
    return new_content


def _find_ast_literal_node(
    root: ast.AST,
    target_var: str,
    key_path: str,
) -> Optional[ast.AST]:
    """Locate AST node of the literal value corresponding to target_var['key']['path']."""
    parts = key_path.split(".")
    dict_node = None

    # 1. Find assignment to target_var
    for node in ast.walk(root):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == target_var:
                    if isinstance(node.value, ast.Dict):
                        dict_node = node.value
                        break
        if dict_node:
            break

    if not dict_node:
        return None

    # 2. Walk down nested dicts
    curr = dict_node
    for i, part in enumerate(parts):
        is_leaf = (i == len(parts) - 1)
        found_val = None
        for k_node, v_node in zip(curr.keys, curr.values):
            if isinstance(k_node, ast.Constant) and k_node.value == part:
                found_val = v_node
                break

        if not found_val:
            return None

        if is_leaf:
            return found_val
        elif isinstance(found_val, ast.Dict):
            curr = found_val
        else:
            return None

    return None


def patch_config_defaults_literal(
    key_path: str,
    new_value_repr: str,
    target_file: Path = Path("config_defaults.py"),
) -> str:
    """
    Patch a literal in config_defaults.py via AST source span replacement (CFG-05).
    - Locates the node via AST
    - Preserves comments and surrounding code
    - Creates a .bak backup
    - Validates via subprocess import; rolls back on failure
    - Returns unified diff
    """
    if not target_file.exists():
        raise FileNotFoundError(f"Target file {target_file} not found")

    orig_text = target_file.read_text(encoding="utf-8")
    parsed = ast.parse(orig_text, filename=str(target_file))

    node = _find_ast_literal_node(parsed, "OCR_TRAINING_CONFIG", key_path)
    if node is None or not hasattr(node, "lineno"):
        raise ValueError(f"Could not locate literal AST node for key_path '{key_path}' in {target_file}")

    lines = orig_text.splitlines(keepends=True)
    # 1-indexed lineno to 0-indexed line array
    start_line_idx = node.lineno - 1
    end_line_idx = node.end_lineno - 1
    start_col = node.col_offset
    end_col = node.end_col_offset

    # Build new text
    before = "".join(lines[:start_line_idx]) + lines[start_line_idx][:start_col]
    after = lines[end_line_idx][end_col:] + "".join(lines[end_line_idx + 1:])
    new_text = before + new_value_repr + after

    # Write backup
    bak_file = target_file.with_suffix(target_file.suffix + ".bak")
    bak_file.write_text(orig_text, encoding="utf-8")

    # Write new text
    target_file.write_text(new_text, encoding="utf-8")

    # Generate unified diff
    diff_lines = list(
        difflib.unified_diff(
            orig_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(target_file),
            tofile=str(target_file),
        )
    )
    diff_str = "".join(diff_lines)

    # Subprocess validation check
    check_code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('test_defaults', '{target_file}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
    )
    res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
    if res.returncode != 0:
        # Roll back
        target_file.write_text(orig_text, encoding="utf-8")
        raise ValueError(f"Patch validation failed, rolled back to original: {res.stderr}")

    return diff_str
