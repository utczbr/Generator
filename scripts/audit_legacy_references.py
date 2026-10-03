#!/usr/bin/env python3
"""
scripts/audit_legacy_references.py

AST-based verification engine to audit the repository for legacy semantic references
originating from themes.py.

Scans all Python files (excluding tests/ and virtual environments) for:
1. `ImportFrom` nodes where `node.module == "themes"` importing banned legacy constants.
2. `Attribute` access where `node.value.id == "themes"` accessing banned legacy constants.

Exit Codes:
- 0: Clean scan (0 violations found)
- 1: Violations detected (>= 1 violation found)
"""
import ast
import os
import sys
from pathlib import Path
from typing import List, NamedTuple, Set


BANNED_LEGACY_CONSTANTS: Set[str] = {
    "SCIENTIFIC_X_LABELS",
    "SCIENTIFIC_Y_LABELS",
    "BUSINESS_X_LABELS",
    "BUSINESS_Y_LABELS",
    "CHART_TITLES",
    "COMPARATIVE_LABELS",
    "HISTOGRAM_Y_LABELS",
    "SCIENTIFIC_DOMAIN_DICT",
    "BUSINESS_DOMAIN_DICT",
    "HEATMAP_XLABELS_SCIENTIFIC",
    "HEATMAP_YLABELS_SCIENTIFIC",
    "HEATMAP_XLABELS_BUSINESS",
    "HEATMAP_YLABELS_BUSINESS",
    "COLORBAR_TITLES_SCIENTIFIC",
    "COLORBAR_TITLES_BUSINESS",
    "CONTEXT_CONFIGURATIONS",
    "STRUCTURAL_THEMES",
}

EXCLUDED_DIR_NAMES: Set[str] = {
    "tests",
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "graphify-out",
}


class Violation(NamedTuple):
    filepath: str
    line: int
    column: int
    symbol: str


class LegacyReferenceVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.violations: List[Violation] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "themes":
            for alias in node.names:
                if alias.name in BANNED_LEGACY_CONSTANTS or alias.name == "*":
                    self.violations.append(
                        Violation(
                            filepath=self.filepath,
                            line=node.lineno,
                            column=node.col_offset,
                            symbol=alias.name,
                        )
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "themes":
            if node.attr in BANNED_LEGACY_CONSTANTS:
                self.violations.append(
                    Violation(
                        filepath=self.filepath,
                        line=node.lineno,
                        column=node.col_offset,
                        symbol=node.attr,
                    )
                )
        self.generic_visit(node)


def should_skip_directory(dir_path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in dir_path.parts)


def scan_file(filepath: Path) -> List[Violation]:
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as exc:
        print(f"WARNING: Could not parse {filepath}: {exc}", file=sys.stderr)
        return []

    visitor = LegacyReferenceVisitor(filepath=str(filepath))
    visitor.visit(tree)
    return visitor.violations


def audit_repository(root_path: Path) -> List[Violation]:
    all_violations: List[Violation] = []

    if root_path.is_file() and root_path.suffix == ".py":
        return scan_file(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):
        current_dir = Path(dirpath)
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]

        if should_skip_directory(current_dir):
            continue

        for filename in filenames:
            if filename.endswith(".py"):
                file_path = current_dir / filename
                # Exclude themes.py itself since it defines the constants
                if file_path.name == "themes.py":
                    continue
                all_violations.extend(scan_file(file_path))

    return all_violations


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    violations = audit_repository(target.resolve())

    if violations:
        print(f"FAILED: Found {len(violations)} legacy semantic reference(s):")
        for v in violations:
            print(f"{v.filepath}:{v.line}:{v.column}: {v.symbol}")
        return 1

    print("PASSED: Zero legacy semantic references detected in production codebase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
