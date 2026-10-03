"""
config_loader.py

Standard library-only configuration loader for synthetic chart generation.
Implements layered configuration resolution, provenance tracking, and deep-copy guarantees.
Precedence:
  config_defaults.py
  -> custom_config.py (optional user overrides; skipped if use_custom=False)
  -> --config file (.py or .json)
  -> env: DEBUG_MODE
  -> overrides / CLI options (--set, --num, --output, --mode)
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional
import warnings


class ResolvedConfig(dict):
    """
    Resolved dictionary configuration with an attached provenance map.
    Mutations on ResolvedConfig do not affect default configurations.
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None, provenance: Optional[Dict[str, str]] = None):
        super().__init__(data or {})
        self.provenance: Dict[str, str] = dict(provenance or {})

    def copy(self) -> "ResolvedConfig":
        return ResolvedConfig(copy.deepcopy(dict(self)), copy.deepcopy(self.provenance))

    def __deepcopy__(self, memo: Any) -> "ResolvedConfig":
        return ResolvedConfig(copy.deepcopy(dict(self), memo), copy.deepcopy(self.provenance, memo))


def _deep_merge_with_provenance(
    base: Dict[str, Any],
    overlay: Dict[str, Any],
    source_tag: str,
    provenance: Dict[str, str],
    prefix: str = "",
) -> Dict[str, Any]:
    """Recursively merge overlay into base while recording provenance for each leaf and branch."""
    for key, val in overlay.items():
        path = f"{prefix}.{key}" if prefix else key
        provenance[path] = source_tag
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge_with_provenance(base[key], val, source_tag, provenance, prefix=path)
        else:
            base[key] = copy.deepcopy(val)
    return base


def _load_py_config(path: Path) -> Dict[str, Any]:
    """Load configuration from a python file."""
    spec = importlib.util.spec_from_file_location("dynamic_config", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "OCR_TRAINING_CONFIG", None)
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"Module at {path} does not define OCR_TRAINING_CONFIG dictionary")
    return cfg


def _load_json_config(path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON config at {path} must be an object")
    return data


def parse_set_override(set_expr: str) -> tuple[str, Any]:
    """
    Parse a 'key.path=value' expression into (key_path, parsed_value).
    Value is parsed as JSON with fallback to raw string.
    """
    if "=" not in set_expr:
        raise ValueError(f"Invalid --set expression '{set_expr}'; must be in format 'key.path=value'")
    key_path, val_str = set_expr.split("=", 1)
    key_path = key_path.strip()
    val_str = val_str.strip()
    try:
        val = json.loads(val_str)
    except Exception:
        val = val_str
    return key_path, val


def apply_dot_override(target: Dict[str, Any], key_path: str, value: Any, provenance: Dict[str, str], source_tag: str = "cli") -> None:
    """Set a nested leaf in target dict given a dot-separated key path."""
    parts = key_path.split(".")
    curr = target
    for i, part in enumerate(parts[:-1]):
        if part not in curr or not isinstance(curr[part], dict):
            curr[part] = {}
        curr = curr[part]
    curr[parts[-1]] = value
    provenance[key_path] = source_tag
    # Also record prefix components
    for j in range(1, len(parts) + 1):
        provenance[".".join(parts[:j])] = source_tag


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    use_custom: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> ResolvedConfig:
    """
    Load and resolve layered configuration with provenance tracking.
    Precedence:
      1. config_defaults.py (source: default)
      2. custom_config.py (if use_custom=True and present; source: custom)
      3. path (.py or .json; source: file)
      4. env DEBUG_MODE (source: env)
      5. overrides dict / CLI arguments (source: cli)
    """
    if env is None:
        env = os.environ

    # 1. Defaults
    from config_defaults import OCR_TRAINING_CONFIG as DEFAULT_CONFIG
    base = copy.deepcopy(DEFAULT_CONFIG)
    provenance: Dict[str, str] = {}
    for k in base:
        provenance[k] = "default"

    # 2. Custom config
    if use_custom:
        custom_path = Path("custom_config.py")
        if custom_path.exists():
            try:
                custom_cfg = _load_py_config(custom_path)
            except Exception as e:
                sys.stderr.write(f"[ERROR] Broken custom_config.py: {e}\n")
                sys.exit(2)
            _deep_merge_with_provenance(base, custom_cfg, "custom", provenance)

    # 3. File path override (--config)
    if path:
        p = Path(path)
        if not p.exists():
            sys.stderr.write(f"[ERROR] Configuration file not found: {path}\n")
            sys.exit(2)
        try:
            if p.suffix.lower() == ".json":
                file_cfg = _load_json_config(p)
            elif p.suffix.lower() == ".py":
                file_cfg = _load_py_config(p)
            else:
                sys.stderr.write(f"[ERROR] Unsupported config file format: {path}. Use .py or .json.\n")
                sys.exit(2)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to load config from {path}: {e}\n")
            sys.exit(2)
        _deep_merge_with_provenance(base, file_cfg, "file", provenance)

    # 4. Environment variables
    if "DEBUG_MODE" in env:
        debug_val = env["DEBUG_MODE"].strip().lower() in ("true", "1", "yes")
        base["debug_mode"] = debug_val
        provenance["debug_mode"] = "env"

    # 5. Overrides
    if overrides:
        for k, v in overrides.items():
            if "." in k:
                apply_dot_override(base, k, v, provenance, source_tag="cli")
            else:
                base[k] = copy.deepcopy(v)
                provenance[k] = "cli"

    # CFG-07: Handle top-level scientific_ratio & legacy sync
    bar_sci_source = provenance.get("bar_chart_config.scientific_ratio")
    top_sci_source = provenance.get("scientific_ratio")
    if bar_sci_source in ("custom", "file", "cli") and top_sci_source == "default":
        warnings.warn(
            "bar_chart_config.scientific_ratio is deprecated; use top-level scientific_ratio instead",
            DeprecationWarning,
            stacklevel=2,
        )
        base["scientific_ratio"] = base["bar_chart_config"]["scientific_ratio"]
        provenance["scientific_ratio"] = bar_sci_source
    elif "scientific_ratio" not in base:
        bar_cfg = base.get("bar_chart_config", {})
        if "scientific_ratio" in bar_cfg:
            warnings.warn(
                "bar_chart_config.scientific_ratio is deprecated; use top-level scientific_ratio instead",
                DeprecationWarning,
                stacklevel=2,
            )
            base["scientific_ratio"] = bar_cfg["scientific_ratio"]
            provenance["scientific_ratio"] = provenance.get("bar_chart_config.scientific_ratio", "default")
    else:
        # Keep legacy nested location in sync
        if "bar_chart_config" in base and isinstance(base["bar_chart_config"], dict):
            base["bar_chart_config"]["scientific_ratio"] = base["scientific_ratio"]

    return ResolvedConfig(base, provenance)
