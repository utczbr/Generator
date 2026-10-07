"""
synth/semantics/admin/configspec.py

Declarative configuration specification and validator (CFG-03).
Provides validation rules for generation configurations:
- Chart types, weights, probabilities
- Subdomain weights & scientific domain matching
- Domain validity via list_domains()
- Dataset format, engine, coverage, and realism effects
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from generator import EFFECT_REGISTRY


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    path: str
    msg: str


@dataclass(frozen=True)
class ConfigParamSpec:
    path: str
    expected_type: type | tuple[type, ...]
    doc: str
    group: str
    valid_range: Optional[tuple[float, float]] = None
    allowed_values: Optional[Set[Any]] = None
    danger: bool = False


def list_domains(domains_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Read-only helper to inspect declared domains from YAML manifests / cache.
    Release A does not depend on SEM-01 registry refactoring.
    """
    if domains_dir is None:
        try:
            from synth.semantics.loader import registry
            if registry and registry.domains:
                return {
                    dom_id: {
                        "display_name": info.display_name,
                        "is_scientific": info.is_scientific,
                        "default_weight": info.default_weight,
                    }
                    for dom_id, info in registry.domains.items()
                }
        except Exception:
            pass
        domains_dir = Path(__file__).resolve().parent.parent / "domains"

    cache_file = domains_dir / ".cache.json"
    if cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            if raw:
                return {
                    dom_id: {
                        "display_name": doc.get("display_name", dom_id.capitalize()),
                        "is_scientific": bool(doc.get("is_scientific", False)),
                        "default_weight": doc.get("default_weight", 0.0),
                    }
                    for dom_id, doc in raw.items()
                }
        except Exception:
            pass

    import yaml
    try:
        from yaml import CSafeLoader as FastYamlLoader
    except ImportError:
        from yaml import SafeLoader as FastYamlLoader  # type: ignore

    res: Dict[str, Dict[str, Any]] = {}
    for yf in sorted(domains_dir.glob("*.yaml")):
        try:
            with open(yf, "rb") as f:
                doc = yaml.load(f, Loader=FastYamlLoader)
                if doc and "domain_id" in doc:
                    res[doc["domain_id"]] = {
                        "display_name": doc.get("display_name", doc["domain_id"].capitalize()),
                        "is_scientific": bool(doc.get("is_scientific", False)),
                        "default_weight": doc.get("default_weight", 0.0),
                    }
        except Exception:
            pass
    return res


CONFIG_SPECS: list[ConfigParamSpec] = [
    ConfigParamSpec("num_images", int, "Number of images to generate", "general", valid_range=(1, 1_000_000)),
    ConfigParamSpec("engine", str, "Rendering backend engine", "general", allowed_values={"matplotlib", "vegalite"}),
    ConfigParamSpec("dataset_format", str, "Dataset format mode", "general", allowed_values={"detection", "classification", "multi_chart_detection"}),
    ConfigParamSpec("scientific_ratio", (int, float), "Proportion of scientific domain charts", "domains", valid_range=(0.0, 1.0)),
    ConfigParamSpec("use_parallel", bool, "Run generation in parallel worker processes", "performance"),
    ConfigParamSpec("strict", bool, "Fail on any single chart generation error", "robustness"),
    ConfigParamSpec("heatmap_validation.min_cell_coverage", (int, float), "Minimum cell coverage for heatmap validation", "heatmap", valid_range=(0.0, 1.0)),
]


def validate(cfg: Dict[str, Any], domains_dir: Optional[Path] = None) -> List[Issue]:
    """
    Validate resolved configuration dictionary against declarative rules (CFG-03).
    Returns list of validation issues (errors and warnings).
    """
    issues: List[Issue] = []

    # 1. Basic specs
    if "num_images" in cfg:
        num = cfg["num_images"]
        if not isinstance(num, int) or num < 1:
            issues.append(Issue("error", "num_images", f"num_images must be an integer >= 1 (got {num})"))

    if "engine" in cfg:
        engine = cfg["engine"]
        if engine not in {"matplotlib", "vegalite"}:
            issues.append(Issue("error", "engine", f"engine must be 'matplotlib' or 'vegalite' (got {engine})"))

    if "dataset_format" in cfg:
        fmt = cfg["dataset_format"]
        if fmt not in {"detection", "classification", "multi_chart_detection"}:
            issues.append(Issue("error", "dataset_format", f"dataset_format must be 'detection', 'classification', or 'multi_chart_detection' (got {fmt})"))

    sci_ratio = cfg.get("scientific_ratio", cfg.get("bar_chart_config", {}).get("scientific_ratio", 0.6))
    if not (0.0 <= sci_ratio <= 1.0):
        issues.append(Issue("error", "scientific_ratio", f"scientific_ratio must be between 0.0 and 1.0 (got {sci_ratio})"))

    if "annotation_min_side_px" in cfg:
        msp = cfg["annotation_min_side_px"]
        if not isinstance(msp, (int, float)) or msp <= 0:
            issues.append(Issue("error", "annotation_min_side_px", f"annotation_min_side_px must be a positive number (got {msp})"))

    # 2. Chart types & weights
    chart_types = cfg.get("chart_types", {})
    if not isinstance(chart_types, dict):
        issues.append(Issue("error", "chart_types", "chart_types must be a dictionary"))
    else:
        enabled_positive_count = 0
        for cname, ccfg in chart_types.items():
            if not isinstance(ccfg, dict):
                continue
            enabled = ccfg.get("enabled", False)
            weight = ccfg.get("weight", 0)
            if weight < 0:
                issues.append(Issue("error", f"chart_types.{cname}.weight", f"Weight cannot be negative (got {weight})"))
            if enabled and weight > 0:
                enabled_positive_count += 1
        if enabled_positive_count == 0:
            issues.append(Issue("error", "chart_types", "At least one chart type must be enabled with weight > 0"))

    # 3. Scenario weights
    scenario_weights = cfg.get("scenario_weights", {})
    if isinstance(scenario_weights, dict):
        for sname, sweight in scenario_weights.items():
            if sweight < 0:
                issues.append(Issue("error", f"scenario_weights.{sname}", f"Scenario weight cannot be negative (got {sweight})"))

    # 4. Probabilities in bar_chart_config & multi_chart_detection
    for group_name in ("bar_chart_config", "multi_chart_detection"):
        grp = cfg.get(group_name, {})
        if isinstance(grp, dict):
            for k, v in grp.items():
                if "probability" in k and isinstance(v, (int, float)):
                    if not (0.0 <= v <= 1.0):
                        issues.append(Issue("error", f"{group_name}.{k}", f"Probability must be in [0.0, 1.0] (got {v})"))

    # 5. Domains & Subdomain weights
    known_domains = list_domains(domains_dir)
    sci_domains = {d: info for d, info in known_domains.items() if info["is_scientific"]}
    non_sci_domains = {d: info for d, info in known_domains.items() if not info["is_scientific"]}

    # Check scientific_subdomain_weights
    sci_sub = cfg.get("scientific_subdomain_weights", {})
    if isinstance(sci_sub, dict):
        sci_total = 0.0
        for dom, w in sci_sub.items():
            if w < 0:
                issues.append(Issue("error", f"scientific_subdomain_weights.{dom}", f"Weight cannot be negative (got {w})"))
            if dom not in known_domains:
                issues.append(Issue("error", f"scientific_subdomain_weights.{dom}", f"Unknown domain '{dom}'; not found in manifest domain list"))
            elif dom not in sci_domains:
                issues.append(Issue("error", f"scientific_subdomain_weights.{dom}", f"Domain '{dom}' is not scientific (manifest has is_scientific=False)"))
            else:
                sci_total += w
        if sci_ratio > 0.0 and sci_total <= 0.0:
            issues.append(Issue("error", "scientific_subdomain_weights", f"Total scientific subdomain weight must be > 0 when scientific_ratio > 0 (got {sci_total})"))

    # Check non_scientific_subdomain_weights
    non_sci_sub = cfg.get("non_scientific_subdomain_weights", {})
    if isinstance(non_sci_sub, dict):
        non_sci_total = 0.0
        for dom, w in non_sci_sub.items():
            if w < 0:
                issues.append(Issue("error", f"non_scientific_subdomain_weights.{dom}", f"Weight cannot be negative (got {w})"))
            if dom not in known_domains:
                issues.append(Issue("error", f"non_scientific_subdomain_weights.{dom}", f"Unknown domain '{dom}'; not found in manifest domain list"))
            elif dom not in non_sci_domains:
                issues.append(Issue("error", f"non_scientific_subdomain_weights.{dom}", f"Domain '{dom}' is scientific (manifest has is_scientific=True)"))
            else:
                non_sci_total += w
        if sci_ratio < 1.0 and non_sci_total <= 0.0:
            issues.append(Issue("error", "non_scientific_subdomain_weights", f"Total non-scientific subdomain weight must be > 0 when scientific_ratio < 1.0 (got {non_sci_total})"))

    # Check forced domain keys
    for dom_key in ("synthetic_domain", "semantic_domain"):
        val = cfg.get(dom_key)
        if val is not None and val not in known_domains:
            issues.append(Issue("error", dom_key, f"Domain '{val}' is not a recognized manifest domain"))

    # Warn about domains with no weight
    all_weighted = set(sci_sub.keys()) | set(non_sci_sub.keys())
    for dname in known_domains:
        if dname == "common":
            continue
        if dname not in all_weighted or (sci_sub.get(dname, 0) == 0 and non_sci_sub.get(dname, 0) == 0):
            issues.append(Issue("warning", f"domain_weights.{dname}", f"Domain '{dname}' has weight 0 or is absent from weights — it will never be sampled"))

    # 6. Heatmap validation
    heatmap_val = cfg.get("heatmap_validation", {})
    if isinstance(heatmap_val, dict) and "min_cell_coverage" in heatmap_val:
        cov = heatmap_val["min_cell_coverage"]
        if not (0.0 <= cov <= 1.0):
            issues.append(Issue("error", "heatmap_validation.min_cell_coverage", f"min_cell_coverage must be in [0.0, 1.0] (got {cov})"))

    # 7. Realism effects
    effects = cfg.get("realism_effects", {})
    prof = cfg.get("profile", "legacy")
    is_non_legacy = bool(prof and prof != "legacy")

    if isinstance(effects, dict):
        for eff_name, eff_cfg in effects.items():
            if eff_name not in EFFECT_REGISTRY:
                issues.append(Issue("error", f"realism_effects.{eff_name}", f"Effect '{eff_name}' is not in EFFECT_REGISTRY"))
                continue
            if isinstance(eff_cfg, dict):
                p_val = eff_cfg.get("p", 1.0)
                if not (0.0 <= p_val <= 1.0):
                    issues.append(Issue("error", f"realism_effects.{eff_name}.p", f"Probability p must be in [0.0, 1.0] (got {p_val})"))

                # FR-091: Parameter validation against function signature
                func = EFFECT_REGISTRY[eff_name]
                target = func.func if hasattr(func, "func") else func
                sig = inspect.signature(target)
                accepted = set()
                for pname, param in sig.parameters.items():
                    if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                        if pname not in ("pil_img", "image", "img"):
                            accepted.add(pname)
                if eff_name in ("perspective", "perspective_warp"):
                    accepted.add("magnitude")
                    accepted.add("distortion_factor")
                if eff_name == "uneven_lighting":
                    accepted.add("intensity_range")

                # Extract params from "params" subdict or direct keys
                params = {}
                if "params" in eff_cfg and isinstance(eff_cfg["params"], dict):
                    params.update(eff_cfg["params"])
                for k, v in eff_cfg.items():
                    if k not in ("p", "params"):
                        params[k] = v

                for param_name in params:
                    if param_name not in accepted:
                        level = "error" if is_non_legacy else "warning"
                        issues.append(Issue(level, f"realism_effects.{eff_name}.params.{param_name}", f"Unknown parameter '{param_name}' for effect '{eff_name}'. Accepted parameters: {sorted(list(accepted))}"))

                # FR-091: reject perspective.magnitude > 0.15 in non-legacy profiles
                if eff_name in ("perspective", "perspective_warp"):
                    mag = params.get("magnitude") or params.get("distortion_factor")
                    if mag is not None:
                        if isinstance(mag, (int, float)) and mag > 0.15:
                            if is_non_legacy:
                                issues.append(Issue("error", f"realism_effects.{eff_name}.params.magnitude", f"perspective magnitude {mag} exceeds maximum allowed value 0.15 for non-legacy profile '{prof}'"))
                            else:
                                issues.append(Issue("warning", f"realism_effects.{eff_name}.params.magnitude", f"perspective magnitude {mag} exceeds recommended value 0.15"))
                        elif isinstance(mag, (list, tuple)) and len(mag) == 2 and mag[1] > 0.15:
                            if is_non_legacy:
                                issues.append(Issue("error", f"realism_effects.{eff_name}.params.magnitude", f"perspective magnitude {mag} exceeds maximum allowed value 0.15 for non-legacy profile '{prof}'"))

        # FR-112: canvas_reframe ordering and parameter bounds in non-legacy profiles
        if "canvas_reframe" in effects:
            reframe_cfg = effects["canvas_reframe"]
            if is_non_legacy:
                effect_keys = list(effects.keys())
                reframe_idx = effect_keys.index("canvas_reframe")
                geom_effects = {"scan_rotation", "perspective", "perspective_warp", "page_curl", "non_rigid_mesh", "mesh_warp", "clipping", "pdf_document_context"}
                for ge in geom_effects:
                    if ge in effect_keys:
                        ge_idx = effect_keys.index(ge)
                        if reframe_idx < ge_idx:
                            issues.append(Issue("error", "realism_effects.canvas_reframe", f"canvas_reframe must follow geometric effect '{ge}', but appears at index {reframe_idx} before index {ge_idx}"))
                if "resize" in effect_keys:
                    resize_idx = effect_keys.index("resize")
                    if reframe_idx > resize_idx:
                        issues.append(Issue("error", "realism_effects.canvas_reframe", f"canvas_reframe must precede 'resize', but appears at index {reframe_idx} after index {resize_idx}"))

            if isinstance(reframe_cfg, dict):
                rf_params = {}
                if "params" in reframe_cfg and isinstance(reframe_cfg["params"], dict):
                    rf_params.update(reframe_cfg["params"])
                for k, v in reframe_cfg.items():
                    if k not in ("p", "params"):
                        rf_params[k] = v

                for p_key in ("p_tight", "p_match_bg"):
                    if p_key in rf_params:
                        pv = rf_params[p_key]
                        if not (0.0 <= pv <= 1.0):
                            issues.append(Issue("error" if is_non_legacy else "warning", f"realism_effects.canvas_reframe.params.{p_key}", f"{p_key} must be in [0.0, 1.0] (got {pv})"))

                if "margin_frac_range" in rf_params:
                    mfr = rf_params["margin_frac_range"]
                    vals = [mfr] if isinstance(mfr, (int, float)) else list(mfr) if isinstance(mfr, (list, tuple)) else []
                    for v in vals:
                        if not (0.0 <= v <= 0.25):
                            issues.append(Issue("error" if is_non_legacy else "warning", "realism_effects.canvas_reframe.params.margin_frac_range", f"margin_frac_range values must be in [0.0, 0.25] (got {mfr})"))

                if "tight_px" in rf_params:
                    tpx = rf_params["tight_px"]
                    vals = [tpx] if isinstance(tpx, (int, float)) else list(tpx) if isinstance(tpx, (list, tuple)) else []
                    for v in vals:
                        if v < 0 or v > 12:
                            issues.append(Issue("error" if is_non_legacy else "warning", "realism_effects.canvas_reframe.params.tight_px", f"tight_px values must be <= 12 (got {tpx})"))

    # 8. Profile validation
    prof = cfg.get("profile")
    if prof is not None and prof != "legacy":
        from config_loader import get_available_profiles
        available = get_available_profiles()
        if prof not in available:
            issues.append(Issue("error", "profile", f"Unknown profile '{prof}'. Available: {', '.join(available)}"))

    return issues
