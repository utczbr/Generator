"""
font_registry.py

Deterministic bundled font management and glyph validation (Phase 1, T234, FR-089, OD-1).

Provides:
- Deterministic registration of bundled fonts under `assets/fonts/` via `font_manager.addfont`.
- Open-licensed font families covering sans-serif, serif, and monospace.
- Glyph coverage validation for the required scientific/math character inventory:
  Latin, digits, µ μ ± ° α β ² ³ ⁻ × ≥ ≤ Δ.
- Alias resolution table mapping system names (Arial, Helvetica, Times New Roman, Courier New)
  to bundled metric-compatible fonts.
- Strict error raising in non-legacy profiles for missing fonts or glyphs.
- Fast import time (lazy registration) to respect import-guard benchmarks (NFR-C).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
MANIFEST_PATH = FONTS_DIR / "manifest.json"

CHARACTER_INVENTORY = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "µμ±°αβ²³⁻×≥≤Δ"
)

# Metric-compatible alias mapping for standard journal/system typography (FR-089, N-17)
FONT_ALIASES: Dict[str, str] = {
    "Arial": "Arimo",
    "Helvetica": "Arimo",
    "Liberation Sans": "Arimo",
    "Calibri": "Arimo",
    "Montserrat": "Arimo",
    "Roboto": "Arimo",
    "DejaVu Sans": "Arimo",
    "Times New Roman": "Tinos",
    "Times": "Tinos",
    "Georgia": "DejaVu Serif",
    "Liberation Serif": "Tinos",
    "Courier New": "DejaVu Sans Mono",
    "Courier": "DejaVu Sans Mono",
    "Liberation Mono": "DejaVu Sans Mono",
}

_FONTS_REGISTERED: bool = False
_MANIFEST_CACHE: Optional[Dict[str, Any]] = None


class FontRegistryError(Exception):
    """Base exception for font registry issues."""


class MissingFontError(FontRegistryError):
    """Raised when a requested font is missing in non-legacy profiles."""


class MissingGlyphError(FontRegistryError):
    """Raised when a font lacks required glyphs in non-legacy profiles."""


def get_bundled_manifest() -> Dict[str, Any]:
    """Load and cache the bundled font manifest."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    if not MANIFEST_PATH.exists():
        raise FontRegistryError(f"Font manifest missing at {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        _MANIFEST_CACHE = json.load(f)
    return _MANIFEST_CACHE


def ensure_fonts_registered(fonts_dir: Optional[Path] = None) -> List[str]:
    """
    Register all bundled fonts deterministically via matplotlib font_manager.addfont.
    Loads lazily on first access to preserve fast module import latency.
    """
    global _FONTS_REGISTERED
    target_dir = Path(fonts_dir) if fonts_dir else FONTS_DIR
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        raise FontRegistryError(f"Font manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    from matplotlib import font_manager as fm

    registered_paths: List[str] = []
    # Register in sorted order of filename for deterministic registry population
    for font_entry in sorted(manifest.get("fonts", []), key=lambda entry: entry["file"]):
        font_path = str(target_dir / font_entry["file"])
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            registered_paths.append(font_path)

    _FONTS_REGISTERED = True
    return registered_paths


def resolve_font_name(font_name: str, strict: bool = False) -> str:
    """
    Resolve font name through alias table to bundled family.
    If strict=True, validates that the resolved name is present in bundled fonts.
    """
    resolved = FONT_ALIASES.get(font_name, font_name)
    if strict:
        manifest = get_bundled_manifest()
        bundled_families = {entry["family"] for entry in manifest.get("fonts", [])}
        if resolved not in bundled_families:
            raise MissingFontError(
                f"Font '{font_name}' (resolved to '{resolved}') not in bundled fonts: {sorted(bundled_families)}"
            )
    return resolved


def check_glyph_coverage(font_path: str, chars: str = CHARACTER_INVENTORY) -> Tuple[bool, List[str]]:
    """
    Check if a font file contains all glyphs in chars.
    Returns (True, []) if all glyphs are present, or (False, missing_chars).
    """
    from matplotlib.ft2font import FT2Font

    font = FT2Font(font_path)
    charmap = font.get_charmap()
    missing = [c for c in chars if ord(c) not in charmap]
    return (len(missing) == 0, missing)


def get_bundled_families(category: Optional[str] = None) -> List[str]:
    """
    Return sorted list of distinct font family names from the bundled registry,
    optionally filtered by category ('sans-serif', 'serif', 'monospace').
    """
    manifest = get_bundled_manifest()
    families = set()
    for entry in manifest.get("fonts", []):
        if category is None or entry.get("category") == category:
            families.add(entry["family"])
    return sorted(families)


def sample_bundled_font(
    rng: Any,
    category: Optional[str] = None,
    domain: str = "scientific",
) -> Dict[str, Any]:
    """
    Sample a font family deterministically using a feature-scoped RNG.
    Never samples ttflist (D1-14).
    """
    manifest = get_bundled_manifest()
    if category is None:
        if domain == "scientific":
            category = rng.choices(["sans-serif", "serif"], weights=[0.7, 0.3], k=1)[0]
        else:
            category = rng.choices(["sans-serif", "serif"], weights=[0.8, 0.2], k=1)[0]

    families = get_bundled_families(category=category)
    if not families:
        families = get_bundled_families()

    chosen_family = rng.choice(families)
    return {
        "family": chosen_family,
        "category": category,
        "font_name": chosen_family,
    }


def validate_font_and_glyphs(
    font_name: str,
    active_profile: Optional[str] = None,
    chars: str = CHARACTER_INVENTORY,
) -> str:
    """
    Validate that font_name exists in bundled registry and contains required glyphs.
    In non-legacy profiles, raises MissingFontError or MissingGlyphError.
    In legacy profiles, performs lenient fallback without raising.
    Returns the resolved font family name.
    """
    resolved = resolve_font_name(font_name)
    if active_profile is None or active_profile == "legacy":
        return resolved

    manifest = get_bundled_manifest()
    matching_entries = [e for e in manifest.get("fonts", []) if e["family"] == resolved]
    if not matching_entries:
        raise MissingFontError(
            f"Font '{font_name}' (resolved to '{resolved}') is not in bundled fonts. Non-legacy profile '{active_profile}' requires bundled fonts."
        )

    for entry in matching_entries:
        font_path = str(FONTS_DIR / entry["file"])
        ok, missing = check_glyph_coverage(font_path, chars=chars)
        if not ok:
            raise MissingGlyphError(
                f"Font '{resolved}' ({entry['file']}) is missing required glyphs in profile '{active_profile}': {missing}"
            )

    return resolved


def get_distinct_rendered_faces(fonts_dir: Optional[Path] = None) -> Set[str]:
    """
    Verify distinct rendered faces resolved by matplotlib findfont across all bundled fonts.
    """
    from matplotlib import font_manager as fm

    ensure_fonts_registered(fonts_dir)
    manifest = get_bundled_manifest()
    rendered: Set[str] = set()

    for entry in manifest.get("fonts", []):
        fam = entry["family"]
        style = entry.get("style", "normal")
        weight = entry.get("weight", "normal")
        prop = fm.FontProperties(family=fam, style=style, weight=weight)
        resolved = fm.findfont(prop, fallback_to_default=True)
        rendered.add(os.path.realpath(resolved))

    return rendered
