"""
synth/semantics/admin/normalize.py

Label normalization and concept stem generator (ADM-02, ADM-03).
Enforces:
- H4: Stored label normalization: NFC, collapse whitespace, NBSP -> space, U+00B5 -> U+03BC.
- H4: Dedup key: casefold(NFKC(collapse_whitespace(label))).
- B4: make_stem(label): NFKD ASCII-fold, non-alphanumerics -> '_', collapsed and trimmed.
      Matches pattern ^[a-z0-9_]+$.
"""
import re
import unicodedata

_WS_COLLAPSE_RE = re.compile(r"[\s\u00A0]+")
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9_]+")
_UNDERSCORES_RE = re.compile(r"_+")
_UNIT_PAREN_RE = re.compile(r"\s*\(.*?\)\s*")


def collapse_whitespace(text: str) -> str:
    """Replace non-breaking spaces and collapse any whitespace runs to a single space."""
    return _WS_COLLAPSE_RE.sub(" ", text).strip()


def normalize_label(label: str) -> str:
    """
    Format stored label string (H4):
    - Replaces non-breaking spaces (U+00A0) with standard space
    - Replaces micro sign (U+00B5) with Greek small letter mu (U+03BC)
    - Collapses whitespace
    - Normalizes to Unicode NFC without losing superscripts or subscripts (e.g. kg/m² preserved)
    """
    if not label:
        return ""
    # Replace micro sign U+00B5 with greek mu U+03BC before normalization
    cleaned = label.replace("\u00a0", " ").replace("\u00b5", "\u03bc")
    collapsed = collapse_whitespace(cleaned)
    return unicodedata.normalize("NFC", collapsed)


def make_dedup_key(label: str) -> str:
    """
    Compute collision dedup key (H4):
    Uses NFKC compatibility decomposition and casefolding so that
    different unicode representations of equivalent text (e.g., µM and μM) collide.
    """
    normalized = normalize_label(label)
    nfkc = unicodedata.normalize("NFKC", normalized)
    return collapse_whitespace(nfkc).casefold()


def strip_unit_suffix(label: str) -> str:
    """Strip parenthetical unit expressions for fuzzy similarity comparison (H3)."""
    return _UNIT_PAREN_RE.sub(" ", label).strip()


def make_stem(label: str) -> str:
    """
    Compute standardized concept_stem conforming to ^[a-z0-9_]+$ (ADM-03, B4):
    - Strips parenthetical units
    - Decomposes accents via NFKD and strips non-ASCII combining marks
    - Replaces any non-alphanumeric character with underscore
    - Collapses multiple underscores and trims leading/trailing underscores
    - Returns empty string if no valid alphanumeric characters remain
    """
    if not label:
        return ""

    # Strip units first so 'Voltage (mV)' -> 'Voltage'
    base = strip_unit_suffix(label)
    if not base:
        base = label

    # ASCII-fold via NFKD decomposition
    nfkd = unicodedata.normalize("NFKD", base)
    ascii_bytes = nfkd.encode("ascii", "ignore")
    ascii_str = ascii_bytes.decode("ascii").lower()

    # Convert non-alphanumeric chars to underscore
    underscored = _NON_ALPHANUM_RE.sub("_", ascii_str)
    stem = _UNDERSCORES_RE.sub("_", underscored).strip("_")
    return stem
