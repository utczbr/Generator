#!/usr/bin/env python3
"""
scripts/env_fingerprint.py

Deterministic environment fingerprint for reproducible chart generation (FR-079, FR-081).
Records OS, Python version, Matplotlib version, FreeType build, and font file sha256 checksums.
"""

import hashlib
import json
import os
import platform
import sys
from typing import Dict, Any

try:
    import matplotlib
    import matplotlib.font_manager as fm
    from matplotlib import ft2font
except ImportError as e:
    matplotlib = None
    fm = None
    ft2font = None


def _hash_file(filepath: str) -> str:
    """Return SHA256 hex digest of a file if it exists."""
    if not os.path.isfile(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_env_fingerprint() -> Dict[str, Any]:
    """Capture environment fingerprint as a deterministic dictionary."""
    data: Dict[str, Any] = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "matplotlib": {
            "version": matplotlib.__version__ if matplotlib else None,
            "freetype_version": getattr(ft2font, "__freetype_version__", None) if ft2font else None,
            "freetype_build": getattr(ft2font, "__freetype_build_type__", None) if ft2font else None,
        },
        "font_files": {},
    }

    if fm:
        core_fonts = [
            "DejaVu Sans",
            "DejaVu Sans:weight=bold",
            "DejaVu Serif",
            "DejaVu Sans Mono",
        ]
        for font_spec in core_fonts:
            try:
                font_path = fm.findfont(font_spec)
                data["font_files"][font_spec] = {
                    "path": font_path,
                    "sha256": _hash_file(font_path),
                }
            except Exception as e:
                data["font_files"][font_spec] = {"error": str(e)}

    # Compute aggregate hash of the normalized JSON string
    serialized = json.dumps(data, sort_keys=True)
    data["fingerprint_hash"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return data


def main():
    fingerprint = get_env_fingerprint()
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        out_file = sys.argv[1]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2, sort_keys=True)
        print(f"Fingerprint written to {out_file}")
    else:
        print(json.dumps(fingerprint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
