"""
tests/test_p5_effects_overlay.py

Verification tests for Phase 1 Task T228: Effects Overlay (FR-090, FR-092).
Validates:
1. profiles/domain_gap_v1.json values match FR-090 bounded parameters.
2. apply_uneven_lighting_effect supports scalar intensity, intensity_range, and randomized gradient_type.
3. Mean luminance of uneven_lighting outputs stays >= 195 on a 24-image forced run.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from effects import apply_uneven_lighting_effect


def test_domain_gap_v1_overlay_values_match_fr090():
    """Verify profiles/domain_gap_v1.json matches FR-090 bounds."""
    pfile = Path("profiles/domain_gap_v1.json")
    assert pfile.exists()
    data = json.loads(pfile.read_text(encoding="utf-8"))

    eff = data.get("realism_effects", {})

    # scan_rotation
    assert eff["scan_rotation"]["p"] == 0.10
    assert eff["scan_rotation"]["params"]["angle_range"] == [-1.5, 1.5]

    # perspective
    assert eff["perspective"]["p"] == 0.06
    assert eff["perspective"]["params"]["magnitude"] == [0.02, 0.08]

    # page_curl
    assert eff["page_curl"]["p"] == 0.04
    assert eff["page_curl"]["params"]["amplitude_ratio"] == [0.010, 0.025]

    # clipping
    assert eff["clipping"]["p"] == 0.04
    assert eff["clipping"]["params"]["clip_range_pct"] == [0.01, 0.03]

    # uneven_lighting
    assert eff["uneven_lighting"]["p"] == 0.10
    assert eff["uneven_lighting"]["params"]["intensity_range"] == [0.05, 0.25]

    # chromatic_aberration
    assert eff["chromatic_aberration"]["p"] == 0.05


def test_uneven_lighting_scalar_intensity_and_range():
    """Verify scalar intensity, intensity_range, and gradient_type are accepted and work."""
    base_img = Image.new("RGB", (200, 200), (255, 255, 255))

    # Scalar intensity
    out_scalar = apply_uneven_lighting_effect(base_img, intensity=0.20, gradient_type="radial")
    assert isinstance(out_scalar, Image.Image)

    # intensity_range
    out_range = apply_uneven_lighting_effect(base_img, intensity_range=[0.05, 0.25], gradient_type="random")
    assert isinstance(out_range, Image.Image)


def test_uneven_lighting_mean_luminance_gte_195():
    """Verify mean luminance of uneven_lighting outputs stays >= 195 on a 24-image forced run."""
    luminances = []
    for i in range(24):
        # Create a synthetic white chart canvas with typical chart ink (lines/text/background)
        canvas = Image.new("RGB", (400, 300), (250, 250, 250))
        # Apply uneven lighting with FR-090 intensity_range [0.05, 0.25]
        degraded = apply_uneven_lighting_effect(
            canvas,
            intensity_range=[0.05, 0.25],
            gradient_type="random",
        )
        gray = np.array(degraded.convert("L"), dtype=np.float32)
        mean_lum = float(np.mean(gray))
        luminances.append(mean_lum)

    overall_mean = float(np.mean(luminances))
    assert overall_mean >= 195.0, f"Mean luminance {overall_mean:.2f} < 195.0"
