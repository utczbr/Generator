"""
tests/test_p2_tabular_synth.py

Unit tests for Phase 7 (P2) Realistic Tabular Data Synthesis & Multivariate Distributions:
- Multivariate correlation fidelity via Gaussian copula and native Cholesky decomposition.
- Strict physical bounds and non-negativity without post-hoc clipping.
- Domain-consistent semantic schemas with zero cross-domain label bleeding.
- End-to-end integration into Matplotlib and Vega-Lite generation pipelines.
"""
import copy
import json
import os
import numpy as np
import pytest
import scipy.stats as stats

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart
from backends.vegalite_backend import generate_single_vegalite_chart
from synth.tabular import (
    DOMAIN_PRESETS,
    sample_coherent_labels,
    sample_domain_schema,
    sample_dose_response_series,
    sample_gaussian_copula,
    sample_multivariate_table,
    transform_to_marginal,
)


def test_multivariate_correlation_fidelity_and_bounds():
    """
    Verify that Gaussian copula with Cholesky factorization preserves target correlation
    within statistical tolerance (error < 0.08) and guarantees physical bounds via PIT.
    """
    rng = np.random.default_rng(42)
    target_rho = 0.75
    R = np.array([
        [1.0, target_rho, -0.60],
        [target_rho, 1.0, -0.45],
        [-0.60, -0.45, 1.0],
    ])

    num_rows = 6000
    table = sample_multivariate_table(
        num_rows=num_rows,
        num_series=3,
        domain="business",
        correlation_spec=R,
        rng=rng,
    )

    data = table["data"]
    assert data.shape == (num_rows, 3)

    # 1. Check empirical correlation fidelity against target R
    # Spearman rank correlation invariant under monotonic marginal PIT
    for i in range(3):
        for j in range(i + 1, 3):
            spearman_corr, _ = stats.spearmanr(data[:, i], data[:, j])
            target_val = R[i, j]
            # Gaussian copula Spearman correlation rho_s = (6/pi) * arcsin(rho/2)
            expected_spearman = (6.0 / np.pi) * np.arcsin(target_val / 2.0)
            diff = abs(spearman_corr - expected_spearman)
            assert diff < 0.08, (
                f"Correlation fidelity mismatch for pair ({i}, {j}): "
                f"empirical={spearman_corr:.3f}, expected={expected_spearman:.3f}, diff={diff:.3f}"
            )

    # 2. Check strict bounds: no clipping, pure analytical PIT
    # Financial metrics are lognormal -> strictly positive (> 0)
    for col_idx in range(3):
        col = data[:, col_idx]
        assert np.all(col > 0.0), f"Series {col_idx} violated non-negativity constraint!"
        assert np.all(np.isfinite(col)), f"Series {col_idx} contains non-finite values!"

    # 3. Test beta distribution bounded within [0, 100]
    u = rng.uniform(0.001, 0.999, size=1000)
    beta_vals = transform_to_marginal(u, dist_type="beta", params={"a": 3.0, "b": 4.0, "scale": 100.0}, bounds=(0.0, 100.0))
    assert np.all(beta_vals >= 0.0) and np.all(beta_vals <= 100.0)


def test_dose_response_curves_fidelity_and_bounds():
    """Verify multi-series dose-response curves generate realistic sigmoidal responses with positive constraints."""
    rng = np.random.default_rng(123)
    num_points = 18
    num_series = 3
    dr = sample_dose_response_series(num_points=num_points, num_series=num_series, rng=rng)

    assert dr["data"].shape == (num_points, num_series)
    assert len(dr["doses"]) == num_points
    assert dr["x_label"] == "Concentration (log M)"
    assert dr["y_label"] == "Response (%)"
    assert dr["title"] == "Dose–Response Curve"

    # Verify all response values are strictly positive and finite
    assert np.all(dr["data"] > 0.0)
    assert np.all(np.isfinite(dr["data"]))

    # Verify concentration points span orders of magnitude (pM to mM range)
    assert dr["doses"][0] < dr["doses"][-1]
    assert np.isclose(dr["log_conc"][0], -9.0)
    assert np.isclose(dr["log_conc"][-1], -4.0)


def test_domain_schema_coherence_no_cross_domain_bleeding():
    """Verify that sampled schemas, axis titles, and metric categories never bleed across domains."""
    rng = np.random.default_rng(999)

    # Sample 1000 schemas across all presets and verify 100% domain purity
    for dom in DOMAIN_PRESETS.keys():
        allowed_x = set(DOMAIN_PRESETS[dom]["x_labels"])
        allowed_titles = set(DOMAIN_PRESETS[dom]["titles"])
        allowed_metrics = {m["name"] for m in DOMAIN_PRESETS[dom]["metrics"]}

        for _ in range(250):
            schema = sample_domain_schema(domain=dom, rng=rng)
            assert schema["domain"] == dom

            # The sampled items must be in the current domain's allowed vocabularies
            assert schema["x_label"] in allowed_x
            assert schema["title"] in allowed_titles
            for m in schema["metrics"]:
                assert m["name"] in allowed_metrics

            # Assert absence of bleed from distinct domains
            for other_dom, other_preset in DOMAIN_PRESETS.items():
                if other_dom == dom:
                    continue
                distinct_x = set(other_preset["x_labels"]) - allowed_x
                distinct_titles = set(other_preset["titles"]) - allowed_titles
                assert schema["x_label"] not in distinct_x
                assert schema["title"] not in distinct_titles

    # Test sample_coherent_labels with scientific ratio constraints
    for _ in range(100):
        sci_labels = sample_coherent_labels(is_scientific=True, rng=rng)
        assert sci_labels["domain"] in ("biomedical", "engineering")

        biz_labels = sample_coherent_labels(is_scientific=False, rng=rng)
        assert biz_labels["domain"] in ("business", "demographic")


def test_matplotlib_pipeline_integration_synthetic_data_engine(tmp_path):
    """Verify Matplotlib chart generation with use_synthetic_data_engine=True across all major chart types."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["use_synthetic_data_engine"] = True
    cfg["debug_mode"] = False
    cfg["num_images"] = 1
    cfg["engine"] = "matplotlib"
    cfg["annotation_schema_version"] = "v3.0"

    run_dir = str(tmp_path / "matplotlib_synth_test")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    chart_types_to_test = ["bar", "line", "scatter", "area"]

    for idx, ctype in enumerate(chart_types_to_test):
        # Configure chart type specifically
        for k in cfg["chart_types"]:
            cfg["chart_types"][k]["enabled"] = (k == ctype)
            cfg["chart_types"][k]["weight"] = 1.0 if k == ctype else 0.0

        res = generate_single_chart(idx, cfg, img_dir, lbl_dir, run_dir)
        base_name = f"chart_{idx:05d}"

        # 1. Check generated PNG image exists
        png_path = os.path.join(img_dir, f"{base_name}.png")
        assert os.path.isfile(png_path), f"Expected PNG file not found at {png_path}"
        assert os.path.getsize(png_path) > 0

        # 2. Check YOLO labels exist
        txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
        assert os.path.isfile(txt_path), f"Expected YOLO labels file not found at {txt_path}"

        # 3. Check detailed.json exists and contains valid structure
        json_path = os.path.join(lbl_dir, f"{base_name}_detailed.json")
        assert os.path.isfile(json_path), f"Expected detailed JSON not found at {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            det_data = json.load(f)
            assert "axis_title" in det_data
            assert len(det_data["axis_title"]) > 0


def test_vegalite_pipeline_integration_synthetic_data_engine(tmp_path):
    """Verify Vega-Lite vector graphics generation with use_synthetic_data_engine=True."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["use_synthetic_data_engine"] = True
    cfg["engine"] = "vegalite"
    cfg["vegalite_scale"] = 1.5

    run_dir = str(tmp_path / "vegalite_synth_test")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for idx, ctype in enumerate(["bar", "line", "scatter"]):
        cfg["chart_type"] = ctype
        cfg["bar_style"] = "grouped" if ctype == "bar" else "standard"
        cfg["synthetic_domain"] = "business" if idx % 2 == 0 else "biomedical"

        res = generate_single_vegalite_chart(idx, cfg, img_dir, lbl_dir, run_dir)
        base_name = f"chart_{idx:05d}"

        png_path = os.path.join(img_dir, f"{base_name}.png")
        assert os.path.isfile(png_path)
        assert os.path.getsize(png_path) > 0

        txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
        assert os.path.isfile(txt_path)
        assert os.path.getsize(txt_path) > 0

        det_path = os.path.join(lbl_dir, f"{base_name}_detailed.json")
        assert os.path.isfile(det_path)
        with open(det_path, "r", encoding="utf-8") as f:
            det = json.load(f)
            assert det["chart_type"] == ctype
            assert len(det["resolution"]) == 2
