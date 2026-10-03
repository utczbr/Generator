"""
tests/benchmarks/benchmark_backends.py

Standalone throughput and latency benchmark suite for Phase 8:
Compares Matplotlib (analytical visibility) vs. Vega-Lite (SVG DOM) backends,
and Parametric sampling vs. Synthetic Tabular Copula sampling across batch sizes.
Outputs summary table and writes tests/benchmarks/benchmark_results.json.
"""
import copy
import io
import json
import os
import resource
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart
import backends.vegalite_backend as vb
from synth.tabular import sample_multivariate_table


def get_current_rss_mb() -> float:
    """Return resident set size in megabytes."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024.0 * 1024.0)


def benchmark_tabular_sampling(batch_size: int) -> Dict[str, Any]:
    """Compare parametric random sampling vs copula-based tabular synthesis."""
    # 1. Parametric baseline
    t0 = time.perf_counter()
    for _ in range(batch_size):
        _ = np.random.uniform(10.0, 100.0, size=(20, 3))
    t_parametric = time.perf_counter() - t0

    # 2. Copula synthesis
    t0 = time.perf_counter()
    for _ in range(batch_size):
        _ = sample_multivariate_table(num_rows=20, num_series=3, domain="financial")
    t_copula = time.perf_counter() - t0

    return {
        "batch_size": batch_size,
        "parametric_seconds": round(t_parametric, 4),
        "parametric_throughput_cps": round(batch_size / max(t_parametric, 1e-6), 2),
        "copula_seconds": round(t_copula, 4),
        "copula_throughput_cps": round(batch_size / max(t_copula, 1e-6), 2),
        "overhead_multiplier": round(t_copula / max(t_parametric, 1e-6), 2),
    }


def profile_chart_breakdown(engine: str, n_samples: int = 20) -> Dict[str, float]:
    """Measure rendering vs. annotation/DOM extraction breakdown."""
    render_times = []
    extract_times = []

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = n_samples
    cfg["debug_mode"] = False
    cfg["engine"] = engine
    # Test across canonical bar chart
    cfg["chart_type"] = "bar"
    cfg["chart_types"] = {"bar": {"weight": 100, "enabled": True}}

    if engine == "vegalite":
        spec = vb.build_vegalite_bar_spec(
            categories=["A", "B", "C", "D", "E"],
            values=[25.0, 40.0, 15.0, 60.0, 35.0],
            title="Profile Benchmark",
        )
        for _ in range(n_samples):
            # Render phase
            t0 = time.perf_counter()
            svg_content = vb.vlc.vegalite_to_svg(spec)
            _ = vb.vlc.vegalite_to_png(spec, scale=2.0)
            t_render = time.perf_counter() - t0
            render_times.append(t_render)

            # Extraction phase
            t0 = time.perf_counter()
            _ = vb.extract_svg_annotations(svg_content, scale_factor=2.0, chart_type="bar")
            t_extract = time.perf_counter() - t0
            extract_times.append(t_extract)
    else:
        import matplotlib.pyplot as plt
        from chart import _generate_bar_chart
        from generator import get_granular_annotations, get_detailed_annotations

        theme_cfg = {"colors": ["#1f77b4"], "background_color": "#ffffff", "font_family": "DejaVu Sans"}
        style_cfg = {"style": "standard", "orientation": "vertical", "is_scientific": False, "pattern": "none"}

        for _ in range(n_samples):
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            t0 = time.perf_counter()
            _generate_bar_chart(ax, "default", theme_cfg, style_cfg, debug_mode=False)
            fig.tight_layout(pad=2.0)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            t_render = time.perf_counter() - t0
            render_times.append(t_render)

            t0 = time.perf_counter()
            chart_info_map = {ax: {"chart_type_str": "bar", "data_artists": ax.patches}}
            cls_map = {0: "chart", 1: "bar", 2: "axis_title", 5: "legend", 6: "chart_title", 8: "axis_labels"}
            _ = get_granular_annotations(fig, chart_info_map, cls_map)
            t_extract = time.perf_counter() - t0
            extract_times.append(t_extract)
            plt.close(fig)

    mean_render = float(np.mean(render_times))
    mean_extract = float(np.mean(extract_times))
    total = mean_render + mean_extract
    return {
        "avg_render_ms": round(mean_render * 1000.0, 2),
        "avg_extract_ms": round(mean_extract * 1000.0, 2),
        "render_pct": round((mean_render / max(total, 1e-9)) * 100.0, 1),
        "extract_pct": round((mean_extract / max(total, 1e-9)) * 100.0, 1),
    }


def run_benchmark_batch(engine: str, batch_size: int, use_synthetic: bool = False) -> Dict[str, Any]:
    """Run an end-to-end batch generation benchmark for a backend."""
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = batch_size
    cfg["debug_mode"] = False
    cfg["engine"] = engine
    cfg["use_synthetic_data_engine"] = use_synthetic

    rss_start = get_current_rss_mb()
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp_dir:
        images_dir = os.path.join(tmp_dir, "images")
        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for i in range(batch_size):
            generate_single_chart(i, cfg, images_dir, labels_dir, tmp_dir)

    wall_clock = time.perf_counter() - t_start
    rss_end = get_current_rss_mb()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux KB -> MB

    throughput_cps = round(batch_size / max(wall_clock, 1e-6), 2)
    sec_per_100 = round((wall_clock / batch_size) * 100.0, 2)

    return {
        "engine": engine,
        "use_synthetic": use_synthetic,
        "batch_size": batch_size,
        "wall_clock_sec": round(wall_clock, 2),
        "sec_per_100": sec_per_100,
        "throughput_cps": throughput_cps,
        "rss_start_mb": round(rss_start, 1),
        "rss_end_mb": round(rss_end, 1),
        "peak_rss_mb": round(peak_rss, 1),
    }


def main():
    print("=" * 80)
    print("  PHASE 8: VECTOR & MATPLOTLIB BACKEND BENCHMARK SUITE")
    print("=" * 80)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": os.uname().sysname if hasattr(os, "uname") else "Unknown",
        "cpu_count": os.cpu_count(),
        "breakdown": {},
        "tabular_benchmarks": [],
        "batch_benchmarks": [],
    }

    # 1. Profile rendering vs extraction breakdown
    print("\n[1/3] Profiling Rendering vs. Annotation/DOM Extraction...")
    mpl_breakdown = profile_chart_breakdown("matplotlib", n_samples=25)
    vlc_breakdown = profile_chart_breakdown("vegalite", n_samples=25)
    results["breakdown"]["matplotlib"] = mpl_breakdown
    results["breakdown"]["vegalite"] = vlc_breakdown
    print(f"  Matplotlib : Render {mpl_breakdown['avg_render_ms']}ms ({mpl_breakdown['render_pct']}%) | Extract {mpl_breakdown['avg_extract_ms']}ms ({mpl_breakdown['extract_pct']}%)")
    print(f"  Vega-Lite  : Render {vlc_breakdown['avg_render_ms']}ms ({vlc_breakdown['render_pct']}%) | DOM Extract {vlc_breakdown['avg_extract_ms']}ms ({vlc_breakdown['extract_pct']}%)")

    # 2. Benchmark tabular sampling (parametric vs copula)
    print("\n[2/3] Benchmarking Tabular Data Engines (Parametric vs. Copula)...")
    for b_size in [50, 200]:
        tab_res = benchmark_tabular_sampling(b_size)
        results["tabular_benchmarks"].append(tab_res)
        print(f"  Batch {b_size:3d}: Parametric {tab_res['parametric_throughput_cps']:>8.1f} tables/s | Copula {tab_res['copula_throughput_cps']:>8.1f} tables/s (Overhead: {tab_res['overhead_multiplier']}x)")

    # 3. End-to-end batches across 50 and 200 charts
    print("\n[3/3] Profiling End-to-End Generation Across Batch Sizes (50, 200)...")
    configs = [
        ("matplotlib", False, "Matplotlib (Analytical)"),
        ("vegalite", False, "Vega-Lite (Vector DOM)"),
        ("vegalite", True, "Vega-Lite + Copula Synth"),
    ]

    for batch_size in [50, 200]:
        print(f"\n  --- Batch Size: {batch_size} Charts ---")
        for engine, syn, label in configs:
            print(f"  Executing {label:30s} ... ", end="", flush=True)
            res = run_benchmark_batch(engine, batch_size, use_synthetic=syn)
            results["batch_benchmarks"].append(res)
            print(f"{res['throughput_cps']:>6.2f} charts/sec ({res['sec_per_100']:>5.2f}s/100) | Peak RSS: {res['peak_rss_mb']:.1f} MB")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Engine / Configuration':<30} | {'Batch':<6} | {'Charts/sec':<11} | {'s/100 charts':<13} | {'Peak RSS':<9}")
    print("-" * 80)
    for res in results["batch_benchmarks"]:
        name = f"{res['engine']}" + (" + copula" if res["use_synthetic"] else "")
        print(f"{name:<30} | {res['batch_size']:<6} | {res['throughput_cps']:>10.2f} | {res['sec_per_100']:>12.2f}s | {res['peak_rss_mb']:>7.1f}MB")
    print("=" * 80)

    # Save JSON report
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark report successfully written to: {out_path}\n")


if __name__ == "__main__":
    main()
