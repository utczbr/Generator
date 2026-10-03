"""
scripts/generate_baseline_snapshot.py

Generate baseline vocabulary distribution snapshot across 1,000 charts
under a balanced 8-chart mix ({bar: 150, line: 150, scatter: 150, box: 150, area: 150, histogram: 100, pie: 100, heatmap: 50})
prior to vocabulary modifications, saving to tests/golden/baseline_semantic_distribution.json.
"""
import copy
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart


CHART_MIX = {
    "bar": 150,
    "line": 150,
    "scatter": 150,
    "box": 150,
    "area": 150,
    "histogram": 100,
    "pie": 100,
    "heatmap": 50,
}


def _generate_worker(item):
    idx, chart_type, root_tmp = item
    run_dir = os.path.join(root_tmp, f"run_{idx:05d}")
    img_dir = os.path.join(run_dir, "images")
    lbl_dir = os.path.join(run_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["realism_effects"] = {}
    cfg["debug_mode"] = False
    cfg["num_images"] = 1

    for k in cfg["chart_types"]:
        cfg["chart_types"][k]["enabled"] = (k == chart_type)
        cfg["chart_types"][k]["weight"] = 1.0 if k == chart_type else 0.0

    generate_single_chart(idx, cfg, img_dir, lbl_dir, run_dir)

    det_path = os.path.join(lbl_dir, f"chart_{idx:05d}_detailed.json")
    with open(det_path, "r", encoding="utf-8") as f:
        det = json.load(f)

    # Extract info
    title = None
    titles = det.get("chart_title", [])
    if titles and isinstance(titles, list) and "text" in titles[0]:
        title = titles[0]["text"]

    x_labels = []
    y_labels = []
    for at in det.get("axis_title", []):
        axis = at.get("axis", "x")
        txt = at.get("text")
        if txt:
            if axis == "x":
                x_labels.append(txt)
            else:
                y_labels.append(txt)

    is_sci = det.get("is_scientific", False)

    # Clean up worker temp dir to conserve disk space
    shutil.rmtree(run_dir, ignore_errors=True)

    return {
        "chart_type": chart_type,
        "title": title,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "is_scientific": is_sci,
    }


def main():
    start_time = time.time()
    golden_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")
    os.makedirs(golden_dir, exist_ok=True)
    out_file = os.path.join(golden_dir, "baseline_semantic_distribution.json")

    items = []
    curr_idx = 0
    for ctype, count in CHART_MIX.items():
        for _ in range(count):
            items.append((curr_idx, ctype))
            curr_idx += 1

    total = len(items)
    assert total == 1000, f"Expected 1000 items, got {total}"
    print(f"Generating {total} baseline charts across 4 worker processes...")

    with tempfile.TemporaryDirectory() as root_tmp:
        worker_items = [(idx, ctype, root_tmp) for idx, ctype in items]
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(_generate_worker, worker_items, chunksize=10))

    title_counts = Counter()
    x_label_counts = Counter()
    y_label_counts = Counter()
    chart_type_counts = Counter()
    domain_counts = Counter()

    for r in results:
        chart_type_counts[r["chart_type"]] += 1
        if r["title"]:
            title_counts[r["title"]] += 1
        for xl in r["x_labels"]:
            x_label_counts[xl] += 1
        for yl in r["y_labels"]:
            y_label_counts[yl] += 1
        if r["is_scientific"]:
            domain_counts["scientific"] += 1
        else:
            domain_counts["business"] += 1

    snapshot = {
        "total_charts": total,
        "chart_type_counts": dict(chart_type_counts),
        "title_frequencies": dict(title_counts.most_common()),
        "x_label_frequencies": dict(x_label_counts.most_common()),
        "y_label_frequencies": dict(y_label_counts.most_common()),
        "domain_counts": dict(domain_counts),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Baseline snapshot saved to {out_file} in {elapsed:.1f}s.")
    print(f"Total charts: {total}")
    print(f"Unique titles observed: {len(title_counts)}")
    print(f"Unique X labels observed: {len(x_label_counts)}")
    print(f"Unique Y labels observed: {len(y_label_counts)}")
    print(f"Domain counts: {dict(domain_counts)}")


if __name__ == "__main__":
    main()
