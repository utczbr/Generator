"""
parity_v3_v4.py

Differential check to run BEFORE retiring annotation schema v3.0.

It generates the same charts (same seeds) twice - once with schema v3.0 and once with v4.0 -
and reports:

1. Invariants: every file other than ``*_detailed.json`` (images, YOLO labels, ...) must be
   byte-identical. If not, the schema flag perturbs generation and flipping the default would
   change your images, not just the JSON.
2. Element parity: each v3 element (per-class lists) is matched by IoU against the v4
   ``annotations`` list. Matching is geometric on purpose, so it does not depend on class names.
3. Metadata parity: chart type, domain, series, stacking, style/pattern, baselines.

Run it with realism effects OFF to check parity, and ON to quantify coordinate drift: v3 builds
some boxes from the live figure (pre-effect) and others from post-effect annotations, whereas v4
carries everything through the effects. A large drop in the 'placed' rate with effects ON is that drift.

Usage (from the repo root, with your real config available):
    python scripts/parity_v3_v4.py --n 25 --effects both --strict
"""
import argparse
import contextlib
import copy
import hashlib
import io
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

# Canonical per-class keys of the v3 ``_detailed.json`` (legacy aliases such as "scalelabels"
# are deliberately excluded so elements are not double counted).
V3_ELEMENT_KEYS = (
    "scale_labels", "tick_labels", "chart_title", "axis_title", "legend", "bar", "data_point",
    "error_bar", "significance_marker", "data_label", "box", "median_line", "range_indicator",
    "outlier", "wedge", "line_segment", "area_boundary", "cell", "color_bar", "color_bar_label",
    "color_bar_title", "connector_line",
)


def iou(a, b):
    """Intersection-over-union of two [x0, y0, x1, y1] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(A, B):
    """Pairwise IoU between two (n, 4) and (m, 4) arrays of xyxy boxes."""
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    union = area(A)[:, None] + area(B)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _v4_annotations(v4):
    return [a for a in v4.get("annotations", []) if isinstance(a, dict) and a.get("xyxy")]


def match_elements(v3, v4, loose=0.5, tight=0.8):
    """One-to-one (greedy, highest IoU first) match of v3 elements to v4 annotations, per element type.

    One-to-one matters: with best-IoU-against-anything, a drifted bar or heatmap cell can "match"
    its neighbour and hide the drift. ``loose`` counts as found; ``tight`` counts as correctly
    placed. Returns ({key: stats}, number of v4 annotations no v3 element claimed).
    """
    anns = _v4_annotations(v4)
    B = [a["xyxy"] for a in anns]
    claimed, report = set(), {}
    for key in V3_ELEMENT_KEYS:
        items = [e for e in (v3.get(key) or []) if isinstance(e, dict) and e.get("xyxy")]
        if not items:
            continue
        best = np.zeros(len(items))
        assigned = {}
        if B:
            M = iou_matrix([e["xyxy"] for e in items], B)
            rows, cols = np.nonzero(M > 0)
            used_r, used_c = set(), set()
            for k in np.argsort(-M[rows, cols], kind="stable"):
                r, c = int(rows[k]), int(cols[k])
                if r in used_r or c in used_c:
                    continue
                used_r.add(r); used_c.add(c)
                best[r] = M[r, c]; assigned[r] = c
        hit = [r for r in assigned if best[r] >= loose]
        classes = Counter(anns[assigned[r]].get("class_name") for r in hit)
        claimed.update(assigned[r] for r in hit)
        report[key] = {"n_v3": len(items), "loose": len(hit), "tight": int((best >= tight).sum()),
                       "median_iou": float(np.median(best)), "min_iou": float(best.min()),
                       "v4_classes": dict(classes.most_common(3))}
        report[key]["loose_frac"] = report[key]["loose"] / len(items)
        report[key]["tight_frac"] = report[key]["tight"] / len(items)
    return report, len(anns) - len(claimed)


def compare_metadata(v3, v4):
    """Compare v3 top-level metadata with v4 root / primary-subplot fields."""
    sub = (v4.get("subplots") or [{}])[0]
    anns = v4.get("annotations", [])
    v4_series = sub.get("series")
    rows = [
        ("chart_type", v3.get("chart_type"), v4.get("chart_type")),
        ("orientation", v3.get("orientation"), sub.get("orientation")),
        ("semantic_domain", v3.get("semantic_domain"), v4.get("semantic_domain")),
        ("is_scientific", v3.get("is_scientific"), v4.get("is_scientific")),
        ("is_composite", v3.get("is_composite"), v4.get("is_composite")),
        ("composite_chart_types", v3.get("composite_chart_types") or [], v4.get("composite_chart_types") or []),
        ("series_count", v3.get("series_count", 1), len(v4_series) if v4_series else 1),
        ("series_names", list(v3.get("series_names") or []),
         [s.get("name") for s in v4_series] if v4_series else []),
        ("stacking_mode", v3.get("stacking_mode"), sub.get("stacking_mode")),
        ("style", v3.get("style"), sub.get("style")),
        ("pattern", v3.get("pattern"), sub.get("pattern")),
        ("n_baselines", len(v3.get("baselines") or []),
         sum(1 for a in anns if a.get("class_name") == "baseline")),
        ("n_bars_with_baseline", len(v3.get("bars_with_baseline") or []),
         sum(1 for a in anns if (a.get("attrs") or {}).get("baseline_id") is not None)),
    ]
    out = []
    for name, a, b in rows:
        if a == b:
            status = "match"
        elif b in (None, [], {}) and a not in (None, [], {}):
            status = "absent-in-v4"
        else:
            status = "differs"
        out.append({"field": name, "v3": a, "v4": b, "status": status})
    return out


def compare(v3, v4, loose=0.5, tight=0.8):
    elements, v4_unmatched = match_elements(v3, v4, loose, tight)
    ids = [a.get("id") for a in v4.get("annotations", [])]
    return {"elements": elements, "v4_unmatched": v4_unmatched,
            "metadata": compare_metadata(v3, v4),
            "v4_duplicate_ids": len(ids) - len(set(ids))}


def aggregate(reports):
    """Merge per-image reports into one per-key summary."""
    agg, meta = {}, Counter()
    for r in reports:
        for key, e in r["elements"].items():
            a = agg.setdefault(key, {"n_v3": 0, "loose": 0, "tight": 0, "min_iou": 1.0, "v4_classes": Counter()})
            a["n_v3"] += e["n_v3"]; a["loose"] += e["loose"]; a["tight"] += e["tight"]
            a["min_iou"] = min(a["min_iou"], e["min_iou"])
            a["v4_classes"].update(e["v4_classes"])
        for m in r["metadata"]:
            meta[(m["field"], m["status"])] += 1
    for a in agg.values():
        a["loose_frac"], a["tight_frac"] = a["loose"] / a["n_v3"], a["tight"] / a["n_v3"]
    return agg, meta


def format_summary(agg, meta, v4_unmatched, label, loose=0.5, tight=0.8):
    lines = [f"\n=== element parity [{label}]  found = IoU>={loose}, placed = IoU>={tight} ==="]
    for key, a in sorted(agg.items(), key=lambda kv: kv[1]["tight_frac"]):
        top = ", ".join(f"{c}:{n}" for c, n in a["v4_classes"].most_common(2))
        lines.append(f"  {key:20s} n={a['n_v3']:<5d} found {a['loose_frac']:6.1%}  placed {a['tight_frac']:6.1%}  min IoU {a['min_iou']:.2f}  -> {top}")
    lines.append(f"  v4 annotations with no v3 counterpart (expected: polygons, keypoints, graph entries): {v4_unmatched}")
    lines.append(f"=== metadata parity [{label}] ===")
    for field in dict.fromkeys(f for f, _ in meta):
        n = {s: meta[(field, s)] for s in ("match", "differs", "absent-in-v4") if meta[(field, s)]}
        lines.append(f"  {field:22s} {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# driver (needs the real generator + config; not exercised by the unit tests)
# --------------------------------------------------------------------------- #
def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _tree(root):
    return {os.path.relpath(os.path.join(d, f), root): os.path.join(d, f)
            for d, _, fs in os.walk(root) for f in fs}


def diff_trees(root_a, root_b):
    """Return (only_in_a, only_in_b, differing) for every file except *_detailed.json."""
    a, b = _tree(root_a), _tree(root_b)
    skip = lambda p: p.endswith("_detailed.json")
    ka, kb = {k for k in a if not skip(k)}, {k for k in b if not skip(k)}
    return (sorted(ka - kb), sorted(kb - ka),
            sorted(k for k in ka & kb if _sha(a[k]) != _sha(b[k])))


def _generate(n, cfg, schema, root, verbose):
    import generator as G
    cfg = copy.deepcopy(cfg)
    cfg["annotation_schema_version"] = cfg["detailed_schema"] = schema
    cfg["num_images"] = n
    cfg.setdefault("debug_mode", False)
    images, labels = os.path.join(root, "images"), os.path.join(root, "labels")
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    sink = sys.stdout if verbose else io.StringIO()
    for i in range(n):
        with contextlib.redirect_stdout(sink):
            _, ok, err = G.generate_single_chart_task((i, cfg, images, labels, root))
        if not ok:
            raise RuntimeError(f"{schema} generation failed on image {i}: {err}")


def run(n, effects, loose=0.5, tight=0.8, verbose=False):
    import json
    import generator as G
    base = copy.deepcopy(G.GENERATION_CONFIG)
    if effects == "off":
        base["realism_effects"] = {}
    with tempfile.TemporaryDirectory() as tmp:
        r3, r4 = os.path.join(tmp, "v3"), os.path.join(tmp, "v4")
        _generate(n, base, "v3.0", r3, verbose)
        _generate(n, base, "v4.0", r4, verbose)
        only3, only4, differing = diff_trees(r3, r4)
        reports = []
        for rel, p3 in sorted(_tree(r3).items()):
            if rel.endswith("_detailed.json") and os.path.exists(os.path.join(r4, rel)):
                with open(p3) as f3, open(os.path.join(r4, rel)) as f4:
                    reports.append(compare(json.load(f3), json.load(f4), loose, tight))
    return reports, only3, only4, differing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=25, help="charts per run")
    ap.add_argument("--effects", choices=("on", "off", "both"), default="both")
    ap.add_argument("--loose-iou", type=float, default=0.5, help="IoU for 'found'")
    ap.add_argument("--tight-iou", type=float, default=0.8, help="IoU for 'correctly placed' (uncalibrated default)")
    ap.add_argument("--min-placed", type=float, default=0.95,
                    help="--strict: minimum 'placed' rate per element type with effects off (uncalibrated default)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on invariant failure / low parity")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    failed = False
    for mode in (("off", "on") if args.effects == "both" else (args.effects,)):
        reports, only3, only4, differing = run(args.n, mode, args.loose_iou, args.tight_iou, args.verbose)
        agg, meta = aggregate(reports)
        print(format_summary(agg, meta, sum(r["v4_unmatched"] for r in reports), f"effects {mode}", args.loose_iou, args.tight_iou))
        inv_ok = not (only3 or only4 or differing)
        print(f"=== invariants [effects {mode}]: non-JSON files identical across schemas: {inv_ok}")
        for label, items in (("only in v3 run", only3), ("only in v4 run", only4), ("bytes differ", differing)):
            for it in items[:5]:
                print(f"     {label}: {it}")
        dup = sum(r["v4_duplicate_ids"] for r in reports)
        print(f"=== v4 duplicate annotation ids: {dup}")
        failed |= not inv_ok or dup > 0
        if mode == "off":
            failed |= any(a["tight_frac"] < args.min_placed for a in agg.values())
    if args.strict and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
