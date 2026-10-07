#!/usr/bin/env python3
"""Reproduce the measurements cited in specs/005-domain-gap-closure/research.md.

Usage (run from anywhere; nothing is written into the repo):
    REPO=/path/to/Generator python repro_005.py <experiment> [--n 24]

Experiments
    sizes        E1  output size / aspect-ratio distribution under default config
    figsize      E2  layout damage for non-default figsize/DPI (errors, tight_layout
                     warnings, text collisions, empty text boxes)
    ink          E4  per-class pixel "ink ratio" + what flat/class gates would flag
    effects      E5  force each realism effect on; check annotations stay pixel-aligned
    ticks        E7  tick-label attachment for rotation x horizontal-alignment
    fonts        E8  what the configured font names really render as + glyph coverage
    multi-axes   E9  scenario=multi 2x1: which half of the image carries annotations
    projection   E10 share of text boxes that MIN_BBOX_SIZE=8 would drop after downscale
    legend       E14 legend presence per chart type + whether `legend._ncol` changes layout
    streams      E15 line charts: labels/ vs line_obj_labels/ class ids, cross-stream
                     duplicates in v4.0 (same class_name, IoU >= 0.8)

    margins      E11 framing prior: per-side margin of the annotation union / ink box
    rng          E12 does a new default `realism_effects` key (p=0) change seeded output?
    reframe      E13 canvas_reframe integrity + p sweep (needs the effect, T254; else prints a notice)
    legend-markers E16 decomposed-legend extractor: marker box extents / ink per handle type
    dpi          E17 throughput of native high-DPI rendering vs the legacy DPI set

All experiments seed per image (seed + i) and switch realism effects off unless the
experiment itself turns one on, so runs are comparable across conditions.
"""
import argparse, contextlib, copy, glob, hashlib, io, itertools, json, logging, os, random, sys
import tempfile, time, warnings

REPO = os.path.abspath(os.environ.get("REPO", "."))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from config_defaults import OCR_TRAINING_CONFIG  # noqa: E402
import generator as G  # noqa: E402

TEXT = {"axis_title", "axis_labels", "chart_title", "legend", "data_label"}


# ----------------------------------------------------------------------------- core
def make_cfg(n, seed=7, effects_off=True, chart_types=None):
    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg.update(num_images=n, seed=seed, debug_mode=False, use_parallel=False)
    if effects_off:
        for v in cfg["realism_effects"].values():
            v["p"] = 0.0
    if chart_types:
        for k in cfg["chart_types"]:
            cfg["chart_types"][k]["weight"] = chart_types.get(k, 0)
    return cfg


def run(cfg, n, figsize=None, dpi=None, layout=None, seed=7):
    """Generate n images. figsize/dpi override the single-chart (7,5)/[96,120,150]
    choice *consistently* (figure dpi == savefig dpi); layout forces the scenario=multi grid."""
    out = tempfile.mkdtemp()
    imgs, lbls = os.path.join(out, "images"), os.path.join(out, "labels")
    os.makedirs(imgs), os.makedirs(lbls)
    orig_sub, orig_choice = plt.subplots, random.choice

    def sub(*a, **k):
        if figsize is not None and k.get("figsize") == (7, 5):
            k = {**k, "figsize": figsize}
        return orig_sub(*a, **k)

    def choice(seq):
        s = list(seq)
        if dpi is not None and s == [96, 120, 150]:
            return dpi
        if layout is not None and (1, 2) in s and (2, 1) in s:
            return layout
        return orig_choice(seq)

    plt.subplots, random.choice = sub, choice
    errors = tl_warn = 0
    try:
        for i in range(n):
            random.seed(seed + i)
            np.random.seed(seed + i)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        G.generate_single_chart(i, cfg, imgs, lbls, out)
                except Exception:
                    errors += 1
                tl_warn += sum("tight_layout" in str(x.message) for x in w)
            plt.close("all")
    finally:
        plt.subplots, random.choice = orig_sub, orig_choice
    return out, tl_warn, errors


def load(out):
    det = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "labels", "*_detailed.json")))]
    paths = sorted(glob.glob(os.path.join(out, "images", "*.png")))
    return det, [Image.open(p).size for p in paths], paths


def ink_ratio(img, box):
    W, H = img.size
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    arr = np.asarray(img.convert("L")).astype(int)
    bg = np.median(np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]))
    return float((np.abs(arr[y0:y1, x0:x1] - bg) > 25).mean())


def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


# ----------------------------------------------------------------------- experiments
def e_sizes(n):
    out, _, err = run(make_cfg(n), n)
    det, sizes, _ = load(out)
    print(f"{len(sizes)} images, {err} errors")
    print("distinct (w,h):", sorted(set(sizes)))
    print("aspect ratios:", sorted({round(w / h, 2) for w, h in sizes}))


def stats(det, sizes, paths):
    ann = text_n = coll = empty = 0
    for d, ip in zip(det, paths):
        img = Image.open(ip)
        T = [a for a in d["annotations"] if a["class_name"] in TEXT]
        ann += len(d["annotations"])
        text_n += len(T)
        for a, b in itertools.combinations(T, 2):
            m = min(area(a["xyxy"]), area(b["xyxy"]))
            if m > 0 and inter(a["xyxy"], b["xyxy"]) / m > 0.3:
                coll += 1
        for a in T:
            r = ink_ratio(img, a["xyxy"])
            empty += r is not None and r < 0.02
    k = max(1, len(det))
    return ann / k, text_n / k, coll / k, 100 * empty / max(1, text_n)


def e_figsize(n):
    conds = [("A baseline 7x5 @96/120/150", None, None), ("B 4x3 @72", (4, 3), 72),
             ("C 5.5x4.1 @80", (5.5, 4.1), 80), ("D 16:9 5.5x3.1 @80", (5.5, 3.09), 80),
             ("E 3:4 5.5x7.3 @80", (5.5, 7.33), 80), ("F 10x7.5 @300", (10, 7.5), 300)]
    print(f"{'condition':28s} {'err':>3} {'tl':>3} {'ann/img':>8} {'text/img':>9} {'coll/img':>9} {'%empty text':>12}")
    for name, fs, dp in conds:
        out, tl, err = run(make_cfg(n), n, figsize=fs, dpi=dp)
        det, sizes, paths = load(out)
        a, t, c, e = stats(det, sizes, paths)
        print(f"{name:28s} {err:3d} {tl:3d} {a:8.1f} {t:9.1f} {c:9.2f} {e:12.1f}")


def e_ink(n):
    cfg = make_cfg(n, chart_types={k: 1 for k in ("bar", "box", "line", "scatter", "area", "histogram")})
    out, _, _ = run(cfg, n)
    det, _, paths = load(out)
    byc = {}
    for d, ip in zip(det, paths):
        img = Image.open(ip)
        for a in d["annotations"]:
            r = ink_ratio(img, a["xyxy"])
            if r is not None:
                byc.setdefault(a["class_name"], []).append(r)
    tot = sum(len(v) for v in byc.values())
    unk = len(byc.get("unknown", []))
    print(f"annotations: {tot}; class_name=='unknown': {unk} ({100 * unk / tot:.1f}%)")
    allr = np.concatenate([np.array(v) for v in byc.values()])
    print(f"flat 0.30 gate flags {100 * (allr < 0.3).mean():.1f}% of a CLEAN baseline")
    print(f"{'class':18s} {'n':>5} {'p5':>6} {'median':>7} {'%<0.30':>7}")
    for c, v in sorted(byc.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 8:
            v = np.array(v)
            print(f"{c:18s} {len(v):5d} {np.percentile(v, 5):6.2f} {np.median(v):7.2f} {100 * (v < 0.3).mean():7.1f}")
    for c, t in (("bar", .5), ("axis_labels", .1), ("line_segment", .02), ("data_point", .01)):
        v = np.array(byc.get(c, []))
        if len(v):
            print(f"  proposed thr {c}={t}: false-flag {100 * (v < t).mean():.1f}%")


def e_effects(n):
    cases = [("baseline", {}),
             ("scan_rotation +-1.5", {"scan_rotation": {"p": 1, "params": {"angle_range": [-1.5, 1.5]}}}),
             *[(f"perspective {m}", {"perspective": {"p": 1, "params": {"magnitude": m}}}) for m in (.05, .15, .3, .5)],
             ("page_curl amp .03", {"page_curl": {"p": 1, "params": {"curl_axis": "y", "amplitude_ratio": .03, "wavelength_ratio": 1.}}}),
             ("clipping 1-4%", {"clipping": {"p": 1, "params": {"clip_range_pct": [.01, .04]}}}),
             ("pdf_document_context", {"pdf_document_context": {"p": 1, "params": {}}}),
             ("uneven_lighting default", {"uneven_lighting": {"p": 1, "params": {}}}),
             ("uneven_lighting 0.25", {"uneven_lighting": {"p": 1, "params": {"intensity": .25}}}),
             ("chromatic_aberration", {"chromatic_aberration": {"p": 1, "params": {}}}),
             ("grid_occlusion", {"grid_occlusion": {"p": 1, "params": {}}})]
    print(f"{'effect':26s} {'err':>3} {'ann/img':>8} {'text n':>7} {'%empty(<5% ink)':>16} {'mean lum':>9}")
    for name, eff in cases:
        cfg = make_cfg(n)
        cfg["realism_effects"].update(eff)
        out, _, err = run(cfg, n)
        det, sizes, paths = load(out)
        empty = tn = 0
        for d, ip in zip(det, paths):
            img = Image.open(ip)
            for a in d["annotations"]:
                if a["class_name"] in TEXT:
                    r = ink_ratio(img, a["xyxy"])
                    if r is not None:
                        tn += 1
                        empty += r < .05
        lum = np.mean([np.asarray(Image.open(p).convert("L")).mean() for p in paths])
        print(f"{name:26s} {err:3d} {sum(len(d['annotations']) for d in det) / max(1, len(det)):8.1f} "
              f"{tn:7d} {100 * empty / max(1, tn):16.1f} {lum:9.1f}")


def e_ticks(_n):
    def gap(rot, ha, mode):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        ax.bar([0], [3]); ax.set_xticks([0]); ax.set_xticklabels(["Treatment group A"])
        ax.set_xlim(-1, 1); ax.set_yticks([])
        ax.tick_params(axis="x", length=0, labelrotation=rot)
        for s in ax.spines.values():
            s.set_visible(False)
        for l in ax.get_xticklabels():
            l.set_horizontalalignment(ha); l.set_rotation_mode(mode)
        fig.subplots_adjust(bottom=.35); fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        ink = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int).sum(axis=2) < 600
        ink[:H - int(ax.get_window_extent().y0) + 1, :] = False
        top = np.where(ink.any(axis=1))[0].min()
        xs = np.where(ink[top:top + 3].any(axis=0))[0]
        tx = ax.transData.transform((0, 0))[0]
        plt.close(fig)
        return xs.mean() - tx
    print("px between tick and the label end nearest the axis (|gap| small = visually attached)")
    print(f"{'rot':>5} | {'center/default':>14} {'right/default':>14} {'left/default':>13} {'right/anchor':>13} {'left/anchor':>12}")
    for rot in (30, 45, 60, -30, -45, -60):
        c = [gap(rot, "center", "default"), gap(rot, "right", "default"), gap(rot, "left", "default"),
             gap(rot, "right", "anchor"), gap(rot, "left", "anchor")]
        print(f"{rot:5d} | " + " ".join(f"{x:13.1f} " for x in c))

    print("\nUnder new rule (FR-088: ha by angle sign, default rotation_mode):")
    print(f"{'rot':>5} | {'ha':>8} {'gap (px)':>10} {'|gap| <= 15':>12}")
    for rot in (30, 45, 60, -30, -45, -60, 90, -90):
        ha = "center" if abs(rot) in (0, 90) else ("right" if rot > 0 else "left")
        g = gap(rot, ha, "default")
        print(f"{rot:5d} | {ha:>8} {g:10.1f} {str(abs(g) <= 15):>12}")


def e_fonts(_n):
    import string
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import FT2Font
    names = [f for fam in ("sans-serif", "serif") for f in __import__("themes").FONT_FAMILIES[fam]]
    res = {n: os.path.basename(fm.findfont(fm.FontProperties(family=n), fallback_to_default=True)) for n in names}
    for n, f in res.items():
        print(f"{n:18s} -> {f}")
    print("distinct rendered faces:", len(set(res.values())), "of", len(names), "sampled names")
    fams = {}
    for f in fm.fontManager.ttflist:
        fams.setdefault(f.name, f.fname)
    basic, sci = string.ascii_letters + string.digits, "µ±°αβ"
    cnt = {"no_latin": 0, "no_sci": 0, "safe": 0}
    for fname in fams.values():
        try:
            cm = FT2Font(fname).get_charmap()
        except Exception:
            cnt["no_latin"] += 1; continue
        cnt["no_latin" if any(ord(c) not in cm for c in basic) else
            "no_sci" if any(ord(c) not in cm for c in sci) else "safe"] += 1
    print(f"ttflist families: {len(fams)} -> {cnt}  (environment-dependent!)")


def e_multi_axes(n):
    cfg = make_cfg(n, chart_types={"bar": 1})
    cfg["scenario_weights"] = {"single": 0, "multi": 100}
    out, _, err = run(cfg, n, layout=(2, 1))
    det, sizes, _ = load(out)
    top = bot = 0
    for d, (W, H) in zip(det, sizes):
        for a in d["annotations"]:
            if (a["xyxy"][1] + a["xyxy"][3]) / 2 < H / 2:
                top += 1
            else:
                bot += 1
    print(f"2x1 composite, {len(det)} images: annotations top half={top}, bottom half={bot}")


def e_projection(n):
    cfg = make_cfg(n, chart_types={k: 1 for k in ("bar", "box", "line", "scatter", "area", "histogram")})
    out, _, _ = run(cfg, n)
    det, _, _ = load(out)
    m = np.array([min(a["xyxy"][2] - a["xyxy"][0], a["xyxy"][3] - a["xyxy"][1])
                  for d in det for a in d["annotations"] if a["class_name"] in TEXT])
    print(f"text boxes n={len(m)}; min-side p5/p50/p95 = {np.percentile(m, 5):.0f}/{np.percentile(m, 50):.0f}/{np.percentile(m, 95):.0f}px")
    print("(upper bound: scatter/box/heatmap are exempt from the size filter, generator.py:4057-4076)")
    for s in (1, .75, .5, .35, .25):
        print(f"  downscale {s:<4}: {100 * (m * s < 8).mean():5.1f}% of text boxes fall under MIN_BBOX_SIZE=8")


def e_legend(n):
    """E14: which chart types ever carry a legend, and whether legend._ncol does anything."""
    out, _, err = run(make_cfg(n), n)
    det, _, _ = load(out)
    per = {}
    for d in det:
        has = any(a["class_name"] == "legend" for a in d["annotations"])
        t = per.setdefault(d.get("chart_type"), [0, 0])
        t[0] += has
        t[1] += 1
    print(f"{len(det)} images, {err} errors; legend present per chart type:")
    for k, (h, t) in sorted(per.items(), key=lambda kv: str(kv[0])):
        print(f"  {k:10s} {h}/{t}")
    fig, ax = plt.subplots()
    for i in range(8):
        ax.plot([0, 1], [i, i], label=f"s{i}")
    r = fig.canvas.get_renderer()
    lg = ax.legend()
    fig.canvas.draw()
    a = lg.get_window_extent(r)
    lg._ncol = 2
    fig.canvas.draw()
    b = lg.get_window_extent(r)
    c = ax.legend(ncols=2)
    fig.canvas.draw()
    c = c.get_window_extent(r)
    print(f"matplotlib {matplotlib.__version__}: 8-entry legend {a.width:.0f}x{a.height:.0f} px; "
          f"after _ncol=2 {b.width:.0f}x{b.height:.0f}; ax.legend(ncols=2) {c.width:.0f}x{c.height:.0f}")
    plt.close("all")


def e_streams(n):
    """E15: line charts — class-id spaces of labels/ vs line_obj_labels/, and v4.0 duplicates."""
    out, _, err = run(make_cfg(n, chart_types={"line": 100}), n)
    det, _, _ = load(out)
    ids = {}
    for sub in ("labels", "line_obj_labels"):
        c = {}
        for p in glob.glob(os.path.join(out, sub, "*.txt")):
            for ln in open(p):
                if ln.strip():
                    c[int(ln.split()[0])] = c.get(int(ln.split()[0]), 0) + 1
        ids[sub] = dict(sorted(c.items()))
    print(f"{len(det)} line images, {err} errors")
    for k, v in ids.items():
        print(f"  {k:16s} class_id -> count {v}")
    dups = only_obj = 0
    for d in det:
        anns = d["annotations"]
        for i, a in enumerate(anns):
            for b in anns[i + 1:]:
                if a["class_name"] == b["class_name"] and a["stream"] != b["stream"]:
                    u = area(a["xyxy"]) + area(b["xyxy"]) - inter(a["xyxy"], b["xyxy"])
                    if u > 0 and inter(a["xyxy"], b["xyxy"]) / u >= 0.8:
                        dups += 1
        prim = {a["class_name"] for a in anns if a["stream"] == "annotations"}
        only_obj += sum(1 for a in anns if a["stream"] == "annotations_obj"
                        and a["class_name"] in TEXT and a["class_name"] not in prim)
    print(f"  cross-stream same-name pairs with IoU>=0.8: {dups}")
    print(f"  text annotations present only in annotations_obj: {only_obj}")


# ------------------------------------------------------------ framing (E11-E13)
def _ink_bbox(img, tol=25):
    a = np.asarray(img.convert("L")).astype(int)
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]))
    ys, xs = np.where(np.abs(a - bg) > tol)
    return None if len(xs) == 0 else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def _union_margins(det, paths):
    """Per image: distance of the annotation union from L,T,R,B as a fraction of W/H."""
    rows = []
    for d, p in zip(det, paths):
        W, H = Image.open(p).size
        bx = [a["xyxy"] for a in d["annotations"]]
        if bx:
            rows.append([min(b[0] for b in bx) / W, min(b[1] for b in bx) / H,
                         (W - max(b[2] for b in bx)) / W, (H - max(b[3] for b in bx)) / H])
    return np.array(rows)


def _margin_line(name, M):
    mn, mx = M.min(axis=1), M.max(axis=1)
    sd = [round(float(c.std()), 3) for c in M.T]
    print(f"{name:34s} n={len(M):3d} per-side std(L,T,R,B)={sd} tightest<1%: {100 * (mn < .01).mean():3.0f}% "
          f"loosest>6%: {100 * (mx > .06).mean():3.0f}% median(L,T,R,B)={[round(float(np.median(c)), 3) for c in M.T]}")


def e_margins(n):
    """E11: baseline margins are constants; pdf_document_context never gets tight."""
    out, _, err = run(make_cfg(n), n)
    det, _, paths = load(out)
    _margin_line(f"baseline (err={err})", _union_margins(det, paths))
    cfg = make_cfg(n)
    cfg["realism_effects"]["pdf_document_context"] = {"p": 1, "params": {}}
    out, _, err = run(cfg, n)
    det, _, paths = load(out)
    _margin_line(f"pdf_document_context forced (err={err})", _union_margins(det, paths))


def e_rng(n):
    """E12: one global RNG draw per `realism_effects` key. Sequential seeding, like the real CLI."""
    def seq(cfg, k, seed=7):
        out = tempfile.mkdtemp()
        imgs, lbls = os.path.join(out, "images"), os.path.join(out, "labels")
        os.makedirs(imgs), os.makedirs(lbls)
        random.seed(seed), np.random.seed(seed)
        for i in range(k):
            with contextlib.redirect_stdout(io.StringIO()):
                G.generate_single_chart(i, cfg, imgs, lbls, out)
            plt.close("all")
        h = hashlib.sha256()
        for p in sorted(glob.glob(os.path.join(lbls, "*.txt"))):
            h.update(open(p, "rb").read())
        return h.hexdigest()[:16]
    k = min(n, 6)
    base = make_cfg(k, effects_off=False)
    h0, h0b = seq(copy.deepcopy(base), k), seq(copy.deepcopy(base), k)
    G.EFFECT_REGISTRY["_probe"] = lambda img, **kw: img
    try:
        h_reg = seq(copy.deepcopy(base), k)                      # registry entry only
        c = copy.deepcopy(base)
        c["realism_effects"]["_probe"] = {"p": 0.0, "params": {}}
        h_key = seq(c, k)                                         # + default key at p = 0
    finally:
        del G.EFFECT_REGISTRY["_probe"]
    print(f"baseline {h0} (repeat {h0b}: deterministic={h0 == h0b})")
    print(f"registry entry only : {h_reg}  identical={h_reg == h0}")
    print(f"+ default key, p=0.0: {h_key}  identical={h_key == h0}")


def e_reframe(n):
    """E13: canvas_reframe integrity (runs only once the effect exists, T254)."""
    if "canvas_reframe" not in G.EFFECT_REGISTRY:
        print("canvas_reframe is not registered yet (T254) - nothing to measure.")
        return
    base_out, _, _ = run(make_cfg(n), n)
    bdet, _, bpaths = load(base_out)
    base_counts = sum(len(d["annotations"]) for d in bdet)

    def stats(name, eff):
        cfg = make_cfg(n)
        cfg["realism_effects"].update(eff)
        out, _, err = run(cfg, n)
        det, sizes, paths = load(out)
        tn = empty = obb_n = obb_out = touch = ann_n = 0
        for d, p in zip(det, paths):
            img = Image.open(p)
            W, H = img.size
            for a in d["annotations"]:
                ann_n += 1
                x0, y0, x1, y1 = a["xyxy"]
                touch += x0 <= .5 or y0 <= .5 or x1 >= W - .5 or y1 >= H - .5
                if a.get("obb"):
                    obb_n += 1
                    obb_out += any(x < -.5 or y < -.5 or x > W + .5 or y > H + .5 for x, y in a["obb"])
                if a["class_name"] in TEXT:
                    r = ink_ratio(img, a["xyxy"])
                    if r is not None:
                        tn += 1
                        empty += r < .05
        M = _union_margins(det, paths)
        sd = [round(float(c.std()), 3) for c in M.T]
        print(f"{name:34s} err={err} kept={100 * ann_n / max(1, base_counts):5.1f}% empty_text={100 * empty / max(1, tn):4.1f}% "
              f"obb_outside={obb_out}/{obb_n} edge_touch={100 * touch / max(1, ann_n):4.1f}% "
              f"aspects={len({round(w / h, 2) for w, h in sizes})} std={sd}")

    comp = {"scan_rotation": {"p": 1, "params": {"angle_range": [-1.5, 1.5]}},
            "perspective": {"p": 1, "params": {"magnitude": .05}},
            "page_curl": {"p": 1, "params": {"curl_axis": "y", "amplitude_ratio": .02, "wavelength_ratio": 1.}}}
    rf = lambda p=1.0, **kw: {"canvas_reframe": {"p": p, "params": kw}}
    stats("baseline", {})
    stats("reframe default, p=1", rf())
    stats("tight-only, zero slack", rf(p_tight=1.0, tight_px=[0, 0]))
    stats("negative control (use_guard False)", rf(p_tight=1.0, tight_px=[0, 0], use_guard=False))
    stats("composed, no reframe", comp)
    stats("composed + reframe, zero slack", {**comp, **rf(p_tight=1.0, tight_px=[0, 0])})
    for p in (.3, .5, .8):
        stats(f"p sweep p={p}", rf(p))


# ------------------------------------------------ legends / resolution (E16-E17)
def e_legend_markers(n):
    """E16: box extent and ink ratio of legend handles, by handle type (extractor of research 9.2)."""
    import chart as C
    def boxes(ax, r, img_h):
        lg = ax.get_legend()
        return [(h.get_window_extent(r), t.get_text()) for t, h in zip(lg.get_texts(), lg.legend_handles)]
    res = {"line, no marker": [], "line, marker": [], "area patch": []}
    for kind in res:
        for s in range(max(10, n)):
            random.seed(s), np.random.seed(s)
            fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
            k, x = random.randint(2, 4), np.linspace(0, 1, 20)
            for i in range(k):
                if kind == "area patch":
                    ax.fill_between(x, 0, np.random.rand(20) + i * .1, alpha=.6, label=f"S{i}")
                else:
                    ax.plot(x, np.random.rand(20) + i, lw=random.choice([1., 1.5, 2.]),
                            marker="o" if kind == "line, marker" else None, label=f"S{i}")
            C.apply_legend_variation(ax, k)
            fig.tight_layout(), fig.canvas.draw()
            H = fig.canvas.get_width_height()[1]
            arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int).mean(axis=2)
            bg = np.median(np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]))
            for b, _ in boxes(ax, fig.canvas.get_renderer(), H):
                x0, y0, x1, y1 = [int(round(v)) for v in (b.x0, H - b.y1, b.x1, H - b.y0)]
                w, h = b.width, b.height
                ink = None if (x1 <= x0 or y1 <= y0) else float((np.abs(arr[max(0, y0):y1, max(0, x0):x1] - bg) > 25).mean())
                res[kind].append((w, h, ink))
            plt.close(fig)
    for kind, v in res.items():
        h = np.array([t[1] for t in v])
        i = [t[2] for t in v if t[2] is not None]
        print(f"{kind:16s} n={len(v):3d} height px p50={np.median(h):5.1f}  height<2px: {100 * (h < 2).mean():3.0f}%  "
              f"measurable ink: {len(i)}/{len(v)}  median ink={np.median(i) if i else float('nan'):.2f}")


def e_dpi(n):
    """E17: throughput of native high-DPI rendering (FR-085b) vs the legacy DPI set."""
    k = min(n, 12)
    for label, dpi in (("legacy 120 dpi", 120), ("native ~2400 px (340 dpi)", 340)):
        t = time.time()
        out, _, err = run(make_cfg(k), k, dpi=dpi)
        dt = time.time() - t
        _, sizes, _ = load(out)
        print(f"{label:26s} {k} imgs in {dt:5.1f}s = {k / dt:4.2f} img/s, err={err}, max size {max(sizes)}")


EXPERIMENTS = {"sizes": e_sizes, "figsize": e_figsize, "ink": e_ink, "effects": e_effects,
               "ticks": e_ticks, "fonts": e_fonts, "multi-axes": e_multi_axes, "projection": e_projection,
               "legend": e_legend, "streams": e_streams, "margins": e_margins, "rng": e_rng,
               "reframe": e_reframe, "legend-markers": e_legend_markers, "dpi": e_dpi}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    EXPERIMENTS[a.experiment](a.n)
