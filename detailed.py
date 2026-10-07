"""
detailed.py

Builds the single-list ``<image_id>_detailed.json`` (annotation schema v4.0).

Design rules
------------
* ONE list of annotations (``annotations``) carries every element, its pixel
  geometry (final image frame, top-left origin) and its data attributes.
  There are no parallel per-class lists, no legacy aliases, no second copy.
* Per-figure information lives in ``subplots`` (one record per visible axes).
  Root scalars are limited to what the spec mandates (primary subplot).
* Everything that has pixel coordinates is produced BEFORE realism effects and
  carried through them by the existing annotation transform machinery, so the
  whole file is in one coordinate frame (the saved image).

Two phases
----------
1. ``prepare_detail_annotations``  (before ``apply_realism_effects``)
     - tags annotations with ``subplot`` and per-element ``attrs``
     - creates "graph" pseudo-annotations (baselines, line/area keypoint series)
2. ``build_detailed_json``          (after effects / filtering)
     - merges all streams, resolves class names per stream, dedupes,
       assigns ids, resolves links, builds ``subplots``.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Stream = (name, annotation_list, class_map, kind)
#   kind "mat"   : ``bbox`` is a matplotlib-frame box (y up); polygons/obb/keypoints image frame
#   kind "pose"  : normalized (cx, cy, w, h) bbox + normalized keypoints (image frame)
#   kind "graph" : pseudo-annotations with pixel keypoints only
Stream = Tuple[str, List[Dict[str, Any]], Optional[Dict[Any, str]], str]

_SCI_DOMAINS = frozenset({"biomedical", "engineering"})
_NON_SCI_DOMAINS = frozenset({"business", "demographic"})
_BAR_CLASS = "bar"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def derive_is_scientific(domain: Optional[str], fallback: Any = False) -> bool:
    """``is_scientific`` is a function of the semantic domain (single source of truth)."""
    if domain:
        try:
            from synth.semantics.loader import registry
            if domain in registry.domains:
                return registry.domains[domain].is_scientific
        except Exception:
            pass
    if domain in _SCI_DOMAINS:
        return True
    if domain in _NON_SCI_DOMAINS:
        return False
    return bool(fallback)


def to_py(v: Any) -> Any:
    """numpy / non-finite safe conversion to plain JSON types."""
    if isinstance(v, dict):
        return {str(k): to_py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_py(x) for x in v]
    if isinstance(v, np.ndarray):
        return to_py(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if v is None or isinstance(v, (int, str)):
        return v
    return None  # artists, callables, ... are never serialisable metadata


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def _finite(*vals: float) -> bool:
    return all(math.isfinite(float(v)) for v in vals)


def _extents(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) from Bbox / BoundingBox namedtuple / tuple / list / dict.

    NOTE: ``.x0`` is tested first on purpose. ``BoundingBox`` is a namedtuple, so
    an ``isinstance(..., tuple)`` test placed first would treat it as a plain tuple.
    """
    if bbox is None:
        return None
    try:
        if hasattr(bbox, "x0") and hasattr(bbox, "y1"):
            return float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)
        if hasattr(bbox, "extents"):
            e = bbox.extents
            return float(e[0]), float(e[1]), float(e[2]), float(e[3])
        if isinstance(bbox, dict):
            return (float(bbox["x0"]), float(bbox["y0"]), float(bbox["x1"]), float(bbox["y1"]))
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _clip_xyxy(xyxy: Sequence[float], w: float, h: float) -> Optional[List[float]]:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    if not _finite(x0, y0, x1, y1):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None  # entirely outside the image
    x0, x1 = max(0.0, x0), min(float(w), x1)
    y0, y1 = max(0.0, y0), min(float(h), y1)
    if x1 - x0 < 1.0:  # thin lines (baselines, medians) keep >= 1 px extent
        x1 = min(float(w), x0 + 1.0)
        x0 = max(0.0, x1 - 1.0)
    if y1 - y0 < 1.0:
        y1 = min(float(h), y0 + 1.0)
        y0 = max(0.0, y1 - 1.0)
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _pts(points: Any) -> Optional[List[List[float]]]:
    if not points:
        return None
    out = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if _finite(x, y):
            out.append([round(x, 2), round(y, 2)])
    return out or None


def _envelope(points: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def is_numeric_text(text: str) -> bool:
    """True for '12', '-3.5', '1,200', '45%', and typographic-minus values ('−5')."""
    s = str(text).strip().replace("\u2212", "-").replace("%", "").replace(",", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def resolve_class(cls_map: Optional[Dict[Any, str]], class_id: Any) -> Tuple[Optional[int], str]:
    """Resolve (numeric class_id | None, class_name) against the map of the ANNOTATION'S OWN stream."""
    if class_id is None:
        return None, "unknown"
    if isinstance(class_id, str) and not class_id.strip().lstrip("-").isdigit():
        return None, class_id  # virtual classes, e.g. 'color_bar_region'
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return None, "unknown"
    name = None
    if cls_map:
        name = cls_map.get(cid)
        if name is None:
            name = cls_map.get(str(cid))
    return cid, (name or "unknown")


# --------------------------------------------------------------------------- #
# subplot bookkeeping
# --------------------------------------------------------------------------- #
def _subplot_axes(fig, chart_info_map) -> List[Dict[str, Any]]:
    """Visible axes that carry chart info, in figure order. ``group`` = axes + twins."""
    subs: List[Dict[str, Any]] = []
    for ax in fig.axes:
        if not ax.get_visible() or ax not in chart_info_map:
            continue
        info = chart_info_map.get(ax) or {}
        group = [ax]
        for aux in info.get("aux_axes") or []:
            if aux is not None and all(aux is not g for g in group):
                group.append(aux)
        subs.append({"index": len(subs), "ax": ax, "info": info, "group": group})
    return subs


def _subplot_of_axes(subs, axes) -> Optional[int]:
    for s in subs:
        if any(axes is g for g in s["group"]):
            return s["index"]
    return None


def _infer_subplot(subs, x_disp: float, y_disp: float) -> Optional[int]:
    best, best_area = None, None
    for s in subs:
        try:
            b = s["ax"].get_window_extent()
        except Exception:
            continue
        if b.x0 <= x_disp <= b.x1 and b.y0 <= y_disp <= b.y1:
            area = b.width * b.height
            if best is None or area < best_area:
                best, best_area = s["index"], area
    return best


# --------------------------------------------------------------------------- #
# phase 1: BEFORE effects
# --------------------------------------------------------------------------- #
def _bar_geometry(patch, orientation: str) -> Tuple[float, float, float, Tuple[float, float]]:
    x, y, w, h = patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()
    if orientation == "horizontal":
        return y + h / 2.0, x, x + w, (y, y + h)
    return x + w / 2.0, y, y + h, (x, x + w)


def _match_bar_info(pos, bottom, top, cands, used) -> Optional[int]:
    for k, bi in enumerate(cands):
        if k in used:
            continue
        try:
            c, b, t = bi.get("center"), bi.get("bottom"), bi.get("top")
            if c is None or b is None or t is None:
                continue
            if _close(c, pos) and _close(b, bottom) and _close(t, top):
                return k
        except (TypeError, ValueError):
            continue
    return None


def _baseline_pseudo(ax_b, orientation: str, sub_idx: Optional[int], h_img: float, is_secondary: bool):
    """Zero line of the value axis of ``ax_b`` as a 2-keypoint pseudo-annotation (pre-effects)."""
    box = ax_b.get_window_extent()
    clamped = False
    if orientation == "horizontal":
        ym = sum(ax_b.get_ylim()) / 2.0
        try:
            x = float(ax_b.transData.transform((0.0, ym))[0])
        except Exception:
            x = float("nan")
        if not math.isfinite(x) or x < box.x0 or x > box.x1:
            x = box.x0 if not math.isfinite(x) else min(max(x, box.x0), box.x1)
            clamped = True
        pts = [[x, h_img - box.y0, 2], [x, h_img - box.y1, 2]]
    else:
        xm = sum(ax_b.get_xlim()) / 2.0
        try:
            y = float(ax_b.transData.transform((xm, 0.0))[1])
        except Exception:
            y = float("nan")
        if not math.isfinite(y) or y < box.y0 or y > box.y1:
            y = box.y0 if not math.isfinite(y) else min(max(y, box.y0), box.y1)
            clamped = True
        pts = [[box.x0, h_img - y, 2], [box.x1, h_img - y, 2]]
    return {
        "_key": ("baseline", id(ax_b)),
        "class_id": None,
        "class_name": "baseline",
        "subplot": sub_idx,
        "keypoints": pts,
        "attrs": {
            "orientation": orientation,
            "value": 0.0,
            "clamped": clamped,
            "axis": "secondary" if is_secondary else "primary",
        },
    }


def _label_attrs(fig, ax, cname: str, text: str, ext, scale_axes) -> Optional[Dict[str, Any]]:
    """Axis + role of a title/tick-label annotation, by matching the live text artist."""
    if ax is None or not text or ext is None:
        return None
    cx, cy = (ext[0] + ext[2]) / 2.0, (ext[1] + ext[3]) / 2.0
    best, best_d = None, 3.0  # px tolerance
    if cname == "axis_title":
        cands = [("x", ax.xaxis.label), ("y", ax.yaxis.label)]
    elif cname == "axis_labels":
        cands = [("x", t) for t in ax.get_xticklabels()] + [("y", t) for t in ax.get_yticklabels()]
    else:
        return None
    for axis_name, art in cands:
        try:
            if art.get_text().strip() != text or not art.get_visible():
                continue
            e = art.get_window_extent()
            d = math.hypot((e.x0 + e.x1) / 2.0 - cx, (e.y0 + e.y1) / 2.0 - cy)
        except Exception:
            continue
        if d <= best_d:
            best, best_d = axis_name, d
    if best is None:
        return None
    if cname == "axis_title":
        return {"axis": best}
    role = "scale" if (is_numeric_text(text) and best in scale_axes) else "tick"
    return {"axis": best, "label_role": role}


def prepare_detail_annotations(fig, chart_info_map, streams: Iterable[Stream], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Tag annotations in place and return graph pseudo-annotations. Call BEFORE realism effects.

    ``img_w``/``img_h`` MUST be the pre-effects image size (what effects use as their source frame).
    """
    subs = _subplot_axes(fig, chart_info_map)
    graph: List[Dict[str, Any]] = []
    baselines: Dict[int, Dict[str, Any]] = {}
    used_bar_info: Dict[int, set] = {}
    wedge_counter: Dict[int, int] = {}

    def scale_axes_of(sub) -> set:
        sai = (sub["info"].get("scale_axis_info") or {}) if sub else {}
        return {a for a in (sai.get("primary_scale_axis", "y"), sai.get("secondary_scale_axis")) if a}

    for name, anns, cls_map, kind in streams:
        for ann in anns:
            art = ann.pop("_artist", None)
            _, cname = resolve_class(cls_map, ann.get("class_id"))
            if ann.get("class_name"):
                cname = str(ann["class_name"])

            # -- subplot ------------------------------------------------------
            sub_idx = None
            ax_index = ann.pop("ax_index", None)
            if ax_index is not None and 0 <= ax_index < len(fig.axes):
                sub_idx = _subplot_of_axes(subs, fig.axes[ax_index])
            if sub_idx is None and kind == "mat":
                e = _extents(ann.get("bbox"))
                if e:
                    sub_idx = _infer_subplot(subs, (e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0)
                elif ann.get("polygon"):
                    p = _envelope(ann["polygon"])
                    sub_idx = _infer_subplot(subs, (p[0] + p[2]) / 2.0, img_h - (p[1] + p[3]) / 2.0)
            elif sub_idx is None and kind == "pose" and ann.get("keypoints"):
                kx = [k[0] for k in ann["keypoints"]]
                ky = [k[1] for k in ann["keypoints"]]
                sub_idx = _infer_subplot(subs, (sum(kx) / len(kx)) * img_w, img_h - (sum(ky) / len(ky)) * img_h)
            if sub_idx is not None:
                ann["subplot"] = sub_idx
            sub = subs[sub_idx] if sub_idx is not None else None
            ctype = (sub["info"].get("chart_type_str") if sub else None)

            attrs = dict(ann.get("attrs") or {})

            # -- bars / histogram bins ----------------------------------------
            if cname == _BAR_CLASS and art is not None and hasattr(art, "get_width") and sub is not None:
                orient = sub["info"].get("orientation") or "vertical"
                pos, bottom, top, span = _bar_geometry(art, orient)
                cands = sub["info"].get("bar_info") or sub["info"].get("bar_info_list") or []
                used = used_bar_info.setdefault(sub_idx, set())
                k = _match_bar_info(pos, bottom, top, cands, used)
                bi = cands[k] if k is not None else {}
                if k is not None:
                    used.add(k)
                is_sec = art.axes is not None and art.axes is not sub["ax"]
                attrs.update({
                    "series_idx": bi.get("series_idx"),
                    "axis": "secondary" if is_sec else "primary",
                    "position": round(pos, 6),
                    "bottom": round(bottom, 6),
                    "top": round(top, 6),
                    "value": round(top - bottom, 6),  # segment size (stacked bars: NOT the cumulative top)
                })
                if ctype == "histogram":
                    attrs["bin"] = [round(min(span), 6), round(max(span), 6)]
                if ctype in ("bar", "histogram") and art.axes is not None:
                    key = id(art.axes)
                    if key not in baselines:
                        baselines[key] = _baseline_pseudo(art.axes, orient, sub_idx, img_h, is_sec)
                    ann["_links"] = {"baseline_id": baselines[key]["_key"]}

            # -- pie wedges -----------------------------------------------------
            elif cname == "wedge" and art is not None and hasattr(art, "theta1"):
                wedge_idx = ann.get("attrs", {}).get("wedge_index")
                if wedge_idx is None:
                    wedge_idx = wedge_counter.get(sub_idx, 0)
                    wedge_counter[sub_idx] = wedge_idx + 1
                span = (float(art.theta2) - float(art.theta1)) % 360.0 or 360.0
                attrs.update({
                    "wedge_index": int(wedge_idx),
                    "start_angle": round(float(art.theta1), 4),
                    "end_angle": round(float(art.theta2), 4),
                    "percentage": round(span / 360.0 * 100.0, 4),
                })

            # -- point-like marks: data coordinates ----------------------------
            elif cname in ("data_point", "outlier") and kind == "mat" and ax_index is not None:
                e = _extents(ann.get("bbox"))
                try:
                    if e:
                        inv = fig.axes[ax_index].transData.inverted()
                        dx, dy = inv.transform(((e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0))
                        if _finite(dx, dy):
                            attrs["data"] = [round(float(dx), 6), round(float(dy), 6)]
                except Exception:
                    pass

            # -- titles / tick labels -------------------------------------------
            elif cname in ("axis_title", "axis_labels") and kind == "mat" and ax_index is not None:
                e = _extents(ann.get("bbox"))
                la = _label_attrs(fig, fig.axes[ax_index], cname, str(ann.get("text", "")).strip(), e, scale_axes_of(sub))
                if la:
                    attrs.update(la)

            if attrs:
                ann["attrs"] = attrs

    # -- line / area keypoint series (pixel keypoints + data coords + tags) ------
    for sub in subs:
        info, ax = sub["info"], sub["ax"]
        if info.get("chart_type_str") not in ("line", "area"):
            continue
        for ser in info.get("keypoint_info") or []:
            if not isinstance(ser, dict):
                continue
            pts = ser.get("plotted_points") or ser.get("all_points") or []
            peaks = {int(i[2]) for i in ser.get("peaks", [])}
            valleys = {int(i[2]) for i in ser.get("valleys", [])}
            infl = {int(i[2]) for i in ser.get("inflections", [])}
            kps, data, tags = [], [], []
            n = len(pts)
            bbox = ax.get_window_extent()
            for j, p in enumerate(pts):
                try:
                    x, y, idx = float(p[0]), float(p[1]), int(p[2])
                    dx, dy = ax.transData.transform((x, y))
                except Exception:
                    continue
                if not _finite(dx, dy):
                    continue
                if j == 0 or j == n - 1:
                    tag = "endpoint"
                elif idx in peaks:
                    tag = "peak"
                elif idx in valleys:
                    tag = "valley"
                elif idx in infl:
                    tag = "inflection"
                else:
                    tag = "vertex"
                inside = bbox.x0 <= dx <= bbox.x1 and bbox.y0 <= dy <= bbox.y1
                kps.append([float(dx), float(img_h - dy), 2 if inside else 1])
                data.append([x, y])
                tags.append(tag)
            if kps:
                attrs_dict = {"series_idx": ser.get("series_idx"), "tags": tags, "data": data}
                if info.get("chart_type_str") == "area" and ser.get("fill_bottom"):
                    attrs_dict["fill_bottom"] = [
                        float(p[1]) if isinstance(p, (list, tuple)) else float(p)
                        for p in ser["fill_bottom"]
                    ]
                graph.append({
                    "class_id": None,
                    "class_name": "series_keypoints",
                    "subplot": sub["index"],
                    "keypoints": kps,
                    "attrs": attrs_dict,
                })

    graph.extend(baselines.values())
    return graph


# --------------------------------------------------------------------------- #
# phase 2: AFTER effects
# --------------------------------------------------------------------------- #
def _serialize(ann: Dict[str, Any], name: str, cls_map, kind: str, w: int, h: int, strict: bool = False) -> Optional[Dict[str, Any]]:
    cid, cname = resolve_class(cls_map, ann.get("class_id"))
    if ann.get("class_name"):
        cname = str(ann["class_name"])

    if cname == "unknown":
        msg = f"Unresolved class '{ann.get('class_id')}' in stream '{name}'"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning)

    amodal = _pts(ann.get("amodal_polygon"))
    modal = _pts(ann.get("modal_polygon"))
    poly = _pts(ann.get("polygon"))
    obb = _pts(ann.get("obb"))

    kps = None
    if ann.get("keypoints"):
        kps = []
        for k in ann["keypoints"]:
            try:
                x, y = float(k[0]), float(k[1])
                v = int(k[2]) if len(k) > 2 else 2
            except (TypeError, ValueError, IndexError):
                continue
            if kind == "pose":
                x, y = x * w, y * h
            if _finite(x, y):
                kps.append([round(x, 2), round(y, 2), v])
        kps = kps or None

    xyxy = None
    raw = ann.get("bbox")
    if kind == "pose":
        if isinstance(raw, (tuple, list)) and len(raw) == 4:
            cx, cy, bw, bh = [float(v) for v in raw]
            xyxy = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
    else:
        e = _extents(raw)
        if e:
            xyxy = [e[0], h - e[3], e[2], h - e[1]]  # matplotlib frame -> image frame
    if xyxy is None:
        env = _envelope(amodal or poly or [k[:2] for k in (kps or [])])
        xyxy = env
    if xyxy is None:
        return None
    xyxy = _clip_xyxy(xyxy, w, h)
    if xyxy is None:
        return None

    rec: Dict[str, Any] = {
        "stream": name,
        "class_id": cid,
        "class_name": cname,
    }
    if ann.get("subplot") is not None:
        rec["subplot"] = int(ann["subplot"])
    rec["xyxy"] = xyxy
    if obb:
        rec["obb"] = obb
    if poly and not amodal:
        rec["polygon"] = poly
    if amodal:
        rec["amodal_polygon"] = amodal
        if modal and modal != amodal:  # absent modal_polygon == identical to amodal_polygon
            rec["modal_polygon"] = modal
    if kps:
        rec["keypoints"] = kps
    text = str(ann.get("text", "")).strip()
    if text:
        rec["text"] = text
    if ann.get("role"):
        rec["role"] = str(ann["role"])
    occluded = bool(ann.get("occluded", False))
    rec["visibility"] = 0 if (occluded or ann.get("visibility") == 0) else 1
    rec["occluded"] = occluded

    attrs = to_py(dict(ann.get("attrs") or {}))
    if ann.get("series_idx") is not None and "series_idx" not in attrs:
        attrs["series_idx"] = ann["series_idx"]
    attrs = {k: v for k, v in attrs.items() if v is not None}
    if attrs:
        rec["attrs"] = attrs
    if ann.get("ignored"):
        rec["ignored"] = True
        if ann.get("ignore_reason"):
            rec["ignore_reason"] = str(ann["ignore_reason"])
    return rec


def _subplot_record(sub) -> Dict[str, Any]:
    info, ax = sub["info"], sub["ax"]
    ctype = info.get("chart_type_str", "unknown")
    domain = info.get("semantic_domain")
    rec: Dict[str, Any] = {
        "index": sub["index"],
        "chart_type": ctype,
        "semantic_domain": domain,
        "is_scientific": derive_is_scientific(domain, info.get("is_scientific", False)),
    }
    if ctype in ("bar", "box", "histogram"):
        rec["orientation"] = info.get("orientation") or "vertical"

    names = list(info.get("series_names") or [])
    count = int(info.get("series_count") or 1)
    n = max(count, len(names))
    if names or n > 1:
        rec["series"] = [{"idx": i, "name": names[i] if i < len(names) else None} for i in range(n)]

    if ctype in ("bar", "area") and info.get("stacking_mode"):
        rec["stacking_mode"] = info.get("stacking_mode")
    if ctype == "bar":
        if info.get("style"):
            rec["style"] = info.get("style")
        if info.get("pattern"):
            rec["pattern"] = info.get("pattern")

    sai = info.get("scale_axis_info")
    if not isinstance(sai, dict):
        try:  # legacy fallback used by the old writer
            from chart import extract_scale_axis_info
            sai = extract_scale_axis_info(ax, ctype)
        except Exception:
            sai = {}
    if ctype not in ("pie", "heatmap"):
        rec["scale_axes"] = {
            "primary": (sai or {}).get("primary_scale_axis", "y"),
            "secondary": (sai or {}).get("secondary_scale_axis"),
        }

    if ctype == "box" and isinstance(info.get("boxplot_dict"), dict) and info["boxplot_dict"]:
        bp = info["boxplot_dict"]
        rec["box"] = {
            "num_groups": int(bp.get("num_groups", 0) or 0),
            "box_width": to_py(bp.get("box_width")),
            "medians": [
                {
                    "group_index": m.get("group_index"),
                    "median_value": to_py(m.get("median_value")),
                    "position": to_py(m.get("center_x") if m.get("center_x") is not None else m.get("center_y")),
                }
                for m in bp.get("medians", [])
            ],
        }
    if ctype == "pie":
        pie: Dict[str, Any] = {}
        for art in info.get("data_artists") or []:
            if hasattr(art, "theta1") and hasattr(art, "center"):
                pie["center"] = [round(float(art.center[0]), 6), round(float(art.center[1]), 6)]
                pie["radius"] = round(float(art.r), 6)
                break
        meta = to_py(info.get("pie_metadata"))
        if meta:
            pie["metadata"] = meta
        if pie:
            rec["pie"] = pie
    if ctype == "histogram":
        meta = to_py(info.get("histogram_metadata"))
        if meta:
            rec["histogram"] = meta
    if ctype == "heatmap":
        hm: Dict[str, Any] = {}
        data = info.get("heatmap_data")
        try:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 2 and arr.size:
                hm.update({"rows": int(arr.shape[0]), "cols": int(arr.shape[1]),
                           "vmin": float(np.nanmin(arr)), "vmax": float(np.nanmax(arr))})
        except Exception:
            pass
        meta = info.get("heatmap_meta")
        if isinstance(meta, dict) and isinstance(meta.get("context_id"), str):
            hm["context_id"] = meta["context_id"]
        if hm:
            rec["heatmap"] = to_py(hm)
    return rec


def merge_streams(streams: Iterable[Stream], w: int, h: int, strict: bool = False, filter_stats: Optional[Dict[str, Dict[str, int]]] = None) -> List[Dict[str, Any]]:
    """Serialise + dedupe all streams into ONE list; assign ids; resolve links."""
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple, int] = {}
    key_to_id: Dict[Any, int] = {}
    pending: List[Tuple[int, Any]] = []

    for name, anns, cls_map, kind in streams:
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            rec = _serialize(ann, name, cls_map, kind, w, h, strict=strict)
            if rec is None:
                continue
            if kind == "mat" and "polygon" not in rec and "amodal_polygon" not in rec:
                is_dup = False
                dk = (rec["class_name"], tuple(round(v, 1) for v in rec["xyxy"]), rec.get("text", ""))
                if dk in seen:
                    is_dup = True
                    match_idx = seen[dk]
                else:
                    # Harmonize cross-stream duplicate pairs with IoU >= 0.8
                    for idx, existing in enumerate(out):
                        if existing["class_name"] == rec["class_name"] and existing.get("subplot") == rec.get("subplot"):
                            b1, b2 = existing["xyxy"], rec["xyxy"]
                            inter = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                            if inter > 0:
                                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                                u = a1 + a2 - inter
                                if u > 0 and inter / u >= 0.8:
                                    is_dup = True
                                    match_idx = idx
                                    break
                if is_dup:
                    if filter_stats is not None:
                        c_name = rec.get("class_name", "unknown")
                        filter_stats["duplicate"][c_name] = filter_stats["duplicate"].get(c_name, 0) + 1
                    prev = out[match_idx]
                    if "attrs" in rec and "attrs" not in prev:
                        prev["attrs"] = rec["attrs"]
                    if not rec.get("ignored") and prev.get("ignored"):
                        prev.pop("ignored", None)
                        prev.pop("ignore_reason", None)
                    continue
                seen[dk] = len(out)
            rec["id"] = len(out)
            if ann.get("_key") is not None:
                key_to_id[ann["_key"]] = rec["id"]
            link = ann.get("_links", {}).get("baseline_id")
            if link is not None:
                pending.append((rec["id"], link))
            out.append(rec)

    for rid, key in pending:
        if key in key_to_id:
            out[rid].setdefault("attrs", {})["baseline_id"] = key_to_id[key]

    # stable field order: id first
    return [{"id": r.pop("id"), **r} for r in out]


def build_detailed_json(
    fig,
    chart_info_map,
    streams: Sequence[Stream],
    img_w: int,
    img_h: int,
    schema_version: str,
    dataset_version: str,
    image_id: Optional[str] = None,
    strict: bool = False,
    filter_stats: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Assemble the v4.1 ``_detailed.json`` payload. Call AFTER effects/filtering."""
    if filter_stats is None:
        filter_stats = {
            "size": {},
            "aspect": {},
            "viewport": {},
            "duplicate": {},
            "overlap": {},
        }
    subs = _subplot_axes(fig, chart_info_map)
    subplots = [_subplot_record(s) for s in subs]
    primary = subplots[0] if subplots else {}
    payload = {
        "schema_version": schema_version,
        "dataset_version": dataset_version,
        "image": {"width": int(img_w), "height": int(img_h)},
        "chart_type": primary.get("chart_type"),
        "semantic_domain": primary.get("semantic_domain"),
        "is_scientific": primary.get("is_scientific", False),
        "is_composite": len(subplots) > 1,
        "composite_chart_types": [s["chart_type"] for s in subplots] if len(subplots) > 1 else [],
        "composite_domains": [s["semantic_domain"] for s in subplots] if len(subplots) > 1 else [],
        "subplots": subplots,
        "annotations": merge_streams(streams, int(img_w), int(img_h), strict=strict, filter_stats=filter_stats),
        "filter_stats": filter_stats,
    }
    if image_id is not None:
        payload["image_id"] = image_id
    return payload
