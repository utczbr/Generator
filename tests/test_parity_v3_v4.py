"""Unit tests for the pure comparison core of parity_v3_v4.py (no generator / config required)."""
import json
import pytest
from scripts import parity_v3_v4 as P


def v3_doc():
    return {
        "chart_type": "bar", "orientation": "vertical", "semantic_domain": "business", "is_scientific": False,
        "is_composite": False, "composite_chart_types": [], "series_count": 2, "series_names": ["A", "B"],
        "stacking_mode": "grouped", "style": "flat", "pattern": None,
        "bar": [{"xyxy": [10, 100, 50, 200]}, {"xyxy": [60, 120, 100, 200]}],
        "chart_title": [{"xyxy": [20, 5, 180, 25], "text": "Revenue"}],
        "baselines": [{"y": 200}], "bars_with_baseline": [{}, {}],
        # legacy aliases must NOT be double counted
        "charttitle": [{"xyxy": [20, 5, 180, 25], "text": "Revenue"}],
    }


def v4_doc(shift=0.0, drop_bar=False):
    bars = [[10, 100, 50, 200], [60, 120, 100, 200]]
    anns = [{"id": i, "class_name": "bar", "xyxy": [b[0] + shift, b[1], b[2] + shift, b[3]],
             "attrs": {"baseline_id": 9}} for i, b in enumerate(bars[: 1 if drop_bar else 2])]
    anns += [{"id": 5, "class_name": "chart_title", "xyxy": [20 + shift, 5, 180 + shift, 25], "text": "Revenue"},
             {"id": 6, "class_name": "baseline", "xyxy": [0, 199, 200, 201]},
             {"id": 7, "class_name": "data_point", "xyxy": [300, 300, 310, 310]}]
    return {"chart_type": "bar", "semantic_domain": "business", "is_scientific": False, "is_composite": False,
            "composite_chart_types": [],
            "subplots": [{"orientation": "vertical", "stacking_mode": "grouped", "style": "flat",
                          "series": [{"idx": 0, "name": "A"}, {"idx": 1, "name": "B"}]}],
            "annotations": anns}


def test_iou_basics():
    assert P.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert P.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert P.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert P.iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0          # degenerate box, no ZeroDivisionError


def test_identical_frames_fully_match_and_aliases_not_double_counted():
    r = P.compare(v3_doc(), v4_doc())
    assert r["elements"]["bar"]["loose"] == 2 and r["elements"]["bar"]["tight_frac"] == 1.0
    assert r["elements"]["chart_title"]["n_v3"] == 1                       # alias "charttitle" ignored
    assert r["elements"]["bar"]["v4_classes"] == {"bar": 2}
    assert r["v4_unmatched"] == 2                                          # baseline + data_point are v4-only here
    assert r["v4_duplicate_ids"] == 0


def test_frame_drift_is_detected():
    r = P.compare(v3_doc(), v4_doc(shift=60.0))                            # simulates a geometric effect
    assert r["elements"]["bar"]["tight_frac"] == 0.0            # nothing is correctly placed any more
    assert r["elements"]["bar"]["min_iou"] < 0.1                # v3 bar 1 is nowhere near any v4 box (coincidental baseline overlap only)


def test_missing_element_lowers_match_rate():
    r = P.compare(v3_doc(), v4_doc(drop_bar=True))
    assert r["elements"]["bar"]["loose"] == 1 and r["elements"]["bar"]["tight_frac"] == 0.5


def test_metadata_parity_and_gaps():
    meta = {m["field"]: m for m in P.compare(v3_doc(), v4_doc())["metadata"]}
    assert all(meta[f]["status"] == "match" for f in
               ("chart_type", "orientation", "semantic_domain", "series_count", "series_names", "stacking_mode", "style"))
    assert meta["n_baselines"]["status"] == "match"                        # 1 baseline annotation in v4
    assert meta["n_bars_with_baseline"]["status"] == "match"               # 2 bars carry attrs.baseline_id
    v3 = v3_doc(); v3["pattern"] = "hatch"                                 # v4 has no pattern -> reported, not hidden
    assert {m["field"]: m["status"] for m in P.compare(v3, v4_doc())["metadata"]}["pattern"] == "absent-in-v4"
    v3 = v3_doc(); v3["chart_type"] = "line"
    assert {m["field"]: m["status"] for m in P.compare(v3, v4_doc())["metadata"]}["chart_type"] == "differs"


def test_duplicate_v4_ids_flagged():
    v4 = v4_doc(); v4["annotations"][1]["id"] = v4["annotations"][0]["id"]
    assert P.compare(v3_doc(), v4)["v4_duplicate_ids"] == 1


def test_aggregate_across_images():
    agg, meta = P.aggregate([P.compare(v3_doc(), v4_doc()), P.compare(v3_doc(), v4_doc(drop_bar=True))])
    assert agg["bar"]["n_v3"] == 4 and agg["bar"]["tight"] == 3 and agg["bar"]["tight_frac"] == 0.75
    assert meta[("chart_type", "match")] == 2
    assert "bar" in P.format_summary(agg, meta, 3, "effects off")


def test_diff_trees_ignores_detailed_json_and_flags_real_differences(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "images").mkdir(parents=True); (d / "labels").mkdir()
        (d / "images" / "chart_00000.png").write_bytes(b"PNG")
        (d / "labels" / "chart_00000.txt").write_text("0 0.5 0.5 0.1 0.1")
    (a / "labels" / "chart_00000_detailed.json").write_text(json.dumps({"v": 3}))
    (b / "labels" / "chart_00000_detailed.json").write_text(json.dumps({"v": 4}))
    assert P.diff_trees(str(a), str(b)) == ([], [], [])                    # only the JSON differs -> invariant holds
    (b / "images" / "chart_00000.png").write_bytes(b"PNG-different")
    (b / "labels" / "extra.txt").write_text("x")
    only_a, only_b, differing = P.diff_trees(str(a), str(b))
    assert only_a == [] and only_b == [str(__import__("pathlib").Path("labels/extra.txt"))]
    assert differing == [str(__import__("pathlib").Path("images/chart_00000.png"))]


def test_cli_help_does_not_need_generator(capsys):
    with pytest.raises(SystemExit) as e:
        P.main(["--help"])
    assert e.value.code == 0 and "--strict" in capsys.readouterr().out


def test_one_to_one_stops_neighbour_aliasing():
    """A v3 grid shifted by half a cell must not have every cell 'found' via a shared neighbour."""
    cell = lambda i, dx=0: [i * 10 + dx, 0, i * 10 + 10 + dx, 10]
    v3 = {"cell": [{"xyxy": cell(i)} for i in range(5)]}
    v4 = {"annotations": [{"id": i, "class_name": "cell", "xyxy": cell(i, 5)} for i in range(5)]}
    e = P.compare(v3, v4)["elements"]["cell"]
    assert e["tight_frac"] == 0.0                       # IoU of half-shifted neighbours is 1/3
    v4_same = {"annotations": [{"id": i, "class_name": "cell", "xyxy": cell(i)} for i in range(5)]}
    assert P.compare(v3, v4_same)["elements"]["cell"]["tight_frac"] == 1.0


def test_empty_inputs_do_not_crash():
    assert P.compare({}, {})["elements"] == {}
    r = P.compare({"bar": [{"xyxy": [0, 0, 5, 5]}]}, {"annotations": []})["elements"]["bar"]
    assert r["loose"] == 0 and r["min_iou"] == 0.0


def test_iou_matrix_agrees_with_scalar_iou():
    import numpy as np
    rs = np.random.RandomState(0)
    A = np.sort(rs.rand(20, 2, 2) * 100, axis=1).reshape(20, 4)[:, [0, 2, 1, 3]]
    B = np.sort(rs.rand(15, 2, 2) * 100, axis=1).reshape(15, 4)[:, [0, 2, 1, 3]]
    M = P.iou_matrix(A, B)
    assert all(abs(M[i, j] - P.iou(A[i], B[j])) < 1e-12 for i in range(20) for j in range(15))
