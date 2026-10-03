"""
Serialization contract for generator.json_default_fallback.

Replaces tests of the removed ``convert_numpy_types`` helper: every json.dump site passes
``default=json_default_fallback``, so numpy / matplotlib values must serialise through it
exactly as the old pre-conversion step did (verified byte-identical on these payloads).
"""
import json
from collections import namedtuple

import numpy as np
import pytest
from matplotlib.transforms import Bbox

from generator import json_default_fallback

BoundingBox = namedtuple("BoundingBox", "x0 y0 x1 y1")


class _Artist:                      # stands in for an arbitrary matplotlib object
    def __repr__(self):
        return "<Line2D weird>"


def dumps(x):
    return json.loads(json.dumps(x, default=json_default_fallback))


@pytest.mark.parametrize("value, expected", [
    (np.int64(3), 3), (np.uint8(7), 7), (np.float32(1.5), 1.5), (np.float64(2.25), 2.25),
    (np.bool_(True), True), (np.bool_(False), False),
])
def test_numpy_scalars_become_python_scalars(value, expected):
    out = dumps({"v": value})["v"]
    assert out == expected and type(out) is type(expected)


def test_arrays_become_nested_lists():
    assert dumps({"a": np.arange(6).reshape(2, 3)})["a"] == [[0, 1, 2], [3, 4, 5]]
    assert dumps({"a": np.array([1.5, 2.5])})["a"] == [1.5, 2.5]
    assert dumps({"a": np.array([True, False])})["a"] == [True, False]


def test_matplotlib_bbox_serialises_as_xyxy_extents():
    assert dumps({"bb": Bbox([[1, 2], [3, 4]])})["bb"] == [1.0, 2.0, 3.0, 4.0]
    assert dumps({"n": [Bbox([[0, 0], [1, 1]])]})["n"] == [[0.0, 0.0, 1.0, 1.0]]


def test_tuples_and_namedtuples_become_lists():
    assert dumps({"t": (1, 2, (3, np.int32(4)))})["t"] == [1, 2, [3, 4]]
    assert dumps({"bb": BoundingBox(1.0, 2.0, 3.0, np.float32(4.0))})["bb"] == [1.0, 2.0, 3.0, 4.0]


def test_unknown_objects_fall_back_to_str_instead_of_raising():
    assert dumps({"o": _Artist(), "l": [_Artist(), 1]}) == {"o": "<Line2D weird>", "l": ["<Line2D weird>", 1]}


def test_nested_annotation_payload():
    ann = {"annotations": [{"xyxy": [np.int64(1), np.float64(2.5), 3, 4], "obb": np.zeros((4, 2)),
                            "attrs": {"k": (np.float32(1),)}}]}
    assert dumps(ann) == {"annotations": [{"xyxy": [1, 2.5, 3, 4], "obb": [[0.0, 0.0]] * 4, "attrs": {"k": [1.0]}}]}


def test_plain_json_types_pass_through_unchanged():
    plain = {"s": "x", "i": 1, "f": 1.5, "n": None, "l": [1, 2, {"z": True}]}
    assert dumps(plain) == plain


def test_int_keys_are_stringified_by_json_itself():
    assert dumps({1: "a", 2: {3: np.int64(4)}}) == {"1": "a", "2": {"3": 4}}
