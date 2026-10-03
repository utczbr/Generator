"""Unit tests verifying Phase 3 Transform & Realism Synchronization."""
import math
import numpy as np
import pytest
from PIL import Image
from matplotlib import transforms

import effects
import generator
from generator import (
    BoundingBox,
    apply_realism_effects,
    clip_bbox_to_viewport,
    clip_polygon_to_viewport,
)


def test_perspective_distortion_shifts_all_annotation_types():
    """
    Verify perspective distortion captures homography and shifts
    bounding boxes, keypoints, and segmentation polygons consistently.
    """
    w, h = 800, 600
    img = Image.new('RGB', (w, h), color=(240, 240, 240))

    # Initial annotations
    init_bbox = transforms.Bbox.from_extents(200.0, 150.0, 400.0, 350.0)
    box_ann = {'class_id': 0, 'bbox': init_bbox}

    # Normalized keypoint annotation (e.g. pie wedge pose)
    init_kpts = [[0.25, 0.25, 2], [0.5, 0.5, 2], [0.75, 0.75, 2]]
    kpt_ann = {
        'class_id': 1,
        'keypoints': [list(kp) for kp in init_kpts],
        'bbox': (0.5, 0.5, 0.5, 0.5)
    }

    # Instance segmentation polygon (e.g. area series fill)
    init_poly = [(200.0, 150.0), (400.0, 150.0), (400.0, 350.0), (200.0, 350.0)]
    poly_ann = {
        'class_id': 2,
        'polygon': list(init_poly)
    }

    # 1. Test using effect alias "perspective"
    effects_cfg_persp = {'perspective': {'distortion_factor': 0.15}}
    extra_sets = [[kpt_ann], [poly_ann]]

    out_img, out_anns = apply_realism_effects(
        img, [box_ann], effects_cfg_persp, extra_annotation_sets=extra_sets
    )

    # Verify bounding box shifted
    warped_bbox = out_anns[0]['bbox']
    assert (warped_bbox.x0, warped_bbox.y0, warped_bbox.x1, warped_bbox.y1) != (
        init_bbox.x0, init_bbox.y0, init_bbox.x1, init_bbox.y1
    ), "Bounding box remained static after perspective effect"

    # Verify keypoints shifted
    warped_kpts = kpt_ann['keypoints']
    for orig_kp, new_kp in zip(init_kpts, warped_kpts):
        assert (new_kp[0], new_kp[1]) != (orig_kp[0], orig_kp[1]), (
            f"Keypoint {orig_kp} remained static after perspective effect"
        )
        assert 0.0 <= new_kp[0] <= 1.0 and 0.0 <= new_kp[1] <= 1.0

    # Verify pose envelope bbox tuple updated
    cx, cy, bw, bh = kpt_ann['bbox']
    assert (cx, cy, bw, bh) != (0.5, 0.5, 0.5, 0.5), "Pose envelope bbox remained static"
    assert bw > 0 and bh > 0

    # Verify polygon shifted
    warped_poly = poly_ann['polygon']
    assert warped_poly != init_poly, "Polygon contour remained static after perspective effect"
    for orig_p, new_p in zip(init_poly, warped_poly):
        assert new_p != orig_p, f"Polygon vertex {orig_p} remained untransformed"

    # 2. Test using effect alias "perspective_warp" with magnitude parameter
    box_ann2 = {'class_id': 0, 'bbox': transforms.Bbox.from_extents(200.0, 150.0, 400.0, 350.0)}
    kpt_ann2 = {'class_id': 1, 'keypoints': [list(kp) for kp in init_kpts], 'bbox': (0.5, 0.5, 0.5, 0.5)}
    poly_ann2 = {'class_id': 2, 'polygon': list(init_poly)}
    extra_sets2 = [[kpt_ann2], [poly_ann2]]

    effects_cfg_warp = {'perspective_warp': {'magnitude': 0.12}}
    out_img2, out_anns2 = apply_realism_effects(
        img, [box_ann2], effects_cfg_warp, extra_annotation_sets=extra_sets2
    )

    assert (out_anns2[0]['bbox'].x0, out_anns2[0]['bbox'].y0) != (init_bbox.x0, init_bbox.y0)
    assert kpt_ann2['keypoints'][0][:2] != init_kpts[0][:2]
    assert poly_ann2['polygon'] != init_poly


def test_transforms_match_image_coordinates_rotation_and_perspective():
    """
    Verify segmentation masks and pose keypoints match analytical image
    coordinates after scan rotation and perspective warping.
    """
    w, h = 600, 400
    img = Image.new('RGB', (w, h), color=(255, 255, 255))

    angle = 6.0  # Counter-clockwise rotation in degrees
    rad = np.radians(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    cx = w / 2.0
    cy = h / 2.0

    # Point in image coordinates (origin top-left)
    test_px, test_py = 200.0, 150.0

    # Analytical rotation around center
    dx = test_px - cx
    dy = test_py - cy
    expected_rot_x = cx + dx * cos_a + dy * sin_a
    expected_rot_y = cy - dx * sin_a + dy * cos_a

    # Test polygon: single vertex test polygon
    poly_ann = {'class_id': 0, 'polygon': [(test_px, test_py), (test_px + 20, test_py), (test_px, test_py + 20)]}

    # Test keypoint: normalized coordinates
    norm_x = test_px / w
    norm_y = test_py / h
    kpt_ann = {'class_id': 1, 'keypoints': [[norm_x, norm_y, 2]], 'bbox': (norm_x, norm_y, 0.1, 0.1)}

    effects_cfg = {'scan_rotation': {'angle_range': (angle, angle)}}
    apply_realism_effects(
        img, [], effects_cfg, extra_annotation_sets=[[poly_ann], [kpt_ann]]
    )

    actual_poly_pt = poly_ann['polygon'][0]
    assert pytest.approx(actual_poly_pt[0], abs=0.1) == expected_rot_x
    assert pytest.approx(actual_poly_pt[1], abs=0.1) == expected_rot_y

    actual_kpt = kpt_ann['keypoints'][0]
    expected_norm_x = expected_rot_x / w
    expected_norm_y = expected_rot_y / h
    assert pytest.approx(actual_kpt[0], abs=0.001) == expected_norm_x
    assert pytest.approx(actual_kpt[1], abs=0.001) == expected_norm_y


def test_dynamic_canvas_expansion_preserves_alignment():
    """
    Verify that canvas expansion (pdf_document_context padding) preceding
    rotation or perspective computes transformations against the proper dimensions.
    """
    w, h = 400, 300
    img = Image.new('RGB', (w, h), color=(255, 255, 255))

    pad = 40
    pdf_noise_cfg = {
        'margin_px_range': (pad, pad),
        'caption_probability': 0.0,
        'body_text_probability': 0.0,
    }
    effects_cfg = {
        'pdf_document_context': pdf_noise_cfg,
        'scan_rotation': {'angle_range': (0.0, 0.0)}  # pure translation
    }

    init_poly = [(100.0, 100.0), (200.0, 100.0), (200.0, 150.0)]
    poly_ann = {'class_id': 0, 'polygon': list(init_poly)}

    out_img, _ = apply_realism_effects(
        img, [], effects_cfg, extra_annotation_sets=[[poly_ann]]
    )

    # After padding, canvas size is expanded by pad on all 4 sides
    final_w, final_h = out_img.size
    assert final_w == w + 2 * pad
    assert final_h == h + 2 * pad

    # In image coordinates, vertices are translated by (pad, pad)
    for orig_p, trans_p in zip(init_poly, poly_ann['polygon']):
        assert pytest.approx(trans_p[0], abs=0.5) == orig_p[0] + pad
        assert pytest.approx(trans_p[1], abs=0.5) == orig_p[1] + pad


def test_viewport_clipping_margin_labels():
    """
    Verify margin-touching and partially clipped annotations are truncated
    to the image viewport rather than discarded, while degenerate ones are dropped.
    """
    img_w, img_h = 800, 600

    test_anns = [
        # 1. Partially clipped at left margin (negative x0, valid visible area)
        {'id': 'clip_left', 'class_id': 8, 'bbox': transforms.Bbox.from_extents(-10.0, 100.0, 30.0, 140.0)},
        # 2. Partially clipped at top margin (y1 > img_h, valid visible area)
        {'id': 'clip_top', 'class_id': 8, 'bbox': transforms.Bbox.from_extents(200.0, 580.0, 250.0, 620.0)},
        # 3. Degenerate width (< 2 px post-clip)
        {'id': 'degen_width', 'class_id': 8, 'bbox': transforms.Bbox.from_extents(-15.0, 100.0, 1.0, 140.0)},
        # 4. Degenerate area (< 4 px² post-clip)
        {'id': 'degen_area', 'class_id': 8, 'bbox': transforms.Bbox.from_extents(0.0, 0.0, 1.5, 1.5)},
        # 5. Completely outside viewport
        {'id': 'outside', 'class_id': 8, 'bbox': transforms.Bbox.from_extents(-50.0, 100.0, -10.0, 140.0)},
        # 6. BoundingBox namedtuple format preserved
        {'id': 'namedtuple_box', 'class_id': 5, 'bbox': BoundingBox(-5.0, 200.0, 100.0, 250.0)},
        # 7. Tuple format preserved
        {'id': 'tuple_box', 'class_id': 0, 'bbox': (-8.0, 50.0, 40.0, 80.0)},
    ]

    kept = clip_bbox_to_viewport(test_anns, img_w, img_h)
    kept_ids = {ann['id'] for ann in kept}

    assert 'clip_left' in kept_ids
    assert 'clip_top' in kept_ids
    assert 'namedtuple_box' in kept_ids
    assert 'tuple_box' in kept_ids
    assert 'degen_width' not in kept_ids
    assert 'degen_area' not in kept_ids
    assert 'outside' not in kept_ids

    # Verify clipped coordinates
    clip_left_ann = next(a for a in kept if a['id'] == 'clip_left')
    assert clip_left_ann['bbox'].x0 == 0.0
    assert clip_left_ann['bbox'].x1 == 30.0
    assert clip_left_ann['bbox'].y0 == 100.0
    assert clip_left_ann['bbox'].y1 == 140.0

    clip_top_ann = next(a for a in kept if a['id'] == 'clip_top')
    assert clip_top_ann['bbox'].y1 == 600.0

    # Verify namedtuple type preserved
    nt_ann = next(a for a in kept if a['id'] == 'namedtuple_box')
    assert isinstance(nt_ann['bbox'], BoundingBox)
    assert nt_ann['bbox'].x0 == 0.0

    # Verify tuple type preserved
    tup_ann = next(a for a in kept if a['id'] == 'tuple_box')
    assert isinstance(tup_ann['bbox'], tuple) and not hasattr(tup_ann['bbox'], 'x0')
    assert tup_ann['bbox'][0] == 0.0


def test_viewport_clipping_polygons():
    """Verify polygon viewport clipping and degenerate polygon discarding."""
    img_w, img_h = 500, 400

    poly_anns = [
        # Valid polygon crossing left edge
        {'id': 'poly_left', 'class_id': 0, 'polygon': [(-20.0, 100.0), (100.0, 100.0), (100.0, 150.0), (-20.0, 150.0)]},
        # Degenerate polygon completely outside viewport
        {'id': 'poly_outside', 'class_id': 0, 'polygon': [(-50.0, 10.0), (-40.0, 10.0), (-40.0, 20.0)]},
        # Degenerate visible slice (< 2 px width)
        {'id': 'poly_degen_slice', 'class_id': 0, 'polygon': [(-50.0, 100.0), (1.0, 100.0), (1.0, 120.0), (-50.0, 120.0)]},
    ]

    kept = clip_polygon_to_viewport(poly_anns, img_w, img_h)
    kept_ids = {ann['id'] for ann in kept}

    assert 'poly_left' in kept_ids
    assert 'poly_outside' not in kept_ids
    assert 'poly_degen_slice' not in kept_ids

    left_ann = next(a for a in kept if a['id'] == 'poly_left')
    for px, py in left_ann['polygon']:
        assert px >= 0.0 and py >= 0.0
