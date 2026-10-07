"""
tests/test_p5_tick_rotation.py

Acceptance tests for Phase 1 Task T230: Tick Rotation (FR-088).
- tick_rotation config {mode: legacy|profile, angles, weights, p_rotate}
- Under profile, ha by angle sign, centered at |90°|, default rotation_mode
- |tick-to-label gap| <= 15 px at ±30/45/60°
- OBB-vs-ink test passes for each angle x ha
- Matplotlib honors label_angle only under profile, ignores under legacy
"""
import copy
import glob
import json
import os
import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pytest

from config_defaults import OCR_TRAINING_CONFIG
from generator import generate_single_chart, get_text_obb_and_bbox
from chart import apply_typography_variation


def _measure_gap(rot: int, ha: str, mode: str = "default") -> float:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    ax.bar([0], [3])
    ax.set_xticks([0])
    ax.set_xticklabels(["Treatment group A"])
    ax.set_xlim(-1, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelrotation=rot)
    for s in ax.spines.values():
        s.set_visible(False)
    for l in ax.get_xticklabels():
        l.set_horizontalalignment(ha)
        l.set_rotation_mode(mode)
    fig.subplots_adjust(bottom=0.35)
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    ink = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int).sum(axis=2) < 600
    ink[:H - int(ax.get_window_extent().y0) + 1, :] = False
    top = np.where(ink.any(axis=1))[0].min()
    xs = np.where(ink[top:top + 3].any(axis=0))[0]
    tx = ax.transData.transform((0, 0))[0]
    plt.close(fig)
    return float(xs.mean() - tx)


def test_tick_gap_under_new_rule():
    """Verify |tick-to-label gap| <= 15 px for all specified angles."""
    test_angles = [30, 45, 60, -30, -45, -60, 90, -90]
    for rot in test_angles:
        ha = "center" if abs(rot) in (0, 90) else ("right" if rot > 0 else "left")
        gap = _measure_gap(rot, ha, "default")
        assert abs(gap) <= 15.0, (
            f"Angle {rot} with ha='{ha}' produced gap {gap:.1f} px, exceeding 15 px threshold!"
        )


def test_obb_vs_ink_all_angles():
    """Verify OBB covers rendered tick label ink (>85%) across all angles."""
    from matplotlib.path import Path as MplPath

    for rot in [0, 30, 45, 60, 90, -30, -45, -60, -90]:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        ax.bar([0], [3])
        ax.set_xticks([0])
        ax.set_xticklabels(["Treatment group A"])
        ax.set_xlim(-1, 1)
        ax.set_yticks([])

        ha = "center" if abs(rot) in (0, 90) else ("right" if rot > 0 else "left")
        ax.tick_params(axis="x", length=0, labelrotation=rot)
        for l in ax.get_xticklabels():
            l.set_horizontalalignment(ha)

        fig.subplots_adjust(bottom=0.35)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        W, H = fig.canvas.get_width_height()
        label = ax.get_xticklabels()[0]
        bbox, obb = get_text_obb_and_bbox(label, renderer, img_h=H)

        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)

        assert obb is not None, f"OBB is None for rot {rot}"
        assert len(obb) == 4, f"OBB does not have 4 vertices for rot {rot}"

        arr = np.asarray(img.convert("L")).astype(int)
        poly = MplPath(obb)
        y, x = np.mgrid[:H, :W]
        pts = np.vstack((x.flatten(), y.flatten())).T
        inside = poly.contains_points(pts).reshape((H, W))

        ink_mask = (arr < 200) & (y > int(H * 0.65))
        ink_inside = (ink_mask & inside).sum()
        total_ink = ink_mask.sum()
        ratio = ink_inside / max(1, total_ink)
        assert ratio >= 0.85, (
            f"OBB missed text ink for rot {rot}: only {ratio*100:.1f}% inside OBB"
        )


def test_label_angle_profile_vs_legacy(tmp_path):
    """Verify label_angle is honored under profile, but ignored in Matplotlib under legacy."""
    # Under profile (domain_gap_v1): label_angle=45 should rotate xticklabels
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Alpha", "Beta"])
    apply_typography_variation(
        ax, domain="scientific",
        theme_config={"profile": "domain_gap_v1", "label_angle": 45}
    )
    fig.canvas.draw()
    for l in ax.get_xticklabels():
        assert l.get_rotation() == 45.0
        assert l.get_ha() == "right"
    plt.close(fig)

    # Under profile: negative angle (-30) should give ha='left'
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Alpha", "Beta"])
    apply_typography_variation(
        ax, domain="scientific",
        theme_config={"profile": "domain_gap_v1", "label_angle": -30}
    )
    fig.canvas.draw()
    for l in ax.get_xticklabels():
        assert (l.get_rotation() % 360) == (-30 % 360)
        assert l.get_ha() == "left"
    plt.close(fig)

    # Under legacy profile: label_angle=45 must NOT force rotation 45 on all runs
    # In legacy mode, Matplotlib ignores label_angle and only rotates 30% of the time with [0, 45, 90]
    np.random.seed(99)
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Alpha", "Beta"])
    apply_typography_variation(
        ax, domain="scientific",
        theme_config={"profile": "legacy", "label_angle": 45}
    )
    fig.canvas.draw()
    # At seed 99, np.random.random() is 0.672 > 0.3, so rotation stays 0.0 despite label_angle=45
    labels = ax.get_xticklabels()
    assert labels[0].get_rotation() == 0.0
    plt.close(fig)


def test_profile_forced_tick_rotation():
    """Verify configured tick_rotation angles and weights are honored under profile mode."""
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Alpha", "Beta"])
    custom_rot = {
        "mode": "profile",
        "p_rotate": 1.0,
        "angles": [60],
        "weights": [1.0],
    }
    apply_typography_variation(
        ax, domain="scientific",
        theme_config={"tick_rotation": custom_rot}
    )
    fig.canvas.draw()
    for l in ax.get_xticklabels():
        assert l.get_rotation() == 60.0
        assert l.get_ha() == "right"
    plt.close(fig)
