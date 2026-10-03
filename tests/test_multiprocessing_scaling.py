"""
tests/test_multiprocessing_scaling.py

Concurrency stress test and memory hygiene verification across ProcessPoolExecutor.
Validates:
a) Bounded resident memory (RSS) across tasks (no linear memory leaks, plt.close functional).
b) PRNG uniqueness: SHA256 hashes of rendered images and tabular data show 0 collisions.
c) Multi-engine concurrency: mixed Matplotlib and Vega-Lite batches complete without GIL deadlock.
"""
import copy
import hashlib
import json
import os
import multiprocessing as mp
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import pytest

from custom_config import OCR_TRAINING_CONFIG
from generator import generate_single_chart


def _worker_generate_task(args: Tuple[int, str, int, str]) -> Dict[str, Any]:
    """Top-level worker function for ProcessPoolExecutor execution."""
    task_id, engine, seed, output_base = args
    pid = os.getpid()

    # Re-seed worker PRNG state per task
    random.seed(seed)
    np.random.seed(seed)

    cfg = copy.deepcopy(OCR_TRAINING_CONFIG)
    cfg["num_images"] = 1000
    cfg["debug_mode"] = False
    cfg["engine"] = engine
    cfg["seed"] = seed

    # Alternate chart types to stress various paths
    canonical_types = ["bar", "line", "scatter"]
    chosen_type = canonical_types[task_id % len(canonical_types)]
    cfg["chart_type"] = chosen_type
    cfg["chart_types"] = {chosen_type: {"weight": 100, "enabled": True}}

    images_dir = os.path.join(output_base, "images")
    labels_dir = os.path.join(output_base, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    try:
        generate_single_chart(task_id, cfg, images_dir, labels_dir, output_base)

        img_path = os.path.join(images_dir, f"chart_{task_id:05d}.png")
        det_path = os.path.join(labels_dir, f"chart_{task_id:05d}_detailed.json")

        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_hash = hashlib.sha256(img_bytes).hexdigest()

        with open(det_path, "rb") as f:
            det_bytes = f.read()
        data_hash = hashlib.sha256(det_bytes).hexdigest()

        import gc
        gc.collect()

        rss_mb = psutil.Process(pid).memory_info().rss / (1024.0 * 1024.0)

        return {
            "task_id": task_id,
            "engine": engine,
            "pid": pid,
            "img_hash": img_hash,
            "data_hash": data_hash,
            "rss_mb": rss_mb,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "engine": engine,
            "pid": pid,
            "img_hash": None,
            "data_hash": None,
            "rss_mb": 0.0,
            "success": False,
            "error": str(e),
        }


def test_multiprocessing_concurrency_and_memory_hygiene():
    """Execute 100+ concurrent chart tasks across ProcessPoolExecutor workers."""
    num_tasks = 104  # 100+ tasks
    num_workers = os.cpu_count() or 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Build tasks: 50% Matplotlib, 50% Vega-Lite with unique deterministic seeds
        task_args = []
        for i in range(num_tasks):
            engine = "matplotlib" if (i % 2 == 0) else "vegalite"
            task_seed = 10000 + i * 37
            task_args.append((i, engine, task_seed, tmp_dir))

        results = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_worker_generate_task, arg) for arg in task_args]
            for fut in as_completed(futures):
                results.append(fut.result())

        # 1. Concurrency sanity: all tasks succeed without deadlock or crashes
        assert len(results) == num_tasks, f"Expected {num_tasks} results, received {len(results)}"
        failed = [r for r in results if not r["success"]]
        assert not failed, f"{len(failed)} tasks failed with errors: {[f['error'] for f in failed[:3]]}"

        # 2. Multi-engine representation
        engines_seen = {r["engine"] for r in results}
        assert engines_seen == {"matplotlib", "vegalite"}, f"Expected both engines, saw: {engines_seen}"

        # 3. PRNG Uniqueness Check: zero duplicate charts or seed collisions
        img_hashes = [r["img_hash"] for r in results]
        data_hashes = [r["data_hash"] for r in results]
        assert len(set(img_hashes)) == len(img_hashes), "Duplicate image hashes detected across concurrent tasks!"
        assert len(set(data_hashes)) == len(data_hashes), "Duplicate data hashes detected across concurrent tasks!"

        # 4. Memory Leak Check: RSS remains bounded per worker process
        tasks_by_pid = {}
        for r in sorted(results, key=lambda x: x["task_id"]):
            tasks_by_pid.setdefault(r["pid"], []).append(r["rss_mb"])

        for pid, rss_seq in tasks_by_pid.items():
            if len(rss_seq) >= 8:
                # Compare steady-state: use midpoint once both engines have initialized
                warm_idx = len(rss_seq) // 2
                warm_rss = rss_seq[warm_idx]
                final_rss = rss_seq[-1]
                growth_mb = final_rss - warm_rss
                # Assert bounded memory footprint: worker RSS should not expand continuously
                assert growth_mb < 65.0, (
                    f"PID {pid} exhibited excessive memory growth: "
                    f"warm={warm_rss:.1f}MB, final={final_rss:.1f}MB (+{growth_mb:.1f}MB)"
                )
