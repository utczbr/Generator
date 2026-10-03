"""
synth/semantics/admin/runner.py

Headless generator subprocess runner with JSONL progress streaming and log capture (GEN-07).
Spawns generator.py in an isolated child process to guard against Agg backend / spawn state leakage.
Tails worker logs on failure and loads run_manifest.json on completion.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional


@dataclass
class GeneratorSubprocessResult:
    returncode: int
    manifest_data: Optional[Dict[str, Any]]
    failure_tail: List[str]
    log_path: str
    manifest_path: str
    progress_events: List[Dict[str, Any]]


def tail_file(path: str | Path, n_lines: int = 40) -> List[str]:
    """Return the last n_lines from a text file."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n_lines:]
    except Exception:
        return []


def run_generator_subprocess(
    args_list: List[str],
    output_dir: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    show_rich_progress: bool = False,
    timeout: Optional[float] = None,
) -> GeneratorSubprocessResult:
    """
    Launch generator.py in a dedicated subprocess (GEN-07).
    - Writes stdout/stderr to <output_dir>/run.log
    - Streams progress events from <output_dir>/progress.jsonl
    - On failure, tails the last 40 lines of run.log
    - Loads <output_dir>/run_manifest.json
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    progress_path = os.path.join(output_dir, "progress.jsonl")

    # Clear previous logs if they exist
    if os.path.exists(progress_path):
        os.remove(progress_path)

    cmd = [
        sys.executable,
        "generator.py",
        *args_list,
        "--output",
        output_dir,
        "--progress-jsonl",
        progress_path,
        "--log-file",
        log_path,
    ]

    progress_bar = None
    task_id = None
    if show_rich_progress:
        try:
            from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
            progress_bar = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress_bar.start()
        except Exception:
            progress_bar = None

    proc = subprocess.Popen(cmd)

    events: List[Dict[str, Any]] = []
    seen_lines = 0

    interrupted = False
    try:
        while proc.poll() is None:
            time.sleep(0.05)
            if os.path.exists(progress_path):
                try:
                    with open(progress_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    if len(lines) > seen_lines:
                        for l in lines[seen_lines:]:
                            l = l.strip()
                            if l:
                                evt = json.loads(l)
                                events.append(evt)
                                if progress_callback:
                                    progress_callback(evt)
                                if progress_bar:
                                    if task_id is None:
                                        total = evt.get("total", 100)
                                        task_id = progress_bar.add_task("Generating charts", total=total)
                                    progress_bar.update(task_id, advance=1)
                        seen_lines = len(lines)
                except Exception:
                    pass
        proc.wait(timeout=timeout)
    except KeyboardInterrupt:
        interrupted = True
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        if progress_bar:
            progress_bar.stop()

    # Capture any final events
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for l in lines[seen_lines:]:
                l = l.strip()
                if l:
                    evt = json.loads(l)
                    events.append(evt)
                    if progress_callback:
                        progress_callback(evt)
        except Exception:
            pass

    failure_tail = []
    if proc.returncode != 0:
        failure_tail = tail_file(log_path, n_lines=40)

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    manifest_data = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            if interrupted and not manifest_data.get("interrupted"):
                manifest_data["interrupted"] = True
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2)
        except Exception:
            pass
    elif interrupted:
        manifest_data = {
            "interrupted": True,
            "counts": {
                "successful": len([e for e in events if e.get("ok")]),
                "failed": len([e for e in events if not e.get("ok")]),
            },
        }
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
        except Exception:
            pass

    retcode = 130 if interrupted else proc.returncode

    return GeneratorSubprocessResult(
        returncode=retcode,
        manifest_data=manifest_data,
        failure_tail=failure_tail,
        log_path=log_path,
        manifest_path=manifest_path,
        progress_events=events,
    )
