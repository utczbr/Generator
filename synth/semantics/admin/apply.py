"""
synth/semantics/admin/apply.py

Transactional manifest modification and disaster-recovery engine (ADM-08, ADM-10).
Guarantees zero half-written state:
1. File locking: O_EXCL with PID and stale lock detection.
2. Timestamped snapshot backup: domains/.backups/<UTC-timestamp>/
3. Candidate in-memory compilation and Pydantic validation.
4. Semantic invariant checks (SEM-03 golden subset, no degenerate Cartesian pools).
5. Atomic round-trip disk write via ruamel.yaml + temp file + fsync + os.replace.
6. Automatic rollback on any failure.
7. Backup listing and restoration API.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set
from ruamel.yaml import YAML

from synth.semantics.admin.plan import IngestPlan
from synth.semantics.loader import (
    CARTESIAN_CHART_TYPES,
    SemanticRegistry,
    get_domains_dir,
    rebuild_cache,
)
from synth.semantics.schema import DomainManifest


class ApplyError(RuntimeError):
    """Raised when plan application or validation fails during transactional apply."""
    pass


class LockError(RuntimeError):
    """Raised when manifest directory lock cannot be acquired."""
    pass


class FileLock:
    """PID-tagged file lock with stale-lock detection (ADM-10)."""

    def __init__(self, lock_file: Path, timeout_secs: float = 10.0, stale_secs: float = 600.0):
        self.lock_file = lock_file
        self.timeout_secs = timeout_secs
        self.stale_secs = stale_secs
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        start = time.time()
        while True:
            try:
                self._fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                pid_bytes = f"{os.getpid()}\n".encode("utf-8")
                os.write(self._fd, pid_bytes)
                return
            except FileExistsError:
                # Check stale lock
                try:
                    mtime = self.lock_file.stat().st_mtime
                    if (time.time() - mtime) > self.stale_secs:
                        try:
                            self.lock_file.unlink()
                            continue
                        except OSError:
                            pass
                except OSError:
                    pass

                if (time.time() - start) >= self.timeout_secs:
                    raise LockError(f"Failed to acquire manifest lock on {self.lock_file} after {self.timeout_secs}s")
                time.sleep(0.1)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def _create_snapshot(domains_dir: Path) -> Path:
    """Create timestamped snapshot directory under domains/.backups/<UTC-ts>/."""
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = domains_dir / ".backups" / utc_now
    backup_dir.mkdir(parents=True, exist_ok=True)

    for yf in domains_dir.glob("*.yaml"):
        shutil.copy2(yf, backup_dir / yf.name)

    return backup_dir


def _restore_snapshot(backup_dir: Path, domains_dir: Path) -> None:
    """Restore all yaml files from backup_dir to domains_dir."""
    if not backup_dir.exists():
        return
    for yf in backup_dir.glob("*.yaml"):
        shutil.copy2(yf, domains_dir / yf.name)


def list_backups(domains_dir: Optional[Path] = None) -> List[str]:
    """List available backup timestamps in reverse chronological order."""
    target_dir = domains_dir or get_domains_dir()
    backups_root = target_dir / ".backups"
    if not backups_root.exists():
        return []

    dirs = [d.name for d in backups_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return sorted(dirs, reverse=True)


def restore_backup(backup_ts: str, domains_dir: Optional[Path] = None) -> bool:
    """Restore manifests to a specific backup timestamp."""
    target_dir = domains_dir or get_domains_dir()
    backup_dir = target_dir / ".backups" / backup_ts
    if not backup_dir.exists():
        raise FileNotFoundError(f"Backup timestamp '{backup_ts}' not found in {target_dir / '.backups'}")

    lock_path = target_dir / ".lock"
    with FileLock(lock_path):
        _restore_snapshot(backup_dir, target_dir)
        rebuild_cache(force=True, domains_dir=target_dir)

    return True


def apply_batch_plans(
    plans: List[IngestPlan],
    domains_dir: Optional[Path] = None,
    verify_tests: bool = False,
) -> Dict[str, Any]:
    """
    Transactional batch apply engine (ADM-08, ADM-10):
    1. Lock directory
    2. Snapshot backup
    3. Validate candidate manifests via Pydantic from accumulated final candidate state
    4. Validate candidate registry invariants
    5. Write round-trip YAMLs via atomic tmp + fsync + os.replace
    6. Rebuild cache once
    7. Optional verify subprocess
    8. Rollback to snapshot on any failure
    """
    if not plans:
        return {
            "success": True,
            "snapshot": None,
            "touched_domains": [],
            "metrics_added": 0,
            "metrics_shared": 0,
            "titles_added": 0,
            "pairs_added": 0,
        }

    for p in plans:
        if p.has_abort_condition:
            src = p.source_file or p.target_domain
            raise ApplyError(f"Plan for '{src}' has abort/error conditions. Refusing to apply batch.")

    target_dir = domains_dir or get_domains_dir()
    lock_path = target_dir / ".lock"

    with FileLock(lock_path):
        snapshot_dir = _create_snapshot(target_dir)
        try:
            # Candidate manifests are taken from the last plan (which chained all mutations)
            final_candidates = plans[-1].candidate_manifests

            # 1. Validate each candidate manifest with Pydantic
            for dom_id, candidate_data in final_candidates.items():
                try:
                    DomainManifest.model_validate(candidate_data)
                except Exception as exc:
                    raise ApplyError(f"Pydantic validation failed for domain '{dom_id}': {exc}")

            # 2. Compile candidate registry in-memory
            candidate_reg = SemanticRegistry.load(raw=final_candidates)

            # Determine touched domains across all plans
            touched_domains: Set[str] = set()
            for p in plans:
                touched_domains.add(p.target_domain)
                for a in p.metric_actions:
                    if a.action == "SHARE":
                        touched_domains.add(a.target_manifest)

            # 3. Check Cartesian pool coverage in candidate registry
            for touched_domain in touched_domains:
                if touched_domain in candidate_reg.domains and touched_domain != "common":
                    dom_metrics = tuple(m for m in candidate_reg.axis_metrics_catalog if touched_domain in m.domain_tags)
                    if not dom_metrics:
                        raise ApplyError(f"Touched domain '{touched_domain}' has zero metrics in candidate registry.")

            # 4. Write modified files via ruamel.yaml round-trip
            yaml = YAML()
            yaml.preserve_quotes = True
            yaml.indent(mapping=2, sequence=2, offset=0)

            for dom_id in touched_domains:
                if dom_id not in final_candidates:
                    continue
                data = final_candidates[dom_id]
                out_path = target_dir / f"{dom_id}.yaml"
                tmp_path = target_dir / f"{dom_id}.yaml.tmp.{os.getpid()}"

                with open(tmp_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(tmp_path, out_path)

            # 5. Rebuild cache
            rebuild_cache(force=True, domains_dir=target_dir)

            # 6. Optional verification test suite
            if verify_tests:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "tests/test_domain_manifests.py"],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise ApplyError(f"Verification test suite failed:\n{proc.stdout}\n{proc.stderr}")

            return {
                "success": True,
                "snapshot": str(snapshot_dir),
                "touched_domains": sorted(touched_domains),
                "metrics_added": sum(len([a for a in p.metric_actions if a.action == "APPEND"]) for p in plans),
                "metrics_shared": sum(len([a for a in p.metric_actions if a.action == "SHARE"]) for p in plans),
                "titles_added": sum(len([a for a in p.title_actions if a.action == "APPEND"]) for p in plans),
                "pairs_added": sum(len([a for a in p.pair_actions if a.action == "APPEND"]) for p in plans),
            }

        except Exception as exc:
            # Automatic rollback to snapshot
            _restore_snapshot(snapshot_dir, target_dir)
            rebuild_cache(force=True, domains_dir=target_dir)
            raise ApplyError(f"Transaction aborted and rolled back to snapshot {snapshot_dir.name}: {exc}") from exc


def apply_plan(
    plan: IngestPlan,
    domains_dir: Optional[Path] = None,
    verify_tests: bool = False,
) -> Dict[str, Any]:
    """
    Transactional apply engine for a single plan (ADM-10).
    Delegates to apply_batch_plans.
    """
    return apply_batch_plans([plan], domains_dir=domains_dir, verify_tests=verify_tests)

