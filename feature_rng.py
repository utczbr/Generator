"""
feature_rng.py

Feature-scoped RNG isolation (FR-082, OC-3).
Provides deterministic, isolated random.Random instances for specific features
derived from (seed, image_idx, name) using hashlib/crc32, never Python's built-in hash().
Ensures new features consume zero draws from global random and remain immune to PYTHONHASHSEED.
"""
from __future__ import annotations

import hashlib
import random
from typing import Union


def feature_seed(seed: int, image_idx: int, name: str) -> int:
    """
    Derive a deterministic integer seed from (seed, image_idx, name).
    Uses hashlib.sha256 to guarantee invariance across process runs and PYTHONHASHSEED.
    """
    token = f"{int(seed)}:{int(image_idx)}:{name}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def feature_rng(seed: int, image_idx: int, name: str) -> random.Random:
    """
    Return an isolated, deterministic random.Random instance for the named feature.
    Mutations on the returned generator do not perturb the global random or np.random state.
    """
    return random.Random(feature_seed(seed, image_idx, name))
