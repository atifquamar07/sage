"""Deterministic hashing and seed derivation.

All per-question randomness in SAGE is derived from ``(seed, question identity)``,
so results do not depend on worker count or on the order in which questions run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """SHA-256 of a canonical JSON encoding (sorted keys, compact, UTF-8)."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_seed(seed: int, purpose: str, question_id: object) -> int:
    """Derive an independent integer seed for one purpose of one question."""
    digest = sha256_json({"seed": int(seed), "purpose": purpose, "question": str(question_id)})
    return int(digest, 16)
