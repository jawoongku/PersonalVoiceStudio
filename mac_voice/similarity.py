"""Speaker embedding similarity primitives; model extraction remains caller-owned."""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embeddings must be non-empty and have equal dimensions")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) ** 2 for a in left))
    right_norm = math.sqrt(sum(float(b) ** 2 for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("embeddings must not be zero vectors")
    return dot / (left_norm * right_norm)
