"""Statement-level splits and the anchor-exclusion gate.

Splitting by row (edge) leaks fragments of the same statement into both sides.
Splitting by statement id prevents that. The gate is issued separately: anchor
statements must never appear in train or validation, and the check prints the
intersection rather than assuming it is empty.
"""
from __future__ import annotations

import random
from typing import Iterable


def split_by_statement(records: Iterable[dict], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list[dict]]:
    rows = list(records)
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = [rows[i] for i in idx[:n_train]]
    val = [rows[i] for i in idx[n_train:n_train + n_val]]
    test = [rows[i] for i in idx[n_train + n_val:]]
    return {"train": train, "validation": val, "test": test}


def anchor_gate(split: dict[str, list[dict]], anchor_ids: Iterable[str]) -> dict:
    """Return the intersection of anchor ids with train/validation.

    The protocol is runnable only when every list is empty; the caller prints
    this dict into the build log.
    """
    anchors = set(anchor_ids)
    out = {}
    for part in ("train", "validation", "test"):
        ids = {str(r.get("id")) for r in split.get(part, [])}
        out[part] = sorted(ids & anchors)
    out["passed"] = not out["train"] and not out["validation"]
    return out
