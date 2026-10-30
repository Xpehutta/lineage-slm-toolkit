"""Edge-level metrics for lineage extraction.

The answer is a *set* of edges, not a class label, so accuracy and AUC do not
apply. Macro-F1 averages per statement and does not let a 190-edge statement
outweigh forty ordinary ones; pooled-F1 aggregates edges and shows the cost on
the stream. Both are reported together.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

Edge = tuple[str, str, str, str]  # target_table, target_column, source_table, source_column


def edge_set(example) -> set[Edge]:
    """All (target, source) tuples of one example."""
    out: set[Edge] = set()
    for e in example.get("edges", []):
        t = e.get("target", {})
        for s in e.get("sources", []):
            out.add((t.get("table", ""), t.get("column", ""), s.get("table", ""), s.get("column", "")))
    return out


def edge_f1(pred: set[Edge], gt: set[Edge]) -> tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(preds: Sequence[set[Edge]], gts: Sequence[set[Edge]]) -> float:
    """Mean per-statement F1. An empty prediction scores 0, not skipped."""
    if not gts:
        return 0.0
    return sum(edge_f1(p, g)[2] for p, g in zip(preds, gts)) / len(gts)


def pooled_f1(preds: Sequence[set[Edge]], gts: Sequence[set[Edge]]) -> float:
    """F1 over all edges at once."""
    tp = sum(len(p & g) for p, g in zip(preds, gts))
    fp = sum(len(p - g) for p, g in zip(preds, gts))
    fn = sum(len(g - p) for p, g in zip(preds, gts))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def by_arity(preds: Sequence[set[Edge]], gts: Sequence[set[Edge]], buckets=(1, 2, 5, 15)) -> dict[str, dict]:
    """Precision and recall split by the number of sources in the reference.

    Wide targets are where models lose sources; a single F1 hides that.
    """
    groups: dict[str, list[tuple[set[Edge], set[Edge]]]] = defaultdict(list)
    for p, g in zip(preds, gts):
        arity = len({(t, c) for (t, c, _, _) in g})
        label = f"> {buckets[-1]}" if arity > buckets[-1] else f"<= {buckets[0]}" if arity <= buckets[0] else "2-5" if arity <= buckets[1] else "6-15"
        groups[label].append((p, g))
    out = {}
    for label, rows in groups.items():
        tp = sum(len(p & g) for p, g in rows)
        fp = sum(len(p - g) for p, g in rows)
        fn = sum(len(g - p) for p, g in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[label] = {"n": len(rows), "precision": round(precision, 4), "recall": round(recall, 4)}
    return dict(sorted(out.items()))


def exact_match(preds: Sequence[set[Edge]], gts: Sequence[set[Edge]]) -> float:
    """Share of statements whose edge set matches the reference exactly."""
    if not gts:
        return 0.0
    return sum(1 for p, g in zip(preds, gts) if p == g) / len(gts)
