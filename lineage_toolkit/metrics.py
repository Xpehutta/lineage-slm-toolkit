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


def _arity_label(arity: int, buckets: Sequence[int]) -> str:
    """Bucket label for a target with this many sources: 1, 2-5, 6-15, >15."""
    low = 1
    for bound in buckets:
        if arity <= bound:
            return f"<= {low}" if low == bound == 1 else f"{low}-{bound}" if low > 1 else f"<= {bound}"
        low = bound + 1
    return f"> {buckets[-1]}"


def by_arity(preds: Sequence[set[Edge]], gts: Sequence[set[Edge]], buckets=(1, 5, 15)) -> dict[str, dict]:
    """Precision and recall split by the number of sources feeding each target.

    Arity belongs to a target column, not to a statement: one SELECT can carry a
    single-source column and a thirty-source column at once. Bucketing per
    target exposes the missing-source failure on wide targets, which a
    statement-level score averages away.

    Predictions for a target absent from the reference carry no arity, so they
    are collected under "no reference target": precision only, recall undefined.
    """
    acc: dict[str, dict] = defaultdict(lambda: {"n_targets": 0, "tp": 0, "fp": 0, "fn": 0})
    for p, g in zip(preds, gts):
        gt_sources: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for (t, c, st, sc) in g:
            gt_sources[(t, c)].add((st, sc))
        pred_sources: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for (t, c, st, sc) in p:
            pred_sources[(t, c)].add((st, sc))
        for target, ref in gt_sources.items():
            cell = acc[_arity_label(len(ref), buckets)]
            cell["n_targets"] += 1
            got = pred_sources.get(target, set())
            cell["tp"] += len(got & ref)
            cell["fp"] += len(got - ref)
            cell["fn"] += len(ref - got)
        for target, got in pred_sources.items():
            if target in gt_sources:
                continue
            cell = acc["no reference target"]
            cell["n_targets"] += 1
            cell["fp"] += len(got)
    out = {}
    for label, cell in acc.items():
        tp, fp, fn = cell["tp"], cell["fp"], cell["fn"]
        out[label] = {
            "n_targets": cell["n_targets"],
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    return dict(sorted(out.items()))


def exact_match(preds: Sequence[set[Edge]], gts: Sequence[set[Edge]]) -> float:
    """Share of statements whose edge set matches the reference exactly."""
    if not gts:
        return 0.0
    return sum(1 for p, g in zip(preds, gts) if p == g) / len(gts)
