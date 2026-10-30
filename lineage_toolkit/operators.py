"""Defect operators for lineage training data.

An edge is {target: {table, column}, sources: [{table, column}, ...]}.
Intensity lam is the share of edges inside one example, not the share of
examples: examples range from 3 to 190 edges, so per-example intensity would
mean a third of the labels in a small statement and one percent in a large one.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Example:
    id: str
    sql: str
    edges: list[dict] = field(default_factory=list)
    schema: list[str] = field(default_factory=list)

    def clone(self) -> "Example":
        return copy.deepcopy(self)


def _source(edge: dict) -> dict:
    return {"table": edge["table"], "column": edge["column"]}


class EdgePool:
    """All sources seen in the corpus, used to draw realistic donors."""

    def __init__(self, examples: Iterable[Example]):
        self.entries: list[tuple[dict, dict]] = []
        for ex in examples:
            for e in ex.edges:
                for s in e.get("sources", []):
                    self.entries.append((e["target"], _source(s)))
        self.tables = sorted({t["table"] for t, _ in self.entries})

    def sample_edge_outside(self, target: dict, rng: np.random.Generator, tries: int = 64):
        """A source belonging to a *different* target column."""
        for _ in range(tries):
            t, s = self.entries[int(rng.integers(0, len(self.entries)))]
            if (t["table"], t["column"]) != (target["table"], target["column"]):
                return dict(s)
        return None

    def nearest_table(self, name: str) -> str | None:
        """Table with the smallest Levenshtein distance — used by edge_confusable."""
        if not self.tables:
            return None
        return min(self.tables, key=lambda t: _levenshtein(name, t))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edge_label(example: Example, lam: float, rng: np.random.Generator, pool: EdgePool):
    """Replace one source of an edge with a source of a different target.

    Returns (example, collisions). A collision is a replacement that happened to
    equal an existing source of the same edge: the edge stays correct, so the
    effective intensity is lower than the nominal lam. Collisions are reported
    next to the effect rather than hidden.
    """
    if not example.edges:
        return example, 0
    n = max(1, int(round(len(example.edges) * lam)))
    collisions = 0
    for i in rng.choice(len(example.edges), size=n, replace=False):
        srcs = example.edges[i].get("sources", [])
        if not srcs:
            continue
        j = int(rng.integers(0, len(srcs)))
        donor = pool.sample_edge_outside(example.edges[i]["target"], rng)
        if donor is None:
            continue
        if donor in srcs:
            collisions += 1
        srcs[j] = donor
    return example, collisions


def edge_confusable(example: Example, lam: float, rng: np.random.Generator, pool: EdgePool):
    """Replace a source with a column of the table whose name is closest.

    This mimics human annotation errors: plausible table, same suffix, often the
    same schema — unlike a uniformly random replacement, which the model can
    reject because the text does not support it.
    """
    if not example.edges:
        return example, 0
    n = max(1, int(round(len(example.edges) * lam)))
    collisions = 0
    for i in rng.choice(len(example.edges), size=n, replace=False):
        srcs = example.edges[i].get("sources", [])
        if not srcs:
            continue
        j = int(rng.integers(0, len(srcs)))
        current = srcs[j]
        twin = pool.nearest_table(current["table"]) if hasattr(pool, "nearest_table") else None
        if not twin or twin == current["table"]:
            continue
        donor = {"table": twin, "column": current["column"]}
        if donor in srcs:
            collisions += 1
        srcs[j] = donor
    return example, collisions


def edge_drop(example: Example, lam: float, rng: np.random.Generator, pool: EdgePool | None = None):
    """Remove a share of edges from the reference set of one example."""
    if not example.edges:
        return example, 0
    n = max(1, int(round(len(example.edges) * lam)))
    drop = set(int(i) for i in rng.choice(len(example.edges), size=n, replace=False))
    example.edges = [e for i, e in enumerate(example.edges) if i not in drop]
    return example, 0


def edge_dup(example: Example, lam: float, rng: np.random.Generator, pool: EdgePool | None = None):
    """Duplicate a share of edges, skewing the example towards frequent targets."""
    if not example.edges:
        return example, 0
    n = max(1, int(round(len(example.edges) * lam)))
    idx = [int(i) for i in rng.choice(len(example.edges), size=n, replace=False)]
    example.edges.extend(copy.deepcopy(example.edges[i]) for i in idx)
    return example, 0


def schema_stale(example: Example, lam: float, rng: np.random.Generator, pool: EdgePool | None = None):
    """Swap a share of schema entries for stale names (renames and drops)."""
    if not example.schema:
        return example, 0
    n = max(1, int(round(len(example.schema) * lam)))
    for i in rng.choice(len(example.schema), size=n, replace=False):
        example.schema[int(i)] = example.schema[int(i)] + "_old"
    return example, 0


def sql_truncate(example: Example, lam: float, rng: np.random.Generator, pool: EdgePool | None = None):
    """Cut the statement, imitating a lost fragment of the input."""
    if not example.sql:
        return example, 0
    keep = max(1, int(len(example.sql) * (1 - lam)))
    example.sql = example.sql[:keep]
    return example, 0


OPERATORS = {
    "edge_label": edge_label,
    "edge_confusable": edge_confusable,
    "edge_drop": edge_drop,
    "edge_dup": edge_dup,
    "schema_stale": schema_stale,
    "sql_truncate": sql_truncate,
}


def apply_operator(name: str, example: Example, lam: float, rng: np.random.Generator, pool: EdgePool):
    if name not in OPERATORS:
        raise KeyError(f"unknown operator: {name}")
    return OPERATORS[name](example.clone() if name != "edge_label" else example, lam, rng, pool)
