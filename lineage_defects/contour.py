"""Statement-level arbitration between two deterministic parsers.

Two parsers fail differently: one is precise on explicit projections, the other
recovers edges from nested subqueries. The result is chosen per statement, not
merged per edge. Disagreements on a target are collected as an audit list and
never averaged into the contour.
"""
from __future__ import annotations

from typing import Iterable

THIN_RATIO = 0.4


def _key(e: dict) -> tuple:
    t = e.get("target", {})
    srcs = tuple(sorted((s.get("table", ""), s.get("column", "")) for s in e.get("sources", [])))
    return (t.get("table", "").lower(), t.get("column", "").lower(), srcs)


def agreement(a: Iterable[dict], b: Iterable[dict]) -> float:
    """Jaccard of edge keys at statement level."""
    sa, sb = {_key(e) for e in a or []}, {_key(e) for e in b or []}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def conflicts(a: Iterable[dict], b: Iterable[dict]) -> list[str]:
    by_target: dict[tuple, set] = {}
    for edges in (a, b):
        for e in edges or []:
            t = e.get("target", {})
            key = (t.get("table", "").lower(), t.get("column", "").lower())
            by_target.setdefault(key, set()).add(
                tuple(sorted((s.get("table", ""), s.get("column", "")) for s in e.get("sources", [])))
            )
    return sorted(f"{k[0]}.{k[1]}" for k, v in by_target.items() if len(v) > 1)


def compose(primary: Iterable[dict], secondary: Iterable[dict], thin_ratio: float = THIN_RATIO):
    """Pick the contour for one statement.

    Rules: secondary empty -> primary; secondary thinner than thin_ratio of
    primary -> primary (marked thin); otherwise secondary. Conflicts are
    reported, not merged.
    """
    a, b = list(primary or []), list(secondary or [])
    info = {"source": "primary", "thin": False}
    if not b:
        chosen, other = a, []
    elif a and len(b) < thin_ratio * len(a):
        chosen, other = a, b
        info["thin"] = True
    else:
        chosen, other = b, a
        info["source"] = "secondary"
    info["conflicts"] = conflicts(chosen, other)
    return chosen, info
