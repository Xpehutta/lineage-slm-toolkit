"""Harness: run the adversarial cases through both parsers and the contour.

The contour module speaks in edge dicts (target/sources), the parsers return
flat tuples. This module owns the translation and the comparison against the
analytic oracle, so the report and the tests share one implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .adversarial import Case, Edge, all_cases
from .contour import compose
from .parsers import sqllineage_edges, sqlglot_edges


@dataclass(frozen=True)
class CaseResult:
    case: Case
    sqllineage: frozenset[Edge]
    sqlglot: frozenset[Edge]
    chosen: frozenset[Edge]
    source: str
    conflicts: list[str]

    @property
    def expected_canon(self) -> frozenset[Edge]:
        return frozenset(canon(e) for e in self.case.expected)

    @property
    def chosen_canon(self) -> frozenset[Edge]:
        return frozenset(canon(e) for e in self.chosen)

    @property
    def hit(self) -> int:
        return len(self.expected_canon & self.chosen_canon)

    @property
    def missed(self) -> frozenset[Edge]:
        return self.expected_canon - self.chosen_canon

    @property
    def extra(self) -> frozenset[Edge]:
        return self.chosen_canon - self.expected_canon

    @property
    def perfect(self) -> bool:
        return not self.missed and not self.extra


def canon(edge: Edge) -> Edge:
    """Compare on the table basename: sqllineage drops the schema prefix."""
    tgt_table, tgt_col, src_table, src_col = edge
    return (tgt_table.split(".")[-1], tgt_col, src_table.split(".")[-1], src_col)


def to_dict_edges(edges: frozenset[Edge]) -> list[dict]:
    by_target: dict[tuple[str, str], list[dict]] = {}
    for tgt_table, tgt_col, src_table, src_col in sorted(edges):
        by_target.setdefault((tgt_table, tgt_col), []).append({"table": src_table, "column": src_col})
    return [
        {"target": {"table": t, "column": c}, "sources": sources}
        for (t, c), sources in by_target.items()
    ]


def run_case(case: Case) -> CaseResult:
    primary = frozenset(sqlglot_edges(case.sql))   # primary contour
    secondary = frozenset(sqllineage_edges(case.sql))  # second opinion
    chosen, info = compose(to_dict_edges(primary), to_dict_edges(secondary))
    chosen_edges = frozenset(
        (e["target"]["table"], e["target"]["column"], s["table"], s["column"])
        for e in chosen for s in e.get("sources", [])
    )
    return CaseResult(
        case=case, sqllineage=secondary, sqlglot=primary,
        chosen=chosen_edges, source=info["source"], conflicts=info.get("conflicts", []),
    )


def run_all() -> list[CaseResult]:
    return [run_case(case) for case in all_cases()]
