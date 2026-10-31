"""Adversarial SQL cases with analytic oracles for the lineage contour.

Each case is a statement whose expected edges are known by construction, not by
running a parser. The point is to stop treating one golden set as a benchmark:
the classes below are the failure modes we kept finding by hand (alias reuse,
self-joins through CTEs, wide projections, correlated subqueries, unions,
quoting, window functions, schema qualification).

Every case is an INSERT ... SELECT, because column lineage only exists when
there is a target: a bare SELECT has no columns to be the origin of.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Edge = tuple[str, str, str, str]  # target_table, target_column, source_table, source_column


@dataclass(frozen=True)
class Case:
    name: str
    klass: str
    sql: str
    expected: frozenset[Edge]
    notes: str = ""
    known_gaps: tuple[str, ...] = field(default_factory=tuple)

    def missing(self, got: frozenset[Edge]) -> frozenset[Edge]:
        return self.expected - got

    def extra(self, got: frozenset[Edge]) -> frozenset[Edge]:
        return got - self.expected


def _edges(*pairs) -> frozenset[Edge]:
    return frozenset(pairs)


def alias_reuse() -> Case:
    sql = """INSERT INTO dm.client_turnover (client_id, amount)
SELECT a.client_id, a.amount
  FROM (SELECT a.client_id, a.amount
          FROM (SELECT c.client_id, p.amount
                  FROM ods.client c
                  JOIN ods.payment p ON p.client_id = c.client_id) a) a"""
    return Case(
        name="alias_reuse", klass="alias reuse", sql=sql,
        expected=_edges(
            ("dm.client_turnover", "client_id", "ods.client", "client_id"),
            ("dm.client_turnover", "amount", "ods.payment", "amount"),
        ),
        notes="the same alias a is bound at two nesting levels",
    )


def cte_self_join() -> Case:
    sql = """WITH base AS (
    SELECT c.client_id, p.amount
      FROM ods.client c
      JOIN ods.payment p ON p.client_id = c.client_id
), paired AS (
    SELECT b1.client_id, b2.amount AS other_amount
      FROM base b1
      JOIN base b2 ON b1.client_id = b2.client_id
)
INSERT INTO dm.client_pair (client_id, other_amount)
SELECT paired.client_id, paired.other_amount FROM paired"""
    return Case(
        name="cte_self_join", klass="cte self-join", sql=sql,
        expected=_edges(
            ("dm.client_pair", "client_id", "ods.client", "client_id"),
            ("dm.client_pair", "other_amount", "ods.payment", "amount"),
        ),
        notes="one CTE joined to itself through a second CTE level",
    )


def nested_subquery() -> Case:
    sql = """INSERT INTO dm.client_peak (client_id, peak_amount)
SELECT c.client_id,
       (SELECT max(p.amount) FROM ods.payment p WHERE p.client_id = c.client_id) AS peak_amount
  FROM ods.client c"""
    return Case(
        name="nested_subquery", klass="correlated subquery", sql=sql,
        expected=_edges(
            ("dm.client_peak", "client_id", "ods.client", "client_id"),
            ("dm.client_peak", "peak_amount", "ods.payment", "amount"),
        ),
        notes="aggregate in a scalar subquery correlated to the outer table",
    )


def wide_projection(sources: int = 12) -> Case:
    branches = []
    expected = [("dm.turnover_wide", "client_id", "ods.client", "client_id")]
    for i in range(sources):
        branches.append(
            f"SELECT c.client_id, p{i}.amount AS amount FROM ods.client c "
            f"JOIN ods.payment_{i} p{i} ON p{i}.client_id = c.client_id"
        )
        expected.append(("dm.turnover_wide", "amount", f"ods.payment_{i}", "amount"))
    sql = ("INSERT INTO dm.turnover_wide (client_id, amount)\n"
           + "\nUNION ALL\n".join(branches))
    return Case(
        name="wide_projection", klass="wide projection", sql=sql,
        expected=_edges(*expected),
        notes=f"{sources} sources reach one target column",
    )


def union_branches() -> Case:
    sql = """INSERT INTO dm.client_amount (client_id, amount)
SELECT c.client_id, c.amount FROM ods.client c
UNION ALL
SELECT h.client_id, h.amount FROM ods.client_hist h"""
    return Case(
        name="union_branches", klass="union branches", sql=sql,
        expected=_edges(
            ("dm.client_amount", "client_id", "ods.client", "client_id"),
            ("dm.client_amount", "amount", "ods.client", "amount"),
            ("dm.client_amount", "client_id", "ods.client_hist", "client_id"),
            ("dm.client_amount", "amount", "ods.client_hist", "amount"),
        ),
        notes="two branches give the same target column different origins",
    )


def quoted_identifiers() -> Case:
    q = chr(34)
    sql = (
        "INSERT INTO dm." + q + "ClientTotal" + q + " (" + q + "ClientId" + q + ", " + q + "Amount" + q + ")\n"
        "SELECT c." + q + "ClientId" + q + ", p." + q + "Amount" + q + "\n"
        "  FROM ods." + q + "Client" + q + " c\n"
        "  JOIN ods." + q + "Payment" + q + " p ON p." + q + "ClientId" + q + " = c." + q + "ClientId" + q
    )
    return Case(
        name="quoted_identifiers", klass="quoted identifiers", sql=sql,
        expected=_edges(
            ("dm.clienttotal", "clientid", "ods.client", "clientid"),
            ("dm.clienttotal", "amount", "ods.payment", "amount"),
        ),
        notes="mixed case and quoting; edges are compared case-folded",
    )


def window_over_derived() -> Case:
    sql = """INSERT INTO dm.client_running (client_id, running_amount)
SELECT d.client_id,
       sum(d.amount) OVER (PARTITION BY d.client_id) AS running_amount
  FROM (SELECT c.client_id, p.amount
          FROM ods.client c
          JOIN ods.payment p ON p.client_id = c.client_id) d"""
    return Case(
        name="window_over_derived", klass="window over derived", sql=sql,
        expected=_edges(
            ("dm.client_running", "client_id", "ods.client", "client_id"),
            ("dm.client_running", "running_amount", "ods.payment", "amount"),
            # the partition key is an input to the window value, not decoration
            ("dm.client_running", "running_amount", "ods.client", "client_id"),
        ),
        notes="window function applied to a derived table",
    )


def schema_qualified() -> Case:
    sql = """INSERT INTO dm.client_branch (client_id, branch_code)
SELECT public.client.client_id, ods.branch.branch_code
  FROM ods.client
  JOIN ods.branch ON ods.branch.client_id = ods.client.client_id"""
    return Case(
        name="schema_qualified", klass="schema qualification", sql=sql,
        expected=_edges(
            ("dm.client_branch", "client_id", "ods.client", "client_id"),
            ("dm.client_branch", "branch_code", "ods.branch", "branch_code"),
        ),
        notes="one table written with a schema predicate, another without",
    )


CASES = (
    alias_reuse,
    cte_self_join,
    nested_subquery,
    wide_projection,
    union_branches,
    quoted_identifiers,
    window_over_derived,
    schema_qualified,
)


def all_cases() -> list[Case]:
    return [factory() for factory in CASES]
