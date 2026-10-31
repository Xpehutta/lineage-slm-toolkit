"""Thin adapters that turn two parsers into the same edge vocabulary.

Both parsers are optional dependencies: the toolkit's core stays dependency-free,
and the adversarial harness imports them lazily with a clear error if missing.

Edge = (target_table, target_column, source_table, source_column), lower-cased.
"""
from __future__ import annotations

Edge = tuple[str, str, str, str]


class ParserUnavailable(RuntimeError):
    pass


def _norm(edge) -> Edge:
    return tuple(str(part).strip().strip('"').lower() for part in edge)  # type: ignore[return-value]


def sqllineage_edges(sql: str) -> set[Edge]:
    """Column-level edges from sqllineage. Requires the [parsers] extra."""
    try:
        from sqllineage.runner import LineageRunner
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ParserUnavailable("sqllineage is not installed") from exc
    out: set[Edge] = set()
    runner = LineageRunner(sql, dialect="postgres")
    for item in runner.get_column_lineage():
        if len(item) < 2:
            continue
        src, tgt = item[0], item[-1]
        src_table = getattr(src.parent, "raw_name", "") or ""
        tgt_table = getattr(tgt.parent, "raw_name", "") or ""
        out.add(_norm((tgt_table, tgt.raw_name, src_table, src.raw_name)))
    return out


def _close(a: str, b: str) -> bool:
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def sqlglot_edges(sql: str) -> set[Edge]:
    """Column-level edges from sqlglot's optimizer scope.

    Deliberately simple: it resolves projection expressions against the scope
    of their own SELECT and maps them positionally to the INSERT column list.
    It is a second opinion for arbitration, not a replacement for sqllineage.
    """
    try:
        import sqlglot
        from sqlglot import exp
        from sqlglot.optimizer.scope import build_scope
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ParserUnavailable("sqlglot is not installed") from exc

    statement = sqlglot.parse_one(sql, read="postgres")
    if not isinstance(statement, exp.Insert):
        raise ValueError("expected an INSERT ... SELECT statement")

    target = statement.this
    tgt_table = target.this.name if hasattr(target, "this") else target.name
    tgt_columns = [c.name for c in getattr(target, "expressions", [])]

    edges: set[Edge] = set()

    def resolve(select: exp.Select, offset: int) -> None:
        scope = build_scope(select)
        sources = scope.sources if scope else {}
        for position, projection in enumerate(select.expressions):
            column_position = offset + position
            if column_position >= len(tgt_columns):
                break
            tgt_column = tgt_columns[column_position]
            for column in projection.find_all(exp.Column):
                table_alias = column.table
                source = sources.get(table_alias) if table_alias else None
                if source is None:
                    candidates = [v for v in sources.values() if hasattr(v, "name")]
                    source = candidates[0] if len(candidates) == 1 else None
                if source is None:
                    continue
                source_table = getattr(source, "name", "") or ""
                edges.add(_norm((tgt_table, tgt_column, source_table, column.name)))

    def walk(node: exp.Expression, offset: int = 0) -> None:
        if isinstance(node, exp.Union):
            walk(node.left, offset)
            walk(node.right, offset)
            return
        if isinstance(node, exp.Select):
            resolve(node, offset)

    walk(statement.expression)
    return edges
