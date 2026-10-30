import numpy as np
import pytest

from lineage_toolkit.contour import agreement, compose
from lineage_toolkit.metrics import exact_match, macro_f1, pooled_f1
from lineage_toolkit.operators import EdgePool, Example, edge_label
from lineage_toolkit.splits import anchor_gate, split_by_statement
from lineage_toolkit.stats import mde, required_n


def _ex(n_edges=3):
    edges = [
        {"target": {"table": "m", "column": f"c{i}"}, "sources": [{"table": "r", "column": f"c{i}"}]}
        for i in range(n_edges)
    ]
    return Example(id="e1", sql="select 1", edges=edges, schema=["m", "r"])


def test_edge_label_empty_example_guard():
    ex = Example(id="empty", sql="", edges=[], schema=[])
    out, collisions = edge_label(ex, 0.2, np.random.default_rng(0), EdgePool([ex]))
    assert out.edges == [] and collisions == 0


def test_edge_label_replaces_one_source_per_edge():
    ex = _ex(3)
    pool = EdgePool([ex])
    out, _ = edge_label(ex, 1.0, np.random.default_rng(0), pool)
    assert len(out.edges) == 3
    for e in out.edges:
        assert len(e["sources"]) == 1


def test_macro_f1_empty_prediction_scores_zero():
    gt = [{"m", "c1", "r", "c1"}]
    assert macro_f1([set()], gt) == 0.0
    assert pooled_f1([set()], gt) == 0.0
    assert exact_match([set()], gt) == 0.0


def test_split_by_statement_keeps_statements_whole():
    rows = [{"id": f"s{i}"} for i in range(100)]
    split = split_by_statement(rows, seed=7)
    ids = [r["id"] for part in split.values() for r in part]
    assert len(ids) == len(set(ids)) == 100


def test_anchor_gate_flags_leak():
    split = {"train": [{"id": "a"}], "validation": [], "test": [{"id": "b"}]}
    gate = anchor_gate(split, ["a"])
    assert gate["passed"] is False and gate["train"] == ["a"]
    assert anchor_gate(split, ["b"])["passed"] is True


def test_mde_and_required_n_consistent():
    assert mde(0.11, 10) == pytest.approx(0.0974, abs=5e-3)
    assert required_n(0.11, 0.05) == 38


def test_compose_prefers_secondary_unless_thin():
    a = [{"target": {"table": "m", "column": f"c{i}"}, "sources": [{"table": "r", "column": f"c{i}"}]} for i in range(3)]
    thin = [{"target": {"table": "m", "column": "c1"}, "sources": [{"table": "r", "column": "c1"}]}]
    chosen, info = compose(a, thin)
    assert info["thin"] is True and chosen == a
    chosen2, info2 = compose(a, a)
    assert info2["source"] == "secondary" and chosen2 == a


def test_agreement_identical_is_one():
    a = [{"target": {"table": "m", "column": "c1"}, "sources": [{"table": "r", "column": "c1"}]}]
    assert agreement(a, a) == 1.0
