import numpy as np
import pytest

from lineage_toolkit.prompts import build_messages, schema_slice
from lineage_toolkit.route import needs_model
from lineage_toolkit.salvage import dedupe_edges, salvage_edges
from lineage_toolkit.train_config import QLoRAConfig


def test_salvage_recovers_complete_objects_from_truncated_text():
    text = ('noise {"target": {"table": "m", "column": "c1"}, "sources": [{"table": "r", "column": "c1"}]} '
            '{"target": {"table": "m", "column": "c2"}, "sour')
    out = salvage_edges(text)
    assert len(out) == 1 and out[0]["target"]["column"] == "c1"


def test_salvage_handles_braces_inside_strings():
    text = '{"target": {"table": "m", "column": "c}"}, "sources": []}'
    out = salvage_edges(text)
    assert len(out) == 1 and out[0]["target"]["column"] == "c}"


def test_dedupe_collapses_repetition_loops():
    obj = {"target": {"table": "m", "column": "c1"}, "sources": [{"table": "r", "column": "c1"}]}
    assert len(dedupe_edges([obj, obj, obj])) == 1


def test_schema_slice_only_keeps_mentioned_tables():
    full = {"a": ["x"], "b": ["y"], "c": ["z"]}
    assert schema_slice(full, ["a", "c", "missing"]) == {"a": ["x"], "c": ["z"]}


def test_build_messages_contains_schema_and_sql():
    msgs = build_messages("select 1", "m", {"m": ["id"]})
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "m(id)" in msgs[1]["content"] and "select 1" in msgs[1]["content"]


def test_routing_prefers_contour_and_escalates_when_empty():
    assert needs_model([], None)[1] == "empty_contour"
    assert needs_model([{"target": {}}], 0.9)[0] is False
    assert needs_model([{"target": {}}], 0.2)[1] == "low_agreement"


def test_qlora_defaults_match_the_article():
    cfg = QLoRAConfig()
    assert cfg.lora_r == 16 and cfg.lora_alpha == 16
    assert cfg.effective_batch() == 16
    assert cfg.packing is False and cfg.epochs == 3


def test_by_arity_buckets_per_target_not_per_statement():
    """One statement can hold a one-source target and an eight-source target."""
    from lineage_toolkit.metrics import by_arity

    narrow = [("dw.f", "dt", "ods.loan", "dt")]
    wide = [("dw.f", "amount", f"ods.p{i}", "amt") for i in range(8)]
    gt = set(narrow) | set(wide)
    pred = set(wide[:5]) | {("dw.f", "amount", "ods.rate", "rate")}

    buckets = by_arity([pred], [gt])

    assert buckets["<= 1"]["n_targets"] == 1
    assert buckets["<= 1"]["recall"] == 0.0
    # five of eight reference sources kept, plus one invented source
    assert buckets["6-15"]["recall"] == 0.625
    assert buckets["6-15"]["precision"] == 0.8333


def test_by_arity_flags_predictions_for_absent_targets():
    from lineage_toolkit.metrics import by_arity

    gt = {("dw.f", "amount", "ods.loan", "a")}
    pred = gt | {("dw.f", "phantom", "ods.loan", "a")}

    buckets = by_arity([pred], [gt])

    assert buckets["no reference target"]["n_targets"] == 1
    assert buckets["no reference target"]["recall"] is None
