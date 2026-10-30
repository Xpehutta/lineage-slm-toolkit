import numpy as np
import pytest

from lineage_defects.prompts import build_messages, schema_slice
from lineage_defects.route import needs_model
from lineage_defects.salvage import dedupe_edges, salvage_edges
from lineage_defects.train_config import QLoRAConfig


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
