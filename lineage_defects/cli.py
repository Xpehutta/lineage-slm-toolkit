"""Command-line demo on synthetic data — no production SQL is shipped."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contour import agreement, compose
from .metrics import by_arity, exact_match, macro_f1, pooled_f1
from .operators import OPERATORS, EdgePool, Example
from .prompts import build_messages, schema_slice
from .route import needs_model
from .salvage import dedupe_edges, salvage_edges
from .splits import anchor_gate, split_by_statement
from .stats import mde, required_n
from .train_config import QLoRAConfig

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "synthetic_statements.json"
TRUNCATED = '{"edges": [{"target": {"table": "m", "column": "c1"}, "sources": [{"table": "r", "column": "c1"}]}, {"target": {"table": "m", "colu'


def load_examples(path: Path = EXAMPLE) -> list[Example]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Example(id=r["id"], sql=r["sql"], edges=r["edges"], schema=r.get("schema", [])) for r in rows]


def demo(lam: float = 0.25, seed: int = 42) -> None:
    examples = load_examples()
    pool = EdgePool(examples)
    rng = np.random.default_rng(seed)

    print(f"examples: {len(examples)}, edges: {sum(len(e.edges) for e in examples)}")
    print(f"{'operator':16s} {'macro F1':>9s} {'pooled':>8s} {'exact':>7s} {'collisions':>11s}")
    for name in OPERATORS:
        preds, gts, cols = [], [], 0
        for ex in examples:
            corrupted, collisions = OPERATORS[name](ex.clone(), lam, rng, pool)
            cols += collisions
            preds.append({(e["target"]["table"], e["target"]["column"], s["table"], s["column"])
                          for e in corrupted.edges for s in e.get("sources", [])})
            gts.append({(e["target"]["table"], e["target"]["column"], s["table"], s["column"])
                        for e in ex.edges for s in e.get("sources", [])})
        print(f"{name:16s} {macro_f1(preds, gts):9.3f} {pooled_f1(preds, gts):8.3f} {exact_match(preds, gts):7.3f} {cols:11d}")

    first = examples[0]
    slice_ = schema_slice({t: ["id", "amount"] for t in first.schema}, first.schema)
    msgs = build_messages(first.sql, first.edges[0]["target"]["table"], slice_)
    print(f"prompt: {len(msgs)} messages, {len(msgs[1]['content'])} chars in the user turn")
    print(f"routing: {needs_model([], agreement([], []))}, {needs_model(first.edges, 0.9)}")

    recovered = dedupe_edges(salvage_edges(TRUNCATED))
    print(f"salvage: {len(recovered)} complete edge(s) from a truncated answer")

    cfg = QLoRAConfig()
    print(f"QLoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}, lr={cfg.learning_rate}, "
          f"effective batch={cfg.effective_batch()}, packing={cfg.packing}")

    split = split_by_statement([{"id": e.id} for e in examples])
    gate = anchor_gate(split, [e.id for e in examples[:2]])
    print(f"anchor gate: {gate}")
    print(f"MDE at n=10, sd=0.11: {mde(0.11, 10):.3f} F1; anchors needed for 0.05: {required_n(0.11, 0.05)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo of the lineage extraction toolkit on synthetic data")
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    demo(args.lam, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
