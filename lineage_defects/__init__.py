"""Measuring instruments for Data Lineage extraction with compact language models.

Two groups:

* contour: arbitration between deterministic parsers, routing to the model,
  quote/case normalisation, JSON salvage for truncated answers;
* protocol: defect operators, edge-level metrics, statement-level splits,
  paired statistics.

The package is data-free: a synthetic example only, no production SQL.
"""
from .contour import agreement, compose, THIN_RATIO
from .route import needs_model, AGREEMENT_THRESHOLD
from .salvage import salvage_edges, dedupe_edges
from .prompts import build_messages, schema_slice, SYSTEM
from .train_config import QLoRAConfig
from .metrics import edge_f1, macro_f1, pooled_f1, by_arity, exact_match, edge_set
from .splits import split_by_statement, anchor_gate
from .stats import paired_bootstrap, mde, required_n
from .operators import Example, EdgePool, OPERATORS, apply_operator

__all__ = [
    "agreement", "compose", "THIN_RATIO",
    "needs_model", "AGREEMENT_THRESHOLD",
    "salvage_edges", "dedupe_edges",
    "build_messages", "schema_slice", "SYSTEM",
    "QLoRAConfig",
    "edge_f1", "macro_f1", "pooled_f1", "by_arity", "exact_match", "edge_set",
    "split_by_statement", "anchor_gate",
    "paired_bootstrap", "mde", "required_n",
    "Example", "EdgePool", "OPERATORS", "apply_operator",
]
__version__ = "0.2.0"
