"""Defect operators and measurement protocol for Data Lineage extraction.

The package is deliberately data-free: it ships with a synthetic example only.
Production SQL and catalog dumps are not part of the distribution.
"""
from .operators import Example, EdgePool, OPERATORS, apply_operator
from .metrics import edge_f1, macro_f1, pooled_f1, by_arity, exact_match
from .splits import split_by_statement, anchor_gate
from .stats import paired_bootstrap, mde
from .contour import agreement, compose

__all__ = [
    "Example", "EdgePool", "OPERATORS", "apply_operator",
    "edge_f1", "macro_f1", "pooled_f1", "by_arity", "exact_match",
    "split_by_statement", "anchor_gate",
    "paired_bootstrap", "mde",
    "agreement", "compose",
]
__version__ = "0.1.0"
