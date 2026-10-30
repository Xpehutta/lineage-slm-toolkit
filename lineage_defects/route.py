"""Routing: when the deterministic contour is not enough.

The model is not called for every statement. It is called when the contour is
empty, or when the two parsers disagree so strongly that the contour cannot be
trusted. Everything else stays deterministic and free.
"""
from __future__ import annotations

AGREEMENT_THRESHOLD = 0.5


def needs_model(contour_edges: list[dict], agreement: float | None) -> tuple[bool, str]:
    if not contour_edges:
        return True, "empty_contour"
    if agreement is not None and agreement < AGREEMENT_THRESHOLD:
        return True, "low_agreement"
    return False, "contour"
