"""Tests for the adversarial SQL harness: the oracle must be a real oracle."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlglot")
pytest.importorskip("sqllineage")

from lineage_toolkit.adversarial import all_cases
from lineage_toolkit.adversarial_harness import run_all

CASES = {case.name: case for case in all_cases()}


def test_every_case_has_edges_and_sql():
    for case in CASES.values():
        # a CTE case starts with WITH, the target still comes from an INSERT
        assert "INSERT INTO" in case.sql.upper()
        assert case.expected, f"{case.name} has an empty oracle"


def test_case_names_unique():
    names = [case.name for case in all_cases()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_contour_recovers_the_expected_edges(case):
    """The composed contour must reproduce the analytic oracle exactly."""
    result = next(r for r in run_all() if r.case.name == case.name)
    assert not result.missed, f"{case.name} missed {sorted(result.missed)}"
    assert not result.extra, f"{case.name} produced extra {sorted(result.extra)}"


def test_wide_projection_really_is_wide():
    case = CASES["wide_projection"]
    sources = {(e[2], e[3]) for e in case.expected if e[1] == "amount"}
    assert len(sources) >= 12
