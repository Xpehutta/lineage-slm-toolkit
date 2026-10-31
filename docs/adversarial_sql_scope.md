# Adversarial SQL harness — scope

## Why

The toolkit ships one synthetic example and one edge-level reference set (DDL10).
A single golden set certifies exactly one shape of SQL, and our own failures were
invisible in it: wide targets collapsed to one source and pooled F1 barely moved.

This harness replaces the golden set with a generator: statements whose expected
edges are known by construction, one class per failure mode. The question it
answers is not "what is the F1" but "which shapes of SQL does the contour break on".

## What a case is

A case is an `INSERT ... SELECT` statement plus a frozen set of expected edges
`(target_table, target_column, source_table, source_column)`. The oracle is
analytic: the generator builds the query and the edge set from the same plan, so
no parser is involved in producing the truth.

Column lineage is undefined without a target, which is why every case writes into
a `dm.*` table rather than being a bare SELECT.

## Case classes (first pass)

| Class | What it probes |
|---|---|
| alias reuse | the same alias bound at two nesting levels |
| cte self-join | one CTE joined to itself through a second level |
| correlated subquery | an aggregate in a scalar subquery tied to the outer row |
| wide projection | twelve sources reaching one target column via UNION ALL |
| union branches | one target column with different origins per branch |
| quoted identifiers | mixed case and quoting; comparison is case-folded |
| window over derived | a window function applied to a derived table |
| schema qualification | the same table written with and without a schema |

## How a case is judged

Both parsers run, the contour arbitrates between them by its own rules, and the
result is compared with the oracle on table basename (sqllineage drops the schema
prefix), column name and case.

A case passes when every expected edge is present and nothing extra is produced.
Known parser gaps are recorded per case in `known_gaps` and surfaced in the
report instead of being silently tolerated.

## Non-goals

- Not a benchmark: there is no score to optimise, only a list of shapes to fix.
- Not a parser test suite: it measures the contour's composed output, which is
  what the pipeline actually consumes.
- No production SQL. Names are invented; the shapes are what matters.
- No model in the loop in this pass. The same cases can later be run against the
  SLM to test whether it inherits the parser's blind spots.

## How to run

```bash
pip install -e ".[parsers]"
python scripts/adversarial_report.py           # table + results/adversarial_report.json
pytest -q tests/test_adversarial.py
```

Exit code is non-zero when any case fails, so CI can gate on it.
