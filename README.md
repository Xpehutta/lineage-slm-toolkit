# lineage-defects

Defect operators, metrics and a statistical protocol for measuring how **training-data defects** affect **Data Lineage extraction** models.

Companion code for the article *«Протокол измерения и базовые линии: что уже измерено и что предстоит измерить»* (How to measure the sensitivity of a compact language model to training-data defects).

The repository is deliberately **data-free**: it ships a small synthetic example only. No production SQL, no catalog dumps, no bank schemas.

## What problem it solves

A data-quality report says *how much* of the data is broken. It does not say *how much worse the model gets*. This package provides the measuring instruments:

- **defect operators** that corrupt the *labels* of a lineage example (an edge = target column plus its source columns), not just the inputs;
- **metrics** suited to structural extraction (macro/pooled edge F1, precision/recall by target arity, exact match);
- **protocol helpers**: statement-level splits, anchor-exclusion gate, paired bootstrap, minimum detectable effect;
- **contour arbitration** between two deterministic parsers, with an audit list of conflicts instead of a silent merge.

## Install

```bash
pip install -e ".[dev]"
pytest -q
lineage-defects --lam 0.25
```

## Defect operators

| Operator | What it does |
|---|---|
| `edge_label` | replaces **one** source of an edge with a source of a **different** target |
| `edge_confusable` | replaces a source with a column of the table whose name is closest (Levenshtein) |
| `edge_drop` | removes a share of edges from the reference set |
| `edge_dup` | duplicates edges, skewing the example towards frequent targets |
| `schema_stale` | renames schema entries, imitating stale catalog data |
| `sql_truncate` | cuts the statement, imitating a lost input fragment |

Intensity λ is the share of **edges inside one example**, not the share of examples — statements range from 3 to 190 edges, so per-example intensity would mean a third of the labels in a small statement and one percent in a large one.

`edge_label` counts **collisions**: replacements that accidentally equal an existing source and therefore leave the edge correct. Effective intensity is lower than nominal λ by that share, so collisions are reported next to the effect.

## Metrics

- **macro F1** — mean per-statement F1; an empty prediction scores 0, not skipped;
- **pooled F1** — F1 over all edges; shows the cost on the stream;
- **precision/recall by arity** — 1 source, 2–5, 6–15, more than 15. Wide targets are where models lose sources; a single F1 hides it;
- **exact match** — strict set equality per statement.

## Protocol helpers

`split_by_statement` splits by statement id, so fragments of one query cannot straddle the split. `anchor_gate` returns the intersection of evaluation-anchor ids with train/validation — a non-empty list means the run is contaminated and must not be reported.

`paired_bootstrap` resamples statements inside each seed and averages across seeds; `mde` derives the minimum detectable effect instead of asserting it: with sd = 0.11 and ten anchors, Δ ≈ 0.10 F1, and reaching 0.05 needs about 38 anchors.

## Synthetic example

`examples/synthetic_statements.json` contains four invented statements over invented tables (`mart.daily_sales`, `raw.orders`, …). It exists to make the CLI runnable and the tests self-contained.

## Scope and limitations

The package implements the **measuring instruments**, not a measured defect matrix. It does not train models. Effects reported with these tools are only as good as the evaluation anchors and the number of seeds; see the article's section on minimum detectable effect.

## License

MIT — see `LICENSE`.
