# lineage-defects

Toolkit for extracting **column-level Data Lineage** from SQL with a **compact language model**: deterministic contour first, a fine-tuned 8B model on the residual, and a measurement protocol for both.

Companion code for the article *«Как дообучить компактную языковую модель разбирать SQL и строить Data Lineage»*.

The repository is **data-free**: a small synthetic example only. No production SQL, no catalog dumps, no bank schemas.

## What is inside

| Module | What it does |
|---|---|
| `contour.py` | statement-level arbitration between two parsers: empty → primary, thinner than `THIN_RATIO = 0.4` → primary, otherwise secondary; conflicts returned as an audit list, never merged |
| `route.py` | when to call the model: empty contour or agreement below `AGREEMENT_THRESHOLD` |
| `prompts.py` | ChatML messages and schema slicing — the model sees only the tables of its statement |
| `salvage.py` | recovers complete JSON objects from a truncated answer, then collapses repetition loops |
| `train_config.py` | QLoRA defaults with the reason behind each value (r=16, alpha=16, lr 2e-4, 3 epochs, 4096 tokens, packing off) |
| `metrics.py` | edge F1 macro and pooled, precision/recall by target arity, exact match; empty prediction scores 0 |
| `splits.py` | statement-level split and the anchor-exclusion gate |
| `operators.py` | defect operators for training labels: `edge_label`, `edge_confusable`, `edge_drop`, `edge_dup`, `schema_stale`, `sql_truncate` |
| `stats.py` | paired bootstrap over statements, minimum detectable effect |

## Install and run

```bash
pip install -e ".[dev]"
pytest -q
lineage-defects --lam 0.25
```

The demo prints a per-operator metric table, a prompt size, a routing decision, a salvage result from a truncated answer, the QLoRA configuration, the anchor gate, and the MDE.

## How the pieces fit

1. **Contour.** Two parsers disagree by design; the arbiter picks one per statement and records conflicting targets.
2. **Routing.** The model is called only when the contour is empty or the parsers disagree strongly. Everything else stays deterministic and free.
3. **Prompt.** System instruction plus the statement and a schema slice of its own tables — not the whole catalog.
4. **Answer.** Truncated JSON is salvaged object by object; duplicates are collapsed; identifiers then pass the same catalog check as parser output.
5. **Measurement.** macro and pooled edge F1 together, precision and recall by arity, exact match.

## Defect operators

They corrupt the **labels**, not only the inputs, because a wrong edge teaches the model a wrong dependency with high confidence.

| Operator | What it does |
|---|---|
| `edge_label` | replaces **one** source of an edge with a source of a **different** target; collisions counted |
| `edge_confusable` | replaces a source with a column of the table whose name is closest (Levenshtein) |
| `edge_drop` | removes a share of edges from the reference set |
| `edge_dup` | duplicates edges, skewing towards frequent targets |
| `schema_stale` | renames schema entries, imitating a stale catalog |
| `sql_truncate` | cuts the statement, imitating a lost input fragment |

Intensity λ is the share of **edges inside one example**, not the share of examples: statements range from 3 to 190 edges.

## Scope

The toolkit implements the instruments, not a measured defect matrix. It does not train models. Effects measured with it are only as good as the evaluation anchors and the number of seeds.

## License

MIT — see `LICENSE`.
