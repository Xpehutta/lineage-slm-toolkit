#!/usr/bin/env python3
"""Run the adversarial SQL cases through the contour and write a report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lineage_toolkit.adversarial_harness import run_all  # noqa: E402


def main() -> int:
    results = run_all()
    rows = []
    print(f"{'case':22s} {'expected':>8s} {'hit':>4s} {'miss':>5s} {'extra':>5s} {'picked':>10s}")
    for r in results:
        print(f"{r.case.name:22s} {len(r.case.expected):8d} {r.hit:4d} {len(r.missed):5d} {len(r.extra):5d} {r.source:>10s}")
        rows.append({
            "case": r.case.name,
            "class": r.case.klass,
            "expected": len(r.case.expected),
            "hit": r.hit,
            "missed": sorted(list(e) for e in r.missed),
            "extra": sorted(list(e) for e in r.extra),
            "picked": r.source,
            "notes": r.case.notes,
        })
    failed = [row for row in rows if row["missed"] or row["extra"]]
    payload = {
        "cases": len(rows),
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "edges_expected": sum(row["expected"] for row in rows),
        "edges_hit": sum(row["hit"] for row in rows),
        "results": rows,
    }
    out = ROOT / "results" / "adversarial_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncases {payload['passed']}/{payload['cases']} clean, "
          f"edges {payload['edges_hit']}/{payload['edges_expected']}")
    print(f"report: {out}")
    for row in failed:
        print(f"  FAIL {row['case']}: missed {row['missed']} extra {row['extra']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
