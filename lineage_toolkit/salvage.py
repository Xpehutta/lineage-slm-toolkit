"""Recover complete JSON objects from a truncated model answer.

Lineage answers are long: 17-23 thousand characters, often cut by the token
limit. A plain json.loads then fails on the whole string and returns nothing,
halving the measured F1. The scanner below walks the text, tracks string and
brace state, and yields every object that closes properly.
"""
from __future__ import annotations

import json


def salvage_edges(text: str, anchor: str = '{"target"') -> list[dict]:
    """All complete objects starting with the anchor token."""
    out: list[dict] = []
    i = 0
    n = len(text)
    while True:
        j = text.find(anchor, i)
        if j == -1:
            break
        depth, k, in_str, esc = 0, j, False, False
        while k < n:
            ch = text[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            k += 1
        if k >= n:
            break
        try:
            obj = json.loads(text[j:k + 1])
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            pass
        i = k + 1
    return out


def dedupe_edges(objects: list[dict]) -> list[dict]:
    """Collapse repeats: repetition loops produce up to 17 copies of one edge."""
    seen, out = set(), []
    for obj in objects:
        t = obj.get("target") or {}
        key = (t.get("table"), t.get("column"),
               tuple(sorted((s.get("table"), s.get("column")) for s in obj.get("sources") or [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out
