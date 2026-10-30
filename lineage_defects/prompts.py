"""Prompt construction: ChatML template and schema slicing.

The model gets only the tables its statement touches. Passing the whole catalog
inflates the prompt and buries the relevant columns; passing nothing forces the
model to guess identifiers, which the catalog check then rejects.
"""
from __future__ import annotations

SYSTEM = (
    "Ты извлекаешь происхождение данных из SQL. "
    "Верни JSON-объект с полем edges: для каждой целевой колонки перечисли все "
    "исходные колонки. Не сокращай список источников и не добавляй колонок, "
    "которых нет в схеме."
)


def schema_slice(schema: dict[str, list[str]], tables: list[str]) -> dict[str, list[str]]:
    """Only the tables mentioned in the statement, with their columns."""
    return {t: schema[t] for t in tables if t in schema}


def build_messages(sql: str, target: str, schema_slice_: dict[str, list[str]]) -> list[dict]:
    """ChatML-style messages for Qwen; assistant turn is added separately."""
    schema_text = chr(10).join(f"{t}({', '.join(cols)})" for t, cols in schema_slice_.items())
    user = (
        f"Целевая таблица: {target}" + chr(10) +
        f"Схема:{chr(10)}{schema_text}" + chr(10) + chr(10) +
        f"SQL:{chr(10)}{sql}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
