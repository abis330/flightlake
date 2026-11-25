"""Read-only SQL guard for user/LLM-supplied queries against the lake."""
from __future__ import annotations

import re

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|set|"
    r"call|checkpoint|vacuum|read_csv\w*|read_json\w*|read_text|read_blob|glob|write\w*)\b",
    re.IGNORECASE,
)


class UnsafeSQL(ValueError):
    pass


def guard(query: str, max_rows: int = 5000) -> str:
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        raise UnsafeSQL("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise UnsafeSQL("only SELECT / WITH queries are allowed")
    m = FORBIDDEN.search(re.sub(r"'[^']*'", "''", q))  # ignore string literals
    if m:
        raise UnsafeSQL(f"forbidden keyword: {m.group(0)}")
    return f"SELECT * FROM ({q}) AS _q LIMIT {int(max_rows)}"
