"""Row cap for `execute_sql`: at most N rows per call.

A language model behind this server tends to fetch raw rows and then add them up
itself. Beyond a few hundred rows that is slow, expensive, and - worse - silently
wrong once the client or the model drops part of the result. The cap makes the
cut explicit: the query runs as `SELECT * FROM (<sql>) LIMIT N+1`, so the
database stops after N+1 rows, the N+1-th row only proves there was more, and
the response carries a notice telling the model to aggregate in SQL instead.

Configuration (environment):

- `PGMCP_MAX_ROWS` - cap per call, default 500. `0` switches the cap off.
- `PGMCP_MAX_ROWS_EXEMPT_ROLES` - optional regex (full match) of login roles
  that get no cap. Only meaningful with per-request credentials
  (`PGMCP_AUTH_MODE=header`), because only there does the server know who is
  asking; in `dsn` mode the cap applies to every call.

Only a single plain SELECT (including WITH, VALUES, TABLE) is wrapped. Anything
else that the access mode lets through (EXPLAIN, SHOW, several statements) runs
unchanged and is cut in Python afterwards - the notice is the same.

This is a guard against careless result sizes, not a security boundary: the
statement timeout and the grants of the role remain what protects the database.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from typing import Sequence
from typing import TypeVar

import pglast
from pglast.ast import RawStmt
from pglast.ast import SelectStmt

DEFAULT_MAX_ROWS = 500

T = TypeVar("T")


@dataclass(frozen=True)
class RowCapConfig:
    max_rows: int = DEFAULT_MAX_ROWS
    exempt_roles: re.Pattern[str] | None = None

    @classmethod
    def from_env(cls) -> RowCapConfig:
        raw = os.environ.get("PGMCP_MAX_ROWS", "").strip()
        if raw:
            try:
                max_rows = int(raw)
            except ValueError as exc:
                raise ValueError(f"PGMCP_MAX_ROWS must be an integer, got {raw!r}") from exc
            if max_rows < 0:
                raise ValueError("PGMCP_MAX_ROWS must be >= 0 (0 switches the cap off)")
        else:
            max_rows = DEFAULT_MAX_ROWS

        pattern = os.environ.get("PGMCP_MAX_ROWS_EXEMPT_ROLES", "").strip()
        exempt = re.compile(pattern) if pattern else None
        return cls(max_rows=max_rows, exempt_roles=exempt)

    def limit_for(self, role: str | None) -> int | None:
        """Cap for this caller, or None when no cap applies."""
        if self.max_rows == 0:
            return None
        if role and self.exempt_roles is not None and self.exempt_roles.fullmatch(role):
            return None
        return self.max_rows


def wrap_query(sql: str, limit: int) -> str | None:
    """`SELECT * FROM (<sql>) LIMIT limit+1`, or None if `sql` is not one plain SELECT.

    The statement text is cut out by the parser's own offsets, which drops a
    trailing semicolon (a syntax error inside the subquery). The newlines keep a
    trailing `-- comment` from swallowing the closing parenthesis.
    """
    try:
        parsed = pglast.parse_sql(sql)
    except pglast.parser.ParseError:
        return None  # let the normal path report the syntax error
    if len(parsed) != 1:
        return None
    raw = parsed[0]
    if not isinstance(raw, RawStmt) or not isinstance(raw.stmt, SelectStmt):
        return None
    if raw.stmt.intoClause is not None:  # SELECT ... INTO creates a table
        return None
    start = raw.stmt_location or 0
    length = raw.stmt_len or 0
    body = sql[start : start + length] if length else sql[start:]
    return f"SELECT * FROM (\n{body}\n) AS pgmcp_capped LIMIT {limit + 1}"


def truncate(rows: Sequence[T], limit: int) -> tuple[list[T], bool]:
    """First `limit` rows and whether anything was cut."""
    return list(rows[:limit]), len(rows) > limit


def notice(limit: int) -> str:
    """Appended below a truncated result. Written for the model, not for people."""
    return (
        f"HINWEIS: Ergebnis gekürzt auf {limit} Zeilen - die Abfrage liefert mehr. "
        "Diese Zeilen sind NICHT vollständig: Summen, Anteile, Zählungen und Rankings "
        "in SQL rechnen (GROUP BY, count, sum, ORDER BY ... LIMIT), nicht aus diesen Zeilen."
    )


def format_rows(cells: list[Any], limit: int | None, truncated: bool) -> str:
    text = str(cells)
    if truncated and limit is not None:
        text += "\n\n" + notice(limit)
    return text
