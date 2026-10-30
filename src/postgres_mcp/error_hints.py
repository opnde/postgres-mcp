"""Next-step hints appended to tool errors.

The client behind this server is a language model. A bare Postgres message like
`column "status" does not exist` tells it what went wrong but not what to do,
and the audit log of the first six weeks shows what it does instead: it guesses
the next column name, or retries the identical query. A hint naming the tool
that answers the question (`get_object_details`) breaks that loop.

Hints only add text after the original message; the message itself is kept
unchanged so nothing is hidden. Matching is on Postgres/psycopg message text,
which is stable across versions for these cases. Unknown errors get no hint.
"""

from __future__ import annotations

import re

_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r'column "[^"]+"(?: of relation "[^"]+")? does not exist|column [\w.]+ does not exist', re.I),
        "Do not guess column names. Call get_object_details for this object, take the exact column names from there, then run the query again.",
    ),
    (
        re.compile(r'relation "[^"]+" does not exist', re.I),
        "This view or table does not exist under that name. Call list_objects (schema 'pub') and use a name from that list.",
    ),
    (
        re.compile(r"permission denied for (schema|table|view|relation)", re.I),
        "This login may only read the schema it was granted (for analysts: pub). Use a pub view instead.",
    ),
    (
        re.compile(r"timed out|canceling statement due to statement timeout", re.I),
        "Make the query smaller instead of repeating it: narrow the time range, aggregate in SQL "
        "(GROUP BY) instead of fetching rows, select only the columns you need.",
    ),
    (
        re.compile(r"could not resize shared memory segment|no space left on device|out of shared memory", re.I),
        "Temporary memory shortage on the database server. Retry once; if it fails again, make the query smaller.",
    ),
    (
        re.compile(r"credential header must be|missing [\w-]+ header|database login failed|may not connect through this endpoint", re.I),
        "This is a connection setup problem, not a query problem: retrying will not help. Tell the user to check the "
        "database credential in their Langdock connection, format '<role>:<password>' (e.g. analyst_firstname:...).",
    ),
)


def hint_for(error: str) -> str | None:
    """The hint for an error message, or None if no rule matches."""
    for pattern, hint in _HINTS:
        if pattern.search(error):
            return hint
    return None


def with_hint(error: str) -> str:
    """Error message plus hint (if any) on a new line."""
    hint = hint_for(error)
    return f"{error}\nHint: {hint}" if hint else error
