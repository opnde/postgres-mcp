"""Tests for the `execute_sql` row cap (`PGMCP_MAX_ROWS`).

The failure that matters is the silent one: a result cut at N rows that looks
complete. So every truncation must carry the notice, and nothing below the cap
may carry it.
"""

import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pglast
import pytest

import postgres_mcp.server as server
from postgres_mcp.identity import IdentityConfig
from postgres_mcp.row_cap import DEFAULT_MAX_ROWS
from postgres_mcp.row_cap import RowCapConfig
from postgres_mcp.row_cap import notice
from postgres_mcp.row_cap import truncate
from postgres_mcp.row_cap import wrap_query
from postgres_mcp.sql import SafeSqlDriver
from postgres_mcp.sql import SqlDriver

# ------------------------------------------------------------------ config


def test_default_is_500(monkeypatch):
    monkeypatch.delenv("PGMCP_MAX_ROWS", raising=False)
    monkeypatch.delenv("PGMCP_MAX_ROWS_EXEMPT_ROLES", raising=False)
    cfg = RowCapConfig.from_env()
    assert cfg.max_rows == DEFAULT_MAX_ROWS == 500
    assert cfg.limit_for(None) == 500
    assert cfg.limit_for("analyst_jane") == 500


def test_zero_switches_off(monkeypatch):
    monkeypatch.setenv("PGMCP_MAX_ROWS", "0")
    assert RowCapConfig.from_env().limit_for("analyst_jane") is None


@pytest.mark.parametrize("value", ["abc", "-1", "5.5"])
def test_invalid_value_refuses_start(monkeypatch, value):
    monkeypatch.setenv("PGMCP_MAX_ROWS", value)
    with pytest.raises(ValueError, match="PGMCP_MAX_ROWS"):
        RowCapConfig.from_env()


def test_exempt_roles_full_match(monkeypatch):
    monkeypatch.setenv("PGMCP_MAX_ROWS", "100")
    monkeypatch.setenv("PGMCP_MAX_ROWS_EXEMPT_ROLES", "analyst_(oliver|daniel)")
    cfg = RowCapConfig.from_env()
    assert cfg.limit_for("analyst_oliver") is None
    assert cfg.limit_for("analyst_daniel") is None
    # Prefix or suffix of an exempt name is not exempt.
    assert cfg.limit_for("analyst_oliver2") == 100
    assert cfg.limit_for("x_analyst_oliver") == 100
    assert cfg.limit_for("analyst_jane") == 100
    # No known role (dsn mode) never matches.
    assert cfg.limit_for(None) == 100


# ------------------------------------------------------------------ wrapping


@pytest.mark.parametrize(
    "sql",
    [
        "select * from pub.x",
        "SELECT a, b FROM t ORDER BY a LIMIT 10",
        "with c as (select 1 as a) select a from c",
        "values (1), (2)",
        "table pub.x",
        "select 1 union all select 2",
    ],
)
def test_plain_select_is_wrapped(sql):
    wrapped = wrap_query(sql, 500)
    assert wrapped is not None
    assert wrapped.endswith("LIMIT 501")
    assert sql in wrapped
    # Still exactly one SELECT statement.
    parsed = pglast.parse_sql(wrapped)
    assert len(parsed) == 1


def test_trailing_semicolon_is_dropped():
    wrapped = wrap_query("select 1;", 5)
    assert wrapped == "SELECT * FROM (\nselect 1\n) AS pgmcp_capped LIMIT 6"


def test_trailing_line_comment_cannot_swallow_the_limit():
    wrapped = wrap_query("select 1 -- my note", 5)
    assert wrapped is not None
    parsed = pglast.parse_sql(wrapped)
    assert parsed[0].stmt.limitCount is not None  # type: ignore[union-attr]


def test_non_ascii_text_is_sliced_by_characters():
    sql = "select 'Größe ü' as größe;"
    wrapped = wrap_query(sql, 5)
    assert wrapped is not None
    assert "select 'Größe ü' as größe\n" in wrapped


@pytest.mark.parametrize(
    "sql",
    [
        "explain select 1",
        "show statement_timeout",
        "select 1; select 2",
        "select 1 into new_table",
        "this is not sql",
    ],
)
def test_everything_else_runs_unwrapped(sql):
    assert wrap_query(sql, 500) is None


def test_wrapped_query_passes_the_restricted_validator():
    """The pglast riegel must accept the wrapper, or restricted mode breaks."""
    safe = SafeSqlDriver(sql_driver=MagicMock())
    wrapped = wrap_query("select a, count(*) from pub.x where b = 'y' group by a order by 2 desc", 500)
    assert wrapped is not None
    safe._validate(wrapped)  # pyright: ignore[reportPrivateUsage]  # raises on rejection


def test_wrapping_does_not_smuggle_a_write_past_the_validator():
    """A rejected statement stays rejected: it is either not wrapped or still caught."""
    safe = SafeSqlDriver(sql_driver=MagicMock())
    for sql in ["select 1 into t", "select * from x for update", "delete from x"]:
        query = wrap_query(sql, 500) or sql
        with pytest.raises(ValueError):
            safe._validate(query)  # pyright: ignore[reportPrivateUsage]


# ------------------------------------------------------------------ truncation


def test_truncate():
    assert truncate([1, 2, 3], 3) == ([1, 2, 3], False)
    assert truncate([1, 2, 3, 4], 3) == ([1, 2, 3], True)
    assert truncate([], 3) == ([], False)


def test_notice_names_the_limit_and_says_incomplete():
    text = notice(500)
    assert "500" in text
    assert "NICHT vollständig" in text
    assert "SQL" in text


# ------------------------------------------------------------------ execute_sql


def _rows(n):
    return [SqlDriver.RowResult(cells={"i": i}) for i in range(n)]


@pytest.fixture
def fake_session(monkeypatch):
    driver = MagicMock()
    driver.execute_query = AsyncMock()

    @asynccontextmanager
    async def session(_ctx):
        yield driver

    monkeypatch.setattr(server, "sql_session", session)
    monkeypatch.setattr(server, "identity_config", IdentityConfig())
    return driver


async def _call(sql="select i from t"):
    result = await server.execute_sql(MagicMock(), sql=sql)
    return result[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_execute_sql_cuts_and_warns(fake_session, monkeypatch):
    monkeypatch.setattr(server, "row_cap_config", RowCapConfig(max_rows=3))
    fake_session.execute_query.return_value = _rows(4)  # the database returns limit+1

    text = await _call()

    sent = fake_session.execute_query.call_args.args[0]
    assert sent.endswith("LIMIT 4")
    assert "{'i': 2}" in text and "{'i': 3}" not in text
    assert notice(3) in text


@pytest.mark.asyncio
async def test_execute_sql_exactly_at_cap_is_complete(fake_session, monkeypatch):
    monkeypatch.setattr(server, "row_cap_config", RowCapConfig(max_rows=3))
    fake_session.execute_query.return_value = _rows(3)

    text = await _call()

    assert "{'i': 2}" in text
    assert "HINWEIS" not in text


@pytest.mark.asyncio
async def test_execute_sql_unwrappable_statement_is_cut_in_python(fake_session, monkeypatch):
    monkeypatch.setattr(server, "row_cap_config", RowCapConfig(max_rows=3))
    fake_session.execute_query.return_value = _rows(10)

    text = await _call("show all")

    assert fake_session.execute_query.call_args.args[0] == "show all"
    assert "{'i': 3}" not in text
    assert notice(3) in text


@pytest.mark.asyncio
async def test_execute_sql_cap_off(fake_session, monkeypatch):
    monkeypatch.setattr(server, "row_cap_config", RowCapConfig(max_rows=0))
    fake_session.execute_query.return_value = _rows(1000)

    text = await _call()

    assert fake_session.execute_query.call_args.args[0] == "select i from t"
    assert "{'i': 999}" in text
    assert "HINWEIS" not in text


@pytest.mark.asyncio
async def test_execute_sql_exempt_role_in_header_mode(fake_session, monkeypatch):
    monkeypatch.setattr(
        server,
        "identity_config",
        IdentityConfig(enabled=True, credential_header="x-db-credential", role_pattern=re.compile(r"^analyst_\w+$")),
    )
    monkeypatch.setattr(
        server,
        "row_cap_config",
        RowCapConfig(max_rows=3, exempt_roles=re.compile("analyst_oliver")),
    )
    fake_session.execute_query.return_value = _rows(10)

    for role, capped in [("analyst_oliver", False), ("analyst_jane", True)]:
        monkeypatch.setattr(server, "_request_headers", lambda _ctx, role=role: {"x-db-credential": f"{role}:pw"})
        text = await _call()
        assert ("HINWEIS" in text) is capped, role
        assert fake_session.execute_query.call_args.args[0].endswith("LIMIT 4") is capped, role


@pytest.mark.asyncio
async def test_execute_sql_no_results(fake_session, monkeypatch):
    monkeypatch.setattr(server, "row_cap_config", RowCapConfig(max_rows=3))
    fake_session.execute_query.return_value = None
    assert await _call() == "No results"
