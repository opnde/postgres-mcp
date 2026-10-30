"""Tests for the hints appended to tool errors (error_hints.py) and PGMCP_QUERY_TIMEOUT."""

import pytest
from mcp.types import TextContent

import postgres_mcp.server as server
from postgres_mcp.error_hints import hint_for
from postgres_mcp.error_hints import with_hint
from postgres_mcp.sql import SafeSqlDriver


@pytest.mark.parametrize(
    ("error", "needle"),
    [
        ('column "status" does not exist', "get_object_details"),
        ('column "x" of relation "group_deals" does not exist', "get_object_details"),
        ("column d.amount_eur does not exist", "get_object_details"),
        ('relation "pub.group_dealz" does not exist', "list_objects"),
        ("permission denied for schema raw_nwp_dolibarr", "pub"),
        ("Query execution timed out after 55.0 seconds in restricted mode.", "aggregate"),
        ("canceling statement due to statement timeout", "aggregate"),
        ('could not resize shared memory segment "/PostgreSQL.1" to 8388608 bytes: No space left on device', "Retry once"),
        ("credential header must be '<role>:<password>'", "<role>:<password>"),
        ("database login failed", "retrying will not help"),
    ],
)
def test_known_errors_get_a_hint(error, needle):
    hint = hint_for(error)
    assert hint is not None and needle in hint


def test_unknown_error_gets_no_hint():
    assert hint_for("division by zero") is None
    assert with_hint("division by zero") == "division by zero"


def test_original_message_is_kept():
    msg = 'column "status" does not exist'
    out = with_hint(msg)
    assert out.startswith(msg) and "\nHint: " in out


def test_format_error_response_carries_hint():
    [content] = server.format_error_response('column "status" does not exist')
    assert isinstance(content, TextContent)
    assert content.text.startswith("Error: column")
    assert "get_object_details" in content.text


def test_query_timeout_default(monkeypatch):
    monkeypatch.delenv("PGMCP_QUERY_TIMEOUT", raising=False)
    assert server.query_timeout_from_env() == 30.0


def test_query_timeout_from_env(monkeypatch):
    monkeypatch.setenv("PGMCP_QUERY_TIMEOUT", "55")
    assert server.query_timeout_from_env() == 55.0


@pytest.mark.parametrize("raw", ["abc", "0", "-5"])
def test_query_timeout_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv("PGMCP_QUERY_TIMEOUT", raw)
    with pytest.raises(ValueError):
        server.query_timeout_from_env()


@pytest.mark.asyncio
async def test_restricted_driver_uses_configured_timeout(monkeypatch):
    monkeypatch.setattr(server, "current_access_mode", server.AccessMode.RESTRICTED)
    monkeypatch.setattr(server, "query_timeout", 55.0)
    driver = await server.get_sql_driver()
    assert isinstance(driver, SafeSqlDriver)
    assert driver.timeout == 55.0
