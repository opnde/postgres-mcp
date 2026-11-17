"""Tests for the per-role connection cap and capacity retries (connection_gate.py).

The case behind this: one Langdock question fanned out into 15 parallel tool
calls, the role allowed 8 connections, and the refusals reached the model as
"database login failed - check your credential". The model gave up.
"""

import asyncio
import dataclasses

import psycopg
import pytest

import postgres_mcp.server as server
from postgres_mcp.connection_gate import DatabaseBusyError
from postgres_mcp.connection_gate import RoleGate
from postgres_mcp.connection_gate import is_capacity_error
from postgres_mcp.error_hints import hint_for
from postgres_mcp.identity import AuthError

ROLE_FULL = 'connection to server at "10.0.0.1", port 5434 failed: FATAL:  too many connections for role "analyst_x"'
BAD_PASSWORD = 'connection to server at "10.0.0.1", port 5434 failed: FATAL:  password authentication failed for user "analyst_x"'


class FakeConn:
    async def close(self):
        pass


class FakeDb:
    """Stands in for Postgres: enforces a per-role connection limit like CONNECTION LIMIT."""

    def __init__(self, limit: int, held_elsewhere: int = 0, fail_with: str | None = None):
        self.limit = limit
        self.open = held_elsewhere
        self.peak = held_elsewhere
        self.attempts = 0
        self.fail_with = fail_with

    async def connect(self, conninfo):
        self.attempts += 1
        await asyncio.sleep(0)
        if self.fail_with:
            raise psycopg.OperationalError(self.fail_with)
        if self.open >= self.limit:
            raise psycopg.OperationalError(ROLE_FULL)
        self.open += 1
        self.peak = max(self.peak, self.open)
        conn = FakeConn()
        db = self

        async def close():
            db.open -= 1

        conn.close = close  # type: ignore[method-assign]
        return conn


@pytest.fixture
def per_request(monkeypatch):
    """Header mode with the identity checks stubbed out; the gate is what is under test."""
    monkeypatch.setattr(server, "identity_config", dataclasses.replace(server.identity_config, enabled=True))
    monkeypatch.setattr(server, "_request_headers", lambda ctx: {})
    monkeypatch.setattr(server, "conninfo_for", lambda headers, cfg: "host=x")
    monkeypatch.setattr(server, "role_of", lambda headers, cfg: "analyst_x")
    monkeypatch.setattr(server, "RETRY_DELAYS", (0.0, 0.0, 0.0))

    def install(db: FakeDb, gate: RoleGate):
        monkeypatch.setattr(server.psycopg.AsyncConnection, "connect", db.connect)
        monkeypatch.setattr(server, "role_gate", gate)

    return install


async def one_call(hold: float = 0.01):
    async with server.sql_session(None):
        await asyncio.sleep(hold)


# ------------------------------------------------------------------ the burst


@pytest.mark.asyncio
async def test_a_burst_of_15_calls_stays_under_a_limit_of_8(per_request):
    db = FakeDb(limit=8)
    per_request(db, RoleGate(max_per_role=4, wait=5))
    results = await asyncio.gather(*(one_call() for _ in range(15)), return_exceptions=True)
    assert [r for r in results if isinstance(r, Exception)] == []
    assert db.peak <= 4
    assert db.attempts == 15


@pytest.mark.asyncio
async def test_without_the_gate_the_same_burst_is_refused(per_request):
    """Guards the test itself: with a cap above the limit, the fake database refuses."""
    db = FakeDb(limit=8)
    per_request(db, RoleGate(max_per_role=50, wait=5))
    # Hold long enough that the zero-delay retries cannot outlast the first eight calls.
    results = await asyncio.gather(*(one_call(hold=0.3) for _ in range(15)), return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    assert failures and all(isinstance(f, DatabaseBusyError) for f in failures)


@pytest.mark.asyncio
async def test_connections_held_elsewhere_are_waited_out(per_request):
    """psql sessions of the same person count against the limit too; a retry rides it out."""
    db = FakeDb(limit=8, held_elsewhere=8)
    per_request(db, RoleGate(max_per_role=4, wait=5))

    async def psql_disconnects():
        await asyncio.sleep(0)
        db.open -= 2

    await asyncio.gather(one_call(), psql_disconnects())
    assert db.attempts >= 2


@pytest.mark.asyncio
async def test_a_role_that_stays_full_gets_database_busy_not_a_login_error(per_request):
    db = FakeDb(limit=8, held_elsewhere=8)
    per_request(db, RoleGate(max_per_role=4, wait=5))
    with pytest.raises(DatabaseBusyError) as exc_info:
        await one_call()
    assert db.attempts == 4  # first try + three retries
    message = str(exc_info.value)
    assert "10.0.0.1" not in message and "analyst_x" not in message


@pytest.mark.asyncio
async def test_a_wrong_password_is_not_retried(per_request):
    db = FakeDb(limit=8, fail_with=BAD_PASSWORD)
    per_request(db, RoleGate(max_per_role=4, wait=5))
    with pytest.raises(AuthError, match="database login failed"):
        await one_call()
    assert db.attempts == 1


@pytest.mark.asyncio
async def test_waiting_too_long_for_a_slot_is_database_busy(per_request):
    db = FakeDb(limit=8)
    per_request(db, RoleGate(max_per_role=1, wait=0.05))
    results = await asyncio.gather(one_call(hold=0.5), one_call(), return_exceptions=True)
    assert any(isinstance(r, DatabaseBusyError) for r in results)


@pytest.mark.asyncio
async def test_roles_do_not_share_slots():
    gate = RoleGate(max_per_role=1, wait=0.05)
    async with gate.hold("analyst_a"):
        async with gate.hold("analyst_b"):
            pass
        with pytest.raises(DatabaseBusyError):
            async with gate.hold("analyst_a"):
                pass


@pytest.mark.asyncio
async def test_slots_are_released_when_the_call_fails():
    gate = RoleGate(max_per_role=1, wait=0.05)
    with pytest.raises(RuntimeError):
        async with gate.hold("analyst_a"):
            raise RuntimeError("query failed")
    async with gate.hold("analyst_a"):
        pass


# ------------------------------------------------------------------ messages and config


@pytest.mark.parametrize(
    "message",
    [
        ROLE_FULL,
        "FATAL:  sorry, too many clients already",
        "FATAL:  remaining connection slots are reserved for roles with the SUPERUSER attribute",
    ],
)
def test_capacity_errors_are_recognised(message):
    assert is_capacity_error(message)


def test_a_password_failure_is_not_a_capacity_error():
    assert not is_capacity_error(BAD_PASSWORD)


def test_busy_hint_says_retry_and_not_check_credentials():
    hint = hint_for("database busy: too many parallel connections for this login")
    assert hint is not None
    assert "not a credential problem" in hint and "retry" in hint
    assert "retrying will not help" not in hint


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("PGMCP_MAX_CONNECTIONS_PER_ROLE", "3")
    monkeypatch.setenv("PGMCP_CONNECTION_WAIT", "12")
    gate = RoleGate.from_env()
    assert (gate.max_per_role, gate.wait) == (3, 12.0)


def test_config_defaults(monkeypatch):
    monkeypatch.delenv("PGMCP_MAX_CONNECTIONS_PER_ROLE", raising=False)
    monkeypatch.delenv("PGMCP_CONNECTION_WAIT", raising=False)
    gate = RoleGate.from_env()
    assert (gate.max_per_role, gate.wait) == (4, 30.0)


@pytest.mark.parametrize(
    ("name", "value"), [("PGMCP_MAX_CONNECTIONS_PER_ROLE", "0"), ("PGMCP_MAX_CONNECTIONS_PER_ROLE", "x"), ("PGMCP_CONNECTION_WAIT", "-1")]
)
def test_bad_config_is_refused(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        RoleGate.from_env()
