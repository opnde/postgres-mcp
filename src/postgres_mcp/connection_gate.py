"""Keep per-request logins below the database's per-role connection limit.

In `header` / `access-email` mode every tool call opens its own connection as
the person behind it. Clients fan out: on 16.11.2026 Langdock sent 13-15
`get_object_details` calls within one second for a single question, the role
had `CONNECTION LIMIT 8`, and Postgres refused the rest with `FATAL: too many
connections for role`. The server reported that as a failed login, the model
told the user to check their password, and the answer was lost.

Two layers fix that:

1. `RoleGate` caps how many connections this process holds per role at once
   (`PGMCP_MAX_CONNECTIONS_PER_ROLE`). Calls beyond the cap wait for a slot
   instead of failing; a call that waits longer than `PGMCP_CONNECTION_WAIT`
   seconds gets `DatabaseBusyError`.
2. Connections the role holds elsewhere (psql, a second client) still count
   against the database limit, so a refused login for that reason is retried a
   few times with backoff and then reported as `DatabaseBusyError` - never as a
   credential problem.
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

DEFAULT_MAX_CONNECTIONS_PER_ROLE = 4
DEFAULT_CONNECTION_WAIT = 30.0
# Backoff before each retry after the database refused a login for capacity.
RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0)

# Per-role limit, server-wide max_connections, and the reserved-slot variant.
_CAPACITY_RE = re.compile(
    r"too many connections for role|sorry, too many clients already|remaining connection slots are reserved",
    re.I,
)


class DatabaseBusyError(Exception):
    """Too many parallel calls for this login right now; retrying later helps.

    The message names neither host nor role: like the login failure it travels
    to the client and into a chat transcript.
    """


def is_capacity_error(message: str) -> bool:
    """True if Postgres refused the login for lack of connection slots."""
    return bool(_CAPACITY_RE.search(message))


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


class RoleGate:
    """One semaphore per role name; the set of roles is bounded by the allowlist."""

    def __init__(self, max_per_role: int = DEFAULT_MAX_CONNECTIONS_PER_ROLE, wait: float = DEFAULT_CONNECTION_WAIT):
        if max_per_role < 1:
            raise ValueError("max_per_role must be >= 1")
        self.max_per_role = max_per_role
        self.wait = wait
        self._slots: dict[str, asyncio.Semaphore] = {}

    @classmethod
    def from_env(cls) -> RoleGate:
        return cls(
            max_per_role=_positive_int("PGMCP_MAX_CONNECTIONS_PER_ROLE", DEFAULT_MAX_CONNECTIONS_PER_ROLE),
            wait=_positive_float("PGMCP_CONNECTION_WAIT", DEFAULT_CONNECTION_WAIT),
        )

    def _slot(self, role: str) -> asyncio.Semaphore:
        slot = self._slots.get(role)
        if slot is None:
            slot = self._slots[role] = asyncio.Semaphore(self.max_per_role)
        return slot

    @asynccontextmanager
    async def hold(self, role: str | None) -> AsyncIterator[None]:
        """Hold one of the role's slots for the duration of the block."""
        slot = self._slot(role or "")
        try:
            await asyncio.wait_for(slot.acquire(), timeout=self.wait)
        except asyncio.TimeoutError as exc:
            raise DatabaseBusyError("database busy: too many parallel calls for this login") from exc
        try:
            yield
        finally:
            slot.release()
