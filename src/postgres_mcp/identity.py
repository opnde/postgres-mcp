"""Per-request database credentials taken from HTTP headers.

Upstream postgres-mcp reads one DSN at process start and serves every request from
one pooled connection, so every MCP client shares a single database role. That is
fine for a personal tool and wrong for a shared HTTP deployment: the database then
cannot tell users apart, `session_user` in the audit log is the service, per-user
`statement_timeout` / `CONNECTION LIMIT` / role settings never apply, and revoking
one person means rotating one credential for everybody.

This module adds a second mode. In `header` mode the server takes *only* the user
name and password from a request header; host, port and database name stay fixed
in the server configuration. The connection is then a real PostgreSQL login as
that person, which means the role's own GUCs (`statement_timeout`,
`default_transaction_read_only`, `log_min_duration_statement`) and its
`CONNECTION LIMIT` apply automatically. `SET ROLE` does NOT do that: it changes
`current_user` but not `session_user`, and it does not load the target role's
settings.

Design rules, in order of importance:

1. **The client never chooses the connection target.** Host, port and dbname come
   from the server's own configuration. Only user and password are taken from the
   request. Otherwise this server would be an open proxy into any reachable
   database.
2. **Credentials are passed as driver parameters, never concatenated into a DSN
   string.** `psycopg.conninfo.make_conninfo` does the quoting; a hand-built
   string breaks on passwords containing spaces, quotes or backslashes, and a
   crafted value could otherwise override the fixed parameters above.
3. **An allowlist decides which roles may be used at all.** Without it a leaked
   superuser or loader password would turn this endpoint into full database
   access.
4. **Optionally require a verified edge JWT** (Cloudflare Access and similar put
   a signed assertion on every request). Checking that a header merely *exists*
   is worthless - anyone who can reach the origin can set headers - so the
   signature, `aud` and expiry are verified against the issuer's JWKS.
5. **Nothing secret is ever logged.** Errors name the header, never its value.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from psycopg.conninfo import make_conninfo

logger = logging.getLogger(__name__)

# Header names are compared lower-case; HTTP headers are case-insensitive and
# Starlette normalises them on the way in.
DEFAULT_CREDENTIAL_HEADER = "x-db-credential"

# Cloudflare Access publishes its signing keys here. Other issuers that follow
# the same convention work unchanged; anything else needs `PGMCP_JWKS_URL`.
CF_ACCESS_CERTS_PATH = "/cdn-cgi/access/certs"


class AuthError(Exception):
    """Raised when a request cannot be turned into a database login.

    The message is shown to the MCP client, so it must stay free of secrets and
    of anything that helps an attacker enumerate valid roles.
    """


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class IdentityConfig:
    """Server-side half of the connection: everything the client may NOT choose."""

    enabled: bool = False
    credential_header: str = DEFAULT_CREDENTIAL_HEADER
    host: str | None = None
    port: int = 5432
    dbname: str | None = None
    sslmode: str | None = None
    # Which role names may log in through this endpoint. Default refuses
    # everything, so a misconfigured deployment fails closed rather than open.
    role_pattern: re.Pattern[str] = re.compile(r"(?!)")
    connect_timeout: int = 10
    # Edge JWT verification (optional, on by default once a JWKS source exists).
    jwt_required: bool = False
    jwks_url: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_header: str = "cf-access-jwt-assertion"
    # When set, the JWT's `common_name` claim (Cloudflare service tokens) must
    # match. This pins the endpoint to one calling system.
    jwt_common_name: str | None = None

    # Reported to PostgreSQL as `application_name`, so every row the server
    # causes carries the channel it came through.
    #
    # Why this is not cosmetic: the person's role is the same whether they query
    # through this server or through psql on their own machine. Without a name,
    # both land in the audit log as `app=[unknown]` and the only way to tell them
    # apart is the absence of `app=psql` - an inference, not a record. Operators
    # answering "who actually uses this endpoint" should read it, not deduce it.
    application_name: str = "langdock"

    @classmethod
    def from_env(cls) -> IdentityConfig:
        mode = os.environ.get("PGMCP_AUTH_MODE", "dsn").strip().lower()
        if mode not in ("dsn", "header"):
            raise ValueError(f"PGMCP_AUTH_MODE must be 'dsn' or 'header', got {mode!r}")
        if mode == "dsn":
            return cls()

        host = os.environ.get("PGMCP_DB_HOST")
        dbname = os.environ.get("PGMCP_DB_NAME")
        if not host or not dbname:
            raise ValueError("PGMCP_AUTH_MODE=header requires PGMCP_DB_HOST and PGMCP_DB_NAME")

        pattern_src = os.environ.get("PGMCP_ALLOWED_ROLES")
        if not pattern_src:
            raise ValueError(
                "PGMCP_AUTH_MODE=header requires PGMCP_ALLOWED_ROLES, a regular expression "
                "matching the role names that may log in (e.g. '^analyst_[a-z0-9_]+$'). "
                "Without it any leaked password, including a superuser's, would be accepted."
            )

        issuer = os.environ.get("PGMCP_JWT_ISSUER")
        jwks_url = os.environ.get("PGMCP_JWKS_URL")
        if issuer and not jwks_url:
            jwks_url = issuer.rstrip("/") + CF_ACCESS_CERTS_PATH
        jwt_required = _env_flag("PGMCP_JWT_REQUIRED", bool(jwks_url))
        if jwt_required and not jwks_url:
            raise ValueError("PGMCP_JWT_REQUIRED is set but neither PGMCP_JWT_ISSUER nor PGMCP_JWKS_URL is configured")

        return cls(
            enabled=True,
            credential_header=os.environ.get("PGMCP_CREDENTIAL_HEADER", DEFAULT_CREDENTIAL_HEADER).strip().lower(),
            host=host,
            port=int(os.environ.get("PGMCP_DB_PORT", "5432")),
            dbname=dbname,
            sslmode=os.environ.get("PGMCP_DB_SSLMODE") or None,
            role_pattern=re.compile(pattern_src),
            connect_timeout=int(os.environ.get("PGMCP_CONNECT_TIMEOUT", "10")),
            jwt_required=jwt_required,
            jwks_url=jwks_url,
            jwt_issuer=issuer,
            jwt_audience=os.environ.get("PGMCP_JWT_AUDIENCE"),
            jwt_header=os.environ.get("PGMCP_JWT_HEADER", "cf-access-jwt-assertion").strip().lower(),
            jwt_common_name=os.environ.get("PGMCP_JWT_COMMON_NAME") or None,
            application_name=os.environ.get("PGMCP_APPLICATION_NAME", "langdock").strip() or "langdock",
        )


def split_credential(raw: str) -> tuple[str, str]:
    """Split `user:password` on the FIRST colon.

    First colon, not last and not `split(':')`: PostgreSQL passwords may contain
    colons, role names may not. Splitting anywhere else silently corrupts valid
    passwords, and the resulting failure looks like a wrong password rather than
    a parsing bug.

    Deployments often have exactly one free-form field on the client side (for
    example a SaaS chat platform offering a single "API key" input), which is why
    both values share one header instead of using two.
    """
    if ":" not in raw:
        raise AuthError("credential header must be '<role>:<password>'")
    user, password = raw.split(":", 1)
    user = user.strip()
    if not user or not password:
        raise AuthError("credential header must be '<role>:<password>'")
    return user, password


def _verify_jwt(token: str, cfg: IdentityConfig) -> dict[str, Any]:
    """Verify the edge assertion: signature, audience, expiry, optional subject.

    Import is local so that `dsn` mode keeps working without the JWT extra
    installed.
    """
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AuthError("JWT verification requested but PyJWT is not installed") from exc

    if not cfg.jwks_url:
        raise AuthError("JWT verification requested but no JWKS URL is configured")

    # PyJWKClient caches keys in-process; the edge rotates them rarely and a
    # fetch per request would add a round trip to every query.
    client = _jwks_client(cfg.jwks_url)
    try:
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=cfg.jwt_audience,
            issuer=cfg.jwt_issuer,
            options={
                "require": ["exp"],
                "verify_aud": bool(cfg.jwt_audience),
                "verify_iss": bool(cfg.jwt_issuer),
            },
        )
    except Exception as exc:
        # Deliberately terse: the client should not learn why verification failed.
        logger.warning("edge assertion rejected: %s", type(exc).__name__)
        raise AuthError("edge assertion invalid") from exc

    if cfg.jwt_common_name:
        seen = claims.get("common_name")
        if seen != cfg.jwt_common_name:
            logger.warning("edge assertion has unexpected common_name")
            raise AuthError("edge assertion invalid")
    return claims


_JWKS_CLIENTS: dict[str, Any] = {}


def _jwks_client(url: str) -> Any:
    client = _JWKS_CLIENTS.get(url)
    if client is None:
        from jwt import PyJWKClient

        client = PyJWKClient(url, cache_keys=True)
        _JWKS_CLIENTS[url] = client
    return client


def conninfo_for(headers: dict[str, str], cfg: IdentityConfig) -> str:
    """Turn request headers into a psycopg connection string.

    Raises AuthError for anything that is not a complete, allowed identity. The
    caller must not fall back to a shared connection when this raises - failing
    the request is the point.
    """
    if not cfg.enabled:
        raise AuthError("per-request credentials are not enabled on this server")

    if cfg.jwt_required:
        token = headers.get(cfg.jwt_header, "")
        if not token:
            raise AuthError(f"missing {cfg.jwt_header} header")
        _verify_jwt(token, cfg)

    raw = headers.get(cfg.credential_header, "")
    if not raw:
        raise AuthError(f"missing {cfg.credential_header} header")

    user, password = split_credential(raw)
    if not cfg.role_pattern.fullmatch(user):
        # Do not echo the role name: that turns the endpoint into a role oracle.
        logger.warning("login refused for a role outside PGMCP_ALLOWED_ROLES")
        raise AuthError("this role may not connect through this endpoint")

    return make_conninfo(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=user,
        password=password,
        connect_timeout=cfg.connect_timeout,
        application_name=cfg.application_name,
        **({"sslmode": cfg.sslmode} if cfg.sslmode else {}),
    )


def role_of(headers: dict[str, str], cfg: IdentityConfig) -> str | None:
    """Role name for logging, without touching the password. Never raises."""
    try:
        raw = headers.get(cfg.credential_header, "")
        return split_credential(raw)[0] if raw else None
    except AuthError:
        return None
