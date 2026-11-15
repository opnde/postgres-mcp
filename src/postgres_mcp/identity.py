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

`access-email` mode is for clients that cannot send a per-user secret at all (Claude
custom connectors authenticate with OAuth only). There the edge does the login: the
verified edge JWT carries the person's e-mail, and the server maps it to a role
from a fixed file and logs in with that role's password from a second file. The
mapping is an explicit list - an address that is not in it gets nothing, however
valid its token. The same real login applies, so the role's own settings still
hold.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from dataclasses import field
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
    # `header`: role and password come from a request header.
    # `access-email`: the person comes from the verified edge JWT, role and
    # password from the two files below.
    mode: str = "header"
    # access-email only: lower-case e-mail -> role, and role -> password.
    identities: dict[str, str] = field(default_factory=dict)
    passwords: dict[str, str] = field(default_factory=dict)
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
        if mode not in ("dsn", "header", "access-email"):
            raise ValueError(f"PGMCP_AUTH_MODE must be 'dsn', 'header' or 'access-email', got {mode!r}")
        if mode == "dsn":
            return cls()

        host = os.environ.get("PGMCP_DB_HOST")
        dbname = os.environ.get("PGMCP_DB_NAME")
        if not host or not dbname:
            raise ValueError(f"PGMCP_AUTH_MODE={mode} requires PGMCP_DB_HOST and PGMCP_DB_NAME")

        pattern_src = os.environ.get("PGMCP_ALLOWED_ROLES")
        if not pattern_src:
            raise ValueError(
                f"PGMCP_AUTH_MODE={mode} requires PGMCP_ALLOWED_ROLES, a regular expression "
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

        role_pattern = re.compile(pattern_src)
        identities: dict[str, str] = {}
        passwords: dict[str, str] = {}
        if mode == "access-email":
            # The JWT is the only proof of who is asking, so every part of its
            # check is mandatory here - not "on if configured".
            audience = os.environ.get("PGMCP_JWT_AUDIENCE")
            if not issuer or not audience or not jwt_required:
                raise ValueError(
                    "PGMCP_AUTH_MODE=access-email requires PGMCP_JWT_ISSUER and PGMCP_JWT_AUDIENCE, and JWT verification must stay on",
                )
            identities = load_identities(_required_env("PGMCP_IDENTITY_FILE"))
            passwords = load_passwords(_required_env("PGMCP_PASSWORD_FILE"))
            check_mapping(identities, passwords, role_pattern)

        return cls(
            enabled=True,
            mode=mode,
            identities=identities,
            passwords=passwords,
            credential_header=os.environ.get("PGMCP_CREDENTIAL_HEADER", DEFAULT_CREDENTIAL_HEADER).strip().lower(),
            host=host,
            port=int(os.environ.get("PGMCP_DB_PORT", "5432")),
            dbname=dbname,
            sslmode=os.environ.get("PGMCP_DB_SSLMODE") or None,
            role_pattern=role_pattern,
            connect_timeout=int(os.environ.get("PGMCP_CONNECT_TIMEOUT", "10")),
            jwt_required=jwt_required,
            jwks_url=jwks_url,
            jwt_issuer=issuer,
            jwt_audience=os.environ.get("PGMCP_JWT_AUDIENCE"),
            jwt_header=os.environ.get("PGMCP_JWT_HEADER", "cf-access-jwt-assertion").strip().lower(),
            jwt_common_name=os.environ.get("PGMCP_JWT_COMMON_NAME") or None,
            application_name=os.environ.get("PGMCP_APPLICATION_NAME", "langdock").strip() or "langdock",
        )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"PGMCP_AUTH_MODE=access-email requires {name}")
    return value


def _lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def load_identities(path: str) -> dict[str, str]:
    """`<e-mail> <role>` per line. One person may have several addresses."""
    out: dict[str, str] = {}
    for n, line in enumerate(_lines(path), 1):
        parts = line.split()
        if len(parts) != 2 or "@" not in parts[0]:
            raise ValueError(f"{path}, line {n}: expected '<e-mail> <role>'")
        email = parts[0].lower()
        if email in out and out[email] != parts[1]:
            raise ValueError(f"{path}, line {n}: address mapped to two roles")
        out[email] = parts[1]
    return out


def load_passwords(path: str) -> dict[str, str]:
    """`<role>:<password>` per line - the same shape as the credential header."""
    out: dict[str, str] = {}
    for n, line in enumerate(_lines(path), 1):
        try:
            role, password = split_credential(line)
        except AuthError:
            # Never echo the line: it holds a password.
            raise ValueError(f"{path}, line {n}: expected '<role>:<password>'") from None
        out[role] = password
    return out


def check_mapping(identities: dict[str, str], passwords: dict[str, str], role_pattern: re.Pattern[str]) -> None:
    """Fail at start-up, not on the first request of the person it concerns."""
    if not identities:
        raise ValueError("PGMCP_IDENTITY_FILE maps no address - nobody could log in")
    for role in sorted(set(identities.values())):
        if not role_pattern.fullmatch(role):
            raise ValueError(f"role {role} is outside PGMCP_ALLOWED_ROLES")
        if role not in passwords:
            raise ValueError(f"role {role} has no entry in PGMCP_PASSWORD_FILE")


def _email_tag(email: str) -> str:
    """Log a fingerprint, not the address: logs stay free of personal data."""
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:8]


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

    if cfg.mode == "access-email":
        return _conninfo_from_email(headers, cfg)

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


def _conninfo_from_email(headers: dict[str, str], cfg: IdentityConfig) -> str:
    token = headers.get(cfg.jwt_header, "")
    if not token:
        raise AuthError(f"missing {cfg.jwt_header} header")
    claims = _verify_jwt(token, cfg)

    # Service tokens carry `common_name` and no e-mail; only a person may pass.
    email = str(claims.get("email") or "").strip().lower()
    if not email:
        logger.warning("edge assertion without e-mail refused")
        raise AuthError("this identity has no access through this endpoint")

    role = cfg.identities.get(email)
    if role is None:
        logger.warning("no role mapped for address %s", _email_tag(email))
        raise AuthError("this identity has no access through this endpoint")
    if not cfg.role_pattern.fullmatch(role) or role not in cfg.passwords:
        # check_mapping() rules this out at start-up; stay closed if it ever drifts.
        logger.error("mapped role %s is not usable", role)
        raise AuthError("this identity has no access through this endpoint")

    return make_conninfo(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=role,
        password=cfg.passwords[role],
        connect_timeout=cfg.connect_timeout,
        application_name=cfg.application_name,
        **({"sslmode": cfg.sslmode} if cfg.sslmode else {}),
    )


def _unverified_email(token: str) -> str | None:
    try:
        import jwt

        claims = jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return None
    email = claims.get("email")
    return str(email).strip().lower() if email else None


def role_of(headers: dict[str, str], cfg: IdentityConfig) -> str | None:
    """Role name for logging, without touching the password. Never raises.

    In `access-email` mode the token is read WITHOUT verification - callers use
    this only for logging and for the row-cap exemption after `conninfo_for` has
    verified the same token in the same request.
    """
    if cfg.mode == "access-email":
        email = _unverified_email(headers.get(cfg.jwt_header, ""))
        return cfg.identities.get(email) if email else None
    try:
        raw = headers.get(cfg.credential_header, "")
        return split_credential(raw)[0] if raw else None
    except AuthError:
        return None
