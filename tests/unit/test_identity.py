"""Tests for per-request credentials (`PGMCP_AUTH_MODE=header`).

The interesting cases are the refusals. A bug that lets a request through with
the wrong identity is invisible in normal use: queries succeed, results look
right, and the audit trail quietly names the wrong person.
"""

import datetime
import re

import pytest
from psycopg.conninfo import conninfo_to_dict

from postgres_mcp.identity import AuthError
from postgres_mcp.identity import IdentityConfig
from postgres_mcp.identity import conninfo_for
from postgres_mcp.identity import role_of
from postgres_mcp.identity import split_credential

ANALYSTS = re.compile(r"^analyst_[a-z][a-z0-9_]{1,40}$")


def cfg(**kw) -> IdentityConfig:
    base = dict(
        enabled=True,
        credential_header="x-db-credential",
        host="db.internal",
        port=5434,
        dbname="platform",
        role_pattern=ANALYSTS,
    )
    base.update(kw)
    return IdentityConfig(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ splitting


def test_password_may_contain_colons():
    """Role names cannot contain a colon, passwords can. Split on the first one."""
    assert split_credential("analyst_oliver:a:b:c") == ("analyst_oliver", "a:b:c")


@pytest.mark.parametrize("raw", ["analyst_oliver", "", ":secret", "analyst_oliver:", "   :x"])
def test_malformed_credentials_are_refused(raw):
    with pytest.raises(AuthError):
        split_credential(raw)


def test_role_of_never_raises():
    """Used for logging only - it must not turn a bad header into a crash."""
    assert role_of({"x-db-credential": "analyst_lisa:pw"}, cfg()) == "analyst_lisa"
    assert role_of({"x-db-credential": "nonsense"}, cfg()) is None
    assert role_of({}, cfg()) is None


# ------------------------------------------------------------------ conninfo


def test_connection_target_comes_from_the_server_not_the_client():
    got = conninfo_to_dict(conninfo_for({"x-db-credential": "analyst_oliver:s3cret"}, cfg()))
    assert got["host"] == "db.internal"
    assert got["port"] == "5434"
    assert got["dbname"] == "platform"
    assert got["user"] == "analyst_oliver"
    assert got["password"] == "s3cret"


def test_password_with_shell_and_dsn_metacharacters_survives_intact():
    """The reason credentials are passed as parameters instead of a built string.

    A generated password can contain spaces, quotes and backslashes. String
    concatenation either corrupts it (login fails, looks like a wrong password)
    or lets a crafted value append its own keywords.
    """
    nasty = "a b 'c' \"d\" \\e = f"
    got = conninfo_to_dict(conninfo_for({"x-db-credential": f"analyst_oliver:{nasty}"}, cfg()))
    assert got["password"] == nasty
    assert got["host"] == "db.internal"


def test_password_cannot_smuggle_a_different_host():
    """A password that looks like DSN keywords must stay a password."""
    got = conninfo_to_dict(conninfo_for({"x-db-credential": "analyst_oliver:x' host=evil.example port=5432 '"}, cfg()))
    assert got["host"] == "db.internal"
    assert "evil.example" not in got.get("host", "")


def test_roles_outside_the_allowlist_are_refused():
    for role in ("postgres", "selectline_loader", "Analyst_Oliver", "analyst_", "app_reporting_reader"):
        with pytest.raises(AuthError):
            conninfo_for({"x-db-credential": f"{role}:pw"}, cfg())


def test_missing_header_is_refused():
    with pytest.raises(AuthError):
        conninfo_for({}, cfg())


def test_disabled_config_refuses_everything():
    """Fail closed: if the mode is off, this path must never build a connection."""
    with pytest.raises(AuthError):
        conninfo_for({"x-db-credential": "analyst_oliver:pw"}, IdentityConfig())


def test_default_role_pattern_matches_nothing():
    """A half-configured server must refuse, not accept every role."""
    with pytest.raises(AuthError):
        conninfo_for({"x-db-credential": "analyst_oliver:pw"}, cfg(role_pattern=IdentityConfig().role_pattern))


# ------------------------------------------------------------------ env parsing


def test_dsn_mode_is_the_default(monkeypatch):
    monkeypatch.delenv("PGMCP_AUTH_MODE", raising=False)
    assert IdentityConfig.from_env().enabled is False


def test_header_mode_requires_a_role_allowlist(monkeypatch):
    monkeypatch.setenv("PGMCP_AUTH_MODE", "header")
    monkeypatch.setenv("PGMCP_DB_HOST", "db.internal")
    monkeypatch.setenv("PGMCP_DB_NAME", "platform")
    monkeypatch.delenv("PGMCP_ALLOWED_ROLES", raising=False)
    with pytest.raises(ValueError, match="PGMCP_ALLOWED_ROLES"):
        IdentityConfig.from_env()


def test_header_mode_requires_a_database(monkeypatch):
    monkeypatch.setenv("PGMCP_AUTH_MODE", "header")
    monkeypatch.delenv("PGMCP_DB_HOST", raising=False)
    monkeypatch.setenv("PGMCP_DB_NAME", "platform")
    with pytest.raises(ValueError, match="PGMCP_DB_HOST"):
        IdentityConfig.from_env()


def test_issuer_implies_the_cloudflare_certs_url(monkeypatch):
    monkeypatch.setenv("PGMCP_AUTH_MODE", "header")
    monkeypatch.setenv("PGMCP_DB_HOST", "db.internal")
    monkeypatch.setenv("PGMCP_DB_NAME", "platform")
    monkeypatch.setenv("PGMCP_ALLOWED_ROLES", r"^analyst_[a-z0-9_]+$")
    monkeypatch.setenv("PGMCP_JWT_ISSUER", "https://example.cloudflareaccess.com")
    got = IdentityConfig.from_env()
    assert got.jwks_url == "https://example.cloudflareaccess.com/cdn-cgi/access/certs"
    assert got.jwt_required is True


# ------------------------------------------------------------------ edge assertion


@pytest.fixture
def signing():
    """An RSA key plus a helper that mints tokens the verifier should accept."""
    jwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def mint(**overrides):
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        claims = {
            "iss": "https://example.cloudflareaccess.com",
            "aud": "the-application-aud",
            "common_name": "token-abc.access",
            "type": "app",
            "iat": now,
            "exp": now + datetime.timedelta(hours=1),
        }
        claims.update(overrides)
        return jwt.encode(claims, key, algorithm="RS256")

    return key, mint


@pytest.fixture
def verifying(monkeypatch, signing):
    """Point the verifier at our key instead of fetching a real JWKS."""
    key, mint = signing
    from postgres_mcp import identity

    class FakeJWKSClient:
        def __init__(self, public_key):
            self._public_key = public_key

        def get_signing_key_from_jwt(self, token):  # noqa: ARG002
            return type("Key", (), {"key": self._public_key})()

    monkeypatch.setattr(identity, "_jwks_client", lambda url: FakeJWKSClient(key.public_key()))
    return mint


def jwt_cfg(**kw) -> IdentityConfig:
    return cfg(
        jwt_required=True,
        jwks_url="https://example.cloudflareaccess.com/cdn-cgi/access/certs",
        jwt_issuer="https://example.cloudflareaccess.com",
        jwt_audience="the-application-aud",
        jwt_header="cf-access-jwt-assertion",
        **kw,
    )


def headers(token: str) -> dict:
    return {"cf-access-jwt-assertion": token, "x-db-credential": "analyst_oliver:pw"}


def test_valid_assertion_is_accepted(verifying):
    got = conninfo_to_dict(conninfo_for(headers(verifying()), jwt_cfg()))
    assert got["user"] == "analyst_oliver"


def test_missing_assertion_is_refused(verifying):
    with pytest.raises(AuthError, match="cf-access-jwt-assertion"):
        conninfo_for({"x-db-credential": "analyst_oliver:pw"}, jwt_cfg())


def test_expired_assertion_is_refused(verifying):
    past = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=2)
    token = verifying(iat=past, exp=past + datetime.timedelta(minutes=1))
    with pytest.raises(AuthError):
        conninfo_for(headers(token), jwt_cfg())


def test_assertion_for_another_application_is_refused(verifying):
    """The aud claim is what ties a token to THIS application."""
    with pytest.raises(AuthError):
        conninfo_for(headers(verifying(aud="a-different-application")), jwt_cfg())


def test_assertion_from_another_issuer_is_refused(verifying):
    with pytest.raises(AuthError):
        conninfo_for(headers(verifying(iss="https://attacker.example")), jwt_cfg())


def test_wrong_service_token_is_refused(verifying):
    """Pinning common_name keeps other tokens of the same account out."""
    token = verifying(common_name="some-other-token.access")
    with pytest.raises(AuthError):
        conninfo_for(headers(token), jwt_cfg(jwt_common_name="token-abc.access"))


def test_unsigned_token_is_refused(verifying):
    """alg=none and friends: a decoded token is not a verified one."""
    import jwt as pyjwt

    token = pyjwt.encode({"aud": "the-application-aud"}, key="", algorithm="none")
    with pytest.raises(AuthError):
        conninfo_for(headers(token), jwt_cfg())


def test_error_messages_never_contain_the_credential(verifying):
    try:
        conninfo_for({**headers(verifying()), "x-db-credential": "postgres:hunter2"}, jwt_cfg())
    except AuthError as exc:
        assert "hunter2" not in str(exc)
        assert "postgres" not in str(exc)
    else:  # pragma: no cover
        pytest.fail("a role outside the allowlist must be refused")


# ------------------------------------------------------------ transport security


@pytest.fixture
def server_module(monkeypatch):
    from postgres_mcp import server

    monkeypatch.setattr(server.mcp, "settings", server.mcp.settings.model_copy(deep=True))
    return server


def test_declared_hosts_are_allowed(server_module, monkeypatch):
    """The reason this exists: FastMCP fixes allowed_hosts at construction time,
    when the host is still the default 127.0.0.1. Binding elsewhere later leaves
    a server that listens everywhere and refuses every request by name."""
    monkeypatch.setenv("PGMCP_ALLOWED_HOSTS", "mcp.example.com, mcp.example.com:*")
    server_module.apply_transport_security("0.0.0.0")
    sec = server_module.mcp.settings.transport_security
    assert sec is not None
    assert sec.enable_dns_rebinding_protection is True
    assert "mcp.example.com" in sec.allowed_hosts


def test_non_localhost_without_declaration_disables_protection(server_module, monkeypatch):
    """Matches what upstream would have done for this bind host - and is loud
    about it in the log rather than failing every request silently."""
    monkeypatch.delenv("PGMCP_ALLOWED_HOSTS", raising=False)
    server_module.apply_transport_security("0.0.0.0")
    assert server_module.mcp.settings.transport_security is None


def test_localhost_keeps_the_default_protection(server_module, monkeypatch):
    monkeypatch.delenv("PGMCP_ALLOWED_HOSTS", raising=False)
    before = server_module.mcp.settings.transport_security
    server_module.apply_transport_security("127.0.0.1")
    assert server_module.mcp.settings.transport_security is before
