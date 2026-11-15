"""Tests for `PGMCP_AUTH_MODE=access-email`.

Here the person comes from the verified edge JWT and the role from a fixed list.
The cases that matter are the refusals: a valid token for someone who is not on
the list, a service token without a person, and a token that was never checked.
"""

import datetime
import re

import pytest
from psycopg.conninfo import conninfo_to_dict

from postgres_mcp.identity import AuthError
from postgres_mcp.identity import IdentityConfig
from postgres_mcp.identity import check_mapping
from postgres_mcp.identity import conninfo_for
from postgres_mcp.identity import load_identities
from postgres_mcp.identity import load_passwords
from postgres_mcp.identity import role_of

ANALYSTS = re.compile(r"^analyst_[a-z][a-z0-9_]{1,40}$")
ISSUER = "https://example.cloudflareaccess.com"
AUD = "the-application-aud"


@pytest.fixture
def mint(monkeypatch):
    jwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives.asymmetric import rsa

    from postgres_mcp import identity

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class FakeJWKSClient:
        def get_signing_key_from_jwt(self, token):  # noqa: ARG002
            return type("Key", (), {"key": key.public_key()})()

    monkeypatch.setattr(identity, "_jwks_client", lambda url: FakeJWKSClient())

    def _mint(**overrides):
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        claims = {
            "iss": ISSUER,
            "aud": AUD,
            "email": "jane@example.com",
            "type": "app",
            "iat": now,
            "exp": now + datetime.timedelta(minutes=15),
        }
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, key, algorithm="RS256")

    return _mint


def cfg(**kw) -> IdentityConfig:
    base = dict(
        enabled=True,
        mode="access-email",
        host="db.internal",
        port=5434,
        dbname="platform",
        role_pattern=ANALYSTS,
        jwt_required=True,
        jwks_url=ISSUER + "/cdn-cgi/access/certs",
        jwt_issuer=ISSUER,
        jwt_audience=AUD,
        application_name="claude",
        identities={"jane@example.com": "analyst_jane", "jane@other.example": "analyst_jane"},
        passwords={"analyst_jane": "pw:with:colons"},
    )
    base.update(kw)
    return IdentityConfig(**base)  # type: ignore[arg-type]


def hdr(token: str) -> dict[str, str]:
    return {"cf-access-jwt-assertion": token}


def test_mapped_person_logs_in_as_their_role(mint):
    got = conninfo_to_dict(conninfo_for(hdr(mint()), cfg()))
    assert got["user"] == "analyst_jane"
    assert got["password"] == "pw:with:colons"
    assert got["host"] == "db.internal"
    assert got["application_name"] == "claude"


def test_second_address_of_the_same_person(mint):
    got = conninfo_to_dict(conninfo_for(hdr(mint(email="Jane@Other.Example")), cfg()))
    assert got["user"] == "analyst_jane"


def test_valid_token_for_an_unlisted_address_is_refused(mint):
    """Passing the edge is not enough - the address must be on the list."""
    with pytest.raises(AuthError, match="no access"):
        conninfo_for(hdr(mint(email="someone@example.com")), cfg())


def test_service_token_without_a_person_is_refused(mint):
    with pytest.raises(AuthError, match="no access"):
        conninfo_for(hdr(mint(email=None, common_name="abc.access")), cfg())


def test_missing_assertion_is_refused(mint):
    with pytest.raises(AuthError, match="cf-access-jwt-assertion"):
        conninfo_for({}, cfg())


def test_credential_header_is_ignored(mint):
    """A client must not be able to pick another role by adding the old header."""
    got = conninfo_to_dict(conninfo_for({**hdr(mint()), "x-db-credential": "analyst_boss:pw"}, cfg()))
    assert got["user"] == "analyst_jane"


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "another-application"},
        {"iss": "https://attacker.example"},
        {"exp": datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc), "iat": datetime.datetime(1999, 12, 31, tzinfo=datetime.timezone.utc)},
    ],
)
def test_unverifiable_assertion_is_refused(mint, overrides):
    with pytest.raises(AuthError):
        conninfo_for(hdr(mint(**overrides)), cfg())


def test_unsigned_token_is_refused(mint):
    import jwt as pyjwt

    token = pyjwt.encode({"aud": AUD, "iss": ISSUER, "email": "jane@example.com"}, key="", algorithm="none")
    with pytest.raises(AuthError):
        conninfo_for(hdr(token), cfg())


def test_role_of_maps_the_address_and_never_raises(mint):
    assert role_of(hdr(mint()), cfg()) == "analyst_jane"
    assert role_of(hdr("garbage"), cfg()) is None
    assert role_of({}, cfg()) is None


def test_files_are_parsed(tmp_path):
    ids = tmp_path / "identities"
    ids.write_text("# comment\nJane@Example.com analyst_jane\njane@other.example analyst_jane\n")
    pws = tmp_path / "passwords"
    pws.write_text("analyst_jane:a:b c\n")
    assert load_identities(str(ids)) == {"jane@example.com": "analyst_jane", "jane@other.example": "analyst_jane"}
    assert load_passwords(str(pws)) == {"analyst_jane": "a:b c"}


def test_password_file_errors_never_echo_the_line(tmp_path):
    pws = tmp_path / "passwords"
    pws.write_text("hunter2-without-role\n")
    with pytest.raises(ValueError) as exc:
        load_passwords(str(pws))
    assert "hunter2" not in str(exc.value)


def test_address_mapped_to_two_roles_is_refused(tmp_path):
    ids = tmp_path / "identities"
    ids.write_text("jane@example.com analyst_jane\njane@example.com analyst_boss\n")
    with pytest.raises(ValueError, match="two roles"):
        load_identities(str(ids))


@pytest.mark.parametrize(
    ("identities", "passwords", "match"),
    [
        ({}, {}, "no address"),
        ({"a@x.example": "postgres"}, {"postgres": "pw"}, "outside"),
        ({"a@x.example": "analyst_ab"}, {}, "no entry"),
    ],
)
def test_mapping_is_checked_at_start(identities, passwords, match):
    with pytest.raises(ValueError, match=match):
        check_mapping(identities, passwords, ANALYSTS)


def test_from_env_requires_audience_and_files(monkeypatch, tmp_path):
    monkeypatch.setenv("PGMCP_AUTH_MODE", "access-email")
    monkeypatch.setenv("PGMCP_DB_HOST", "db.internal")
    monkeypatch.setenv("PGMCP_DB_NAME", "platform")
    monkeypatch.setenv("PGMCP_ALLOWED_ROLES", ANALYSTS.pattern)
    monkeypatch.setenv("PGMCP_JWT_ISSUER", ISSUER)
    with pytest.raises(ValueError, match="AUDIENCE"):
        IdentityConfig.from_env()

    monkeypatch.setenv("PGMCP_JWT_AUDIENCE", AUD)
    monkeypatch.setenv("PGMCP_JWT_REQUIRED", "false")
    with pytest.raises(ValueError, match="verification must stay on"):
        IdentityConfig.from_env()

    monkeypatch.delenv("PGMCP_JWT_REQUIRED")
    with pytest.raises(ValueError, match="PGMCP_IDENTITY_FILE"):
        IdentityConfig.from_env()

    ids = tmp_path / "identities"
    ids.write_text("jane@example.com analyst_jane\n")
    pws = tmp_path / "passwords"
    pws.write_text("analyst_jane:pw\n")
    monkeypatch.setenv("PGMCP_IDENTITY_FILE", str(ids))
    monkeypatch.setenv("PGMCP_PASSWORD_FILE", str(pws))
    got = IdentityConfig.from_env()
    assert got.mode == "access-email" and got.jwt_required and got.identities == {"jane@example.com": "analyst_jane"}
