"""Integration tests for the LLM role bootstrap.

Cover three contracts:
  1. The role is created and re-runs are idempotent (password rotation).
  2. Existing public tables become readable by the role after bootstrap.
  3. Tables created *after* the bootstrap are readable too (default privileges).
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from sqlalchemy import create_engine, text
from src.utils.db_bootstrap import ensure_llm_role


def _role_exists(sql_uri: str, role_name: str) -> bool:
    engine = create_engine(sql_uri)
    try:
        with engine.connect() as conn:
            return bool(
                conn.execute(
                    text("SELECT 1 FROM pg_roles WHERE rolname = :name"),
                    {"name": role_name},
                ).scalar()
            )
    finally:
        engine.dispose()


def _llm_uri(superuser_uri: str, llm_username: str, llm_password: str) -> str:
    """Rewrite the superuser URI to authenticate as ``llm_username`` instead."""
    parsed = urlparse(superuser_uri)
    netloc = f"{llm_username}:{llm_password}@{parsed.hostname}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def test_ensure_llm_role_creates_and_is_idempotent(postgres_uri: str) -> None:
    role = "llm_reader"
    ensure_llm_role(sql_uri=postgres_uri, llm_username=role, llm_password="secret-1")
    assert _role_exists(postgres_uri, role)

    # Second call should rotate the password without erroring.
    ensure_llm_role(sql_uri=postgres_uri, llm_username=role, llm_password="secret-2")
    assert _role_exists(postgres_uri, role)


def test_ensure_llm_role_skips_when_credentials_blank(postgres_uri: str) -> None:
    # Should not raise and should not create any role.
    ensure_llm_role(sql_uri=postgres_uri, llm_username="", llm_password="")
    assert not _role_exists(postgres_uri, "")


def test_ensure_llm_role_grants_select_on_existing_and_future_tables(
    postgres_uri: str,
) -> None:
    """Pre-existing tables become readable, and so do tables created later."""
    role = "llm_grants_user"
    password = "grant-secret"

    engine = create_engine(postgres_uri)
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS pre_bootstrap_tbl"))
            conn.execute(text("DROP TABLE IF EXISTS post_bootstrap_tbl"))
            conn.execute(text("CREATE TABLE pre_bootstrap_tbl (id INT, label TEXT)"))
            conn.execute(text("INSERT INTO pre_bootstrap_tbl (id, label) VALUES (1, 'pre')"))
    finally:
        engine.dispose()

    ensure_llm_role(sql_uri=postgres_uri, llm_username=role, llm_password=password)

    # Create a table AFTER the bootstrap, still as the superuser. The default
    # privileges set by ensure_llm_role must make it readable too.
    engine = create_engine(postgres_uri)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE post_bootstrap_tbl (id INT, label TEXT)"))
            conn.execute(text("INSERT INTO post_bootstrap_tbl (id, label) VALUES (2, 'post')"))
    finally:
        engine.dispose()

    llm_uri = _llm_uri(postgres_uri, role, password)
    llm_engine = create_engine(llm_uri)
    try:
        with llm_engine.connect() as conn:
            assert (
                conn.execute(text("SELECT label FROM pre_bootstrap_tbl WHERE id = 1")).scalar()
                == "pre"
            )
            assert (
                conn.execute(text("SELECT label FROM post_bootstrap_tbl WHERE id = 2")).scalar()
                == "post"
            )
    finally:
        llm_engine.dispose()
