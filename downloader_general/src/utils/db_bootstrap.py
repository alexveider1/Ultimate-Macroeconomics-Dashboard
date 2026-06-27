"""Idempotent Postgres role provisioning + read-only grants.

Folds in the responsibility that lived in the standalone `db_init` container:
ensure the read-only LLM role exists with the current password from env, and
that it can ``SELECT`` from every existing and future public table.

Runs on every `downloader_general` startup, so rotating
``POSTGRES_LLM_PASSWORD`` in ``.env`` takes effect on the next boot without an
extra container, and a fresh deployment / upgrade re-applies the grants in case
new tables were added since the last bootstrap.

The app superuser (``POSTGRES_USER``) is created natively by the
``postgres:18`` image at first volume init and is the role we connect as
here — so it never needs upserting from this code, and ``ALTER DEFAULT
PRIVILEGES FOR ROLE <superuser>`` correctly targets the role that later
creates the World Bank + Yahoo tables.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def ensure_llm_role(sql_uri: str, llm_username: str, llm_password: str) -> None:
    """Create or refresh the read-only LLM role and grant it SELECT on public.

    ``sql_uri`` must authenticate as a Postgres SUPERUSER so the role can be
    created/altered and grants can be issued on tables it (will) own. Identifier
    and password literals go through ``format()`` with ``%I`` / ``%L`` so SQL
    injection is impossible even though the inputs are server-controlled.

    The function is intentionally idempotent: every statement is safe to re-run
    on each container startup. ``GRANT ... ON ALL TABLES`` covers tables that
    already exist; ``ALTER DEFAULT PRIVILEGES`` covers future tables created by
    the same superuser (the World Bank, Yahoo, and on-demand
    ``downloader_extra`` ingest paths all connect as that role).
    """
    if not llm_username or not llm_password:
        logger.warning(
            "Skipping LLM role bootstrap: POSTGRES_LLM_USER / POSTGRES_LLM_PASSWORD not set."
        )
        return

    role_statement = text(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :llm_username) THEN
                EXECUTE format(
                    'CREATE ROLE %I WITH LOGIN NOSUPERUSER ENCRYPTED PASSWORD %L',
                    :llm_username, :llm_password
                );
            ELSE
                EXECUTE format(
                    'ALTER ROLE %I WITH LOGIN NOSUPERUSER ENCRYPTED PASSWORD %L',
                    :llm_username, :llm_password
                );
            END IF;
        END;
        $$;
        """
    )

    grant_statement = text(
        """
        DO $$
        DECLARE
            superuser_role text := current_user;
        BEGIN
            EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I',
                           current_database(), :llm_username);
            EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', :llm_username);
            EXECUTE format(
                'GRANT SELECT ON ALL TABLES IN SCHEMA public TO %I',
                :llm_username
            );
            EXECUTE format(
                'GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO %I',
                :llm_username
            );
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
                'GRANT SELECT ON TABLES TO %I',
                superuser_role, :llm_username
            );
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
                'GRANT SELECT ON SEQUENCES TO %I',
                superuser_role, :llm_username
            );
        END;
        $$;
        """
    )

    engine = create_engine(sql_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                role_statement,
                {"llm_username": llm_username, "llm_password": llm_password},
            )
            connection.execute(grant_statement, {"llm_username": llm_username})
        logger.info(
            "Postgres LLM role '%s' bootstrapped (role upserted, "
            "SELECT granted on public schema + default privileges set).",
            llm_username,
        )
    finally:
        engine.dispose()
