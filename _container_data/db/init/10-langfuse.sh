#!/bin/bash
# Provision the `langfuse` role + database inside the main Postgres so the
# self-hosted Langfuse stack can share this server instead of a dedicated
# `langfuse_postgres` container.
#
# Runs once, on FIRST volume init, via /docker-entrypoint-initdb.d — Postgres
# executes these scripts against a local socket BEFORE it opens the network
# port, so the `langfuse` database exists before langfuse_web ever connects (no
# crash-loop). On an already-initialised volume this does NOT run; create the
# role + database manually (see CLAUDE.md → Langfuse) or start from a clean volume.
#
# Idempotent: guards on pg_roles / pg_database so re-running is safe. The password
# is injected from the container's LANGFUSE_POSTGRES_PASSWORD env (assumed free of
# single quotes, as generated secrets are).
set -euo pipefail

# Login role owning its own database (least privilege: no access to the main
# analytics tables). CREATE ROLE is guarded so a re-seed doesn't error.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c \
  "DO \$do\$ BEGIN
     IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'langfuse') THEN
       CREATE ROLE langfuse LOGIN PASSWORD '${LANGFUSE_POSTGRES_PASSWORD}';
     END IF;
   END \$do\$;"

# CREATE DATABASE cannot run inside the DO block above (no transaction), so guard
# it separately.
if ! psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -tAc \
  "SELECT 1 FROM pg_database WHERE datname = 'langfuse'" | grep -q 1; then
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE langfuse OWNER langfuse"
fi

echo "langfuse role + database ensured in main Postgres"
