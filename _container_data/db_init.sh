#!/bin/bash
set -euo pipefail

: "${POSTGRES_PASSWORD:?}"
: "${POSTGRES_USERNAME:?}"
: "${POSTGRES_LLM_USERNAME:?}"
: "${POSTGRES_LLM_PASSWORD:?}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

_psql() { psql -h db -U postgres -d postgres "$@"; }

if ! _psql -c "SELECT 1" >/dev/null 2>&1; then
    echo "db_init: ERROR — cannot connect to PostgreSQL as 'postgres'." >&2
    echo "db_init: The postgres_data volume was likely initialised with a different" >&2
    echo "db_init: POSTGRES_PASSWORD.  Recreate it:" >&2
    echo "db_init:   docker compose down" >&2
    echo "db_init:   docker volume rm <project>_postgres_data" >&2
    echo "db_init:   docker compose up -d" >&2
    exit 1
fi

upsert_role() {
    local name="$1" pass="$2" opts="$3"
    local esc_pass exists
    esc_pass="$(printf '%s' "$pass" | sed "s/'/''/g")"
    exists="$(_psql -tAc "SELECT 1 FROM pg_roles WHERE rolname = '${name}'")"
    if [ "${exists}" = "1" ]; then
        _psql -c "ALTER ROLE \"${name}\" WITH ${opts} ENCRYPTED PASSWORD '${esc_pass}'"
    else
        _psql -c "CREATE ROLE \"${name}\" WITH LOGIN ${opts} ENCRYPTED PASSWORD '${esc_pass}'"
    fi
    echo "db_init: role '${name}' ready."
}

upsert_role "${POSTGRES_USERNAME}"     "${POSTGRES_PASSWORD}"     "SUPERUSER"
upsert_role "${POSTGRES_LLM_USERNAME}" "${POSTGRES_LLM_PASSWORD}" "NOSUPERUSER"
echo "db_init: done."
