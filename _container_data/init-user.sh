set -euo pipefail

_app_user="${POSTGRES_USERNAME:-}"
_llm_user="${POSTGRES_LLM_USERNAME:-}"
_pg_super="${POSTGRES_USER:-postgres}"

if [ -z "$_app_user" ] || [ "$_app_user" = "$_pg_super" ]; then
    echo "init-user.sh: POSTGRES_USERNAME unset or matches superuser — skipping."
    exit 0
fi

_esc_app_pass="$(printf '%s' "${POSTGRES_PASSWORD:-}"     | sed "s/'/''/g")"
_esc_llm_pass="$(printf '%s' "${POSTGRES_LLM_PASSWORD:-}" | sed "s/'/''/g")"

psql -v ON_ERROR_STOP=1 -U "$_pg_super" --dbname postgres <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${_app_user}') THEN
    EXECUTE format(
      'CREATE ROLE %I WITH LOGIN SUPERUSER ENCRYPTED PASSWORD %L',
      '${_app_user}', '${_esc_app_pass}'
    );
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${_llm_user}') THEN
    EXECUTE format(
      'CREATE ROLE %I WITH LOGIN NOSUPERUSER ENCRYPTED PASSWORD %L',
      '${_llm_user}', '${_esc_llm_pass}'
    );
  END IF;
END;
\$\$;
SQL

echo "init-user.sh: done."
