#!/bin/bash

_app_user="${POSTGRES_USERNAME:-}"
_pg_super="${POSTGRES_USER:-postgres}"

if [ -z "$_app_user" ] || [ "$_app_user" = "$_pg_super" ]; then
    echo "init-user.sh: POSTGRES_USERNAME matches POSTGRES_USER or is unset — skipping."
else
    echo "init-user.sh: Creating application superuser '${_app_user}' ..."

    if createuser --superuser --login "$_app_user"; then
        _esc_pass="$(printf '%s' "${POSTGRES_PASSWORD:-}" | sed "s/'/''/g")"
        psql -c "ALTER USER \"${_app_user}\" WITH ENCRYPTED PASSWORD '${_esc_pass}'"
        echo "init-user.sh: User '${_app_user}' created successfully."
    else
        echo "init-user.sh: WARNING — could not create user '${_app_user}'." >&2
    fi
fi
