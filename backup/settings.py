"""Typed secrets for the backup service.

Dumps Postgres over the network as the **superuser** role (so ``pg_dump`` sees
every table) and calls the Qdrant snapshot API with its service key. Cloud
credentials are deliberately NOT here — they live in the mounted ``rclone.conf``
so the rclone remote can be any of its 70+ backends without new env vars. The
container is given exactly this set in ``docker-compose.yaml``. ``postgres_db``
falls back to ``config.postgres.database`` when unset.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Postgres superuser credentials + the Qdrant API key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = Field(validation_alias=AliasChoices("POSTGRES_USER"))
    postgres_password: str = Field(validation_alias=AliasChoices("POSTGRES_PASSWORD"))
    postgres_db: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DB"))
    qdrant_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "QDRANT_API_KEY", "QDRANT__API_KEY", "QDRANT__SERVICE__API_KEY"
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, validated on first call."""
    # Fields are populated from the environment, not constructor args.
    return Settings()  # ty: ignore[missing-argument]
