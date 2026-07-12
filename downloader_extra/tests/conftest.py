"""Shared fixtures: a throwaway Postgres container per test session."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from schema import Base
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_uri() -> Iterator[str]:
    with PostgresContainer("postgres:18-alpine") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture(scope="session")
def engine(postgres_uri: str) -> Iterator[Engine]:
    engine = create_engine(postgres_uri, future=True)
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session
        session.rollback()
