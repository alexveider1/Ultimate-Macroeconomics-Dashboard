"""Shared fixtures.

A throwaway Postgres container backs the ORM-read tests; a minimal FastAPI app
wired only with the read routers (and ``get_session`` overridden onto the
container) lets the routers be exercised end-to-end via ``TestClient`` without
importing ``main`` (which needs live config + secrets).
"""

from __future__ import annotations

from collections.abc import Iterator

from db import get_session
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from routers import crypto, eurostat, fred, worldbank, yahoo
from schema import Base
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
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


@pytest.fixture(scope="session")
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
def _clean_tables(engine: Engine) -> Iterator[None]:
    """Wipe every table before each test so cases stay isolated."""
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    yield


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    app = FastAPI()
    for module in (worldbank, yahoo, crypto, fred, eurostat):
        app.include_router(module.router)

    def _override_session() -> Iterator[Session]:
        with session_factory() as db_session:
            yield db_session

    app.dependency_overrides[get_session] = _override_session
    with TestClient(app) as test_client:
        yield test_client
