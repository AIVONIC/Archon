"""
Database backend factory.

Connects directly to Postgres via asyncpg using ``ARCHON_DATABASE_URL`` and
caches the backend as a process-wide singleton.
"""

import os

from ...config.config import ConfigurationError
from .database_backend import DatabaseBackend
from .postgres_backend import PostgresBackend

_backend: DatabaseBackend | None = None


def _create_backend() -> DatabaseBackend:
    dsn = os.getenv("ARCHON_DATABASE_URL")
    if not dsn:
        raise ConfigurationError("ARCHON_DATABASE_URL must be set")
    return PostgresBackend(dsn)


def get_database_backend() -> DatabaseBackend:
    """Return the process-wide Postgres backend."""
    global _backend
    if _backend is None:
        _backend = _create_backend()
    return _backend


def set_database_backend(backend: DatabaseBackend | None) -> None:
    """Override the cached backend (tests and explicit dependency injection)."""
    global _backend
    _backend = backend


def reset_database_backend() -> None:
    """Clear the cached backend so the next call re-reads the environment."""
    global _backend
    _backend = None
