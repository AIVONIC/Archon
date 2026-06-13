"""
Database backend factory.

Selects the active backend from ``STORAGE_BACKEND`` and caches it as a
process-wide singleton.

``STORAGE_BACKEND`` values:
- ``supabase`` (default): wrap the Supabase client. This is the default so that
  an unset variable preserves the pre-cutover behavior and a refactored image
  cannot silently route production at an empty Postgres. The production cutover
  step sets ``postgres`` explicitly.
- ``postgres``: connect directly via asyncpg using ``ARCHON_DATABASE_URL``.

Phase 5 of the cutover removes the ``supabase`` path and this switch, leaving
``postgres`` as the only backend.
"""

import os

from ...config.config import ConfigurationError
from .database_backend import DatabaseBackend
from .postgres_backend import PostgresBackend
from .supabase_backend import SupabaseBackend

_backend: DatabaseBackend | None = None


def _create_backend() -> DatabaseBackend:
    selected = os.getenv("STORAGE_BACKEND", "supabase").strip().lower()

    if selected == "postgres":
        dsn = os.getenv("ARCHON_DATABASE_URL")
        if not dsn:
            raise ConfigurationError(
                "STORAGE_BACKEND=postgres requires ARCHON_DATABASE_URL to be set"
            )
        return PostgresBackend(dsn)

    if selected == "supabase":
        # Imported lazily so a pure-postgres deployment never needs Supabase env.
        from ...utils import get_supabase_client

        return SupabaseBackend(get_supabase_client())

    raise ConfigurationError(
        f"Unknown STORAGE_BACKEND={selected!r} (expected 'postgres' or 'supabase')"
    )


def get_database_backend() -> DatabaseBackend:
    """Return the process-wide backend selected by ``STORAGE_BACKEND``."""
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
