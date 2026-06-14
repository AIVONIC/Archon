"""
Database backend abstraction.

Defines the interface the service layer uses to reach Postgres, independent of
whether the underlying transport is the Supabase client (PostgREST over HTTP) or
a direct asyncpg connection pool. The search layer is the first consumer; the
remaining services migrate onto the same interface incrementally.

Every method is async. The Supabase implementation runs its synchronous client
calls in a worker thread so callers never block the event loop, matching the
asyncpg implementation's non-blocking contract.
"""

from abc import ABC, abstractmethod
from typing import Any


class DatabaseBackend(ABC):
    """Transport-agnostic database access used by the service layer."""

    @abstractmethod
    async def rpc(self, function_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Call a Postgres set-returning function by name.

        Args:
            function_name: Name of the SQL function (e.g. ``match_archon_crawled_pages_multi``).
            params: Mapping of the function's named arguments to values. Python
                types map to Postgres types: ``list[float]`` -> ``vector``,
                ``dict`` -> ``jsonb``, ``None`` -> SQL ``NULL``.

        Returns:
            Each returned row as a dict. ``jsonb`` columns are decoded to dicts
            and numeric columns to native Python numbers, so the shape matches
            what the Supabase client returns.
        """

    @abstractmethod
    def table(self, table: str) -> Any:
        """Start a fluent query against ``table``.

        Returns a builder supporting ``select``/``insert``/``update``/``upsert``/
        ``delete`` terminals, ``eq``/``neq``/``gte``/``lte``/``in_``/``contains``/
        ``or_ilike``/``include_unarchived`` filters, ``order``/``limit``/``single``
        modifiers, and an awaitable ``execute()`` whose result exposes ``.data``
        and ``.count``. The shape mirrors the Supabase client so call sites are
        identical across backends.
        """

    @abstractmethod
    async def select_one(
        self, table: str, columns: str, match: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Fetch a single row matched by equality on every key in ``match``.

        Args:
            table: Table name.
            columns: Comma-separated column list (e.g. ``"id, url, word_count"``).
            match: Column -> value equality filters, AND-ed together.

        Returns:
            The first matching row as a dict, or ``None`` if no row matches.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (connection pool, client sockets)."""
