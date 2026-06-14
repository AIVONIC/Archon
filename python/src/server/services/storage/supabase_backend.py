"""
Supabase backend.

Wraps the synchronous Supabase client behind the async DatabaseBackend
interface. Each call runs in a worker thread so the event loop is never blocked
by the client's blocking HTTP I/O. Behavior is identical to the direct client
calls the service layer used before the abstraction was introduced; this exists
so the same code path can run against either backend during the cutover window.
"""

import asyncio
from typing import Any

from supabase import Client

from .database_backend import DatabaseBackend


class _SupabaseQueryBuilder:
    """Fluent builder that records the chain and forwards to supabase-py.

    Each method calls through to the underlying PostgREST query, so behavior is
    exactly the Supabase client's. ``execute()`` runs the (synchronous) query in
    a worker thread and returns the native response (which already exposes
    ``.data`` and ``.count``).
    """

    def __init__(self, query):
        self._query = query

    def select(
        self, columns: str = "*", count: str | None = None, head: bool = False
    ) -> "_SupabaseQueryBuilder":
        if count:
            self._query = self._query.select(columns, count=count, head=head)
        else:
            self._query = self._query.select(columns)
        return self

    def insert(self, payload) -> "_SupabaseQueryBuilder":
        self._query = self._query.insert(payload)
        return self

    def update(self, payload) -> "_SupabaseQueryBuilder":
        self._query = self._query.update(payload)
        return self

    def upsert(self, payload, on_conflict: str | None = None) -> "_SupabaseQueryBuilder":
        self._query = (
            self._query.upsert(payload, on_conflict=on_conflict)
            if on_conflict
            else self._query.upsert(payload)
        )
        return self

    def delete(self) -> "_SupabaseQueryBuilder":
        self._query = self._query.delete()
        return self

    def eq(self, column: str, value: Any) -> "_SupabaseQueryBuilder":
        self._query = self._query.eq(column, value)
        return self

    def neq(self, column: str, value: Any) -> "_SupabaseQueryBuilder":
        self._query = self._query.neq(column, value)
        return self

    def gte(self, column: str, value: Any) -> "_SupabaseQueryBuilder":
        self._query = self._query.gte(column, value)
        return self

    def lte(self, column: str, value: Any) -> "_SupabaseQueryBuilder":
        self._query = self._query.lte(column, value)
        return self

    def in_(self, column: str, values: list) -> "_SupabaseQueryBuilder":
        self._query = self._query.in_(column, values)
        return self

    def contains(self, column: str, value: dict) -> "_SupabaseQueryBuilder":
        self._query = self._query.contains(column, value)
        return self

    def ilike(self, column: str, pattern: str) -> "_SupabaseQueryBuilder":
        self._query = self._query.ilike(column, pattern)
        return self

    def or_ilike(self, columns: list[str], term: str) -> "_SupabaseQueryBuilder":
        self._query = self._query.or_(",".join(f"{column}.ilike.%{term}%" for column in columns))
        return self

    def include_unarchived(self) -> "_SupabaseQueryBuilder":
        self._query = self._query.or_("archived.is.null,archived.is.false")
        return self

    def order(self, column: str, desc: bool = False) -> "_SupabaseQueryBuilder":
        self._query = self._query.order(column, desc=desc)
        return self

    def limit(self, count: int) -> "_SupabaseQueryBuilder":
        self._query = self._query.limit(count)
        return self

    def range(self, start: int, end: int) -> "_SupabaseQueryBuilder":
        self._query = self._query.range(start, end)
        return self

    def single(self) -> "_SupabaseQueryBuilder":
        self._query = self._query.single()
        return self

    def maybe_single(self) -> "_SupabaseQueryBuilder":
        self._query = self._query.maybe_single()
        return self

    async def execute(self):
        return await asyncio.to_thread(self._query.execute)


class SupabaseBackend(DatabaseBackend):
    """Supabase-client implementation of the database interface."""

    def __init__(self, client: Client):
        self._client = client

    def table(self, table: str) -> _SupabaseQueryBuilder:
        return _SupabaseQueryBuilder(self._client.table(table))

    async def rpc(self, function_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        def _call() -> list[dict[str, Any]]:
            response = self._client.rpc(function_name, params).execute()
            return response.data or []

        return await asyncio.to_thread(_call)

    async def select_one(
        self, table: str, columns: str, match: dict[str, Any]
    ) -> dict[str, Any] | None:
        def _call() -> dict[str, Any] | None:
            query = self._client.table(table).select(columns)
            for column, value in match.items():
                query = query.eq(column, value)
            response = query.maybe_single().execute()
            return response.data if response else None

        return await asyncio.to_thread(_call)

    async def close(self) -> None:
        # The Supabase client manages its own HTTP session lifecycle.
        return None
