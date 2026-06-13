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


class SupabaseBackend(DatabaseBackend):
    """Supabase-client implementation of the database interface."""

    def __init__(self, client: Client):
        self._client = client

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
