"""
Async database-backend test double.

Mirrors the fluent surface of the real ``DatabaseBackend`` query builder
(``table(...).select(...).eq(...).execute()`` and the insert/update/upsert/delete
variants) but returns canned data instead of touching a database. ``execute()`` is
a coroutine, so it works with the ``await`` the converted services now use.

Use this in place of the old synchronous Supabase ``Mock()`` in tests that
construct a service or storage op with a backend.
"""

from typing import Any

from src.server.services.storage import QueryResult


class _FakeQuery:
    """Records the chained calls and returns the backend's canned result."""

    def __init__(self, backend: "FakeBackend"):
        self._backend = backend

    def __getattr__(self, name: str):
        # Any builder method (select/insert/eq/order/upsert/...) just chains and
        # is recorded for assertions. ``execute`` is a real method, so it is never
        # routed through here.
        def chain(*args: Any, **kwargs: Any) -> "_FakeQuery":
            self._backend.calls.append((name, args, kwargs))
            return self

        return chain

    async def execute(self) -> QueryResult:
        return self._backend.next_result()


class FakeBackend:
    """Drop-in async backend returning canned rows.

    Args:
        data: rows returned by every ``execute()`` (defaults to one generic row so
            existence checks pass).
        count: value for ``QueryResult.count`` (count="exact" queries).
    """

    def __init__(self, data: list[dict] | None = None, count: int | None = None):
        self._data = data if data is not None else [{"source_id": "fake"}]
        self._count = count
        self.calls: list[tuple[str, tuple, dict]] = []
        self.tables: list[str] = []

    def table(self, name: str) -> _FakeQuery:
        self.tables.append(name)
        return _FakeQuery(self)

    def next_result(self) -> QueryResult:
        return QueryResult(data=self._data, count=self._count)

    async def rpc(self, function_name: str, params: dict) -> list[dict]:
        self.calls.append(("rpc", (function_name,), params))
        return self._data

    async def select_one(self, table: str, columns: str, match: dict) -> dict | None:
        self.tables.append(table)
        return self._data[0] if self._data else None
