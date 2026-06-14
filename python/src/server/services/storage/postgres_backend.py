"""
Direct Postgres backend using an asyncpg connection pool.

Connects to the self-hosted Archon Postgres (pgvector + psql_bm25s) over a plain
Postgres socket, bypassing PostgREST. Connections register text codecs so that
``jsonb`` columns decode to dicts and ``vector`` columns to ``list[float]``,
keeping the returned row shape identical to the Supabase client's.

RPCs are issued with named-argument notation (``fn(arg => $n)``) so Postgres
infers each parameter's type from the function signature. That lets asyncpg pick
the right codec for ``jsonb`` and ``vector`` arguments without per-call cast
bookkeeping.
"""

import asyncio
import json
import re
from typing import Any

import asyncpg

from ...config.logfire_config import get_logger
from .database_backend import DatabaseBackend
from .query_builder import PostgresQueryBuilder

logger = get_logger(__name__)

# Postgres identifiers we interpolate into SQL come from our own code, never user
# input. Validate them anyway so a future caller cannot turn a typo into an
# injection vector.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, kind: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValueError(f"Invalid {kind} identifier: {name!r}")
    return name


def _validate_columns(columns: str) -> str:
    for col in columns.split(","):
        _validate_identifier(col.strip(), "column")
    return columns


def _encode_vector(value: Any) -> str:
    return "[" + ",".join(str(float(x)) for x in value) + "]"


def _decode_vector(value: str) -> list[float]:
    inner = value.strip("[]")
    return [float(x) for x in inner.split(",")] if inner else []


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register codecs so returned/sent values match the Supabase client shape."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await conn.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    # pgvector installs the ``vector`` type into the schema where the extension
    # lives (``public`` in the Archon database).
    await conn.set_type_codec(
        "vector",
        encoder=_encode_vector,
        decoder=_decode_vector,
        schema="public",
        format="text",
    )


class PostgresBackend(DatabaseBackend):
    """asyncpg-backed implementation of the database interface."""

    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20):
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()
        self._column_type_cache: dict[str, dict[str, str]] = {}
        self._primary_key_cache: dict[str, str] = {}

    @staticmethod
    def validate_identifier(name: str) -> str:
        """Public identifier check used by the query builder."""
        return _validate_identifier(name, "identifier")

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazily create the pool on first use (asyncpg pools need an event loop)."""
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        self._dsn,
                        min_size=self._min_size,
                        max_size=self._max_size,
                        init=_init_connection,
                    )
                    logger.info(
                        f"asyncpg pool created (min={self._min_size}, max={self._max_size})"
                    )
        return self._pool

    async def rpc(self, function_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        _validate_identifier(function_name, "function")
        arg_fragments = []
        values = []
        for position, (name, value) in enumerate(params.items(), start=1):
            _validate_identifier(name, "argument")
            arg_fragments.append(f"{name} => ${position}")
            values.append(value)

        sql = f"SELECT * FROM {function_name}({', '.join(arg_fragments)})"
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *values)
        return [dict(row) for row in rows]

    async def select_one(
        self, table: str, columns: str, match: dict[str, Any]
    ) -> dict[str, Any] | None:
        _validate_identifier(table, "table")
        _validate_columns(columns)
        where_fragments = []
        values = []
        for position, (name, value) in enumerate(match.items(), start=1):
            _validate_identifier(name, "column")
            where_fragments.append(f"{name} = ${position}")
            values.append(value)

        where_clause = " AND ".join(where_fragments) if where_fragments else "TRUE"
        sql = f"SELECT {columns} FROM {table} WHERE {where_clause} LIMIT 1"
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *values)
        return dict(row) if row is not None else None

    def table(self, table: str) -> PostgresQueryBuilder:
        """Start a fluent query against ``table`` (mirrors the Supabase client)."""
        _validate_identifier(table, "table")
        return PostgresQueryBuilder(self, table)

    async def fetch(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        """Run a query and return rows as dicts."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(row) for row in rows]

    async def fetchval(self, sql: str, params: list[Any]) -> Any:
        """Run a query and return the first column of the first row."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(sql, *params)

    async def column_types(self, table: str) -> dict[str, str]:
        """Return ``{column: data_type}`` for ``table`` (cached)."""
        if table not in self._column_type_cache:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = $1",
                    table,
                )
            self._column_type_cache[table] = {r["column_name"]: r["data_type"] for r in rows}
        return self._column_type_cache[table]

    async def primary_key(self, table: str) -> str:
        """Return the comma-joined primary-key column(s) for ``table`` (cached)."""
        if table not in self._primary_key_cache:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT a.attname FROM pg_index i "
                    "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                    "WHERE i.indrelid = $1::regclass AND i.indisprimary "
                    "ORDER BY array_position(i.indkey, a.attnum)",
                    table,
                )
            if not rows:
                raise ValueError(f"No primary key found for upsert on table {table!r}")
            self._primary_key_cache[table] = ", ".join(r["attname"] for r in rows)
        return self._primary_key_cache[table]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
