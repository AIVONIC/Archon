"""
Async query builder for the Postgres backend.

Mirrors the subset of the Supabase/PostgREST fluent API the service layer uses
(``table(...).select(...).eq(...).execute()`` and the insert/update/upsert/delete
variants), compiling each chain to a single parameterized SQL statement. The
Supabase backend implements the same fluent surface by delegating to supabase-py
directly, so call sites are backend-agnostic and behave identically on either.

The builder introspects column types and primary keys (cached per table on the
backend) so it can:
- let ``jsonb`` columns receive ``dict``/``list`` values (the connection's jsonb
  codec serializes them; column-type introspection tells asyncpg the param is
  jsonb),
- cast ISO-8601 string values into ``timestamp``/``timestamptz`` columns (the
  service layer passes ``datetime.now().isoformat()`` strings, which asyncpg
  would otherwise reject), and
- resolve the conflict target for an upsert with no explicit ``on_conflict``.

``execute()`` returns a QueryResult exposing ``.data`` and ``.count``, matching
how call sites read the Supabase response.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .postgres_backend import PostgresBackend

_JSON_TYPES = {"jsonb", "json"}
_TIMESTAMP_TYPES = {"timestamp with time zone", "timestamp without time zone"}


@dataclass
class QueryResult:
    """Result of an executed query, shaped like the Supabase client response."""

    data: Any
    count: int | None = None


class PostgresQueryBuilder:
    """Accumulates a fluent query and compiles it to parameterized SQL."""

    def __init__(self, backend: "PostgresBackend", table: str):
        self._backend = backend
        self._table = table
        self._op = "select"
        self._columns = "*"
        self._count = False
        self._payload: dict | list | None = None
        self._on_conflict: str | None = None
        self._filters: list[tuple[str, str, Any]] = []
        self._or_ilike: tuple[list[str], str] | None = None
        self._unarchived = False
        self._order: list[tuple[str, bool]] = []
        self._limit: int | None = None
        self._single = False

    # Terminal operation selectors ------------------------------------------------

    def select(self, columns: str = "*", count: str | None = None) -> "PostgresQueryBuilder":
        self._op = "select"
        self._columns = columns
        self._count = count is not None
        return self

    def insert(self, payload: dict | list) -> "PostgresQueryBuilder":
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload: dict) -> "PostgresQueryBuilder":
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload: dict | list, on_conflict: str | None = None) -> "PostgresQueryBuilder":
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def delete(self) -> "PostgresQueryBuilder":
        self._op = "delete"
        return self

    # Filters ---------------------------------------------------------------------

    def eq(self, column: str, value: Any) -> "PostgresQueryBuilder":
        self._filters.append(("=", column, value))
        return self

    def neq(self, column: str, value: Any) -> "PostgresQueryBuilder":
        self._filters.append(("<>", column, value))
        return self

    def gte(self, column: str, value: Any) -> "PostgresQueryBuilder":
        self._filters.append((">=", column, value))
        return self

    def lte(self, column: str, value: Any) -> "PostgresQueryBuilder":
        self._filters.append(("<=", column, value))
        return self

    def in_(self, column: str, values: list) -> "PostgresQueryBuilder":
        self._filters.append(("in", column, list(values)))
        return self

    def contains(self, column: str, value: dict) -> "PostgresQueryBuilder":
        self._filters.append(("@>", column, value))
        return self

    def or_ilike(self, columns: list[str], term: str) -> "PostgresQueryBuilder":
        """Match ``term`` case-insensitively against any of ``columns`` (OR)."""
        self._or_ilike = (list(columns), term)
        return self

    def include_unarchived(self) -> "PostgresQueryBuilder":
        """Restrict to rows where ``archived`` is NULL or false."""
        self._unarchived = True
        return self

    # Modifiers -------------------------------------------------------------------

    def order(self, column: str, desc: bool = False) -> "PostgresQueryBuilder":
        self._order.append((column, desc))
        return self

    def limit(self, count: int) -> "PostgresQueryBuilder":
        self._limit = count
        return self

    def single(self) -> "PostgresQueryBuilder":
        self._single = True
        return self

    def maybe_single(self) -> "PostgresQueryBuilder":
        self._single = True
        return self

    # Execution -------------------------------------------------------------------

    async def execute(self) -> QueryResult:
        builders = {
            "select": self._build_select,
            "insert": self._build_insert,
            "update": self._build_update,
            "upsert": self._build_upsert,
            "delete": self._build_delete,
        }
        return await builders[self._op]()

    def _validate(self, name: str) -> None:
        self._backend.validate_identifier(name)

    def _render_where(self, params: list, position: int) -> str:
        """Render a WHERE clause starting at placeholder ``position``.

        ``params`` already holds ``position - 1`` bind values; filter values are
        appended in order.
        """
        fragments: list[str] = []
        for operator, column, value in self._filters:
            self._validate(column)
            if operator == "in":
                fragments.append(f"{column} = ANY(${position})")
                params.append(value)
            elif operator == "@>":
                # Pass the raw dict: the cast makes asyncpg treat the parameter
                # as jsonb, and the connection's jsonb codec serializes it once.
                fragments.append(f"{column} @> ${position}::jsonb")
                params.append(value)
            else:
                fragments.append(f"{column} {operator} ${position}")
                params.append(value)
            position += 1

        if self._or_ilike is not None:
            columns, term = self._or_ilike
            ilikes = []
            for column in columns:
                self._validate(column)
                ilikes.append(f"{column} ILIKE ${position}")
            fragments.append("(" + " OR ".join(ilikes) + ")")
            params.append(f"%{term}%")
            position += 1

        if self._unarchived:
            fragments.append("(archived IS NULL OR archived = false)")

        return (" WHERE " + " AND ".join(fragments)) if fragments else ""

    def _order_limit(self) -> str:
        clause = ""
        if self._order:
            parts = []
            for column, desc in self._order:
                self._validate(column)
                parts.append(f"{column} {'DESC' if desc else 'ASC'}")
            clause += " ORDER BY " + ", ".join(parts)
        if self._limit is not None:
            clause += f" LIMIT {int(self._limit)}"
        return clause

    def _shape(self, rows: list[dict]) -> Any:
        if self._single:
            return rows[0] if rows else None
        return rows

    async def _build_select(self) -> QueryResult:
        self._validate(self._table)
        params: list = []
        where = self._render_where(params, 1)

        if self._count:
            # Consumers of count="exact" read only ``.count``; return the count
            # without materializing rows (these row sets can be very large).
            count = await self._backend.fetchval(f"SELECT count(*) FROM {self._table}{where}", params)
            return QueryResult(data=[], count=count)

        sql = f"SELECT {self._columns} FROM {self._table}{where}{self._order_limit()}"
        rows = await self._backend.fetch(sql, params)
        return QueryResult(data=self._shape(rows), count=None)

    async def _build_insert(self) -> QueryResult:
        rows = await self._insert_rows(self._payload, on_conflict=None)
        return QueryResult(data=rows, count=None)

    async def _build_upsert(self) -> QueryResult:
        conflict = self._on_conflict or await self._backend.primary_key(self._table)
        rows = await self._insert_rows(self._payload, on_conflict=conflict)
        return QueryResult(data=rows, count=None)

    async def _insert_rows(self, payload: dict | list, on_conflict: str | None) -> list[dict]:
        self._validate(self._table)
        records = payload if isinstance(payload, list) else [payload]
        if not records:
            return []

        columns = list(records[0].keys())
        for column in columns:
            self._validate(column)
        col_types = await self._backend.column_types(self._table)

        params: list = []
        value_rows: list[str] = []
        position = 1
        for record in records:
            placeholders = []
            for column in columns:
                value = record.get(column)
                placeholders.append(_placeholder(value, col_types.get(column), position))
                params.append(value)
                position += 1
            value_rows.append("(" + ", ".join(placeholders) + ")")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES {', '.join(value_rows)}"
        if on_conflict is not None:
            for conflict_col in on_conflict.split(","):
                self._validate(conflict_col.strip())
            updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in columns)
            sql += f" ON CONFLICT ({on_conflict}) DO UPDATE SET {updates}"
        sql += " RETURNING *"
        return await self._backend.fetch(sql, params)

    async def _build_update(self) -> QueryResult:
        self._validate(self._table)
        col_types = await self._backend.column_types(self._table)
        params: list = []
        assignments = []
        position = 1
        for column, value in self._payload.items():
            self._validate(column)
            assignments.append(f"{column} = {_placeholder(value, col_types.get(column), position)}")
            params.append(value)
            position += 1

        where = self._render_where(params, position)
        sql = f"UPDATE {self._table} SET {', '.join(assignments)}{where} RETURNING *"
        rows = await self._backend.fetch(sql, params)
        return QueryResult(data=rows, count=None)

    async def _build_delete(self) -> QueryResult:
        self._validate(self._table)
        params: list = []
        where = self._render_where(params, 1)
        sql = f"DELETE FROM {self._table}{where} RETURNING *"
        rows = await self._backend.fetch(sql, params)
        return QueryResult(data=rows, count=None)


def _placeholder(value: Any, col_type: str | None, position: int) -> str:
    """Placeholder text for an insert/update value, adding a cast when needed.

    jsonb columns are left uncast: column-type introspection tells asyncpg the
    parameter is jsonb and the connection codec serializes dict/list values.
    Timestamp columns receiving an ISO string are bound as ``$n::text::timestamptz``:
    the inner ``::text`` forces asyncpg to send the value as text (a bare
    ``::timestamptz`` would make it infer a datetime parameter and reject the
    string), and Postgres then parses the text exactly as the PostgREST path did.
    """
    if col_type in _TIMESTAMP_TYPES and isinstance(value, str):
        return f"${position}::text::timestamptz"
    return f"${position}"
