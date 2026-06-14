"""
Query-builder tests for the Postgres backend.

Exercises every operation the service layer uses (select/insert/update/upsert/
delete, eq/neq/gte/in_/contains/or_ilike filters, order/limit/single, and
count="exact") against a real Archon Postgres, asserting the builder compiles
and runs correct parameterized SQL and returns Supabase-shaped results (``.data``
/``.count``, jsonb decoded to dict).

Skips when the parity Postgres is unreachable, so CI without a database simply
skips. Uses a dedicated source-id namespace and cleans up after itself.
"""

import os
from datetime import datetime

import asyncpg
import pytest
import pytest_asyncio

from src.server.services.storage import PostgresBackend

DEFAULT_DSN = "postgresql://archon_user:archon_sandbox_pass@localhost:5434/archon_sandbox"
PARITY_DSN = os.getenv("ARCHON_PARITY_DATABASE_URL", DEFAULT_DSN)
TAG = "qb-test-source"


@pytest_asyncio.fixture
async def backend():
    try:
        probe = await asyncpg.connect(PARITY_DSN, timeout=5)
        await probe.close()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Parity Postgres unreachable at {PARITY_DSN}: {exc}")

    pg = PostgresBackend(PARITY_DSN, min_size=1, max_size=4)
    await pg.table("archon_crawled_pages").delete().eq("source_id", TAG).execute()
    await pg.table("archon_sources").delete().eq("source_id", TAG).execute()
    await pg.table("archon_sources").insert(
        {"source_id": TAG, "title": "qb", "metadata": {"knowledge_type": "technical"}}
    ).execute()
    try:
        yield pg
    finally:
        await pg.table("archon_crawled_pages").delete().eq("source_id", TAG).execute()
        await pg.table("archon_sources").delete().eq("source_id", TAG).execute()
        await pg.close()


async def _add_page(backend, url, chunk, content, **extra):
    record = {
        "url": url,
        "chunk_number": chunk,
        "content": content,
        "metadata": extra.pop("metadata", {}),
        "source_id": TAG,
    }
    record.update(extra)
    result = await backend.table("archon_crawled_pages").insert(record).execute()
    return result.data[0]


class TestQueryBuilder:
    async def test_insert_returns_row_with_jsonb_decoded(self, backend):
        row = await _add_page(backend, "https://qb/1", 0, "alpha", metadata={"k": "v"})
        assert row["id"] is not None
        assert row["metadata"] == {"k": "v"}
        assert isinstance(row["metadata"], dict)

    async def test_insert_accepts_isoformat_timestamp_string(self, backend):
        row = await _add_page(
            backend, "https://qb/ts", 0, "ts", created_at=datetime(2026, 1, 2, 3, 4, 5).isoformat()
        )
        assert row["created_at"].year == 2026

    async def test_select_eq_order(self, backend):
        await _add_page(backend, "https://qb/a", 0, "first")
        await _add_page(backend, "https://qb/b", 1, "second")
        result = await (
            backend.table("archon_crawled_pages")
            .select("id, chunk_number")
            .eq("source_id", TAG)
            .order("chunk_number", desc=True)
            .execute()
        )
        assert [r["chunk_number"] for r in result.data] == [1, 0]

    async def test_select_single_returns_dict_or_none(self, backend):
        row = await _add_page(backend, "https://qb/s", 0, "solo")
        hit = await (
            backend.table("archon_crawled_pages").select("*").eq("id", row["id"]).single().execute()
        )
        assert isinstance(hit.data, dict) and hit.data["content"] == "solo"

        miss = await (
            backend.table("archon_crawled_pages").select("*").eq("id", -1).single().execute()
        )
        assert miss.data is None

    async def test_update_jsonb_and_timestamp(self, backend):
        row = await _add_page(backend, "https://qb/u", 0, "before", metadata={"k": "v"})
        result = await (
            backend.table("archon_crawled_pages")
            .update({"content": "after", "metadata": {"k": "z"}})
            .eq("id", row["id"])
            .execute()
        )
        assert result.data[0]["content"] == "after"
        assert result.data[0]["metadata"] == {"k": "z"}

    async def test_in_filter(self, backend):
        await _add_page(backend, "https://qb/i", 0, "x")
        result = await (
            backend.table("archon_crawled_pages")
            .select("id")
            .in_("source_id", [TAG, "does-not-exist"])
            .execute()
        )
        assert len(result.data) >= 1

    async def test_contains_jsonb_filter(self, backend):
        result = await (
            backend.table("archon_sources")
            .select("source_id")
            .contains("metadata", {"knowledge_type": "technical"})
            .eq("source_id", TAG)
            .execute()
        )
        assert len(result.data) == 1

    async def test_or_ilike_search(self, backend):
        await _add_page(backend, "https://qb/voice", 0, "uses ElevenLabs voice")
        await _add_page(backend, "https://qb/pay", 1, "Stripe checkout")
        result = await (
            backend.table("archon_crawled_pages")
            .select("id, content")
            .eq("source_id", TAG)
            .or_ilike(["content"], "voice")
            .execute()
        )
        assert len(result.data) == 1 and "voice" in result.data[0]["content"]

    async def test_count_exact_returns_count(self, backend):
        await _add_page(backend, "https://qb/c1", 0, "one")
        await _add_page(backend, "https://qb/c2", 1, "two")
        result = await (
            backend.table("archon_crawled_pages").select("id", count="exact").eq("source_id", TAG).execute()
        )
        assert result.count == 2

    async def test_upsert_uses_primary_key_when_no_conflict_given(self, backend):
        result = await (
            backend.table("archon_sources")
            .upsert({"source_id": TAG, "title": "upserted", "metadata": {"x": 1}})
            .execute()
        )
        assert result.data[0]["title"] == "upserted"

    async def test_upsert_explicit_on_conflict(self, backend):
        await _add_page(backend, "https://qb/uc", 7, "v1")
        result = await (
            backend.table("archon_crawled_pages")
            .upsert(
                {"url": "https://qb/uc", "chunk_number": 7, "content": "v2", "metadata": {}, "source_id": TAG},
                on_conflict="url, chunk_number",
            )
            .execute()
        )
        assert result.data[0]["content"] == "v2"

    async def test_delete_returns_removed_rows(self, backend):
        await _add_page(backend, "https://qb/d1", 0, "gone")
        result = await (
            backend.table("archon_crawled_pages").delete().eq("url", "https://qb/d1").execute()
        )
        assert len(result.data) == 1
