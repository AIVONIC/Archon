"""
Retrieval parity tests for the asyncpg database backend.

These tests seed controlled fixtures at the production embedding dimension (384,
via the local bge-small-en-v1.5 server in production) and drive the refactored
search layer through ``PostgresBackend`` against a real Archon Postgres
(pgvector + psql_bm25s). They prove the asyncpg path reproduces the retrieval
contract the Supabase client provided: correct ordering, source-tag isolation,
dict/float result typing, and working BM25 hybrid fusion at 384.

Two real production read paths are represented as fixtures: an agent KB tag
(``aivonic-spark``) and the Workspace KB tag (``workspace-project-<id>``).

Embeddings are deterministic (hash-seeded), not service-generated, so retrieval
ordering is reproducible and the assertions test retrieval mechanics rather than
embedding quality or a live embedding server.

Backends targeted:
- ``PostgresBackend`` against ``ARCHON_PARITY_DATABASE_URL`` (defaults to the
  local sandbox on port 5434). The whole module skips if that DB is unreachable,
  so CI without a Postgres simply skips rather than fails.
- ``SupabaseBackend`` cross-backend equality is gated behind
  ``ARCHON_PARITY_SUPABASE_URL`` / ``ARCHON_PARITY_SUPABASE_KEY`` and skips when
  unset (no live Supabase required for the default run).
"""

import hashlib
import math
import os
import random

import asyncpg
import pytest
import pytest_asyncio

from src.server.services.search.base_search_strategy import BaseSearchStrategy
from src.server.services.search.hybrid_search_strategy import HybridSearchStrategy
from src.server.services.storage import PostgresBackend

DEFAULT_DSN = "postgresql://archon_user:archon_sandbox_pass@localhost:5434/archon_sandbox"
PARITY_DSN = os.getenv("ARCHON_PARITY_DATABASE_URL", DEFAULT_DSN)

# Production read-path tags: one agent KB, one Workspace KB project.
SPARK_TAG = "parity-aivonic-spark"
WORKSPACE_TAG = "parity-workspace-project-test"

# (source_tag, key, content)
FIXTURES = [
    (SPARK_TAG, "spark-1", "SPARK agent pricing and Stripe checkout flow for Aivonic"),
    (SPARK_TAG, "spark-2", "SPARK voice input uses ElevenLabs multilingual text to speech"),
    (WORKSPACE_TAG, "ws-1", "Workspace KB project chat retrieval augmented generation"),
    (WORKSPACE_TAG, "ws-2", "Workspace sync writes crawled pages to Archon nightly"),
]

EMBEDDING_DIM = 384


def deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a stable unit vector for ``text``.

    The same text always yields the same vector (cosine similarity 1.0 with
    itself), and distinct texts are near-orthogonal, so retrieval ordering is
    predictable and assertable.
    """
    rng = random.Random(hashlib.sha256(text.encode()).hexdigest())
    vector = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(component * component for component in vector)) or 1.0
    return [component / norm for component in vector]


async def _seed(backend: PostgresBackend) -> None:
    pool = await backend._get_pool()
    async with pool.acquire() as conn:
        await _purge(conn)
        for tag in (SPARK_TAG, WORKSPACE_TAG):
            await conn.execute(
                "INSERT INTO archon_sources (source_id, title, created_at, updated_at) "
                "VALUES ($1, $2, now(), now())",
                tag,
                f"parity fixture {tag}",
            )
        for tag, key, content in FIXTURES:
            await conn.execute(
                "INSERT INTO archon_crawled_pages "
                "(url, chunk_number, content, metadata, source_id, embedding_384, "
                "embedding_dimension, created_at) "
                "VALUES ($1, 0, $2, $3, $4, $5, $6, now())",
                f"https://parity.test/{key}",
                content,
                {"tag": tag, "key": key},
                tag,
                deterministic_embedding(content),
                EMBEDDING_DIM,
            )


async def _purge(conn: asyncpg.Connection) -> None:
    tags = [SPARK_TAG, WORKSPACE_TAG]
    await conn.execute("DELETE FROM archon_crawled_pages WHERE source_id = ANY($1::text[])", tags)
    await conn.execute("DELETE FROM archon_sources WHERE source_id = ANY($1::text[])", tags)


@pytest_asyncio.fixture
async def postgres_backend():
    """Seeded PostgresBackend against the parity DB; skips if it is unreachable."""
    try:
        probe = await asyncpg.connect(PARITY_DSN, timeout=5)
        await probe.close()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Parity Postgres unreachable at {PARITY_DSN}: {exc}")

    backend = PostgresBackend(PARITY_DSN, min_size=1, max_size=4)
    await _seed(backend)
    try:
        yield backend
    finally:
        pool = await backend._get_pool()
        async with pool.acquire() as conn:
            await _purge(conn)
        await backend.close()


@pytest_asyncio.fixture
async def strategies(postgres_backend):
    base = BaseSearchStrategy(postgres_backend)
    hybrid = HybridSearchStrategy(postgres_backend, base)
    return base, hybrid


class TestPostgresBackendContract:
    """End-to-end retrieval contract for the asyncpg backend at 384 dimensions."""

    async def test_dense_exact_match_ranks_first(self, strategies):
        base, _ = strategies
        query = deterministic_embedding(FIXTURES[0][2])  # the spark-1 content vector

        results = await base.vector_search(query, match_count=5)

        assert results, "dense search returned no rows"
        top = results[0]
        assert top["metadata"]["key"] == "spark-1"
        assert top["similarity"] == pytest.approx(1.0, abs=1e-6)

    async def test_result_shape_matches_supabase(self, strategies):
        """jsonb decodes to dict and numeric columns to float, as the client returned."""
        base, _ = strategies
        query = deterministic_embedding(FIXTURES[0][2])

        top = (await base.vector_search(query, match_count=5))[0]

        assert isinstance(top["metadata"], dict)
        assert isinstance(top["similarity"], float)
        assert top["id"] is not None
        assert top["source_id"] == SPARK_TAG

    async def test_source_filter_isolates_tags(self, strategies):
        """Agent-KB and Workspace-KB read paths must not bleed into each other."""
        base, _ = strategies
        query = deterministic_embedding(FIXTURES[0][2])

        spark = await base.vector_search(query, match_count=10, filter_metadata={"source": SPARK_TAG})
        workspace = await base.vector_search(
            query, match_count=10, filter_metadata={"source": WORKSPACE_TAG}
        )

        assert {row["source_id"] for row in spark} == {SPARK_TAG}
        assert {row["source_id"] for row in workspace} == {WORKSPACE_TAG}

    async def test_hybrid_384_routes_to_multi_and_bm25_matches(self, strategies):
        """Hybrid fusion works at 384 and BM25 surfaces the keyword row."""
        _, hybrid = strategies
        query_text = "Stripe checkout"

        results = await hybrid.search_documents_hybrid(
            query=query_text,
            query_embedding=deterministic_embedding(query_text),
            match_count=5,
            filter_metadata={"source": SPARK_TAG},
        )

        assert results, "hybrid search returned no rows at 384"
        assert any(row["metadata"].get("key") == "spark-1" for row in results)
        assert all(isinstance(row["metadata"], dict) for row in results)

    async def test_select_one_roundtrip(self, postgres_backend):
        """select_one returns a single row dict, or None when nothing matches."""
        found = await postgres_backend.select_one(
            "archon_sources", "source_id, title", {"source_id": SPARK_TAG}
        )
        assert found is not None
        assert found["source_id"] == SPARK_TAG

        missing = await postgres_backend.select_one(
            "archon_sources", "source_id", {"source_id": "no-such-source"}
        )
        assert missing is None


class TestHybridRpcRouting:
    """Pure routing logic; runs without a database."""

    def test_384_routes_to_multi_with_dimension(self):
        name, params = HybridSearchStrategy._build_hybrid_rpc(
            "hybrid_search_archon_crawled_pages", [0.0] * 384, "q", 5, {}, None
        )
        assert name == "hybrid_search_archon_crawled_pages_multi"
        assert params["embedding_dimension"] == 384

    def test_1536_routes_to_fixed_function(self):
        name, params = HybridSearchStrategy._build_hybrid_rpc(
            "hybrid_search_archon_crawled_pages", [0.0] * 1536, "q", 5, {}, None
        )
        assert name == "hybrid_search_archon_crawled_pages"
        assert "embedding_dimension" not in params


@pytest.mark.skipif(
    not (os.getenv("ARCHON_PARITY_SUPABASE_URL") and os.getenv("ARCHON_PARITY_SUPABASE_KEY")),
    reason="Cross-backend equality needs ARCHON_PARITY_SUPABASE_URL/_KEY (live Supabase).",
)
class TestSupabaseVsPostgresParity:
    """Dense ranking equivalence between the Supabase and Postgres backends.

    Runs only when a Supabase project carrying the same fixtures is configured.
    The dense path is deterministic on identical data, so the two backends must
    return the same rows in the same order. (BM25 hybrid is intentionally a
    Postgres-only capability and is not asserted equal here.)
    """

    async def test_dense_ranking_equivalence(self, postgres_backend):
        from supabase import create_client

        from src.server.services.storage import SupabaseBackend

        supabase = SupabaseBackend(
            create_client(
                os.environ["ARCHON_PARITY_SUPABASE_URL"],
                os.environ["ARCHON_PARITY_SUPABASE_KEY"],
            )
        )
        query = deterministic_embedding(FIXTURES[0][2])

        pg_base = BaseSearchStrategy(postgres_backend)
        sb_base = BaseSearchStrategy(supabase)

        pg_rows = await pg_base.vector_search(query, match_count=5, filter_metadata={"source": SPARK_TAG})
        sb_rows = await sb_base.vector_search(query, match_count=5, filter_metadata={"source": SPARK_TAG})

        assert [r["id"] for r in pg_rows] == [r["id"] for r in sb_rows]
        for pg_row, sb_row in zip(pg_rows, sb_rows, strict=True):
            assert pg_row["similarity"] == pytest.approx(sb_row["similarity"], abs=1e-4)
