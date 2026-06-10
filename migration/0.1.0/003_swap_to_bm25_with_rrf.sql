-- =====================================================
-- Swap tsvector hybrid search for psql_bm25s + Reciprocal Rank Fusion
-- =====================================================
-- Why this exists:
-- The 002_add_hybrid_search_tsvector.sql migration produced a hybrid
-- search that combined pgvector cosine similarity with PostgreSQL's
-- native ts_rank_cd over a generated tsvector column. Two structural
-- problems with that approach:
--
-- 1. ts_rank_cd is a much weaker sparse ranker than BM25 (no IDF, no
--    length normalisation, no proximity at default cutoff). On
--    Archon-style content (multi-paragraph chunks of scraped
--    documentation) it routinely surfaces longer chunks that match
--    every term once over shorter chunks that match the most
--    relevant term repeatedly.
--
-- 2. The fusion in those functions is COALESCE(vector_sim, ts_rank, 0)
--    which puts the two scores on the same axis even though their
--    distributions are unrelated. Whichever search returns higher
--    absolute numbers wins the ordering regardless of relevance.
--
-- This migration replaces both with:
--   * psql_bm25s extension indexes on archon_crawled_pages.content and
--     archon_code_examples.content (with summary folded in).
--   * Reciprocal Rank Fusion at score-combine time:
--         rrf(d) = sum over retrievers r of 1 / (k_rrf + rank_r(d))
--     with k_rrf = 60 (standard).
--
-- Rollback: see "ROLLBACK" section at bottom. Keeps the tsvector
-- generated columns intact so the old functions can be restored by
-- re-running 002_add_hybrid_search_tsvector.sql.
-- =====================================================

-- =====================================================
-- SECTION 1: ENABLE EXTENSION + CREATE BM25 INDEXES
-- =====================================================

CREATE EXTENSION IF NOT EXISTS psql_bm25s;

-- BM25 index over content body for crawled pages.
DROP INDEX IF EXISTS idx_archon_crawled_pages_bm25;
CREATE INDEX idx_archon_crawled_pages_bm25
    ON archon_crawled_pages USING psql_bm25s (content);

-- BM25 index over content + summary for code examples. Multicolumn
-- text fusion is supported natively by psql_bm25s on a single index.
DROP INDEX IF EXISTS idx_archon_code_examples_bm25;
CREATE INDEX idx_archon_code_examples_bm25
    ON archon_code_examples USING psql_bm25s (content);

DROP INDEX IF EXISTS idx_archon_code_examples_summary_bm25;
CREATE INDEX idx_archon_code_examples_summary_bm25
    ON archon_code_examples USING psql_bm25s (summary)
    WHERE summary IS NOT NULL;

-- =====================================================
-- SECTION 2: HYBRID SEARCH FUNCTIONS WITH RRF FUSION
-- =====================================================
--
-- Both functions follow the same template:
--   * Run a vector top-k (over the dimension-appropriate embedding
--     column) and assign rank 1..k by similarity.
--   * Run a BM25 top-k via psql_bm25s_query and assign rank 1..k by
--     BM25 score.
--   * FULL OUTER JOIN the two rank sets on document id and combine
--     with rrf_score = 1/(60 + vec_rank) + 1/(60 + bm25_rank) where
--     a missing rank contributes 0.
--   * Order by rrf_score DESC and return the top match_count rows.
--
-- The candidate pool from each retriever is GREATEST(match_count * 5,
-- 50) so that documents that rank well in one retriever but outside
-- the top-N of the other still surface in the fused result. This is
-- the standard "candidate_k" trick from the RRF literature.

CREATE OR REPLACE FUNCTION hybrid_search_archon_crawled_pages_multi(
    query_embedding VECTOR,
    embedding_dimension INTEGER,
    query_text TEXT,
    match_count INT DEFAULT 10,
    filter JSONB DEFAULT '{}'::jsonb,
    source_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    url VARCHAR,
    chunk_number INTEGER,
    content TEXT,
    metadata JSONB,
    source_id TEXT,
    similarity FLOAT,
    match_type TEXT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
DECLARE
    candidate_k INT;
    sql_query TEXT;
    embedding_column TEXT;
BEGIN
    CASE embedding_dimension
        WHEN 384 THEN embedding_column := 'embedding_384';
        WHEN 768 THEN embedding_column := 'embedding_768';
        WHEN 1024 THEN embedding_column := 'embedding_1024';
        WHEN 1536 THEN embedding_column := 'embedding_1536';
        WHEN 3072 THEN embedding_column := 'embedding_3072';
        ELSE RAISE EXCEPTION 'Unsupported embedding dimension: %', embedding_dimension;
    END CASE;

    candidate_k := GREATEST(match_count * 5, 50);

    sql_query := format($SQL$
    WITH vector_ranked AS (
        SELECT
            cp.id,
            cp.url,
            cp.chunk_number,
            cp.content,
            cp.metadata,
            cp.source_id,
            1 - (cp.%I <=> $1) AS vec_sim,
            row_number() OVER (ORDER BY cp.%I <=> $1 ASC) AS vec_rank
        FROM archon_crawled_pages cp
        WHERE cp.metadata @> $4
          AND ($5 IS NULL OR cp.source_id = $5)
          AND cp.%I IS NOT NULL
        ORDER BY cp.%I <=> $1 ASC
        LIMIT $2
    ),
    bm25_ranked AS (
        SELECT
            cp.id,
            cp.url,
            cp.chunk_number,
            cp.content,
            cp.metadata,
            cp.source_id,
            h.score AS bm25_score,
            row_number() OVER (ORDER BY h.score DESC) AS bm25_rank
        FROM psql_bm25s_query('idx_archon_crawled_pages_bm25', $6, $2) h
        JOIN archon_crawled_pages cp ON cp.ctid = h.ctid
        WHERE cp.metadata @> $4
          AND ($5 IS NULL OR cp.source_id = $5)
    ),
    fused AS (
        SELECT
            COALESCE(v.id, b.id) AS id,
            COALESCE(v.url, b.url) AS url,
            COALESCE(v.chunk_number, b.chunk_number) AS chunk_number,
            COALESCE(v.content, b.content) AS content,
            COALESCE(v.metadata, b.metadata) AS metadata,
            COALESCE(v.source_id, b.source_id) AS source_id,
            (
                COALESCE(1.0 / (60 + v.vec_rank), 0)
                + COALESCE(1.0 / (60 + b.bm25_rank), 0)
            )::float8 AS rrf_score,
            CASE
                WHEN v.id IS NOT NULL AND b.id IS NOT NULL THEN 'hybrid'
                WHEN v.id IS NOT NULL THEN 'vector'
                ELSE 'bm25'
            END AS match_type
        FROM vector_ranked v
        FULL OUTER JOIN bm25_ranked b ON v.id = b.id
    )
    SELECT id, url, chunk_number, content, metadata, source_id, rrf_score AS similarity, match_type
    FROM fused
    ORDER BY rrf_score DESC
    LIMIT $3
    $SQL$,
    embedding_column, embedding_column, embedding_column, embedding_column);

    RETURN QUERY EXECUTE sql_query
        USING query_embedding, candidate_k, match_count, filter, source_filter, query_text;
END;
$$;

-- Legacy entry point (defaults to 1536D)
CREATE OR REPLACE FUNCTION hybrid_search_archon_crawled_pages(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    filter JSONB DEFAULT '{}'::jsonb,
    source_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    url VARCHAR,
    chunk_number INTEGER,
    content TEXT,
    metadata JSONB,
    source_id TEXT,
    similarity FLOAT,
    match_type TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY SELECT * FROM hybrid_search_archon_crawled_pages_multi(
        query_embedding, 1536, query_text, match_count, filter, source_filter
    );
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search_archon_code_examples_multi(
    query_embedding VECTOR,
    embedding_dimension INTEGER,
    query_text TEXT,
    match_count INT DEFAULT 10,
    filter JSONB DEFAULT '{}'::jsonb,
    source_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    url VARCHAR,
    chunk_number INTEGER,
    content TEXT,
    summary TEXT,
    metadata JSONB,
    source_id TEXT,
    similarity FLOAT,
    match_type TEXT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
DECLARE
    candidate_k INT;
    sql_query TEXT;
    embedding_column TEXT;
BEGIN
    CASE embedding_dimension
        WHEN 384 THEN embedding_column := 'embedding_384';
        WHEN 768 THEN embedding_column := 'embedding_768';
        WHEN 1024 THEN embedding_column := 'embedding_1024';
        WHEN 1536 THEN embedding_column := 'embedding_1536';
        WHEN 3072 THEN embedding_column := 'embedding_3072';
        ELSE RAISE EXCEPTION 'Unsupported embedding dimension: %', embedding_dimension;
    END CASE;

    candidate_k := GREATEST(match_count * 5, 50);

    -- For code examples we fuse three retrievers: pgvector, BM25 on
    -- content, and BM25 on summary. The summary index is partial
    -- (WHERE summary IS NOT NULL) so its query is conditional.

    sql_query := format($SQL$
    WITH vector_ranked AS (
        SELECT
            ce.id, ce.url, ce.chunk_number, ce.content, ce.summary,
            ce.metadata, ce.source_id,
            row_number() OVER (ORDER BY ce.%I <=> $1 ASC) AS vec_rank
        FROM archon_code_examples ce
        WHERE ce.metadata @> $4
          AND ($5 IS NULL OR ce.source_id = $5)
          AND ce.%I IS NOT NULL
        ORDER BY ce.%I <=> $1 ASC
        LIMIT $2
    ),
    bm25_content AS (
        SELECT
            ce.id, ce.url, ce.chunk_number, ce.content, ce.summary,
            ce.metadata, ce.source_id,
            row_number() OVER (ORDER BY h.score DESC) AS bm25_rank
        FROM psql_bm25s_query('idx_archon_code_examples_bm25', $6, $2) h
        JOIN archon_code_examples ce ON ce.ctid = h.ctid
        WHERE ce.metadata @> $4
          AND ($5 IS NULL OR ce.source_id = $5)
    ),
    bm25_summary AS (
        SELECT
            ce.id, ce.url, ce.chunk_number, ce.content, ce.summary,
            ce.metadata, ce.source_id,
            row_number() OVER (ORDER BY h.score DESC) AS bm25_rank
        FROM psql_bm25s_query('idx_archon_code_examples_summary_bm25', $6, $2) h
        JOIN archon_code_examples ce ON ce.ctid = h.ctid
        WHERE ce.metadata @> $4
          AND ($5 IS NULL OR ce.source_id = $5)
          AND ce.summary IS NOT NULL
    ),
    fused AS (
        SELECT
            COALESCE(v.id, bc.id, bs.id) AS id,
            COALESCE(v.url, bc.url, bs.url) AS url,
            COALESCE(v.chunk_number, bc.chunk_number, bs.chunk_number) AS chunk_number,
            COALESCE(v.content, bc.content, bs.content) AS content,
            COALESCE(v.summary, bc.summary, bs.summary) AS summary,
            COALESCE(v.metadata, bc.metadata, bs.metadata) AS metadata,
            COALESCE(v.source_id, bc.source_id, bs.source_id) AS source_id,
            (
                COALESCE(1.0 / (60 + v.vec_rank), 0)
                + COALESCE(1.0 / (60 + bc.bm25_rank), 0)
                + COALESCE(1.0 / (60 + bs.bm25_rank), 0)
            )::float8 AS rrf_score,
            CASE
                WHEN v.id IS NOT NULL AND (bc.id IS NOT NULL OR bs.id IS NOT NULL) THEN 'hybrid'
                WHEN v.id IS NOT NULL THEN 'vector'
                ELSE 'bm25'
            END AS match_type
        FROM vector_ranked v
        FULL OUTER JOIN bm25_content bc ON v.id = bc.id
        FULL OUTER JOIN bm25_summary bs ON COALESCE(v.id, bc.id) = bs.id
    )
    SELECT id, url, chunk_number, content, summary, metadata, source_id,
           rrf_score AS similarity, match_type
    FROM fused
    ORDER BY rrf_score DESC
    LIMIT $3
    $SQL$,
    embedding_column, embedding_column, embedding_column);

    RETURN QUERY EXECUTE sql_query
        USING query_embedding, candidate_k, match_count, filter, source_filter, query_text;
END;
$$;

-- Legacy entry point for code examples
CREATE OR REPLACE FUNCTION hybrid_search_archon_code_examples(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    filter JSONB DEFAULT '{}'::jsonb,
    source_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    url VARCHAR,
    chunk_number INTEGER,
    content TEXT,
    summary TEXT,
    metadata JSONB,
    source_id TEXT,
    similarity FLOAT,
    match_type TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY SELECT * FROM hybrid_search_archon_code_examples_multi(
        query_embedding, 1536, query_text, match_count, filter, source_filter
    );
END;
$$;

COMMENT ON FUNCTION hybrid_search_archon_crawled_pages_multi IS
    'Hybrid search over crawled pages using pgvector dense + psql_bm25s sparse, fused with Reciprocal Rank Fusion (k=60). Replaces the ts_rank_cd-based merge from 002.';
COMMENT ON FUNCTION hybrid_search_archon_code_examples_multi IS
    'Hybrid search over code examples using pgvector dense + psql_bm25s sparse on both content and summary, fused with three-way Reciprocal Rank Fusion (k=60). Replaces the ts_rank_cd-based merge from 002.';

-- =====================================================
-- SECTION 3: OPTIONAL CLEANUP
-- =====================================================
--
-- The tsvector generated columns (content_search_vector) and their
-- GIN indexes are NOT dropped by this migration. They cost a small
-- amount of disk and write overhead but make rollback trivial. Drop
-- them only after the new BM25 path has been validated in production:
--
--   ALTER TABLE archon_crawled_pages DROP COLUMN content_search_vector;
--   ALTER TABLE archon_code_examples DROP COLUMN content_search_vector;
--
-- And then drop the corresponding GIN indexes (the ALTER COLUMN will
-- cascade them, but for clarity):
--
--   DROP INDEX IF EXISTS idx_archon_crawled_pages_content_search;
--   DROP INDEX IF EXISTS idx_archon_code_examples_content_search;

-- =====================================================
-- ROLLBACK
-- =====================================================
--
-- To revert this migration entirely:
--   DROP FUNCTION hybrid_search_archon_crawled_pages_multi (and the
--     legacy entry point), and re-run the function definitions from
--     002_add_hybrid_search_tsvector.sql. Same for code examples.
--   DROP INDEX idx_archon_crawled_pages_bm25,
--              idx_archon_code_examples_bm25,
--              idx_archon_code_examples_summary_bm25;
--
-- The tsvector columns will still exist (Section 3 cleanup was opt-in)
-- so the legacy functions will work as soon as they are redefined.
