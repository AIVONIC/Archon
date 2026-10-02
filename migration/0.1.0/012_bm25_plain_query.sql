-- =====================================================
-- 012: BM25 gets the visitor's WORDS, never query syntax
-- =====================================================
-- Found 2026-10-02 (aivonic-12) from two PERSISTENT archon-server alerts, measured
-- directly against psql_bm25s 0.4.14. `psql_bm25s_query(index, text, k)` parses its
-- text as a RAW QUERY, and three things follow for natural-language input:
--
-- 1. It RAISES on an unbalanced `"` or `(`/`)` (so `:)` and `1) price`), a trailing
--    `AND`/`OR`/`NOT`, `x +`, `x -`, or a query of exclusions only. The error aborts the
--    WHOLE hybrid statement, so the vector half is lost too and the caller gets 0 chunks.
--    Real ContentClaw grounding queries hit this, not only injection probes.
-- 2. A word with punctuation attached is one literal term that matches nothing:
--    `pricing?`, `GDPR?`, `pris?`, `plan.`, `e-commerce` all score 0 where `pricing`,
--    `gdpr`, `pris` score 3-5. The last word of most questions was silently dropped.
-- 3. When nothing matches it still returns k rows at score 0, and the callers below
--    rank them 1..k into the RRF fusion, so arbitrary chunks were fused in at the same
--    weight as the best vector hit.
--
-- And `psql_bm25s_query(index, NULL, k)` SEGFAULTS the backend (signal 11; the whole
-- postmaster reinitialises). Measured once, by accident, at 2026-10-02 21:27:07 UTC.
--
-- So every BM25 call goes through ONE function, archon_bm25_hits(), which:
--   * turns the text into plain terms with the extension's OWN tokenizer (lowercase
--     only; stemming, stopwords and diacritics stay with the index exactly as before),
--   * never passes NULL or an empty query (it returns no hits instead),
--   * returns only hits that actually scored (score > 0).
-- The three hybrid functions are otherwise byte-identical to what ran in production on
-- 2026-10-02 (pg_get_functiondef), including _multi_v2, which until now existed only in
-- the database and not in this repo.
--
-- Rollback: restore the three definitions with `psql_bm25s_query(` in place of
-- `archon_bm25_hits(` (that is the whole diff) and DROP FUNCTION archon_bm25_hits.
-- =====================================================

CREATE OR REPLACE FUNCTION public.archon_bm25_hits(index_name regclass, query_text text, k integer)
 RETURNS SETOF psql_bm25s_result_hit
 LANGUAGE plpgsql
 STABLE
AS $function$
DECLARE
    plain_query TEXT;
BEGIN
    plain_query := array_to_string(
        psql_bm25s_tokenize_text(COALESCE(query_text, ''), true, NULL, false, false), ' ');
    IF plain_query IS NULL OR plain_query = '' THEN
        RETURN;
    END IF;
    RETURN QUERY
        SELECT * FROM psql_bm25s_query(index_name, plain_query, k) h WHERE h.score > 0;
END;
$function$;

CREATE OR REPLACE FUNCTION public.hybrid_search_archon_crawled_pages_multi(query_embedding vector, embedding_dimension integer, query_text text, match_count integer DEFAULT 10, filter jsonb DEFAULT '{}'::jsonb, source_filter text DEFAULT NULL::text)
 RETURNS TABLE(id bigint, url character varying, chunk_number integer, content text, metadata jsonb, source_id text, similarity double precision, match_type text)
 LANGUAGE plpgsql
AS $function$
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
        FROM archon_bm25_hits('idx_archon_crawled_pages_bm25', $6, $2) h
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
$function$;

CREATE OR REPLACE FUNCTION public.hybrid_search_archon_crawled_pages_multi_v2(query_embedding vector, embedding_dimension integer, query_text text, match_count integer DEFAULT 10, filter jsonb DEFAULT '{}'::jsonb, source_filter text DEFAULT NULL::text)
 RETURNS TABLE(id bigint, url character varying, chunk_number integer, content text, metadata jsonb, source_id text, similarity double precision, vector_similarity double precision, match_type text)
 LANGUAGE plpgsql
AS $function$
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
        FROM archon_bm25_hits('idx_archon_crawled_pages_bm25', $6, $2) h
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
            COALESCE(v.vec_sim, 0)::float8 AS vec_sim,
            CASE
                WHEN v.id IS NOT NULL AND b.id IS NOT NULL THEN 'hybrid'
                WHEN v.id IS NOT NULL THEN 'vector'
                ELSE 'bm25'
            END AS match_type
        FROM vector_ranked v
        FULL OUTER JOIN bm25_ranked b ON v.id = b.id
    )
    SELECT id, url, chunk_number, content, metadata, source_id, rrf_score AS similarity, vec_sim AS vector_similarity, match_type
    FROM fused
    ORDER BY rrf_score DESC
    LIMIT $3
    $SQL$,
    embedding_column, embedding_column, embedding_column, embedding_column);

    RETURN QUERY EXECUTE sql_query
        USING query_embedding, candidate_k, match_count, filter, source_filter, query_text;
END;
$function$;

CREATE OR REPLACE FUNCTION public.hybrid_search_archon_code_examples_multi(query_embedding vector, embedding_dimension integer, query_text text, match_count integer DEFAULT 10, filter jsonb DEFAULT '{}'::jsonb, source_filter text DEFAULT NULL::text)
 RETURNS TABLE(id bigint, url character varying, chunk_number integer, content text, summary text, metadata jsonb, source_id text, similarity double precision, match_type text)
 LANGUAGE plpgsql
AS $function$
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
        FROM archon_bm25_hits('idx_archon_code_examples_bm25', $6, $2) h
        JOIN archon_code_examples ce ON ce.ctid = h.ctid
        WHERE ce.metadata @> $4
          AND ($5 IS NULL OR ce.source_id = $5)
    ),
    bm25_summary AS (
        SELECT
            ce.id, ce.url, ce.chunk_number, ce.content, ce.summary,
            ce.metadata, ce.source_id,
            row_number() OVER (ORDER BY h.score DESC) AS bm25_rank
        FROM archon_bm25_hits('idx_archon_code_examples_summary_bm25', $6, $2) h
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
$function$;

INSERT INTO archon_migrations (version, migration_name)
VALUES ('0.1.0', '012_bm25_plain_query')
ON CONFLICT (version, migration_name) DO NOTHING;
