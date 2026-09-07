-- ⛔ CHUNK UNIQUENESS IS PER SOURCE, NOT GLOBAL PER URL.
--
-- The old constraint was UNIQUE (url, chunk_number). `url` is the document's own
-- path (`file://LEDGER.md`) and carries NO source identity, so uniqueness on it
-- was GLOBAL: two sources could never hold a file with the same basename,
-- whatever project or tag they belonged to. In a multi-tenant knowledge base that
-- means the first project to hold a README.md prevents every other project from
-- holding one.
--
-- It did not fail loudly. Combined with an upsert on the same key, the second
-- ingest TOOK the first source's rows and kept only the tail under its own id.
-- Measured 2026-09-07 on a live customer workspace:
--
--     url file://LEDGER.md   45 rows   2 source_ids
--       d32f0737   30 chunks   chunk_number  0..29
--       2927d5b6   15 chunks   chunk_number 30..44
--
-- One document, split across two sources, in disjoint ranges. Any source-scoped
-- read got half a document and nothing errored. After this change the next
-- re-ingest wrote it whole: 105 rows, one source, chunk_number 0..104.
--
-- ⛔ THE CODE HALF IS NOT OPTIONAL. document_storage_service.py passes
-- `on_conflict="url,chunk_number"` EXPLICITLY at two call sites. Dropping this
-- constraint without changing them makes every upsert fail with "no unique or
-- exclusion constraint matching the ON CONFLICT specification" - all ingestion
-- stops. Apply in this order, and note archon-server runs `uvicorn --reload`
-- against the bind-mounted src, so the code edit goes live on save:
--     1. create the new index (both coexist; the old one still guards)
--     2. change both call sites to source_id,url,chunk_number
--     3. drop the old constraint
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_cp_source_url_chunk
    ON archon_crawled_pages (source_id, url, chunk_number);

ALTER TABLE archon_crawled_pages
    DROP CONSTRAINT IF EXISTS archon_crawled_pages_url_chunk_number_key;
