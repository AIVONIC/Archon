-- ⛔ THE COMPANION TO per_source_chunk_uniqueness.sql, AND THE HALF IT MISSED.
--
-- That migration made chunk uniqueness per-source on `archon_crawled_pages` and
-- stopped there. `archon_code_examples` kept the identical global constraint:
--
--     archon_code_examples_url_chunk_number_key  UNIQUE (url, chunk_number)
--
-- Same table shape, same `url` carrying no source identity, same consequence: two
-- sources can never hold code examples for files sharing a basename.
--
-- ⛔ WHY NOBODY NOTICED FOR FIVE DAYS. `add_code_examples_to_supabase` returns
-- early when a document has no code blocks, so it never reached the delete or the
-- insert. The E3 INO README destroyed on 2026-09-11 logged
-- `code_examples_stored=0` and its 5 code-example rows survived while every one of
-- its chunks was deleted. The table looked innocent because the code path was
-- skipped, not because it was safe. A defect that has not fired is still a defect.
--
-- ⛔ THIS MIGRATION IS NOT OPTIONAL AND NOT INDEPENDENT.
--
-- The real destroyer was never the constraint, it was a DELETE scoped by `url`
-- alone (code_storage_service.py, document_storage_service.py x2). Those deletes
-- are now scoped by (source_id, url). That fix REQUIRES this index: previously the
-- url-only delete cleared every same-basename row out of the way, so the insert
-- never met the global constraint. Scope the delete and leave the constraint, and
-- the rows that legitimately survive now collide - a silent data loss becomes a
-- hard duplicate-key failure on every second document sharing a basename.
--
-- Apply order (archon-server runs uvicorn --reload against bind-mounted src, so a
-- code edit goes live on save - stage the SQL BEFORE saving the code, or ingestion
-- is briefly broken in the window between them):
--     1. create the new index (both coexist; the old one still guards)
--     2. drop the old constraint
--     3. save the code changes
--
-- Note `archon_code_examples` is inserted with plain insert(), not upsert(), so no
-- ON CONFLICT names either constraint and step 2 breaks no call site. That is the
-- one way this is simpler than its sibling.
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_ce_source_url_chunk
    ON archon_code_examples (source_id, url, chunk_number);

ALTER TABLE archon_code_examples
    DROP CONSTRAINT IF EXISTS archon_code_examples_url_chunk_number_key;
