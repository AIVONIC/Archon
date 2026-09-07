-- ⛔ A CHUNK INHERITS ITS SOURCE'S TAGS. Tag-scoped RAG filters on the CHUNK's
-- metadata (hybrid_search_archon_crawled_pages_multi: WHERE cp.metadata @> filter),
-- but tag propagation in the ingest path is conditional on the CALLER passing tags
-- (storage_services.py: `if tags: meta["tags"] = tags`). Any caller that omits them
-- writes untagged chunks under a correctly-tagged source, and the result is a KB
-- that looks healthy and retrieves nothing for that tag.
--
-- Measured 2026-09-07: 10,278 of 10,685 chunks in that state, tag-scoped retrieval
-- returning 0 for all 12 agent knowledge bases while unscoped retrieval worked.
--
-- This is at the DATABASE rather than in the writer on purpose: there are several
-- ingest paths and the invariant should not depend on each remembering. A caller
-- that DOES pass tags is left alone.
CREATE OR REPLACE FUNCTION archon_chunk_inherit_source_tags()
RETURNS trigger AS $$
DECLARE
    src_tags jsonb;
BEGIN
    IF COALESCE(NEW.metadata, '{}'::jsonb) ? 'tags' THEN
        RETURN NEW;                      -- caller supplied tags; never override
    END IF;
    SELECT s.metadata::jsonb -> 'tags' INTO src_tags
      FROM archon_sources s
     WHERE s.source_id = NEW.source_id
       AND s.metadata::jsonb ? 'tags';
    IF src_tags IS NOT NULL THEN         -- no source row yet is NOT an error
        NEW.metadata := COALESCE(NEW.metadata, '{}'::jsonb)
                        || jsonb_build_object('tags', src_tags);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_archon_chunk_inherit_source_tags ON archon_crawled_pages;
CREATE TRIGGER trg_archon_chunk_inherit_source_tags
    BEFORE INSERT OR UPDATE ON archon_crawled_pages
    FOR EACH ROW EXECUTE FUNCTION archon_chunk_inherit_source_tags();
