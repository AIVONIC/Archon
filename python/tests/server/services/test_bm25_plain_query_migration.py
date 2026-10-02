"""Migration 012: every BM25 call goes through archon_bm25_hits, never the raw parser.

psql_bm25s_query() parses its text as query SYNTAX: an unbalanced quote or paren, or a
trailing AND/OR/NOT, raises and aborts the whole hybrid statement (0 chunks); a word with
punctuation attached ("pricing?") matches nothing; misses come back at score 0 and were
fused into RRF; and a NULL text segfaults the backend. Measured 2026-10-02 against
psql_bm25s 0.4.14 (see the migration header). These checks keep a later edit from putting
the raw call back into a hybrid function.
"""
import re
from pathlib import Path

MIGRATION = Path(__file__).resolve().parents[4] / "migration" / "0.1.0" / "012_bm25_plain_query.sql"
HYBRID = (
    "hybrid_search_archon_crawled_pages_multi",
    "hybrid_search_archon_crawled_pages_multi_v2",
    "hybrid_search_archon_code_examples_multi",
)


def _sql_without_comments() -> str:
    return "\n".join(line for line in MIGRATION.read_text().splitlines() if not line.lstrip().startswith("--"))


def _functions(sql: str) -> dict[str, str]:
    parts = re.split(r"CREATE OR REPLACE FUNCTION public\.", sql)[1:]
    return {p.split("(", 1)[0]: p for p in parts}


def test_every_hybrid_function_is_redefined_and_uses_the_wrapper():
    fns = _functions(_sql_without_comments())
    assert set(HYBRID) <= set(fns), sorted(fns)
    for name in HYBRID:
        assert "archon_bm25_hits(" in fns[name], name
        assert "psql_bm25s_query(" not in fns[name], f"{name} calls the raw parser"


def test_the_raw_parser_is_called_only_inside_the_wrapper():
    fns = _functions(_sql_without_comments())
    callers = [n for n, body in fns.items() if "psql_bm25s_query(" in body]
    assert callers == ["archon_bm25_hits"], callers


def test_the_wrapper_tokenizes_guards_null_and_drops_unscored_hits():
    body = _functions(_sql_without_comments())["archon_bm25_hits"]
    assert "psql_bm25s_tokenize_text(COALESCE(query_text, '')" in body
    assert "plain_query IS NULL OR plain_query = ''" in body and "RETURN;" in body
    assert "h.score > 0" in body


def test_the_migration_records_itself():
    assert "'012_bm25_plain_query'" in MIGRATION.read_text()
