"""Cross-tenant isolation for tag-scoped RAG queries, through the real API.

Hybrid search was enabled for tag-scoped queries. The failure mode of getting
that wrong is silent cross-tenant leakage - one tenant's agent reading another
tenant's knowledge base - so this asserts isolation at the API, not at the SQL
layer where the change was reasoned about.

Method: for every tag, build the set of source ids that legitimately carry it
(straight from the database), then fire queries deliberately WORDED FOR ANOTHER
TENANT at that tag and assert nothing outside the allowed set comes back.
Querying a tag with its own vocabulary would not test much; the interesting case
is a query whose best global match lives somewhere else entirely.

Exit 1 on any leak, so this can gate a deploy.
"""
import json
import subprocess
import sys
import urllib.request

RAG = "http://localhost:8181/api/rag/query"


def psql(sql):
    p = subprocess.run(
        ["docker", "exec", "-i", "archon-postgres", "psql", "-U", "archon_user",
         "-d", "archon", "-tAF", "|", "-c", sql],
        capture_output=True, text=True, timeout=120)
    p.check_returncode()
    return [l for l in p.stdout.strip().splitlines() if l.strip()]


def rag(query, tag, n=10):
    body = json.dumps({"query": query, "tag": tag, "match_count": n,
                       "skip_reranking": True}).encode()
    req = urllib.request.Request(RAG, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    return [str((x.get("metadata") or {}).get("source") or "?")
            for x in (d.get("results") or [])]


# Tags that actually exist, and how many chunks each owns.
rows = psql(
    "SELECT t.tag, count(*) FROM archon_crawled_pages cp, "
    "LATERAL jsonb_array_elements_text(cp.metadata->'tags') AS t(tag) "
    "GROUP BY t.tag HAVING count(*) > 0 ORDER BY count(*) DESC LIMIT 12;")
tags = [r.split("|")[0] for r in rows]
print("tags found: %s\n" % ", ".join(tags))

allowed = {}
for t in tags:
    allowed[t] = set(psql(
        "SELECT DISTINCT source_id FROM archon_crawled_pages "
        "WHERE metadata @> $f${\"tags\":[\"%s\"]}$f$::jsonb;" % t))

# Queries phrased for a DIFFERENT domain than the tag being searched, so the
# globally-best match is outside the tag and a filter failure would show.
PROBES = [
    "bindningstid och provperiod for abonnemang",
    "escalation matrix and dispute contact",
    "vad kostar ritningsanalysen",
    "how do I reset my password",
    "leveransvillkor och frakt",
]

leaks = 0
checked = 0
for t in tags:
    for q in PROBES:
        got = rag(q, t)
        bad = [s for s in got if s not in allowed[t]]
        checked += 1
        if bad:
            leaks += 1
            print("LEAK  tag=%-22s q=%-38s foreign=%s"
                  % (t, q[:37], bad[:3]))

print("\n%s" % ("=" * 72))
print("queries run: %d across %d tags" % (checked, len(tags)))
print("leaks: %d" % leaks)

# A control: with NO tag, foreign sources MUST appear, or this checker is
# only ever seeing an empty result set and would report clean no matter what.
body = json.dumps({"query": "escalation matrix and dispute contact",
                   "match_count": 10, "skip_reranking": True}).encode()
req = urllib.request.Request(RAG, data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=90) as r:
    d = json.loads(r.read())
untagged = [str((x.get("metadata") or {}).get("source") or "?")
            for x in (d.get("results") or [])]
sb = allowed.get("smartbyggai", set())
outside = [s for s in untagged if s not in sb]
print("control (no tag): %d results, %d outside smartbyggai -> %s"
      % (len(untagged), len(outside),
         "checker CAN see foreign sources" if outside
         else "CONTROL FAILED - checker may be blind"))
if not outside:
    print("Treat the leak count as unproven: the control did not demonstrate")
    print("that a foreign source is visible to this checker at all.")
    sys.exit(2)
sys.exit(1 if leaks else 0)
