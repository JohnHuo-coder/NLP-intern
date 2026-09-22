"""Extract leftover for the nDCG 30 queries; no auto pass/fail.

Output is for human review: query vs leftover, plus any catalog the
rule matcher hit and the remarks-ranking text (leftover or full-query fallback).
"""

from __future__ import annotations

import html
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

os.environ["OPENAI_API_KEY"] = (
    (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
)

from query_parser import QueryParser
from semantic_text_llm import SemanticTextLLM, has_catalog_terms, ranking_query

CASES_PATH = EVAL_DIR / "queries" / "ndcg_queries_30.json"
OUT_JSON = EVAL_DIR / "results" / "semantic_text_llm_30.json"
OUT_HTML = EVAL_DIR / "results" / "semantic_text_llm_30.html"

CATALOG_KEYS = (
    "amenities",
    "features",
    "finance",
    "negated_amenities",
    "negated_features",
    "negated_finance",
)

STRUCTURED_KEYS = (
    "city",
    "bedrooms",
    "bedrooms_min",
    "bedrooms_max",
    "bathrooms",
    "bathrooms_min",
    "bathrooms_max",
    "bathroom_half",
    "price",
    "price_min",
    "price_max",
    "sqft",
    "sqft_min",
    "sqft_max",
)


def load_cases() -> list[dict]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["queries"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"No queries in {CASES_PATH}")
    return cases


def slice_filters(filters: dict, keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        if not filters.get(key):
            continue
        val = filters[key]
        if hasattr(val, "item"):
            val = val.item()
        out[key] = val
    return out


def llm_extract(llm: SemanticTextLLM, query: str) -> str:
    last = None
    for attempt in range(3):
        try:
            return llm.extract(query)
        except Exception as exc:
            last = exc
            print(f"  retry {attempt + 1}/3 {type(exc).__name__}", flush=True)
            time.sleep(2)
    print(f"  failed: {type(last).__name__}", flush=True)
    return ""


def render_html(payload: dict) -> str:
    rows = []
    for r in payload["rows"]:
        leftover = r["leftover"] or ""
        empty = not leftover.strip()
        catalog = r.get("catalog") or {}
        rows.append(
            "<tr class='{cls}'>"
            "<td>{id}</td><td>{kind}</td>"
            "<td class='q'>{query}</td>"
            "<td class='left'>{leftover}</td>"
            "<td>{rank}</td>"
            "<td>{catalog}</td>"
            "<td>{struct}</td>"
            "</tr>".format(
                cls="empty" if empty else "ok",
                id=r["id"],
                kind=html.escape(r["kind"]),
                query=html.escape(r["query"]),
                leftover=html.escape(leftover or "(empty)"),
                rank=html.escape(r.get("rank_text") or ""),
                catalog=html.escape(json.dumps(catalog, ensure_ascii=False) if catalog else ""),
                struct=html.escape(json.dumps(r.get("structured") or {}, ensure_ascii=False)),
            )
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Leftover extract — human review</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #1b1f24; }}
.note {{ color: #555; max-width: 960px; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #eef2f6; text-align: left; }}
td.q {{ max-width: 420px; }}
td.left {{ color: #1a4a8a; font-weight: 600; }}
tr.empty td.left {{ color: #a40; }}
</style></head><body>
<h1>Leftover extract (nDCG 30 queries)</h1>
<p class="note">No auto score. Yellow leftover = empty (ranking falls back to the full query).
Catalog column should stay empty on this set.</p>
<table>
<thead><tr>
<th>#</th><th>kind</th><th>query</th><th>leftover</th><th>rank text</th><th>rule catalog</th><th>structured</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")
    cases = load_cases()
    parser = QueryParser()
    llm = SemanticTextLLM()
    rows = []
    t0 = time.time()
    for case in cases:
        query = case["query"]
        print(f"{case['id']:02d} {case.get('kind', '')} ...", flush=True)
        filters = parser.parse(query)
        leftover = llm_extract(llm, query)
        rank_text = ranking_query(query, leftover, filters)
        row = {
            "id": case["id"],
            "kind": case.get("kind") or "",
            "query": query,
            "leftover": leftover,
            "rank_text": rank_text,
            "leftover_fallback": bool(rank_text) and not leftover,
            "structured": slice_filters(filters, STRUCTURED_KEYS),
            "catalog": slice_filters(filters, CATALOG_KEYS),
            "rule_has_catalog": has_catalog_terms(filters),
        }
        rows.append(row)
        print(f"  leftover={leftover!r}", flush=True)

    payload = {
        "n_queries": len(rows),
        "cases_path": str(CASES_PATH),
        "elapsed_s": round(time.time() - t0, 2),
        "note": (
            "Leftover extract only. No pass/fail. Review leftover against the query; "
            "it should keep A/B paraphrase/setting and drop beds/city/price/catalog names."
        ),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {OUT_HTML}", flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
