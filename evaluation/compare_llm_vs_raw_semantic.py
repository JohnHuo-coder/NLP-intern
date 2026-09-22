"""Compare semantic top-10: full query vs LLM leftover (A/B rephrase).

No MySQL: structured filters run on a local listings CSV; ranking uses the
existing FAISS index over ``all_listings_cleaned.csv``.
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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
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

# torch/OpenMP must load before pandas (otherwise WinError 1114 on c10.dll)
from semantic_searcher import SemanticSearcher
import pandas as pd

from query_parser import QueryParser
from semantic_text_llm import SemanticTextLLM, ranking_query

CLEANED_CSV = ROOT / "data" / "processed" / "all_listings_cleaned.csv"
INDEX_PATH = ROOT / "data" / "processed" / "index.faiss"
SQL_DUMP = ROOT / "data" / "raw" / "rets_property.sql"
LISTINGS_CSV = ROOT / "data" / "processed" / "rets_property.csv"
NOTEBOOKS_CSV = ROOT / "notebooks" / "rets_property.csv"
OUT_JSON = EVAL_DIR / "results" / "llm_vs_raw_semantic_30.json"
OUT_HTML = EVAL_DIR / "results" / "llm_vs_raw_semantic_30.html"
CASES_PATH = EVAL_DIR / "queries" / "ndcg_queries_30.json"
TOP_K = 10


def load_cases() -> list[dict]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["queries"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"No queries in {CASES_PATH}")
    return cases


def ensure_listings_csv() -> Path:
    """Use a local table export. SQL dump is phpMyAdmin (~400MB); the notebooks CSV
    already has every FAISS listing id, so we read that instead of re-parsing SQL.
    """
    if LISTINGS_CSV.exists():
        return LISTINGS_CSV
    if NOTEBOOKS_CSV.exists():
        print(f"Using existing table export {NOTEBOOKS_CSV} (skip SQL parse)", flush=True)
        return NOTEBOOKS_CSV
    if SQL_DUMP.exists():
        raise SystemExit(
            f"No listings CSV found next to the dump. Export {SQL_DUMP} to CSV "
            f"or copy it to {NOTEBOOKS_CSV}."
        )
    raise SystemExit("Missing listings CSV and SQL dump.")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def candidate_ids(meta: pd.DataFrame, filters: dict, indexed: set[int]) -> list[int]:
    df = meta[meta["id"].isin(indexed)].copy()
    if "city" in filters and "L_City" in df.columns:
        df = df[df["L_City"].astype(str).str.strip().str.lower() == str(filters["city"]).strip().lower()]
    if "bedrooms" in filters:
        df = df[_num(df["L_Keyword2"]) == filters["bedrooms"]]
    if "bedrooms_min" in filters:
        df = df[_num(df["L_Keyword2"]) >= filters["bedrooms_min"]]
    if "bedrooms_max" in filters:
        df = df[_num(df["L_Keyword2"]) <= filters["bedrooms_max"]]
    if "bathrooms" in filters:
        df = df[_num(df["LM_Dec_3"]) == filters["bathrooms"]]
    if "bathrooms_min" in filters:
        df = df[_num(df["LM_Dec_3"]) >= filters["bathrooms_min"]]
    if "bathrooms_max" in filters:
        df = df[_num(df["LM_Dec_3"]) <= filters["bathrooms_max"]]
    if "price" in filters:
        df = df[_num(df["L_SystemPrice"]) == filters["price"]]
    if "price_min" in filters:
        df = df[_num(df["L_SystemPrice"]) >= filters["price_min"]]
    if "price_max" in filters:
        df = df[_num(df["L_SystemPrice"]) <= filters["price_max"]]
    if "sqft" in filters:
        df = df[_num(df["LM_Int2_3"]) == filters["sqft"]]
    if "sqft_min" in filters:
        df = df[_num(df["LM_Int2_3"]) >= filters["sqft_min"]]
    if "sqft_max" in filters:
        df = df[_num(df["LM_Int2_3"]) <= filters["sqft_max"]]
    return [int(x) for x in df["id"].tolist()]


def structured_slice(filters: dict) -> dict:
    keys = (
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
    return {k: _jsonable(filters[k]) for k in keys if k in filters}


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float):
        return value
    return value


def _meta_fields(lid: int, by_id) -> dict:
    rec = {}
    if int(lid) not in by_id.index:
        return rec
    row = by_id.loc[int(lid)]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    def cell(col):
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val):
            return None
        return val

    rec["address"] = None if cell("L_Address") is None else str(cell("L_Address"))
    rec["city"] = None if cell("L_City") is None else str(cell("L_City"))
    rec["beds"] = _jsonable(cell("L_Keyword2"))
    rec["baths"] = _jsonable(cell("LM_Dec_3"))
    rec["price"] = _jsonable(cell("L_SystemPrice"))
    rec["sqft"] = _jsonable(cell("LM_Int2_3"))
    return rec


def pack_hits(ids_scores, results, meta: pd.DataFrame) -> list[dict]:
    by_id = meta.set_index("id", drop=False)
    rows = []
    for rank, ((lid, score), result) in enumerate(zip(ids_scores, results), start=1):
        chunk = result[2] if len(result) > 2 else result[0]
        rec = {"rank": rank, "id": int(lid), "score": round(float(score), 4)}
        rec.update(_meta_fields(lid, by_id))
        rec["chunk"] = str(chunk or "")[:420]
        rows.append(rec)
    return rows


def pack_sql_hits(
    candidate_ids: list[int] | None,
    meta: pd.DataFrame,
    remarks_by_id: dict[int, str],
) -> list[dict]:
    """Live SQL path with no catalog: same filters, ORDER BY id DESC, no remarks rank."""
    ids = list(candidate_ids or [])
    ids.sort(reverse=True)
    by_id = meta.set_index("id", drop=False)
    rows = []
    for rank, lid in enumerate(ids[:TOP_K], start=1):
        rec = {"rank": rank, "id": int(lid), "score": None}
        rec.update(_meta_fields(lid, by_id))
        rec["chunk"] = (remarks_by_id.get(int(lid)) or "")[:420]
        rows.append(rec)
    return rows


def search_top(
    searcher: SemanticSearcher,
    query: str,
    ids: list[int] | None,
) -> tuple[list, list]:
    if not query.strip():
        return [], []
    if ids:
        results, ranked, _ = searcher.search_hybrid_for_listings(
            query, ids, top_k=TOP_K, return_chunks=True
        )
    else:
        results, ranked, _ = searcher.search_hybrid(
            query, top_k=TOP_K, return_chunks=True
        )
    return results, ranked


def fmt_money(v) -> str:
    try:
        return f"${int(float(v)):,}"
    except (TypeError, ValueError):
        return ""


def card_html(hit: dict, other_ids: dict[int, int]) -> str:
    lid = hit["id"]
    other_rank = other_ids.get(lid)
    badge = (
        f'<span class="both">also #{other_rank} on other side</span>'
        if other_rank
        else '<span class="only">only this side</span>'
    )
    meta = " · ".join(
        p
        for p in (
            hit.get("city") or "",
            f"{hit['beds']} bd" if hit.get("beds") is not None else "",
            f"{hit['baths']} ba" if hit.get("baths") is not None else "",
            fmt_money(hit.get("price")),
            f"{int(hit['sqft'])} sqft" if hit.get("sqft") not in (None, "") else "",
        )
        if p
    )
    return f"""
    <article class="hit {'shared' if other_rank else 'solo'}">
      <div class="hit-head">
        <strong>#{hit['rank']}</strong>
        <span class="lid">id {lid}</span>
        <span class="score">{'' if hit.get('score') is None else hit['score']}</span>
        {badge}
      </div>
      <div class="addr">{html.escape(hit.get('address') or '')}</div>
      <div class="meta">{html.escape(meta)}</div>
      <p class="chunk">{html.escape(hit.get('chunk') or '')}</p>
    </article>
    """


def render_html(payload: dict) -> str:
    parts = [
        """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Raw query vs LLM leftover — semantic top 10</title>
<style>
body { font-family: Georgia, serif; margin: 24px; background: #f6f4ef; color: #222; }
h1 { font-size: 1.4rem; }
.toc a { margin-right: 8px; }
.case { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px 20px; margin: 28px 0; }
.case h2 { margin: 0 0 8px; font-size: 1.15rem; }
.kind { background: #314; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.col h3 { margin: 12px 0 8px; font-size: 1rem; }
.q, .left { background: #f3f6fb; padding: 8px 10px; border-radius: 4px; }
.hit { border: 1px solid #e2e2e2; border-radius: 6px; padding: 8px 10px; margin: 8px 0; }
.hit.shared { border-color: #3a7; background: #f3faf5; }
.hit.solo { background: #fffdf8; }
.hit-head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.lid, .score { font-family: Consolas, monospace; font-size: 0.9rem; }
.both { color: #146c2e; font-size: 0.8rem; }
.only { color: #888; font-size: 0.8rem; }
.chunk { font-size: 0.88rem; line-height: 1.35; color: #333; white-space: pre-wrap; }
.addr { font-weight: 600; margin-top: 4px; }
.meta { color: #555; font-size: 0.9rem; }
.warn { color: #a40; }
</style></head><body>
"""
    ]
    parts.append("<h1>SQL vs full-query vs leftover — top 10</h1>")
    parts.append(
        "<p>Same structured candidate set (CSV city/beds/price/sqft + FAISS ids). "
        "SQL: no catalog rank, ORDER BY id DESC (live SQL-only path). "
        "Full query / leftover: hybrid BM25+embedding. "
        "Green cards also appear in the leftover list.</p>"
    )
    parts.append('<p class="toc">')
    for row in payload["rows"]:
        parts.append(f'<a href="#q{row["id"]}">{row["id"]}</a>')
    parts.append("</p>")
    for row in payload["rows"]:
        raw_ids = {h["id"]: h["rank"] for h in row["raw_top10"]}
        llm_ids = {h["id"]: h["rank"] for h in row["llm_top10"]}
        sql_ids = {h["id"]: h["rank"] for h in row.get("sql_top10") or []}
        overlap = sorted(set(raw_ids) & set(llm_ids), key=lambda i: raw_ids[i])
        warn = ""
        if row.get("filter_fallback"):
            warn = '<p class="warn">Structured filter matched 0 indexed rows; ranked over the full FAISS corpus.</p>'
        parts.append(f'<section class="case" id="q{row["id"]}">')
        parts.append(
            f'<h2>#{row["id"]} <span class="kind">{html.escape(row["kind"])}</span> '
            f'full∩leftover {len(overlap)}/{TOP_K}</h2>'
        )
        parts.append(f'<p class="q"><b>Query:</b> {html.escape(row["query"])}</p>')
        parts.append(
            f'<p><b>Structured:</b> {html.escape(json.dumps(row["structured"], ensure_ascii=False))} '
            f'· candidates {row["n_candidates"]}</p>'
        )
        parts.append(f'<p class="left"><b>LLM leftover:</b> {html.escape(row["llm_text"] or "(empty)")}</p>')
        if row.get("leftover_fallback"):
            parts.append(
                '<p class="warn">Leftover was empty; leftover column ranked the full user query.</p>'
            )
        parts.append(warn)
        parts.append('<div class="grid">')
        parts.append('<div class="col"><h3>SQL only (id DESC)</h3>')
        parts.extend(card_html(h, llm_ids) for h in row.get("sql_top10") or [])
        parts.append("</div>")
        parts.append('<div class="col"><h3>Full-query semantic</h3>')
        parts.extend(card_html(h, llm_ids) for h in row["raw_top10"])
        parts.append("</div>")
        parts.append('<div class="col"><h3>Leftover semantic</h3>')
        parts.extend(card_html(h, sql_ids) for h in row["llm_top10"])
        parts.append("</div></div></section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def llm_extract(llm: SemanticTextLLM, query: str) -> str:
    last = None
    for attempt in range(3):
        try:
            return llm.extract(query)
        except Exception as exc:
            last = exc
            print(f"  llm retry {attempt + 1}/3 {type(exc).__name__}", flush=True)
            time.sleep(2)
    print(f"  llm failed: {type(last).__name__}", flush=True)
    return ""


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing FAISS index: {INDEX_PATH}")
    if not CLEANED_CSV.exists():
        raise SystemExit(f"Missing cleaned listings: {CLEANED_CSV}")

    listings_csv = ensure_listings_csv()
    print(f"Listings CSV: {listings_csv}", flush=True)
    if SQL_DUMP.exists():
        print(f"SQL dump present (not parsed): {SQL_DUMP}", flush=True)

    usecols = ["id", "L_Address", "L_City", "L_Keyword2", "LM_Dec_3", "L_SystemPrice", "LM_Int2_3"]
    meta = pd.read_csv(listings_csv, usecols=lambda c: c in usecols)
    meta["id"] = pd.to_numeric(meta["id"], errors="coerce")
    meta = meta.dropna(subset=["id"])
    meta["id"] = meta["id"].astype(int)

    print("Loading SemanticSearcher (FAISS + MiniLM)...", flush=True)
    searcher = SemanticSearcher(listings_path=CLEANED_CSV)
    indexed = set(int(x) for x in searcher.listings["id"].tolist())
    print(
        f"FAISS ntotal={searcher.index.ntotal} chunks={searcher._n_chunks} listings={len(indexed)}",
        flush=True,
    )
    if int(searcher.index.ntotal) != int(searcher._n_chunks):
        print("WARNING: FAISS size != chunk count; index may have been rebuilt.", flush=True)

    remarks_by_id = {}
    for lid, remark in searcher.listings[["id", "remarks"]].itertuples(index=False, name=None):
        lid = int(lid)
        if lid not in remarks_by_id:
            remarks_by_id[lid] = str(remark or "")[:420]
    parser = QueryParser()
    llm = SemanticTextLLM()
    cases = load_cases()
    print(f"Loaded {len(cases)} queries from {CASES_PATH}", flush=True)
    rows = []
    t0 = time.time()

    for case in cases:
        print(f"{case['id']:02d} {case['kind']} ...", flush=True)
        filters = parser.parse(case["query"])
        structured = structured_slice(filters)
        cands = candidate_ids(meta, structured, indexed)
        fallback = False
        if not cands:
            fallback = True
            cands = None
        leftover = llm_extract(llm, case["query"])
        rank_text = ranking_query(case["query"], leftover, filters)
        leftover_fallback = bool(rank_text) and not leftover
        raw_res, raw_ids = search_top(searcher, case["query"], cands)
        llm_res, llm_ids = search_top(searcher, rank_text, cands)
        raw_hits = pack_hits(raw_ids, raw_res, meta)
        llm_hits = pack_hits(llm_ids, llm_res, meta)
        sql_hits = pack_sql_hits([] if fallback else cands, meta, remarks_by_id)
        overlap = [h["id"] for h in raw_hits if h["id"] in {x["id"] for x in llm_hits}]
        sql_overlap = [h["id"] for h in sql_hits if h["id"] in {x["id"] for x in llm_hits}]
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "query": case["query"],
                "structured": structured,
                "n_candidates": 0 if fallback else len(cands or []),
                "filter_fallback": fallback,
                "llm_text": leftover,
                "search_text": rank_text,
                "leftover_fallback": leftover_fallback,
                "overlap_ids": overlap,
                "overlap_n": len(overlap),
                "sql_leftover_overlap_n": len(sql_overlap),
                "sql_top10": sql_hits,
                "raw_top10": raw_hits,
                "llm_top10": llm_hits,
            }
        )
        print(
            f"  leftover={leftover!r} fallback={leftover_fallback} "
            f"cands={0 if fallback else len(cands or [])} "
            f"full∩left={len(overlap)} sql∩left={len(sql_overlap)}",
            flush=True,
        )

    payload = {
        "n_queries": len(cases),
        "top_k": TOP_K,
        "cases_path": str(CASES_PATH),
        "index": str(INDEX_PATH),
        "listings_csv": str(listings_csv),
        "cleaned_csv": str(CLEANED_CSV),
        "elapsed_s": round(time.time() - t0, 2),
        "note": (
            "Same CSV-filtered candidates. sql_top10 = live SQL-only order (id DESC), "
            "no amenity/feature rank. raw_top10 = hybrid on the full user query. "
            "llm_top10 = hybrid on leftover, or the full query if leftover is empty."
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
