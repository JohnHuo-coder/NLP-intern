"""Score raw-query vs leftover ranking with leftover-only LLM relevance.

Reads ``evaluation/results/llm_vs_raw_semantic_30.json``. Each listing is
graded against leftover A/B text only (not the full query), then nDCG@10 /
MRR / P@k compare the two ranked lists. Overlap is expected; the question is
whether leftover puts the more relevant listings higher.

Also times leftover extract so the quality numbers have a latency cost.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
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

from openai import OpenAI

from semantic_text_llm import SemanticTextLLM

IN_JSON = EVAL_DIR / "results" / "llm_vs_raw_semantic_30.json"
OUT_JSON = EVAL_DIR / "results" / "llm_vs_raw_ndcg_30.json"
OUT_HTML = EVAL_DIR / "results" / "llm_vs_raw_ndcg_30.html"

K = 10
RELEVANT_AT = 2  # rel >= 2 counts as relevant for MRR / Precision
MODEL = "gpt-4.1-mini"

JUDGE_SYSTEM = """
You grade how well a listing remark chunk supports a leftover search phrase.

The leftover is only the soft A/B ask (paraphrased amenity/feature or holistic
setting). Numeric filters and city were already applied. Do not reward bedroom
count, bathrooms, price, square footage, or city names.

Grade each listing independently. Do not let neighboring listings in the
same request pull a score up or down. Paraphrase counts (bright/sunny ≈
natural light; ocean/coastal ≈ next to the beach) but do not infer from
city or price.

Score each listing 0-3:
0 = leftover is not supported
1 = weak / one vague hint
2 = partial: some leftover aspects are clearly present
3 = strong: leftover is clearly supported

For each listing, write a one-sentence why, then the score.
Score only from the leftover phrase and the chunk. Do not invent facts.
Return one object per listing_id you were given.
""".strip()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "listing_id": {"type": "integer"},
                    "why": {"type": "string"},
                    "rel": {"type": "integer", "enum": [0, 1, 2, 3]},
                },
                "required": ["listing_id", "why", "rel"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def dcg(rels: list[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        gain = (2 ** int(rel)) - 1
        total += gain / math.log2(i + 1)
    return total


def ndcg(rels: list[int], k: int, ideal: list[int] | None = None) -> float:
    padded = list(rels) + [0] * k
    actual = dcg(padded, k)
    if ideal is None:
        ideal_rels = sorted(rels, reverse=True)
    else:
        ideal_rels = sorted(ideal, reverse=True)
    best = dcg(ideal_rels + [0] * k, k)
    if best <= 0:
        return 0.0
    return actual / best


def mrr(rels: list[int], min_rel: int = RELEVANT_AT) -> float:
    for i, rel in enumerate(rels, start=1):
        if int(rel) >= min_rel:
            return 1.0 / i
    return 0.0


def precision_at(rels: list[int], k: int, min_rel: int = RELEVANT_AT) -> float:
    if not rels:
        return 0.0
    head = rels[:k]
    if not head:
        return 0.0
    return sum(1 for r in head if int(r) >= min_rel) / len(head)


def leftover_terms(text: str) -> list[str]:
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "with",
        "for",
        "near",
        "next",
    }
    return [
        t
        for t in re.findall(r"[a-z0-9']+", (text or "").lower())
        if t not in stop and len(t) > 2
    ]


def coverage(chunk: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    blob = (chunk or "").lower()
    return sum(1 for t in terms if t in blob) / len(terms)


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def judge_query(client: OpenAI, leftover: str, items: list[dict]) -> dict[int, dict]:
    payload = [
        {"listing_id": int(it["id"]), "chunk": (it.get("chunk") or "")[:700]}
        for it in items
    ]
    user = json.dumps({"leftover": leftover, "listings": payload}, ensure_ascii=False)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "leftover_relevance",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            },
        },
        temperature=0,
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    out: dict[int, dict] = {}
    for row in parsed.get("scores") or []:
        try:
            lid = int(row["listing_id"])
            rel = int(row["rel"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= rel <= 3:
            out[lid] = {
                "rel": rel,
                "why": str(row.get("why") or ""),
            }
    return out


def unique_hits(*hit_lists: list[dict]) -> list[dict]:
    seen: dict[int, dict] = {}
    for hits in hit_lists:
        for hit in hits or []:
            seen[int(hit["id"])] = hit
    # Prefer leftover-side chunk when the same listing appears in several lists.
    if hit_lists:
        for hit in hit_lists[-1] or []:
            seen[int(hit["id"])] = hit
    return list(seen.values())


def rels_for(hits: list[dict], grades: dict[int, dict]) -> list[int]:
    return [int((grades.get(int(h["id"])) or {}).get("rel", 0)) for h in hits]


def rank_map(hits: list[dict]) -> dict[int, int]:
    return {int(h["id"]): int(h["rank"]) for h in hits}


def time_leftover_extract(queries: list[str]) -> list[float]:
    llm = SemanticTextLLM()
    times = []
    for q in queries:
        t0 = time.time()
        llm.extract(q)
        times.append(time.time() - t0)
    return times


def summarize(rows: list[dict]) -> dict:
    def col(key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    def pair_wins(left_key: str, right_key: str) -> tuple[int, int, int]:
        wins = sum(1 for r in rows if r[right_key] > r[left_key] + 1e-9)
        ties = sum(1 for r in rows if abs(r[right_key] - r[left_key]) <= 1e-9)
        losses = len(rows) - wins - ties
        return wins, ties, losses

    wins, ties, losses = pair_wins("ndcg_raw", "ndcg_llm")
    sql_wins, sql_ties, sql_losses = pair_wins("ndcg_sql", "ndcg_llm")
    by_kind = {}
    for kind in ("A", "B", "A+B"):
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        by_kind[kind] = {
            "n": len(subset),
            "ndcg_sql": mean([r["ndcg_sql"] for r in subset]),
            "ndcg_raw": mean([r["ndcg_raw"] for r in subset]),
            "ndcg_llm": mean([r["ndcg_llm"] for r in subset]),
            "mrr_sql": mean([r["mrr_sql"] for r in subset]),
            "mrr_raw": mean([r["mrr_raw"] for r in subset]),
            "mrr_llm": mean([r["mrr_llm"] for r in subset]),
            "p10_sql": mean([r["p10_sql"] for r in subset]),
            "p10_raw": mean([r["p10_raw"] for r in subset]),
            "p10_llm": mean([r["p10_llm"] for r in subset]),
            "ndcg_wins_vs_raw": sum(
                1 for r in subset if r["ndcg_llm"] > r["ndcg_raw"] + 1e-9
            ),
            "ndcg_wins_vs_sql": sum(
                1 for r in subset if r["ndcg_llm"] > r["ndcg_sql"] + 1e-9
            ),
        }
    large = [r for r in rows if int(r["n_candidates"]) >= 30]
    return {
        "n": len(rows),
        "ndcg_sql": mean(col("ndcg_sql")),
        "ndcg_raw": mean(col("ndcg_raw")),
        "ndcg_llm": mean(col("ndcg_llm")),
        "ndcg_union_sql": mean(col("ndcg_union_sql")),
        "ndcg_union_raw": mean(col("ndcg_union_raw")),
        "ndcg_union_llm": mean(col("ndcg_union_llm")),
        "mrr_sql": mean(col("mrr_sql")),
        "mrr_raw": mean(col("mrr_raw")),
        "mrr_llm": mean(col("mrr_llm")),
        "p5_sql": mean(col("p5_sql")),
        "p5_raw": mean(col("p5_raw")),
        "p5_llm": mean(col("p5_llm")),
        "p10_sql": mean(col("p10_sql")),
        "p10_raw": mean(col("p10_raw")),
        "p10_llm": mean(col("p10_llm")),
        "coverage_raw": mean(col("coverage_raw")),
        "coverage_llm": mean(col("coverage_llm")),
        "ndcg_wins": wins,
        "ndcg_ties": ties,
        "ndcg_losses": losses,
        "ndcg_wins_vs_sql": sql_wins,
        "ndcg_ties_vs_sql": sql_ties,
        "ndcg_losses_vs_sql": sql_losses,
        "by_kind": by_kind,
        "n_candidates_ge_30": {
            "n": len(large),
            "ndcg_sql": mean([r["ndcg_sql"] for r in large]),
            "ndcg_raw": mean([r["ndcg_raw"] for r in large]),
            "ndcg_llm": mean([r["ndcg_llm"] for r in large]),
            "mrr_sql": mean([r["mrr_sql"] for r in large]),
            "mrr_raw": mean([r["mrr_raw"] for r in large]),
            "mrr_llm": mean([r["mrr_llm"] for r in large]),
        },
    }


def render_html(payload: dict) -> str:
    s = payload["summary"]
    lat = payload["latency"]
    rows_html = []
    for r in payload["rows"]:
        win = "win" if r["ndcg_llm"] > r["ndcg_raw"] + 1e-9 else (
            "tie" if abs(r["ndcg_llm"] - r["ndcg_raw"]) <= 1e-9 else "loss"
        )
        rows_html.append(
            "<tr class='{win}'>"
            "<td>{id}</td><td>{kind}</td><td>{n_candidates}</td>"
            "<td class='q'>{query}<div class='left'>leftover: {left}</div></td>"
            "<td>{ndcg_sql:.3f}</td><td>{ndcg_raw:.3f}</td><td>{ndcg_llm:.3f}</td>"
            "<td>{mrr_sql:.3f}</td><td>{mrr_raw:.3f}</td><td>{mrr_llm:.3f}</td>"
            "<td>{p10_sql:.2f}</td><td>{p10_raw:.2f}</td><td>{p10_llm:.2f}</td>"
            "</tr>".format(
                win=win,
                id=r["id"],
                kind=html.escape(r["kind"]),
                n_candidates=r["n_candidates"],
                query=html.escape(r["query"]),
                left=html.escape(r["llm_text"]),
                ndcg_sql=r["ndcg_sql"],
                ndcg_raw=r["ndcg_raw"],
                ndcg_llm=r["ndcg_llm"],
                mrr_sql=r["mrr_sql"],
                mrr_raw=r["mrr_raw"],
                mrr_llm=r["mrr_llm"],
                p10_sql=r["p10_sql"],
                p10_raw=r["p10_raw"],
                p10_llm=r["p10_llm"],
            )
        )

    def triple(title: str, sql: float, raw: float, leftover: float) -> str:
        return (
            f"<div class='stat'><div class='k'>{html.escape(title)}</div>"
            f"<div>SQL {sql:.3f}</div>"
            f"<div>full-query {raw:.3f}</div>"
            f"<div>leftover {leftover:.3f}</div></div>"
        )

    extract_mean = lat.get("extract_mean_s")
    extract_html = ""
    if extract_mean not in (None, ""):
        extract_html = (
            "<div class='stat'><div class='k'>leftover extract latency</div>"
            f"<div>mean {float(extract_mean):.2f}s</div></div>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>nDCG / MRR: SQL vs full query vs leftover ({html.escape(payload.get("judge_model") or "")})</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #1b1f24; }}
h1 {{ font-size: 20px; }}
.note {{ color: #555; max-width: 960px; line-height: 1.45; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 24px; }}
.stat {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; min-width: 160px; }}
.k {{ font-size: 12px; color: #666; margin-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ background: #eef2f6; text-align: left; }}
td.q {{ max-width: 360px; }}
.left {{ color: #385d8a; margin-top: 4px; font-size: 12px; }}
tr.win td:nth-child(7) {{ background: #d8f3dc; }}
tr.loss td:nth-child(6) {{ background: #fde2e1; }}
tr.tie td:nth-child(6), tr.tie td:nth-child(7) {{ background: #fff3cd; }}
</style></head><body>
<h1>SQL vs full-query vs leftover (judge: {html.escape(payload.get("judge_model") or "")})</h1>
<p class="note">{html.escape(payload["note"])}</p>
<div class="stats">
{triple("nDCG@10", s["ndcg_sql"], s["ndcg_raw"], s["ndcg_llm"])}
{triple("MRR (first rel≥2)", s["mrr_sql"], s["mrr_raw"], s["mrr_llm"])}
{triple("P@10 (rel≥2)", s["p10_sql"], s["p10_raw"], s["p10_llm"])}
<div class="stat"><div class="k">leftover nDCG vs full-query</div>
<div>{s["ndcg_wins"]} win / {s["ndcg_ties"]} tie / {s["ndcg_losses"]} loss</div></div>
<div class="stat"><div class="k">leftover nDCG vs SQL</div>
<div>{s["ndcg_wins_vs_sql"]} win / {s["ndcg_ties_vs_sql"]} tie / {s["ndcg_losses_vs_sql"]} loss</div></div>
{extract_html}
</div>
<table>
<thead><tr>
<th>#</th><th>kind</th><th>cands</th><th>query</th>
<th>nDCG SQL</th><th>nDCG full</th><th>nDCG left</th>
<th>MRR SQL</th><th>MRR full</th><th>MRR left</th>
<th>P@10 SQL</th><th>P@10 full</th><th>P@10 left</th>
</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
<p class="note">Green leftover nDCG = leftover beats full-query semantic.
SQL is structured filters only, ordered by id DESC (no remarks ranker).</p>
</body></html>
"""


def main() -> None:
    src = json.loads(IN_JSON.read_text(encoding="utf-8"))
    client = OpenAI(timeout=90.0)

    prev = {}
    if OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    latency = {
        "search_elapsed_s": src.get("elapsed_s"),
        "extract_mean_s": (prev.get("latency") or {}).get("extract_mean_s"),
        "note": (
            "search_elapsed_s is leftover extract plus two hybrid searches per query. "
            "SQL top-10 is computed from the same candidate ids (id DESC), no extra search."
        ),
    }
    extract_by_id = {
        int(r["id"]): r.get("extract_s") or 0.0 for r in (prev.get("rows") or [])
    }

    scored = []
    t0 = time.time()
    for row in src["rows"]:
        leftover = row["llm_text"]
        raw_hits = row["raw_top10"]
        llm_hits = row["llm_top10"]
        sql_hits = row.get("sql_top10") or []
        uniq = unique_hits(sql_hits, raw_hits, llm_hits)
        print(
            f"{row['id']:02d} judge {len(uniq)} listings leftover={leftover!r}",
            flush=True,
        )
        grades = judge_query(client, leftover, uniq)
        missing = [int(h["id"]) for h in uniq if int(h["id"]) not in grades]
        if missing:
            print(f"  missing grades default 0: {missing[:8]}", flush=True)

        raw_rels = rels_for(raw_hits, grades)
        llm_rels = rels_for(llm_hits, grades)
        sql_rels = rels_for(sql_hits, grades)
        union_rels = [
            int((grades.get(int(h["id"])) or {}).get("rel", 0)) for h in uniq
        ]
        terms = leftover_terms(leftover)
        raw_cov = [coverage(h.get("chunk") or "", terms) for h in raw_hits]
        llm_cov = [coverage(h.get("chunk") or "", terms) for h in llm_hits]

        raw_ranks = rank_map(raw_hits)
        llm_ranks = rank_map(llm_hits)
        shared = [i for i in raw_ranks if i in llm_ranks]
        deltas = [raw_ranks[i] - llm_ranks[i] for i in shared]
        mean_delta = round(sum(deltas) / len(deltas), 3) if deltas else 0.0

        scored.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "query": row["query"],
                "llm_text": leftover,
                "search_text": row.get("search_text") or leftover,
                "leftover_fallback": bool(row.get("leftover_fallback")),
                "n_candidates": row["n_candidates"],
                "overlap_n": row["overlap_n"],
                "sql_leftover_overlap_n": row.get("sql_leftover_overlap_n", 0),
                "grades": {str(k): v for k, v in grades.items()},
                "sql_rels": sql_rels,
                "raw_rels": raw_rels,
                "llm_rels": llm_rels,
                "ndcg_sql": round(ndcg(sql_rels, K), 4),
                "ndcg_raw": round(ndcg(raw_rels, K), 4),
                "ndcg_llm": round(ndcg(llm_rels, K), 4),
                "ndcg_union_sql": round(ndcg(sql_rels, K, union_rels), 4),
                "ndcg_union_raw": round(ndcg(raw_rels, K, union_rels), 4),
                "ndcg_union_llm": round(ndcg(llm_rels, K, union_rels), 4),
                "mrr_sql": round(mrr(sql_rels), 4),
                "mrr_raw": round(mrr(raw_rels), 4),
                "mrr_llm": round(mrr(llm_rels), 4),
                "p5_sql": round(precision_at(sql_rels, 5), 4),
                "p5_raw": round(precision_at(raw_rels, 5), 4),
                "p5_llm": round(precision_at(llm_rels, 5), 4),
                "p10_sql": round(precision_at(sql_rels, 10), 4),
                "p10_raw": round(precision_at(raw_rels, 10), 4),
                "p10_llm": round(precision_at(llm_rels, 10), 4),
                "coverage_raw": round(sum(raw_cov) / len(raw_cov), 4) if raw_cov else 0.0,
                "coverage_llm": round(sum(llm_cov) / len(llm_cov), 4) if llm_cov else 0.0,
                "mean_shared_rank_delta": mean_delta,
                "extract_s": round(extract_by_id.get(int(row["id"]), 0.0), 3),
            }
        )
        print(
            f"  nDCG sql={scored[-1]['ndcg_sql']} raw={scored[-1]['ndcg_raw']} "
            f"left={scored[-1]['ndcg_llm']}",
            flush=True,
        )

    summary = summarize(scored)
    payload = {
        "n_queries": len(scored),
        "k": K,
        "relevant_at": RELEVANT_AT,
        "judge_model": MODEL,
        "source": str(IN_JSON),
        "elapsed_s": round(time.time() - t0, 2),
        "note": (
            "Relevance is leftover-only (A/B phrase vs remark chunk). "
            "nDCG@10 uses gain 2^rel-1. SQL is structured filters only, id DESC "
            "(no remarks ranker). Full-query and leftover are hybrid rankers on "
            "the same candidate set. Empty leftover falls back to the full query."
        ),
        "latency": latency,
        "summary": summary,
        "rows": scored,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"summary": summary, "latency": latency}, indent=2), flush=True)
    print(f"Wrote {OUT_HTML}", flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
