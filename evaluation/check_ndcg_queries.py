"""Sanity-check nDCG queries: structured hits + no catalog terms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from query_parser import QueryParser
from semantic_text_llm import has_catalog_terms

CASES_PATH = EVAL_DIR / "queries" / "ndcg_queries_30.json"
CLEANED_CSV = ROOT / "data" / "processed" / "all_listings_cleaned.csv"
LISTINGS_CSV = ROOT / "data" / "processed" / "rets_property.csv"
NOTEBOOKS_CSV = ROOT / "notebooks" / "rets_property.csv"

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


def listings_path() -> Path:
    if LISTINGS_CSV.exists():
        return LISTINGS_CSV
    if NOTEBOOKS_CSV.exists():
        return NOTEBOOKS_CSV
    raise SystemExit("Missing listings CSV")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def candidate_ids(meta: pd.DataFrame, filters: dict, indexed: set[int]) -> list[int]:
    df = meta[meta["id"].isin(indexed)].copy()
    if "city" in filters and "L_City" in df.columns:
        df = df[
            df["L_City"].astype(str).str.strip().str.lower()
            == str(filters["city"]).strip().lower()
        ]
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
    out = {}
    for k in STRUCTURED_KEYS:
        if k not in filters:
            continue
        val = filters[k]
        if hasattr(val, "item"):
            val = val.item()
        out[k] = val
    return out


def main() -> None:
    cases = load_cases()
    csv_path = listings_path()
    print(f"queries: {CASES_PATH}")
    print(f"listings: {csv_path}", flush=True)
    meta = pd.read_csv(
        csv_path,
        usecols=lambda c: c
        in {
            "id",
            "L_City",
            "L_Keyword2",
            "LM_Dec_3",
            "L_SystemPrice",
            "LM_Int2_3",
        },
    )
    meta["id"] = pd.to_numeric(meta["id"], errors="coerce")
    meta = meta.dropna(subset=["id"])
    meta["id"] = meta["id"].astype(int)
    indexed = set(
        int(x)
        for x in pd.read_csv(CLEANED_CSV, usecols=["id"])["id"].tolist()
        if pd.notna(x)
    )
    print(f"meta rows={len(meta)} faiss listing ids={len(indexed)}", flush=True)

    parser = QueryParser()
    n_empty = 0
    n_catalog = 0
    n_no_city = 0
    all_ids = set(meta["id"].tolist())
    for case in cases:
        filters = parser.parse(case["query"])
        structured = structured_slice(filters)
        catalog = {k: filters[k] for k in CATALOG_KEYS if filters.get(k)}
        cands = candidate_ids(meta, structured, indexed)
        csv_only = candidate_ids(meta, structured, all_ids)
        ok_hits = len(cands) > 0
        ok_catalog = not has_catalog_terms(filters)
        if not ok_hits:
            n_empty += 1
        if not ok_catalog:
            n_catalog += 1
        if "city" not in structured:
            n_no_city += 1
        flag = []
        if not ok_hits:
            flag.append("NO_HITS")
        if not ok_catalog:
            flag.append("CATALOG")
        if "city" not in structured:
            flag.append("NO_CITY")
        mark = " ".join(flag) if flag else "ok"
        print(
            f"{case['id']:02d} {case['kind']:<4} {mark:12} "
            f"faiss={len(cands):4d} csv={len(csv_only):4d}  "
            f"{json.dumps(structured, ensure_ascii=False)}"
            + (f"  CATALOG={catalog}" if catalog else ""),
            flush=True,
        )

    print(
        f"\n{len(cases)} queries  empty_faiss={n_empty}  "
        f"catalog_leaks={n_catalog}  no_city={n_no_city}",
        flush=True,
    )


if __name__ == "__main__":
    main()
