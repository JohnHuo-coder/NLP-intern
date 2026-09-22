"""End-to-end listing search: rule parser → leftover LLM → SQL / remarks ranker."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from query_parser import QueryParser
from query_sql import LISTING_SELECT
from schema_validator import CITY_PATH, STATS_PATH, SchemaValidator
from semantic_searcher import SemanticSearcher
from semantic_text_llm import SemanticTextLLM, has_catalog_terms, ranking_query

DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 50
MAX_SEMANTIC_WINDOW = 500

_STRUCTURED_KEYS = (
    "price_max",
    "price_min",
    "price",
    "bedrooms",
    "bedrooms_min",
    "bedrooms_max",
    "bathrooms",
    "bathrooms_min",
    "bathrooms_max",
    "bathroom_half",
    "sqft_min",
    "sqft_max",
    "sqft",
    "city",
)


def _error(
    message: str,
    *,
    code: str = "bad_request",
    details: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
        }
    }


def _has_structured(filters: dict) -> bool:
    return any(key in filters for key in _STRUCTURED_KEYS)


def _coerce_mysql_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_mysql_params(params: Any) -> Any:
    if params is None:
        return None
    if isinstance(params, (list, tuple)):
        return [_coerce_mysql_scalar(p) for p in params]
    return _coerce_mysql_scalar(params)


def _where_clause(sql: str) -> str | None:
    lower = sql.lower()
    idx = lower.find(" where ")
    if idx == -1:
        return None
    start = idx + len(" where ")
    tail = sql[start:]
    lowered_tail = lower[start:]
    cut_at = len(tail)
    for token in (" order by ", " limit ", " offset "):
        token_at = lowered_tail.find(token)
        if token_at != -1:
            cut_at = min(cut_at, token_at)
    return tail[:cut_at].rstrip()


def _parse_photos_field(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, list):
        return [str(u) for u in raw if u is not None and str(u).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(data, list):
            return [str(u) for u in data if u is not None and str(u).strip()]
        return None
    return None


def _normalize_listing_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if "photos" in row:
            row["photos"] = _parse_photos_field(row.get("photos"))
    return rows


def _coerce_ui_filters(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in raw.items() if v is not None}
    for key in ("amenities", "features", "finance"):
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, str):
            s = val.strip()
            out[key] = [s] if s else []
        elif not isinstance(val, list):
            out[key] = []
    for key in ("negated_amenities", "negated_features", "negated_finance"):
        out.setdefault(key, [])
    out.setdefault("amenities", [])
    out.setdefault("features", [])
    out.setdefault("finance", [])
    city = out.get("city")
    if isinstance(city, str) and not city.strip():
        out.pop("city", None)
    elif isinstance(city, str):
        out["city"] = city.strip()
    return out


def _ok(
    listings: list[dict[str, Any]],
    filters: dict[str, Any],
    *,
    where_clause: str,
    params: list[Any],
    mode: str,
    limit: int,
    offset: int,
    total: int,
) -> dict[str, Any]:
    return {
        "listings": listings,
        "filter": filters,
        "where_clause": where_clause,
        "params": params,
        "mode": mode,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
        },
    }


class QuerySearcher:
    """NL query → filters → listings.

    1. Rule-based parser (numbers, city, catalog amenity / feature / finance).
    2. If no catalog hits, leftover LLM for paraphrases / holistic setting text.
    3. Catalog SQL is hard filters. Leftover text ranks remarks (optionally inside SQL candidates).
       Empty leftover (and no catalog hit) falls back to ranking the full user query.
    """

    def __init__(
        self,
        pool,
        *,
        parser: QueryParser | None = None,
        semantic: SemanticSearcher | None = None,
        leftover_llm: SemanticTextLLM | None = None,
        cities_path: str = CITY_PATH,
        stats_path: str = STATS_PATH,
        table: str = "rets_property",
        max_semantic_window: int = MAX_SEMANTIC_WINDOW,
        enable_llm: bool = True,
    ):
        self.pool = pool
        self.parser = parser or QueryParser()
        self.semantic = semantic or SemanticSearcher()
        self.leftover_llm = leftover_llm if leftover_llm is not None else (
            SemanticTextLLM() if enable_llm else None
        )
        self.cities_path = cities_path
        self.stats_path = stats_path
        self.table = table
        self.max_semantic_window = max_semantic_window

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return _error("Query cannot be empty.")

        safe_limit = min(MAX_PAGE_LIMIT, max(1, int(limit)))
        safe_offset = max(0, int(offset))

        filters = self.parser.parse(q)
        validator = SchemaValidator(filters, self.cities_path, self.stats_path)
        valid, errors = validator.validate_query()
        if not valid:
            return _error(
                "Query validation failed.",
                code="validation_error",
                details=errors,
            )

        leftover = self._leftover_text(q, filters)
        if self.leftover_llm is None:
            rank_text = leftover
        else:
            rank_text = ranking_query(q, leftover, filters)
        if leftover:
            filters = {**filters, "semantic_text": leftover}
        elif rank_text:
            filters = {**filters, "semantic_text": rank_text}

        if has_catalog_terms(filters) or not rank_text:
            return self._sql_page(filters, safe_limit, safe_offset)

        if _has_structured(filters):
            return self._hybrid_page(filters, rank_text, safe_limit, safe_offset)
        return self._semantic_page(filters, rank_text, safe_limit, safe_offset)

    def search_filters(
        self,
        filters: dict[str, Any],
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """UI-edited filter dict → SQL only (no leftover LLM / remarks ranker)."""
        safe_limit = min(MAX_PAGE_LIMIT, max(1, int(limit)))
        safe_offset = max(0, int(offset))
        filters = _coerce_ui_filters(filters)
        validator = SchemaValidator(filters, self.cities_path, self.stats_path)
        valid, errors = validator.validate_query()
        if not valid:
            return _error(
                "Filter validation failed.",
                code="validation_error",
                details=errors,
            )
        return self._sql_page(filters, safe_limit, safe_offset)

    def _leftover_text(self, query: str, filters: dict) -> str:
        if self.leftover_llm is None:
            return ""
        if query.strip().lower() == "browse":
            return ""
        try:
            return self.leftover_llm.extract_if_unmatched(query, filters)
        except Exception:
            return ""

    def _sql_page(self, filters: dict, limit: int, offset: int) -> dict[str, Any]:
        sql, params = self.parser.to_sql(
            filters=filters,
            table=self.table,
            limit=limit,
            offset=offset,
            order_by="id DESC",
        )
        where = _where_clause(sql)
        if not where:
            return _error("Could not build SQL where clause.", code="internal_error")

        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, _coerce_mysql_params(params))
            rows = cursor.fetchall()
            cursor.close()
            _normalize_listing_rows(rows)
            total = self._count(conn, where, params)
        finally:
            conn.close()
        return _ok(
            rows,
            filters,
            where_clause=where,
            params=params,
            mode="sql",
            limit=limit,
            offset=offset,
            total=total,
        )

    def _semantic_page(
        self,
        filters: dict,
        semantic_text: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        top_k = min(self.max_semantic_window, offset + limit)
        _, ranked_ids, _ = self.semantic.search_hybrid(semantic_text, top_k=top_k)
        ranked_ids = [lid for lid, _ in ranked_ids]
        return self._page_ranked_ids(
            filters,
            ranked_ids,
            limit,
            offset,
            mode="semantic",
            where_clause="",
            params=[],
        )

    def _hybrid_page(
        self,
        filters: dict,
        semantic_text: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        candidate_sql, candidate_params = self.parser.to_sql(
            filters=filters,
            table=self.table,
            limit=self.max_semantic_window,
            offset=0,
            order_by="id DESC",
        )
        where = _where_clause(candidate_sql)
        if not where:
            return _error("Could not build SQL where clause.", code="internal_error")

        conn = self.pool.get_connection()
        try:
            candidate_ids = self._ids_for_where(conn, where, candidate_params)
        finally:
            conn.close()

        top_k = min(self.max_semantic_window, offset + limit)
        _, ranked_ids, _ = self.semantic.search_hybrid_for_listings(
            semantic_text,
            candidate_ids,
            top_k=top_k,
        )
        ranked_ids = [lid for lid, _ in ranked_ids]
        return self._page_ranked_ids(
            filters,
            ranked_ids,
            limit,
            offset,
            mode="hybrid",
            where_clause=where,
            params=candidate_params,
        )

    def _page_ranked_ids(
        self,
        filters: dict,
        ranked_ids: list[int],
        limit: int,
        offset: int,
        *,
        mode: str,
        where_clause: str,
        params: list[Any],
    ) -> dict[str, Any]:
        total = len(ranked_ids)
        paged_ids = ranked_ids[offset : offset + limit]
        if not paged_ids:
            return _error("No matching listing found.", code="not_found")
        conn = self.pool.get_connection()
        try:
            rows = self._fetch_rows_by_ids(conn, paged_ids)
        finally:
            conn.close()
        return _ok(
            rows,
            filters,
            where_clause=where_clause,
            params=params,
            mode=mode,
            limit=limit,
            offset=offset,
            total=total,
        )

    def _ids_for_where(self, conn, where_clause: str, params: list[Any]) -> list[int]:
        query = f"SELECT id FROM {self.table} WHERE {where_clause}"
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, _coerce_mysql_params(params))
        rows = cursor.fetchall()
        cursor.close()
        return [int(row["id"]) for row in rows]

    def _count(self, conn, where_clause: str, params: list[Any]) -> int:
        query = f"SELECT COUNT(*) AS total FROM {self.table} WHERE {where_clause}"
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, _coerce_mysql_params(params))
        row = cursor.fetchone()
        cursor.close()
        return int((row or {}).get("total", 0))

    def _fetch_rows_by_ids(self, conn, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        sql = LISTING_SELECT.format(
            table=self.table,
            where_clause=f"id IN ({placeholders})",
            order_by="id DESC",
            limit=max(1, len(ids)),
            offset=0,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, _coerce_mysql_params(ids))
        rows = cursor.fetchall()
        cursor.close()
        by_id = {int(row["id"]): row for row in rows}
        ordered = [by_id[i] for i in ids if i in by_id]
        return _normalize_listing_rows(ordered)
