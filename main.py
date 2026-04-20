from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from typing import Any


import sys
from pathlib import Path

from scripts.query_parser import QueryParser
from scripts.schema_validator import SchemaValidator
from scripts.prepare_property_detail import process_result
from scripts.semantic_searcher import SemanticSearcher
import pandas as pd
import json
from mysql.connector.pooling import MySQLConnectionPool

ROOT = Path(__file__).resolve().parent

parser = QueryParser()
searcher = SemanticSearcher()
cities_path = str(ROOT / "data" / "processed" / "distinct_cities.csv")
stats_path = str(ROOT / "data" / "processed" / "stats.json")

pool = MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    host="localhost",
    user="root",
    password="root",
    database="real_estate"
)


def _coerce_mysql_params(params: Any) -> Any:
    """mysql-connector C API does not accept numpy scalar types."""
    if params is None:
        return None
    if isinstance(params, (list, tuple)):
        return [_coerce_mysql_scalar(p) for p in params]
    return _coerce_mysql_scalar(params)


def _coerce_mysql_scalar(p: Any) -> Any:
    if isinstance(p, np.generic):
        return p.item()
    return p


def _parse_photos_field(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, list):
        return [str(u) for u in raw if u is not None and str(u).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            data = json.loads(s)
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

def _get_where_clause(sql: str) -> str | None:
    lower = sql.lower()
    idx = lower.find(" where ")
    if idx == -1:
        return None
    start = idx + len(" where ")
    tail = sql[start:]
    limit_at = lower[start:].find(" limit ")
    if limit_at != -1:
        tail = tail[:limit_at].rstrip()
    return tail

def _get_lid_for_semantic(conn,params, where_clause):
    query = f"""
            SELECT id
            FROM rets_property WHERE {where_clause}"""
    cursor = conn.cursor(dictionary = True)
    cursor.execute(query, _coerce_mysql_params(params))
    rows = cursor.fetchall()
    cursor.close()
    return [row["id"] for row in rows]

def process_query(query: str):
    q = query.strip()
    if not q:
        return "Can not be empty"

    conn = pool.get_connection()
    filter = parser.parse(query)
    validator = SchemaValidator(filter, cities_path, stats_path)
    valid, errors = validator.validate_query()
    if not valid:
        conn.close()
        return {
            "error": errors
        }

    sql, params = parser.to_sql()
    params_to_show = params
    where_clause = _get_where_clause(sql)
    mode = "regex"

    if not ({"amenities", "negated_amenities", "features", "negated_features", "finance"} & filter.keys()):
        if not ({"price_max", "price_min", "price","bedrooms", "bedrooms_min", "bedrooms_max", "bathrooms", 
                "bathrooms_min", "bathrooms_max", "bathroom_half", "sqft_min", "sqft_max", "sqft", "city"} & filter.keys()):
                results, ids, latency_ms = searcher.search_hybrid(q, top_k = 50)
                ids = [id for id, _ in ids]
                if len(ids) > 0:
                    mode = "semantic"
                    sql = f"""
                            SELECT id,
                                L_Address as address,
                                L_Zip as zipcode,
                                L_City as city,
                                L_State as state,
                                L_Keyword2 as bedrooms,
                                LM_Dec_3 as bathrooms,
                                L_SystemPrice as price,
                                LM_Int2_3 as living_area,
                                L_Remarks as remark,
                                L_Photos as photos,
                                Flooring as flooring,
                                ViewYN,
                                PoolPrivateYN,
                                AttachedGarageYN,
                                FireplaceYN,
                                HeatingYN,
                                Appliances,
                                CoolingYN,
                                GarageYN,
                                SpaYN,
                                BathroomsHalf as half_bathrooms,
                                AssociationAmenities,
                                StructureType,
                                ArchitecturalStyle,
                                Cooling,
                                Heating,
                                View,
                                FireplaceFeatures,
                                InteriorFeatures,
                                PoolFeatures,
                                CommunityFeatures,
                                SecurityFeatures,
                                SpaFeatures
                            FROM rets_property WHERE id IN ({','.join(['%s'] * len(ids))}) LIMIT 50"""
                    params = ids
                    where_clause = ""
                    params_to_show = []
                else:
                    conn.close()
                    return {
                        "error": "no matching found"
                    }
        else:
            ids = _get_lid_for_semantic(conn, params, where_clause)
            results, ids, latency_ms = searcher.search_hybrid_for_listings(q, ids, top_k = 50)
            ids = [id for id, _ in ids]
            if len(ids) > 0:
                mode = "hybrid"
                sql = f"""
                        SELECT id,
                            L_Address as address,
                            L_Zip as zipcode,
                            L_City as city,
                            L_State as state,
                            L_Keyword2 as bedrooms,
                            LM_Dec_3 as bathrooms,
                            L_SystemPrice as price,
                            LM_Int2_3 as living_area,
                            L_Remarks as remark,
                            L_Photos as photos,
                            Flooring as flooring,
                            ViewYN,
                            PoolPrivateYN,
                            AttachedGarageYN,
                            FireplaceYN,
                            HeatingYN,
                            Appliances,
                            CoolingYN,
                            GarageYN,
                            SpaYN,
                            BathroomsHalf as half_bathrooms,
                            AssociationAmenities,
                            StructureType,
                            ArchitecturalStyle,
                            Cooling,
                            Heating,
                            View,
                            FireplaceFeatures,
                            InteriorFeatures,
                            PoolFeatures,
                            CommunityFeatures,
                            SecurityFeatures,
                            SpaFeatures
                        FROM rets_property WHERE id IN ({','.join(['%s'] * len(ids))}) LIMIT 50"""
                params = ids
            else:
                conn.close()
                return {
                    "error": "no matching found"
                }

    cursor = conn.cursor(dictionary = True)
    cursor.execute(sql, _coerce_mysql_params(params))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    _normalize_listing_rows(rows)
    return {
        "listings": rows,
        "filter": filter,
        "where_clause": where_clause,
        "params": params_to_show,
        "mode": mode
    }

def _coerce_ui_filters(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in raw.items() if v is not None}
    for key in ("amenities", "features"):
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, str):
            s = val.strip()
            out[key] = [s] if s else []
        elif not isinstance(val, list):
            out[key] = []
    for key in ("negated_amenities", "negated_features"):
        out.setdefault(key, [])
    out.setdefault("amenities", [])
    out.setdefault("features", [])
    c = out.get("city")
    if isinstance(c, str) and not c.strip():
        out.pop("city", None)
    elif isinstance(c, str):
        out["city"] = c.strip()
    return out


def process_filter(filters: dict[str, Any]):
    filters = _coerce_ui_filters(filters)
    validator = SchemaValidator(filters, cities_path, stats_path)
    valid, errors = validator.validate_query()
    if not valid:
        return {"error": errors}
    sql, params = parser.to_sql(filters=filters)
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, _coerce_mysql_params(params))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    _normalize_listing_rows(rows)
    return {"listings": rows}

def process_detail(id: int):

    conn = pool.get_connection()
    query = """
            SELECT id,
                L_Address as address,
                L_Zip as zipcode,
                L_City as city,
                L_State as state,
                L_Keyword2 as bedrooms,
                LM_Dec_3 as bathrooms,
                L_SystemPrice as price,
                LM_Int2_3 as living_area,
                L_Remarks as remark,
                L_Photos as photos,
                Flooring as flooring,
                ViewYN,
                PoolPrivateYN,
                AttachedGarageYN,
                FireplaceYN,
                HeatingYN,
                Appliances,
                CoolingYN,
                GarageYN,
                SpaYN,
                BathroomsHalf as half_bathrooms,
                AssociationAmenities,
                StructureType,
                ArchitecturalStyle,
                Cooling,
                Heating,
                View,
                FireplaceFeatures,
                InteriorFeatures,
                PoolFeatures,
                CommunityFeatures,
                SecurityFeatures,
                SpaFeatures
            FROM rets_property WHERE id = %s"""
    params = [id]
    
    cursor = conn.cursor(dictionary = True)
    cursor.execute(query, _coerce_mysql_params(params))
    row = cursor.fetchone()
    result = process_result(
        row, 
        ["AssociationAmenities", "View", "PoolFeatures", "CommunityFeatures", "SecurityFeatures", "SpaFeatures",
        "flooring", "Appliances", "Cooling", "Heating", "InteriorFeatures"]
    )
    cursor.close()
    conn.close()
    if "photos" in result:
        result["photos"] = _parse_photos_field(result.get("photos"))
    return {
        "listing": result
    }


class QueryBody(BaseModel):
    searchQuery: str

class DetailQuery(BaseModel):
    listingID: int

app = FastAPI(title="real_estate_search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/query")
def run_query(body: QueryBody):
    result = process_query(body.searchQuery)
    return result

@app.post("/api/filter")
async def run_filter(request: Request) -> Any:
    body = await request.json()
    if not isinstance(body, dict):
        return {"error": ["Request body must be a JSON object"]}
    return process_filter(body)

@app.post("/api/detail")
async def run_detail(body: DetailQuery) -> Any:
    id = body.listingID
    return process_detail(id)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
