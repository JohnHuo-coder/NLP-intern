from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import boto3
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mysql.connector.pooling import MySQLConnectionPool
import numpy as np
from pydantic import BaseModel, Field

from scripts.prepare_property_detail import process_result
from scripts.query_searcher import QuerySearcher

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
REQUIRED_PROCESSED_FILES = (
    "amenities.json",
    "features.json",
    "finance.json",
    "distinct_cities.csv",
    "stats.json",
    "all_listings_cleaned.csv",
)
OPTIONAL_PROCESSED_FILES = ("index.faiss",)

DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 50

LISTING_SELECT_FIELDS = """
id,
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
""".strip()


@dataclass
class RuntimeState:
    query_searcher: QuerySearcher


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [value.strip() for value in raw.split(",") if value.strip()]
    return origins or ["http://localhost:3000"]


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


def _ensure_nltk_data() -> None:
    """Download NLTK corpora/tokenizers if missing (needed on fresh environments)."""
    import nltk

    bundles: tuple[tuple[str, str], ...] = (
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
    )
    for resource_path, download_name in bundles:
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            print(f"[startup] NLTK: downloading {download_name!r} ...")
            try:
                nltk.download(download_name, quiet=True)
            except Exception:
                pass

    # NLTK 3.8.2+ can need punkt_tab for word_tokenize.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except (LookupError, OSError):
        try:
            print("[startup] NLTK: downloading 'punkt_tab' ...")
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def _ensure_processed_data_from_s3() -> None:
    """
    Ensure required files exist in data/processed.
    If missing, download them from S3.
    """
    missing = [name for name in REQUIRED_PROCESSED_FILES if not (PROCESSED_DIR / name).exists()]
    if not missing:
        print(
            f"[startup] data/processed: all {len(REQUIRED_PROCESSED_FILES)} required files present "
            f"under {PROCESSED_DIR}; skip S3 download."
        )
        return

    print(
        f"[startup] data/processed: missing {len(missing)} file(s): {missing}; "
        "will fetch from S3 if configured."
    )

    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "Missing required processed data files and S3_BUCKET is not set. "
            f"Missing files: {missing}"
        )

    prefix = os.getenv("S3_PREFIX", "data/processed").strip("/\\")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    s3_client = boto3.client("s3", region_name=region)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for filename in missing:
        object_key = f"{prefix}/{filename}" if prefix else filename
        local_path = PROCESSED_DIR / filename
        print(f"[startup] Downloading s3://{bucket}/{object_key} ...")
        s3_client.download_file(bucket, object_key, str(local_path))

    prefetch_optional = _parse_bool_env("S3_PREFETCH_OPTIONAL", True)
    if not prefetch_optional:
        return

    for filename in OPTIONAL_PROCESSED_FILES:
        local_path = PROCESSED_DIR / filename
        if local_path.exists():
            continue
        object_key = f"{prefix}/{filename}" if prefix else filename
        try:
            print(f"[startup] Prefetching optional file s3://{bucket}/{object_key} ...")
            s3_client.download_file(bucket, object_key, str(local_path))
        except Exception:
            print(f"[startup] Optional file not downloaded: s3://{bucket}/{object_key}")


def _get_mysql_config() -> dict[str, Any]:
    mysql_url = (
        os.getenv("MYSQL_URL")
        or os.getenv("MYSQL_PUBLIC_URL")
        or os.getenv("DATABASE_URL")
    )

    config: dict[str, Any] = {
        "host": os.getenv("MYSQLHOST", "localhost"),
        "port": int(os.getenv("MYSQLPORT", "3306")),
        "user": os.getenv("MYSQLUSER", "root"),
        "password": os.getenv("MYSQLPASSWORD", "root"),
        "database": os.getenv("MYSQLDATABASE", "real_estate"),
    }

    if mysql_url:
        parsed = urlparse(mysql_url)
        if parsed.scheme and not parsed.scheme.startswith("mysql"):
            raise ValueError(f"Unsupported DB scheme in URL: {parsed.scheme}")
        if parsed.hostname:
            config["host"] = parsed.hostname
        if parsed.port:
            config["port"] = parsed.port
        if parsed.username:
            config["user"] = unquote(parsed.username)
        if parsed.password:
            config["password"] = unquote(parsed.password)
        if parsed.path and parsed.path != "/":
            config["database"] = parsed.path.lstrip("/")

    # MySQL 8 caching_sha2_password often requires TLS.
    if _parse_bool_env("MYSQL_SSL_DISABLED", False):
        config["ssl_disabled"] = True

    return config


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


def _runtime_from_request(request: Request) -> RuntimeState | None:
    return getattr(request.app.state, "runtime", None)


def process_query(
    runtime: RuntimeState,
    query: str,
    *,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    return runtime.query_searcher.search(query, limit=limit, offset=offset)


def process_filter(
    runtime: RuntimeState,
    filters: dict[str, Any],
    *,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    return runtime.query_searcher.search_filters(filters, limit=limit, offset=offset)


def process_detail(runtime: RuntimeState, listing_id: int) -> dict[str, Any]:
    conn = runtime.query_searcher.pool.get_connection()
    query = f"""
            SELECT {LISTING_SELECT_FIELDS}
            FROM rets_property WHERE id = %s"""
    params = [listing_id]

    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, _coerce_mysql_params(params))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row is None:
        return _error("Listing not found.", code="not_found")

    result = process_result(
        row,
        [
            "AssociationAmenities",
            "View",
            "PoolFeatures",
            "CommunityFeatures",
            "SecurityFeatures",
            "SpaFeatures",
            "flooring",
            "Appliances",
            "Cooling",
            "Heating",
            "InteriorFeatures",
        ],
    )
    if "photos" in result:
        result["photos"] = _parse_photos_field(result.get("photos"))
    return {
        "listing": result
    }


class QueryBody(BaseModel):
    searchQuery: str
    limit: int = Field(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(default=0, ge=0)


class DetailQuery(BaseModel):
    listingID: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = None
    app.state.startup_error = None
    try:
        _ensure_processed_data_from_s3()
        _ensure_nltk_data()
        pool = MySQLConnectionPool(
            pool_name=os.getenv("MYSQL_POOL_NAME", "mypool"),
            pool_size=int(os.getenv("MYSQL_POOL_SIZE", "5")),
            **_get_mysql_config(),
        )
        app.state.runtime = RuntimeState(
            query_searcher=QuerySearcher(
                pool,
                cities_path=str(PROCESSED_DIR / "distinct_cities.csv"),
                stats_path=str(PROCESSED_DIR / "stats.json"),
            ),
        )
    except Exception as exc:  # pragma: no cover - startup diagnostics
        app.state.startup_error = str(exc)
        print(f"[startup] failed: {exc}")
    yield


app = FastAPI(title="real_estate_search API", lifespan=lifespan)

allowed_origins = _parse_allowed_origins()
allow_credentials = _parse_bool_env("CORS_ALLOW_CREDENTIALS", True)
if "*" in allowed_origins and allow_credentials:
    # Wildcard with credentials breaks browser CORS behavior.
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/query")
def run_query(body: QueryBody, request: Request):
    runtime = _runtime_from_request(request)
    if runtime is None:
        return _error(
            "Service is not ready yet.",
            code="service_unavailable",
            details=[getattr(request.app.state, "startup_error", "startup in progress")],
        )
    return process_query(
        runtime,
        body.searchQuery,
        limit=body.limit,
        offset=body.offset,
    )


@app.post("/api/filter")
async def run_filter(request: Request) -> Any:
    runtime = _runtime_from_request(request)
    if runtime is None:
        return _error(
            "Service is not ready yet.",
            code="service_unavailable",
            details=[getattr(request.app.state, "startup_error", "startup in progress")],
        )
    body = await request.json()
    if not isinstance(body, dict):
        return _error("Request body must be a JSON object.")

    try:
        limit = int(body.get("limit", DEFAULT_PAGE_LIMIT))
        offset = int(body.get("offset", 0))
    except (TypeError, ValueError):
        return _error("limit and offset must be integers.", code="validation_error")

    if "filters" in body:
        filters = body["filters"]
        if not isinstance(filters, dict):
            return _error("filters must be a JSON object.", code="validation_error")
    else:
        filters = {
            key: value
            for key, value in body.items()
            if key not in {"limit", "offset"}
        }
    return process_filter(runtime, filters, limit=limit, offset=offset)


@app.post("/api/detail")
async def run_detail(body: DetailQuery, request: Request) -> Any:
    runtime = _runtime_from_request(request)
    if runtime is None:
        return _error(
            "Service is not ready yet.",
            code="service_unavailable",
            details=[getattr(request.app.state, "startup_error", "startup in progress")],
        )
    return process_detail(runtime, body.listingID)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, Any]:
    runtime = _runtime_from_request(request)
    startup_error = getattr(request.app.state, "startup_error", None)
    if startup_error:
        return _error(
            "Service startup failed.",
            code="startup_failed",
            details=[startup_error],
        )
    if runtime is None:
        return _error("Service is still starting.", code="service_unavailable")
    missing = [
        name for name in REQUIRED_PROCESSED_FILES if not (PROCESSED_DIR / name).exists()
    ]
    if missing:
        return _error(
            "Required data artifacts are missing.",
            code="missing_artifacts",
            details=missing,
        )
    try:
        conn = runtime.query_searcher.pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        return _error(
            "Database connection is not ready.",
            code="db_unavailable",
            details=[str(exc)],
        )
    if runtime.query_searcher.semantic.index is None:
        return _error("Search index is not ready.", code="search_unavailable")
    return {"status": "ok"}
