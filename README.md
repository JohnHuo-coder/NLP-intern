# NLP Internship - Real Estate Query Understanding and Search

This project builds a natural-language real estate search backend.
It parses user queries into structured filters, validates them, and retrieves listings with a hybrid semantic + keyword search pipeline.

## What This Project Does

- Parses free-form housing queries (price, beds, baths, sqft, city, amenities, features).
- Validates extracted filters against schema constraints and data stats.
- Runs SQL filtering on a MySQL `real_estate` database.
- Uses hybrid retrieval (Sentence Transformers embeddings + BM25 + FAISS) for semantic matching.
- Exposes API endpoints with FastAPI for query search, filter search, and listing details.

## Tech Stack

- Python 3.11+
- FastAPI
- MySQL (Docker compose is included)
- pandas, numpy
- sentence-transformers
- FAISS
- rank-bm25
- pytest

## Project Structure (Runtime)

```text
.
├── main.py                         # FastAPI app entrypoint
├── requirements.txt
├── Dockerfile                      # Runtime image for ECS Fargate
├── docker-compose.yml
├── infra/cdk/                      # RDS + optional Fargate CDK stacks
├── scripts/                        # Runtime parser / search
├── evaluation/                     # Offline eval (not on the API path)
│   ├── queries/                    # Held-out nDCG queries
│   ├── results/                    # HTML/JSON reports
│   ├── compare_llm_vs_raw_semantic.py
│   ├── eval_llm_vs_raw_ndcg.py
│   └── check_ndcg_queries.py
├── docs/
│   ├── runtime_scope.md
│   └── offline_assets_migration.md
└── data/processed/
```

Offline ETL / notebook assets are tracked separately from runtime scope. See:

- `docs/runtime_scope.md`
- `docs/offline_assets_migration.md`

## Prerequisites

1. Python 3.11 or newer
2. MySQL available locally or from a managed provider (for example Railway)
3. Required data artifacts under `data/processed/` (for parser and semantic search), such as:
   - `all_listings_cleaned.csv`
   - `distinct_cities.csv`
   - `stats.json`
   - taxonomy/features/amenities JSON files

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start MySQL with Docker (optional; skip if you use Railway MySQL):

```bash
docker-compose up -d
```

## Use Railway MySQL

Set environment variables before starting the API.

Option A (Railway standard variables):

```bash
MYSQLHOST=your-host
MYSQLPORT=3306
MYSQLUSER=your-user
MYSQLPASSWORD=your-password
MYSQLDATABASE=railway
```

Option B (single URL):

```bash
MYSQL_URL=mysql://user:password@host:3306/database
```

Optional pool/SSL settings:

```bash
MYSQL_POOL_SIZE=5
MYSQL_POOL_NAME=mypool
MYSQL_SSL_DISABLED=true
```

If your provider requires SSL, set `MYSQL_SSL_DISABLED=false` and provide the extra SSL parameters supported by `mysql-connector-python`.

## Load `data/processed` from S3 (Railway)

The backend checks `data/processed` at startup. If required files are missing, it downloads them from S3 before creating `QueryParser` and `SemanticSearcher`.

Set these environment variables in Railway:

```bash
S3_BUCKET=your-bucket-name
S3_PREFIX=data/processed
AWS_REGION=us-east-1
```

Optional:

```bash
S3_PREFETCH_OPTIONAL=true
```

Required files expected in the S3 prefix:

- `amenities.json`
- `features.json`
- `finance.json`
- `distinct_cities.csv`
- `stats.json`
- `all_listings_cleaned.csv`

Optional (recommended):

- `index.faiss` (to avoid rebuilding index on cold start)

## Run the API

From the project root:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health checks:

- `GET /health`
- `GET /ready`

Core endpoints:

- `POST /api/query` with body:

```json
{
  "searchQuery": "Find me a 3 bedroom home in Irvine under $900000 with a pool",
  "limit": 20,
  "offset": 0
}
```

- `POST /api/filter` with a JSON filter object (or `{ "filters": {...}, "limit": 20, "offset": 0 }`)
- `POST /api/detail` with body:

```json
{
  "listingID": 12345
}
```

## Run Tests

```bash
pip install -r requirements-test.txt
pytest -q tests/test_text_cleaning_and_query_parse.py tests/test_semantic_text_llm.py
```

`tests/test_setup.py` talks to local Docker MySQL and is skipped by default (`pytest.ini`). CI uses the same two modules. GitHub Actions on `main` also `cdk synth`s; Fargate deploy/destroy is a **manual** workflow (see `infra/cdk/README.md` section 8).

## Deploy on AWS (ECS Fargate)

See `infra/cdk/README.md` for RDS + Fargate CDK deploy and GitHub Actions CD. Local image smoke test:

```bash
docker build -t real-estate-api .
```

## Notes

- Runtime dependencies include `langchain-core` and `langchain-text-splitters` for chunk splitting.
- The semantic index file is expected at `data/processed/index.faiss`; it is created automatically if missing.
- CORS origins are controlled by `CORS_ALLOWED_ORIGINS` (comma-separated).
