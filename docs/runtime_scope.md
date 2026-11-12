# Runtime Scope (MVP Backend)

This file marks the minimum code/data that must stay in this repository for the API server to run.

## Required Runtime Code

- `main.py`
- `scripts/query_parser.py`
- `scripts/query_numeric.py`
- `scripts/query_sql.py`
- `scripts/query_searcher.py`
- `scripts/semantic_text_llm.py`
- `scripts/schema_validator.py`
- `scripts/semantic_searcher_v2.py`
- `scripts/entity_extractor.py`
- `scripts/text_cleaning.py`
- `scripts/prepare_property_detail.py`

## Required Runtime Artifacts

Under `data/processed/`:

- `amenities.json`
- `features.json`
- `finance.json`
- `distinct_cities.csv`
- `stats.json`
- `all_listings_cleaned.csv`

Optional but recommended:

- `index.faiss` (faster startup; avoids index rebuild)

## Required Runtime Config

- DB env vars (`MYSQL_URL` or `MYSQLHOST`/`MYSQLPORT`/`MYSQLUSER`/`MYSQLPASSWORD`/`MYSQLDATABASE`)
- Optional S3 vars for artifact download (`S3_BUCKET`, `S3_PREFIX`, `AWS_REGION`, `S3_PREFETCH_OPTIONAL`)
- Optional CORS allowlist (`CORS_ALLOWED_ORIGINS`)

## Also In This Repo (Not On The Request Path)

- `evaluation/` — parser stress + leftover nDCG scoring
- `tests/` — pytest for cleaner / numeric parser
- `infra/cdk/` — RDS + Fargate deploy
- `docs/`

Local ETL dumps stay out of git (`notebooks/`, large `data/` files). Small parser catalogs (`amenities.json`, `features.json`, `finance.json`, `distinct_cities.csv`, `stats.json`) are tracked so CI can run `QueryParser` tests.
