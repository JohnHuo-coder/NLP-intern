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

## Project Structure

```text
.
├── main.py                         # FastAPI app entrypoint
├── requirements.txt
├── docker-compose.yml
├── scripts/
│   ├── semantic_searcher.py        # Embedding/BM25/FAISS hybrid search
│   ├── query_parser.py             # NL query -> structured filters
│   ├── schema_validator.py         # Filter validation
│   ├── entity_extractor.py
│   ├── text_cleaning.py
│   ├── prepare_property_detail.py
│   ├── categorize_taxonomy.py
│   ├── generators/
│   │   ├── user_query_generator.py
│   │   └── generate_synthetic_remarks.py
│   └── builders/
│       ├── house_features_builder.py
│       ├── amenities_builder.py
│       ├── finance_opt_builder.py
│       ├── taxonomy_builder.py
│       └── build_from_text_col.py
├── tests/
│   ├── test_setup.py
│   ├── test_week1.py
│   └── test_week2.py
└── notebooks/
```

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

3. Install additional packages used by the current code (if not already installed):

```bash
pip install fastapi uvicorn sentence-transformers rank-bm25 faiss-cpu
```

4. Start MySQL with Docker (optional; skip if you use Railway MySQL):

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

## Run the API

From the project root:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

- `GET /health`

Core endpoints:

- `POST /api/query` with body:

```json
{
  "searchQuery": "Find me a 3 bedroom home in Irvine under $900000 with a pool"
}
```

- `POST /api/filter` with a JSON filter object (UI-style structured filters)
- `POST /api/detail` with body:

```json
{
  "listingID": 12345
}
```

## Run Tests

```bash
pytest -q
```

You can also run setup checks:

```bash
python tests/test_setup.py
```

## Notes

- `main.py` currently creates a global MySQL connection pool and a global `SemanticSearcher` instance at startup.
- The semantic index file is expected at `data/processed/index.faiss`; it is created automatically if missing.
- The API CORS config currently allows local frontend origins on port `3000`.
