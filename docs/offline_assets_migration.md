# Offline Assets

This repo keeps runtime API code, evaluation, tests, and AWS infra.

Local datasets are gitignored (`data/`, `notebooks/`). Runtime artifacts (`amenities.json`, `features.json`, `finance.json`, `distinct_cities.csv`, `stats.json`, `all_listings_cleaned.csv`, optional `index.faiss`) are loaded from S3 via `S3_BUCKET` + `S3_PREFIX`.

ETL builders, taxonomy generators, and research notebooks were removed from this repo. Rebuild those artifacts in a sibling data workspace if needed, then publish the processed files to S3.
