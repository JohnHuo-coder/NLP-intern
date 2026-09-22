"""Rule-based NL query → structured filters → listing SQL.

Numeric / city regex lives in ``query_numeric``. Named amenity / feature /
finance terms come from ``EntityExtractor``. Column routing lives in ``query_sql``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from entity_extractor import EntityExtractor, group_spans_by_connector
from query_numeric import fill_structured_fields
from query_sql import build_listing_sql
from text_cleaning import TextCleaner


class QueryParser:
    def __init__(self):
        self.normalizer = TextCleaner()
        self.extractor = EntityExtractor()
        self.filters = {}

    def parse(self, query: str) -> dict:
        self.filters = {}
        cleaned = self.normalizer.clean_text(query)
        fill_structured_fields(cleaned, self.filters)
        self._parse_catalog_terms(cleaned)
        return self.filters

    def _parse_catalog_terms(self, query: str) -> None:
        neg_amenity, neg_amenity_kept = self.extractor.extract_negated_amenities_features(
            query, type="amenity"
        )
        amenity_terms, amenity_kept = self.extractor.extract_amenities_features(
            query, type="amenity", neg_kept=neg_amenity_kept
        )
        if neg_amenity:
            self.filters["negated_amenities"] = neg_amenity
        if amenity_terms:
            self.filters["amenities"] = amenity_terms
            self.filters["amenity_groups"] = group_spans_by_connector(query, amenity_kept)

        neg_feature, neg_feature_kept = self.extractor.extract_negated_amenities_features(
            query, type="feature"
        )
        feature_terms, feature_kept = self.extractor.extract_amenities_features(
            query, type="feature", neg_kept=neg_feature_kept
        )
        if neg_feature:
            self.filters["negated_features"] = neg_feature
        if feature_terms:
            self.filters["features"] = feature_terms
            self.filters["feature_groups"] = group_spans_by_connector(query, feature_kept)

        neg_finance, neg_finance_kept = self.extractor.extract_negated_financing_options(query)
        finance_terms, finance_kept = self.extractor.extract_financing_options(
            query, neg_kept=neg_finance_kept
        )
        if neg_finance:
            self.filters["negated_finance"] = neg_finance
        if finance_terms:
            self.filters["finance"] = finance_terms
            self.filters["finance_groups"] = group_spans_by_connector(query, finance_kept)

    def to_sql(
        self,
        table: str = "rets_property",
        filters: dict | None = None,
        *,
        order_by: str = "id DESC",
        limit: int = 50,
        offset: int = 0,
    ):
        if filters is None:
            filters = self.filters
        return build_listing_sql(
            self.extractor,
            filters,
            table=table,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )
