"""LLM leftover extractor for remarks search.

Rule-based ``QueryParser`` still owns numbers, city, and named amenity / interior
feature / finance terms. This extractor still runs when those catalog terms were
found, because the same query can also describe atmosphere or setting. Keep at most:

  (A) amenity/feature paraphrases the term matcher will miss
      (e.g. lots of natural light)
  (B) holistic property/setting asks (quiet and cozy, next to the beach, great view)

Filler (no rush, just browsing), explicit catalog *names*, and negations stay out.
Keep paraphrases even when they mean a catalog amenity the regex missed
(securely leave my car, room to store clothes, a place to wash clothes).
If neither class is present, ``semantic_text`` is "". With catalog hits, that
empty string stays empty. With no catalog hit, the searcher falls back to
ranking with the full user query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from text_cleaning import TextCleaner

# Positive or negated hits mean the regex already handled catalog terms.
_CATALOG_KEYS = (
    "amenities",
    "features",
    "finance",
    "negated_amenities",
    "negated_features",
    "negated_finance",
)

SYSTEM_PROMPT = """
You extract a short semantic_text for real-estate remark search.

A rule-based matcher already covers named catalog amenities, interior features,
and finance terms (pool, gym, garage, hardwood, fireplace, FHA, VA, cash, ...).
Those named terms may already be in the query. Still extract leftover A/B text
that sits beside them, and drop the catalog names themselves.
Your job is to keep the leftover A/B ask so remarks search can still use it.

Your only job is semantic_text. Default is "".
Do not answer the user. Do not invent listings. Output must match the JSON schema.

============================================================
SEMANTIC TEXT — keep a fragment only if it is (A) or (B)
============================================================

(A) Implicit amenity / house-feature paraphrase
    The buyer describes a facility or interior quality without using a catalog name.
    Keep the buyer's wording even if you know it MEANS a catalog amenity.
    Named catalog terms are already handled elsewhere. Leftover is the remarks
    signal for wording that was not taken as a catalog name.
    Keep: "lots of natural light", "open floor plan", "updated throughout",
          "chef's kitchen", "indoor-outdoor living", "sun-filled rooms",
          "securely leave my car", "enough area to keep clothes out of sight",
          "a dedicated area for washing and drying clothes"
    Drop only if they used an explicit catalog name: pool, gym, garage, parking,
          hardwood, fireplace, carpet, basement, HOA, elevator, doorman, patio,
          yard, washer, dryer, laundry, storage, central air, FHA, VA, cash,
          USDA, conventional.
    Do not rewrite a paraphrase into those catalog names.

(B) Holistic property / setting requirement
    A vibe, setting, or location-quality of the home as a whole.
    Keep: "quiet and cozy", "next to the beach", "great view",
          "walkable to cafes", "tree-lined street", "family-friendly neighborhood"

Never include:
- Filler / search intent: "just browsing", "no rush", "casual look",
  "I'm looking for", "show me", "pls", "if possible", "flex on", "loosely"
- Bare property type: house, condo, townhouse, apartment, SFR, home, rental
- Explicit catalog names and finance terms, even next to a vibe word
- Negations: "no carpet", "without pool", "no HOA", "prefer no basement"
  (embeddings cannot encode NOT; the UI lets the user pick those)
- Numbers, price, sqft, city names

How to write it:
- Copy the buyer's wording for kept fragments. Do not paraphrase or add requirements.
- Keep commas, "and", and "with" that connect two kept fragments so the output is a
  readable phrase (good for embeddings). Never smash fragments into a word pile.
  Bad:  "lots of natural light open floor plan"
  Good: "lots of natural light and an open floor plan"
  Good: "chef's kitchen, updated throughout"
- Only strip glue that belonged to a dropped span
  ("3 bedroom house with lots of natural light" → "lots of natural light").
- If nothing qualifies, return "". Empty means there is no leftover remarks signal.

============================================================
EXAMPLES
============================================================
User: "3 bedroom 2 bathroom house in Chicago under $500000 with a quiet street"
{"semantic_text": "quiet street"}

User: "2+ bedroom Naperville, pool and gym, no carpet"
{"semantic_text": ""}

User: "pool and gym, quiet and cozy near the beach"
{"semantic_text": "quiet and cozy near the beach"}

User: "hardwood floors and a fireplace, family-friendly neighborhood with a tree-lined street"
{"semantic_text": "family-friendly neighborhood with a tree-lined street"}

User: "around 3 bedrooms, roughly $600000, Evanston, walkable to cafes"
{"semantic_text": "walkable to cafes"}

User: "looking for a cozy sun-filled home near parks"
{"semantic_text": "cozy sun-filled home near parks"}

User: "4 bedrooms"
{"semantic_text": ""}

User: "Chi anything with 2 bedroom under $500000? just browsing, no rush pls"
{"semantic_text": ""}

User: "lots of natural light and an open floor plan"
{"semantic_text": "lots of natural light and an open floor plan"}

User: "chef's kitchen, updated throughout"
{"semantic_text": "chef's kitchen, updated throughout"}

User: "3 bedroom house with lots of natural light, quiet and cozy, next to the beach"
{"semantic_text": "lots of natural light, quiet and cozy, next to the beach"}

User: "2-bedroom condo in San Francisco under $1.4 million, where I can securely leave my car"
{"semantic_text": "securely leave my car"}

User: "enough area to keep clothes, luggage, and household items out of sight"
{"semantic_text": "enough area to keep clothes, luggage, and household items out of sight"}

User: "dedicated area for washing and drying clothes rather than having the machines in the kitchen"
{"semantic_text": "dedicated area for washing and drying clothes rather than having the machines in the kitchen"}
""".strip()

PARSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semantic_text": {
            "type": "string",
            "description": (
                "Buyer wording for (A) amenity/feature paraphrases (even if they "
                "mean a catalog amenity the matcher missed) or (B) holistic "
                "setting. Empty string if neither. Do not rewrite into catalog names."
            ),
        }
    },
    "required": ["semantic_text"],
    "additionalProperties": False,
}


def has_catalog_terms(filters: dict | None) -> bool:
    """True if rule-base already extracted amenity, interior feature, or finance."""
    if not filters:
        return False
    return any(filters.get(key) for key in _CATALOG_KEYS)


def ranking_query(query: str, leftover: str, filters: dict | None = None) -> str:
    """Text used for remarks ranking: leftover, else the full query.

    Empty leftover with catalog hits stays empty (SQL already covers those terms).
    """
    text = (leftover or "").strip()
    if text:
        return text
    if has_catalog_terms(filters):
        return ""
    q = (query or "").strip()
    if not q or q.lower() == "browse":
        return ""
    return q


class SemanticTextLLM:
    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        client_kwargs: dict[str, Any] = {"timeout": 60.0}
        if api_key:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)
        self.normalizer = TextCleaner()

    def extract_if_unmatched(self, query: str, filters: dict | None = None) -> str:
        """Extract A/B leftover even when catalog terms were already extracted.

        ``filters`` is accepted for callers that already parsed the query.
        Catalog names stay out of the prompt output; an empty string means
        the query had no leftover remarks signal.
        """
        _ = filters
        return self.extract(query)

    def extract(self, query: str) -> str:
        """Call the LLM for A/B leftover text. Empty string if neither is present."""
        q = (query or "").strip()
        if not q:
            return ""
        q = self.normalizer.clean_text(q)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_text_extract",
                    "strict": True,
                    "schema": PARSE_JSON_SCHEMA,
                },
            },
            temperature=self.temperature,
        )
        raw = response.choices[0].message.content
        if not raw:
            return ""
        parsed = json.loads(raw)
        text = parsed.get("semantic_text") or ""
        if not isinstance(text, str):
            text = str(text)
        return " ".join(text.split())
