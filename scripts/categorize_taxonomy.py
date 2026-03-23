"""
Load taxonomy.json, assign a category to each term via OpenAI, write enriched JSON.

Requires OPENAI_API_KEY. Install: pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

Category = Literal[
    "amenity",
    "room",
    "layout",
    "measurement",
    "condition",
    "location",
    "interior feature",
    "other",
]

_CATEGORY_FIELD = Field(
    ...,
    description="""
        Classify the given real estate phrase into ONE of the following categories:

        - 'amenity': Facilities or features that enhance living experience but are not essential structural components
        (e.g., pool, gym, parking, air conditioning, tennis courts, central heating).

        - 'room': Specific rooms or spaces within the property
        (e.g., bedroom, bathroom, living room, dining room, office).

        - 'layout': Descriptions of spatial arrangement or structure
        (e.g., open floor plan, main level, second floor, open concept).

        - 'measurement': Size, area, or numeric-related expressions
        (e.g., sq ft, square feet, lot size).

        - 'condition': Property condition, updates, or renovation status
        (e.g., newly renovated, brand new, move-in ready).

        - 'location': Geographic or accessibility-related descriptions
        (e.g., near downtown, walking distance, prime location).

        - 'interior feature': Materials, finishes, or interior design elements
        (e.g., hardwood floors, granite countertops, recessed lighting, vaulted ceiling).

        - 'other': Marketing phrases, filler language, or irrelevant descriptions
        (e.g., must see, beautiful home, great opportunity).

        Return ONLY one category label.
        """,
)


class phrase_type(BaseModel):
    """categorize the phrase into the most relevant category"""

    category: Category = _CATEGORY_FIELD


class TermCategoryItem(BaseModel):
    id: str
    category: Category


class BatchCategoryResponse(BaseModel):
    class Config:
        extra = "ignore"

    items: List[TermCategoryItem]


SYSTEM_PROMPT = """You classify short phrases (mostly two-word bigrams) from U.S. real estate listing remarks.
Assign each phrase exactly ONE category:

- amenity: Facilities or features that enhance living (pool, gym, parking, air conditioning, tennis courts, central heating).
- room: Specific rooms or spaces (bedroom, bathroom, living room, dining room, office).
- layout: Spatial arrangement (open floor plan, main level, second floor, open concept).
- measurement: Size or area expressions (sq ft, square feet, lot size).
- condition: Condition or renovation status (newly renovated, brand new, move-in ready).
- location: Geography or accessibility (near downtown, walking distance, prime location, conveniently located).
- interior feature: Materials, finishes, lighting (hardwood floors, granite countertops, recessed lighting).
- other: Marketing filler or vague phrasing (must see, beautiful home, home offers, highlights include).

Use each id string exactly as given. Every id in the user message must appear once in your JSON."""


def _chunks(items: List[dict], size: int) -> List[List[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def categorize_batch(
    client: OpenAI,
    model: str,
    batch: List[dict],
    max_retries: int = 3,
) -> Dict[str, str]:
    """Return mapping term id -> category for this batch."""
    lines = "\n".join(f'{t["id"]} | {t["term"]}' for t in batch)
    user_content = (
        "Classify each line (id | phrase). Respond with JSON only.\n\n"
        f"{lines}\n\n"
        'Format: {"items": [{"id": "T0001", "category": "room"}, ...]}\n'
        "Categories must be exactly one of: amenity, room, layout, measurement, condition, "
        "location, interior feature, other."
    )
    expected_ids = {t["id"] for t in batch}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = resp.choices[0].message.content
            if not raw:
                raise ValueError("empty model response")
            data = BatchCategoryResponse.parse_raw(raw)
            out: Dict[str, str] = {}
            for item in data.items:
                out[item.id] = item.category
            missing = expected_ids - set(out.keys())
            extra = set(out.keys()) - expected_ids
            if missing or extra:
                raise ValueError(f"missing ids {missing}, extra ids {extra}")
            return out
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def run(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    model: str,
    sleep_s: float,
    limit: Optional[int],
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it in your environment before running this script."
        )

    with input_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    terms: List[dict] = list(payload.get("terms") or [])
    if not terms:
        raise SystemExit(f"No terms found in {input_path}")
    if limit is not None:
        terms = terms[:limit]
        payload = {**payload, "terms": terms}

    client = OpenAI()
    id_to_category: Dict[str, str] = {}
    batches = _chunks(terms, batch_size)

    for i, batch in enumerate(batches, start=1):
        mapping = categorize_batch(client, model, batch)
        id_to_category.update(mapping)
        print(f"batch {i}/{len(batches)} ({len(batch)} terms) ok")
        if sleep_s > 0 and i < len(batches):
            time.sleep(sleep_s)

    for t in terms:
        tid = t["id"]
        t["category"] = id_to_category.get(tid, "other")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {output_path} ({len(terms)} terms)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_in = root / "data" / "processed" / "taxonomy.json"
    default_out = root / "data" / "processed" / "taxonomy_categorized.json"

    p = argparse.ArgumentParser(description="Add category field to each taxonomy term.")
    p.add_argument("--input", type=Path, default=default_in, help="Input taxonomy JSON")
    p.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Output JSON (includes original fields plus category)",
    )
    p.add_argument("--batch-size", type=int, default=20, help="Terms per API request")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between batches (rate limiting)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only classify the first N terms (for testing)",
    )
    args = p.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        model=args.model,
        sleep_s=args.sleep,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
