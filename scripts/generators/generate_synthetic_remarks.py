"""
Generate synthetic listing remarks with gold labels (beds, baths, price, sqft, amenity spans).

Remarks are passed through ``TextCleaner.clean_text`` (``scripts/text_cleaning.py``): no
abbreviations, normalized prices (no commas; k/m expanded), and measurements as
``<n> square feet``.

Writes JSON to data/processed/synthetic_remarks_labeled.json by default.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from text_cleaning import TextCleaner

LabelScalar = Union[int, float, str]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = ROOT / "data" / "processed" / "taxonomy_categorized.json"
DEFAULT_OUT = ROOT / "data" / "processed" / "synthetic_remarks_labeled.json"

EXTRA_AMENITIES = (
    "pool",
    "spa",
    "gym",
    "garage",
    "balcony",
    "patio",
    "elevator",
    "concierge",
    "tennis court",
    "golf course",
    "clubhouse",
    "sauna",
    "jacuzzi",
)


def load_amenity_phrases(taxonomy_path: Path, cleaner: TextCleaner) -> List[str]:
    """Load amenity phrases that are unchanged by ``clean_text`` when used alone (no abbrev conflicts)."""
    with taxonomy_path.open(encoding="utf-8") as f:
        data = json.load(f)
    terms = [
        t["term"].lower()
        for t in data.get("terms", [])
        if t.get("category") == "amenity"
    ]
    seen = set()
    out = []
    for p in sorted(set(terms) | set(EXTRA_AMENITIES), key=len, reverse=True):
        if p in seen:
            continue
        seen.add(p)
        if not _amenity_stable_under_clean(cleaner, p):
            continue
        out.append(p)
    return out


def _amenity_stable_under_clean(cleaner: TextCleaner, phrase: str) -> bool:
    """Reject phrases that ``clean_text`` would rewrite (e.g. contains token ``ac``, ``gar``)."""
    t = cleaner.clean_text(phrase)
    return t.lower().strip() == phrase.strip().lower()


def _rand_price(rng: random.Random) -> int:
    return rng.randint(175_000, 2_250_000)


def _rand_sqft(rng: random.Random) -> int:
    return rng.choice(
        list(range(650, 1900, 25))
        + list(range(1900, 4200, 50))
        + list(range(4200, 9000, 100))
    )


def _pick_amenities(rng: random.Random, pool: List[str], k: int) -> List[str]:
    k = min(k, len(pool))
    if k <= 0:
        return []
    return rng.sample(pool, k=k)


def _none_scalar(v: Optional[LabelScalar]) -> LabelScalar:
    return v if v is not None else "none"


def _amenity_span_pattern(phrase: str) -> re.Pattern:
    """Same word-boundary rules as ``entity_extractor._amenity_feature_pattern`` (avoids ``spa`` inside ``space``)."""
    words = phrase.split()
    if len(words) == 1:
        return re.compile(rf"\b{re.escape(phrase)}\b", re.I)
    return re.compile(
        r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b",
        re.I,
    )


def _raw_price_fragment(rng: random.Random, price_val: int) -> str:
    """Messy listing-style price before ``normalize_prices``."""
    p = price_val
    choices: List[str] = []
    choices.append(f"Asking ${p:,}. ")
    choices.append(f"Priced at ${p:,}. ")
    choices.append(f"Price ${p:,}. ")
    if p % 1000 == 0 and p >= 1000:
        k = p // 1000
        choices.append(f"Asking ${k}k. ")
        choices.append(f"Priced at {k}k dollars. ")
        choices.append(f"Selling for ${k}k. ")
    return rng.choice(choices)


def _raw_sqft_fragment(rng: random.Random, sq: int) -> str:
    """Messy sqft before ``normalize_measurements`` (all normalize to ``<n> square feet``)."""
    variants = [
        f"Approximately {sq:,} sq ft of living space. ",
        f"The home spans {sq} sqft. ",
        f"Enjoy {sq:,} square feet. ",
        f"Interior measures {sq} square feet. ",
        f"About {sq:,} sq ft of space. ",
    ]
    return rng.choice(variants)


def _raw_bed_bath_fragment(
    rng: random.Random,
    bedrooms: int,
    bathrooms: float,
) -> str:
    """Use spaced ``br`` / ``ba`` so ``expand_abbreviations`` rewrites to bedroom / bathroom."""
    ba_str = str(int(bathrooms)) if bathrooms == int(bathrooms) else str(bathrooms)
    patterns = [
        f"{bedrooms} br / {ba_str} ba",
        f"{bedrooms} br, {ba_str} ba",
        f"{bedrooms} bedrooms and {ba_str} baths",
        f"{bedrooms} bedroom, {ba_str} bathroom",
        f"{bedrooms} beds, {ba_str} baths",
    ]
    return rng.choice(patterns) + rng.choice([". ", " — ", "; "])


def generate_one(
    rng: random.Random,
    amenity_pool: List[str],
    cleaner: TextCleaner,
) -> Tuple[str, Dict[str, Any]]:
    """Build raw remark, clean with ``TextCleaner``, return cleaned text and labels (spans on cleaned)."""
    text_parts: List[str] = []

    def append(s: str) -> None:
        text_parts.append(s)

    def append_amenity(phrase: str) -> None:
        text_parts.append(phrase.lower())

    include_bed = rng.random() < 0.82
    include_bath = rng.random() < 0.82
    include_price = rng.random() < 0.78
    include_sqft = rng.random() < 0.72
    n_amenities = rng.choices([0, 1, 2, 3, 4], weights=[0.08, 0.25, 0.35, 0.22, 0.10])[0]

    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    price_val: Optional[int] = None
    sqft_val: Optional[int] = None

    if include_bed:
        bedrooms = rng.randint(1, 6)
    if include_bath:
        bathrooms = float(rng.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]))

    if include_price:
        price_val = _rand_price(rng)
    if include_sqft:
        sqft_val = _rand_sqft(rng)

    # Ensure at least one signal (bed/bath/price/sqft/amenity)
    if not any(
        [
            include_bed,
            include_bath,
            include_price,
            include_sqft,
            n_amenities > 0,
        ]
    ):
        n_amenities = max(1, n_amenities)
        include_sqft = True
        sqft_val = _rand_sqft(rng)

    intros = [
        "Welcome to this thoughtfully updated residence.",
        "Rare opportunity in a sought-after neighborhood.",
        "Step inside and discover comfortable everyday living.",
        "This home offers a practical layout and great natural light.",
        "Beautifully presented property ready for its next chapter.",
        "Ideal for buyers seeking space and convenience.",
        "Move-in ready with appealing finishes throughout.",
    ]
    append(rng.choice(intros) + " ")

    # Beds / baths (raw: abbreviations expanded by TextCleaner)
    if include_bed and include_bath:
        append(_raw_bed_bath_fragment(rng, bedrooms, bathrooms))
    elif include_bed:
        append(
            rng.choice(
                [
                    f"{bedrooms} bedroom home. ",
                    f"{bedrooms} bedrooms. ",
                    f"{bedrooms} bedroom. ",
                ]
            )
        )
    elif include_bath:
        ba_str = str(int(bathrooms)) if bathrooms == float(int(bathrooms)) else str(bathrooms)
        append(
            rng.choice(
                [
                    f"{ba_str} bathrooms. ",
                    f"{ba_str} baths. ",
                ]
            )
        )

    if include_price and price_val is not None:
        append(_raw_price_fragment(rng, price_val))

    if include_sqft and sqft_val is not None:
        append(_raw_sqft_fragment(rng, sqft_val))

    # Narrative glue + amenities
    mids = [
        "Highlights include ",
        "You will love ",
        "Notable features: ",
        "Amenities include ",
        "The property features ",
        "Enjoy ",
    ]
    ams = _pick_amenities(rng, amenity_pool, n_amenities)
    if ams:
        append(rng.choice(mids))
        for i, phrase in enumerate(ams):
            if i:
                append(rng.choice([", ", " and ", ", plus ", " along with "]))
            append_amenity(phrase)
        append(". ")

    outros = [
        "Schedule your showing today.",
        "A must-see in person.",
        "Conveniently located near shopping and dining.",
        "Easy access to major routes.",
        "Don't miss this opportunity.",
    ]
    append(rng.choice(outros))

    raw = "".join(text_parts)
    raw = re.sub(r"\s+", " ", raw).strip()
    text = cleaner.clean_text(raw)
    amenity_spans = _respans_after_normalize(text, ams)

    if bathrooms is None:
        bath_out: LabelScalar = "none"
    elif bathrooms == int(bathrooms):
        bath_out = int(bathrooms)
    else:
        bath_out = bathrooms

    labels = {
        "bedrooms": _none_scalar(bedrooms),
        "bathrooms": bath_out,
        "price": _none_scalar(price_val),
        "sqft": _none_scalar(sqft_val),
        "amenities": amenity_spans if amenity_spans else "none",
    }

    _validate_labels(text, labels, ams)
    return text, labels  # cleaned text


def _respans_after_normalize(text: str, phrases_in_order: List[str]) -> List[Dict[str, Any]]:
    """Find phrase spans with word boundaries, left-to-right (matches entity extraction)."""
    spans: List[Dict[str, Any]] = []
    search_from = 0
    for phrase in phrases_in_order:
        p = phrase.lower()
        pat = _amenity_span_pattern(phrase)
        m = pat.search(text, search_from)
        if not m:
            raise RuntimeError(
                f"amenity {p!r} not found in generated text after position {search_from}"
            )
        s, e = m.span()
        spans.append({"name": p, "start": s, "end": e})
        search_from = e
    return spans


def _validate_labels(text: str, labels: Dict[str, Any], ams: List[str]) -> None:
    am = labels["amenities"]
    if am == "none":
        assert not ams
        return
    assert isinstance(am, list)
    for span in am:
        s, e = span["start"], span["end"]
        assert text[s:e].lower() == span["name"].lower()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate labeled synthetic listing remarks.")
    ap.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of remarks (default: random 200-300)",
    )
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    ap.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = args.count if args.count is not None else rng.randint(200, 300)
    cleaner = TextCleaner()
    amenity_pool = load_amenity_phrases(args.taxonomy, cleaner)

    records: List[Dict[str, Any]] = []
    for i in range(n):
        text, labels = generate_one(rng, amenity_pool, cleaner)
        records.append({"id": i, "text": text, "labels": labels})

    payload = {
        "meta": {
            "count": n,
            "seed": args.seed,
            "cleaning": "text_cleaning.TextCleaner.clean_text",
        },
        "remarks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {n} remarks to {args.output}")


if __name__ == "__main__":
    main()
