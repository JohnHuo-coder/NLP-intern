"""Regex extractors for beds / baths / price / sqft / city."""

from __future__ import annotations

import csv
import re
from pathlib import Path

# support later: school district, HOA fee range, fencing type

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CITY_CSV = _PROJECT_ROOT / "data" / "processed" / "distinct_cities.csv"

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_WORD_NUM = "|".join(_WORD_TO_NUM.keys())
# 5+ digits: cleaner expands 500k → 500000. 4-digit spans are almost always
# sqft (or years), not CA list prices (typically $10k+).
_PRICE = r"\$?(\d{5,})(?!\s*(?:square feet))"
# `$` is non-word, so `\b$800000+` fails; `(?<!\w)` still matches after space/comma.
_PRICE_PLUS = r"(?<!\w)\$?(\d{5,})\s*\+"

_CITY_CUE_PREFIX = re.compile(
    r"(?:in|near|at|around|located\s+in)\s+(?:the\s+city\s+of\s+)?\s*$",
    re.I,
)
# Catalog names that are also common housing English. Keep only with a location cue
# so "cool house in Irvine" / "walnut floors" do not become the city.
_CITY_NEED_CUE = frozenset({
    "bell", "challenge", "commerce", "cool", "fellows", "freedom",
    "nice", "orange", "paradise", "pioneer", "rescue", "thermal",
    "vista", "walnut", "weed", "winters", "woodland",
})

_city_re: re.Pattern[str] | None = None
_city_canonical: dict[str, str] = {}


def _spell_num(word):
    if not word:
        return None
    return _WORD_TO_NUM.get(str(word).lower())


def _split_number(x):
    whole = int(x)
    return whole, x - whole


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _as_int(raw):
    return int(raw) if _is_number(raw) else _spell_num(raw)


def _as_bath_full(raw):
    num = float(raw) if _is_number(raw) else _spell_num(raw)
    full, _ = _split_number(num)
    return full


def _unit_patterns(cat: str) -> tuple[str, str, str]:
    """Return (with_unit, symbol_unit, plus_pattern) for min/max/range."""
    if cat == "bedrooms":
        unit = r"(?:bed|bedroom|br|bd)s?"
        with_unit = rf"(\d+|{_WORD_NUM})\s*-?\s*{unit}"
        symbol_unit = rf"(\d+)\s*-?\s*{unit}"
        plus = rf"\b(\d+)\s*\+\s*{unit}\b"
    elif cat == "bathrooms":
        unit = r"(?:bath|bathroom)s?"
        with_unit = rf"(\d+(?:\.\d+)?|{_WORD_NUM})\s*-?\s*{unit}"
        symbol_unit = rf"(\d+(?:\.\d+)?)\s*-?\s*{unit}"
        plus = rf"\b(\d+(?:\.\d+)?)\s*\+\s*(bath|bathroom)s?\b"
    elif cat == "price":
        with_unit = symbol_unit = _PRICE
        plus = _PRICE_PLUS
    elif cat == "sqft":
        with_unit = symbol_unit = r"(\d+(?:\.\d+)?)\s*(?:square feet)"
        plus = r"\b(\d+(?:\.\d+)?)\s*\+\s*(?:square feet)\b"
    else:
        raise ValueError(cat)
    return with_unit, symbol_unit, plus


def _range_patterns(cat: str) -> tuple[str, str, str]:
    """prefix, suffix, a-b symbol pattern for range (from X to Y / between / -)."""
    if cat == "bedrooms":
        prefix = rf"(\d+|{_WORD_NUM})"
        suffix = rf"(\d+|{_WORD_NUM})\s*(?:bed|bedroom|br|bd)s?"
        symbol = r"\b(\d+)\s*(?:-|~)\s*(\d+)\s*(?:bed|bedroom|br|bd)s?\b"
    elif cat == "bathrooms":
        prefix = rf"(\d+(?:\.\d+)?|{_WORD_NUM})"
        suffix = rf"(\d+(?:\.\d+)?|{_WORD_NUM})\s*(?:bath|bathroom)s?"
        symbol = r"\b(\d+(?:\.\d+)?)\s*(?:-|~)\s*(\d+(?:\.\d+)?)\s*(bath|bathroom)s?\b"
    elif cat == "price":
        prefix = suffix = _PRICE
        symbol = rf"\b{prefix}\s*(?:-|~)\s*{suffix}\b"
    elif cat == "sqft":
        prefix = r"(\d+(?:\.\d+)?)"
        suffix = r"(\d+(?:\.\d+)?)\s*(?:square feet)"
        symbol = rf"\b{prefix}\s*(?:-|~)\s*{suffix}\b"
    else:
        raise ValueError(cat)
    return prefix, suffix, symbol


def _store_bound(filters: dict, cat: str, raw, *, key: str) -> None:
    filters[key] = _as_bath_full(raw) if cat == "bathrooms" else _as_int(raw)


def parse_range(query: str, filters: dict, cat: str) -> bool:
    prefix, suffix, symbol = _range_patterns(cat)
    patterns = [
        rf"\b(?:from\s+)?{prefix}\s+(?:to|through)\s+{suffix}\b",
        rf"\b(?:between)\s+{prefix}\s+(?:and)\s+{suffix}\b",
        symbol,
    ]
    for p in patterns:
        match = re.search(p, query, re.I)
        if match:
            _store_bound(filters, cat, match.group(1), key=f"{cat}_min")
            _store_bound(filters, cat, match.group(2), key=f"{cat}_max")
            return True
    return False


def parse_max(query: str, filters: dict, cat: str) -> bool:
    num, num_symbol, _ = _unit_patterns(cat)
    # `(?<!\bno\s)less than` so "no less than" stays a min cue.
    # `\b` in the lookbehind avoids "Fresno less than" false-firing.
    patterns = []
    if cat == "price":
        # "maximum $550k" is in the generic list; these cover "maximum price of"
        # and "price ceiling of $830k".
        patterns.append(rf"\b(?:maximum|max)\s+price(?:\s+of)?\s+{num}\b")
        patterns.append(rf"\bprice\s+(?:ceiling|maximum)(?:\s+of)?\s+{num}\b")
        patterns.append(rf"\b(?:hard\s+)?spending\s+(?:limit|ceiling)(?:\s+of)?\s+{num}\b")
        patterns.append(rf"\blimit\s+of\s+{num}\b")
        patterns.append(rf"\bnot\s+pricier\s+than\s+{num}\b")
    patterns.extend([
        rf"\b(?:under|below|at most(?:\s+of)?|(?<!\bno\s)(?<!\bnot\s)less than|cheaper than|up to|maximum(?:\s+of)?|max(?:\s+of)?|no more than|not more than|no pricier than|capped at|not over|no greater than)\s+{num}\b",
        rf"\b{num}\s+(?:or less)\b",
        rf"<\s*{num_symbol}\b",
    ])
    for p in patterns:
        match = re.search(p, query, re.I)
        if match:
            _store_bound(filters, cat, match.group(1), key=f"{cat}_max")
            return True
    return False


def parse_min(query: str, filters: dict, cat: str) -> bool:
    num, symbol_gt, plus = _unit_patterns(cat)
    # `starting from` before `from` so the longer cue wins. Range `from X to Y`
    # is handled first in parse_range, so bare `from $500k` is a min bound.
    # `(?<!\bno\s)more than` so "no more than" stays a max cue.
    patterns = [
        rf"\b(?:at least|above|(?<!\bnot\s)over|(?<!\bno\s)(?<!\bnot\s)(?:more|greater|larger) than|upwards of|starting from|from|no less than|not less than|no fewer than|no smaller than|minimum(?:\s+of)?|min)\s+{num}\b",
        rf"\b{num}\s+(?:or more)\b",
        rf">\s*{symbol_gt}\b",
        plus,
    ]
    for p in patterns:
        match = re.search(p, query, re.I)
        if match:
            _store_bound(filters, cat, match.group(1), key=f"{cat}_min")
            return True
    return False


def _parse_bounds_then_approx_exact(
    query: str,
    filters: dict,
    cat: str,
    *,
    approx_re: str,
    exact_re: str,
    on_approx,
    on_exact,
) -> None:
    if parse_range(query, filters, cat):
        return
    max_hit = parse_max(query, filters, cat)
    min_hit = parse_min(query, filters, cat)
    if min_hit or max_hit:
        return
    if match := re.search(approx_re, query, re.I):
        on_approx(match)
        return
    if match := re.search(exact_re, query, re.I):
        on_exact(match)


def _first_num(match) -> str:
    return next(g for g in match.groups() if g is not None)


def parse_price(query: str, filters: dict) -> None:
    _parse_bounds_then_approx_exact(
        query,
        filters,
        "price",
        approx_re=rf"(?:(?:\b(?:around|about|approximately|roughly|near|close to)\b|~)\s*{_PRICE}|{_PRICE}\s*-?\s*ish\b)",
        exact_re=rf"\b(?:at|exactly)?\s+{_PRICE}\b",
        on_approx=lambda m: filters.update(
            price_min=int(_first_num(m)) * 0.8,
            price_max=int(_first_num(m)) * 1.2,
        ),
        on_exact=lambda m: filters.__setitem__("price", int(m.group(1))),
    )


def parse_sqft(query: str, filters: dict) -> None:
    _parse_bounds_then_approx_exact(
        query,
        filters,
        "sqft",
        approx_re=(
            r"(?:(?:\b(?:around|about|approximately|roughly|near|close to)\b|~)\s*(\d+(?:\.\d+)?)\s*(?:square feet)\b"
            r"|(\d+(?:\.\d+)?)\s*-?\s*ish\s*(?:square feet)\b)"
        ),
        exact_re=r"\b(?:at|exactly)?\s+(\d+(?:\.\d+)?)\s*(?:square feet)\b",
        on_approx=lambda m: filters.update(
            sqft_min=int(float(_first_num(m))) * 0.8,
            sqft_max=int(float(_first_num(m))) * 1.2,
        ),
        on_exact=lambda m: filters.__setitem__("sqft", int(m.group(1))),
    )


def parse_bedroom(query: str, filters: dict) -> None:
    words = _WORD_NUM
    _parse_bounds_then_approx_exact(
        query,
        filters,
        "bedrooms",
        approx_re=rf"(?:(?:\b(?:around|about|approximately|roughly|close to)\b|~)\s*(\d+|{words})\s*(?:bed|bedroom|br|bd)s?\b|(\d+)\s*-?\s*ish\s*(?:bed|bedroom|br|bd)s?\b)",
        exact_re=rf"\b(\d+|{words})(?:\s*-\s*|\s+)(bed|bedroom|br|bd)s?\b",
        on_approx=_approx_bedrooms(filters),
        on_exact=lambda m: filters.__setitem__(
            "bedrooms", int(m.group(1)) if m.group(1).isdigit() else _spell_num(m.group(1))
        ),
    )


def _approx_bedrooms(filters: dict):
    def _apply(match):
        raw = _first_num(match)
        num = int(raw) if raw.isdigit() else _spell_num(raw)
        filters["bedrooms_min"] = num - 1 if num - 1 > 0 else num
        filters["bedrooms_max"] = num + 1
    return _apply


def parse_bathroom(query: str, filters: dict) -> None:
    # 2.5 bath → 2 full + 1 half (table has LM_Dec_3 + BathroomsHalf).
    # Range/min/max use full-bath floor only. Exact 2.5 sets bathroom_half=1.
    words = _WORD_NUM
    _parse_bounds_then_approx_exact(
        query,
        filters,
        "bathrooms",
        approx_re=rf"(?:(?:\b(?:around|about|approximately|roughly|close to)\b|~)\s*(\d+|{words})\s*(?:bath|bathroom)s?\b|(\d+(?:\.\d+)?)\s*-?\s*ish\s*(?:bath|bathroom)s?\b)",
        exact_re=rf"\b(\d+(?:\.\d+)?|{words})(?:\s*-\s*|\s+|\s*full\s+)(?:bath|bathroom)s?\b",
        on_approx=_approx_bathrooms(filters),
        on_exact=_exact_bathrooms(filters),
    )


def _approx_bathrooms(filters: dict):
    def _apply(match):
        raw = _first_num(match)
        num = float(raw) if _is_number(raw) else _spell_num(raw)
        full, _ = _split_number(num)
        filters["bathrooms_min"] = full - 1 if full - 1 > 0 else full
        filters["bathrooms_max"] = full + 1
    return _apply


def _exact_bathrooms(filters: dict):
    def _apply(match):
        raw = match.group(1)
        num = float(raw) if _is_number(raw) else _spell_num(raw)
        full, half = _split_number(num)
        if half:
            filters["bathroom_half"] = 1
        filters["bathrooms"] = full
    return _apply


def _load_city_index() -> None:
    """Compile longest-first city patterns from ``distinct_cities.csv`` once."""
    global _city_re, _city_canonical
    if _city_re is not None:
        return
    names: list[str] = []
    if _CITY_CSV.exists():
        with _CITY_CSV.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                raw = (row.get("L_City") or "").strip()
                if not raw or raw.lower() == "other":
                    continue
                names.append(raw)
    _city_canonical = {name.lower(): name for name in names}
    if not names:
        _city_re = re.compile(r"(?!)")
        return
    parts = [re.escape(name) for name in sorted(names, key=len, reverse=True)]
    _city_re = re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


def parse_city(query: str, filters: dict) -> None:
    """Gazetteer match against listing cities; longest span wins.

    Multi-word names work without a preposition (``Oak Park 3 bd``).
    Ambiguous single words like ``cool`` / ``walnut`` still need ``in`` / ``near``.
    """
    _load_city_index()
    if _city_re is None:
        return
    scan = query.replace("-", " ")
    best: str | None = None
    best_key: tuple[int, int] | None = None
    for match in _city_re.finditer(scan):
        canon = _city_canonical.get(match.group(0).lower())
        if not canon:
            continue
        has_cue = 1 if _CITY_CUE_PREFIX.search(scan[: match.start()]) else 0
        if canon.lower() in _CITY_NEED_CUE and not has_cue:
            continue
        key = (len(match.group(0)), has_cue)
        if best_key is None or key > best_key:
            best_key = key
            best = canon
    if best:
        filters["city"] = best


def canonical_city_name(raw: str) -> str:
    """Return the gazetteer spelling, or ``raw`` if the name is unknown."""
    _load_city_index()
    return _city_canonical.get(raw.strip().lower(), raw)


def fill_structured_fields(query: str, filters: dict) -> None:
    parse_bedroom(query, filters)
    parse_bathroom(query, filters)
    parse_sqft(query, filters)
    parse_price(query, filters)
    parse_city(query, filters)
