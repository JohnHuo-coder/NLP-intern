"""Intended structured gold for the unique 100-query stress set.

Wording family = (id-1) % 20. Numbers and cities differ on every query.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from query_numeric import canonical_city_name  # noqa: E402

_WORD = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_DASH = r"[–—-]"
_NUM = r"(?:[\d.]+|one|two|three|four|five|six|seven|eight|nine|ten)"


def e(**kwargs):
    return {k: v for k, v in kwargs.items() if v not in (None, [], {})}


def _n(raw) -> float:
    s = str(raw).strip().lower()
    if s in _WORD:
        return float(_WORD[s])
    return float(s.replace(",", ""))


def _int(raw) -> int:
    return int(_n(raw))


def _money(num, unit: str | None = None) -> int:
    n = _n(num)
    if unit is None:
        return int(round(n))
    u = unit.lower()
    if u == "k":
        return int(round(n * 1000))
    if u == "m":
        return int(round(n * 1_000_000))
    if u == "grand":
        return int(round(n * 1000))
    return int(round(n))


def _approx(n: float, pct: float = 0.2) -> tuple[float, float]:
    return n * (1 - pct), n * (1 + pct)


def _bath_exact(raw) -> dict:
    n = _n(raw)
    full = int(n)
    out = {"bathrooms": full}
    if n - full >= 0.4:
        out["bathroom_half"] = 1
    return out


def _bath_min(raw) -> dict:
    return {"bathrooms_min": int(_n(raw))}


def _bath_max(raw) -> dict:
    return {"bathrooms_max": int(_n(raw))}


def _city(query: str) -> str | None:
    hits = list(
        re.finditer(
            r"\bin\s+([A-Z][a-zA-Z]+(?:\s+(?:del|de|la|las|los|[A-Z][a-zA-Z]+))*)\b",
            query,
        )
    )
    if not hits:
        return None
    return canonical_city_name(hits[0].group(1))


_BED_UNIT = r"(?:bedrooms?|beds?|br|bdrms?|bdrm|bds?|bd)\b"
_BATH_UNIT = r"(?:bathrooms?|baths?|bthrms?|bthrm|bths?|ba|bth)\b"
_SQ_UNIT = r"(?:sq\.?\s*ft\.?|sqft|square feet|sf)"
_MONEY = r"\$?\s*([\d,.]+)(?:\s*(k|m|grand|dollar)s?)?"


def _as_money(num, unit=None) -> int:
    if unit and unit.lower().startswith("dollar"):
        return _int(num)
    return _money(num, unit)


def _ok_price(num, unit=None) -> bool:
    n = _as_money(num, unit)
    if unit:
        return True
    return n >= 10_000


_NOT_PRICE_AFTER = re.compile(
    r"\s*(?:k\s*)?(?:sq\.?\s*ft|sqft|square feet|sf|bedrooms?|beds?|bathrooms?|baths?|"
    r"brs?\b|bas?\b|bdrms?|bthrms?|bds?\b|bths?\b)\b",
    re.I,
)


def _price_ok_at(m: re.Match, query: str) -> bool:
    if not _ok_price(m.group(1), m.group(2)):
        return False
    return not _NOT_PRICE_AFTER.match(query[m.end() :])


def _search_price(pattern: str, query: str):
    for m in re.finditer(pattern, query, re.I):
        if _price_ok_at(m, query):
            return m
    return None


def _intended_price(query: str) -> dict:
    range_re = (
        rf"(?:priced from|from|listed between|listed|ranging from|price range|anywhere from|for)\s+"
        rf"{_MONEY}\s+(?:to|through|and)\s+{_MONEY}"
    )
    for m in re.finditer(range_re, query, re.I):
        if _ok_price(m.group(1), m.group(2)) and _ok_price(m.group(3), m.group(4)):
            return {
                "price_min": _as_money(m.group(1), m.group(2)),
                "price_max": _as_money(m.group(3), m.group(4)),
            }
    if m := re.search(
        rf"in the {_MONEY}(?:{_DASH}| to ){_MONEY} range",
        query,
        re.I,
    ):
        if _ok_price(m.group(1), m.group(2)) and _ok_price(m.group(3), m.group(4)):
            return {
                "price_min": _as_money(m.group(1), m.group(2)),
                "price_max": _as_money(m.group(3), m.group(4)),
            }
    if m := re.search(
        rf"give or take \$50k",
        query,
        re.I,
    ) and (t := _search_price(rf"(?:target price of|budget)\s+{_MONEY}", query)):
        pr = _as_money(t.group(1), t.group(2))
        return {"price_min": pr - 50_000, "price_max": pr + 50_000}
    if m := _search_price(
        rf"(?:no more than|not more than|under|below|less than|cheaper than|no pricier than|not pricier than|"
        rf"capped at|maximum price of|price ceiling of|price maximum of|hard spending (?:limit|ceiling) of|"
        rf"limit of|maximum|max|not over|at most|for less than|but not over)\s+{_MONEY}",
        query,
    ):
        return {"price_max": _as_money(m.group(1), m.group(2))}
    if m := _search_price(rf"<\s*{_MONEY}", query):
        return {"price_max": _as_money(m.group(1), m.group(2))}
    if m := _search_price(
        rf"(?:around|about|approximately|roughly|near|close to|~)\s*{_MONEY}",
        query,
    ):
        pr = _as_money(m.group(1), m.group(2))
        a, b = _approx(pr)
        return {"price_min": a, "price_max": b}
    if m := re.search(rf"roughly\s+(\d+)\s+grand", query, re.I):
        pr = _money(m.group(1), "grand")
        a, b = _approx(pr)
        return {"price_min": a, "price_max": b}
    if m := _search_price(
        rf"(?:exactly|target price of|at a price of(?:\s+exactly)?)\s+{_MONEY}",
        query,
    ):
        return {"price": _as_money(m.group(1), m.group(2))}
    return {}


def _intended_beds(query: str) -> dict:
    if m := re.search(rf"three or four {_BED_UNIT}|(\d+) or (\d+) {_BED_UNIT}", query, re.I):
        if m.group(1):
            return {"bedrooms_min": _int(m.group(1)), "bedrooms_max": _int(m.group(2))}
        return {"bedrooms_min": 3, "bedrooms_max": 4}
    if m := re.search(rf"(\d+){_DASH}(\d+) {_BED_UNIT}", query, re.I):
        return {"bedrooms_min": _int(m.group(1)), "bedrooms_max": _int(m.group(2))}
    if m := re.search(rf"(\d+)/(\d+)\s+bds?\b", query, re.I):
        return {"bedrooms_min": _int(m.group(1)), "bedrooms_max": _int(m.group(2))}
    if m := re.search(rf"(\d+)\s+to\s+(\d+)\s+{_BED_UNIT}", query, re.I):
        return {"bedrooms_min": _int(m.group(1)), "bedrooms_max": _int(m.group(2))}
    if m := re.search(rf"(?:at most|no greater than|<)\s*({_NUM})[-\s]*{_BED_UNIT}", query, re.I):
        return {"bedrooms_max": _int(m.group(1))}
    if m := re.search(
        rf"(?:at least|no less than|no fewer than|>)\s*({_NUM})[-\s]*{_BED_UNIT}",
        query,
        re.I,
    ):
        return {"bedrooms_min": _int(m.group(1))}
    if m := re.search(rf"({_NUM})\+\s*{_BED_UNIT}", query, re.I):
        return {"bedrooms_min": _int(m.group(1))}
    if m := re.search(
        rf"(?:about|around|approximately|roughly|close to)\s+({_NUM})\s*{_BED_UNIT}",
        query,
        re.I,
    ):
        n = _int(m.group(1))
        return {"bedrooms_min": n - 1 if n > 1 else n, "bedrooms_max": n + 1}
    if m := re.search(rf"exactly\s+({_NUM})\s*{_BED_UNIT}", query, re.I):
        return {"bedrooms": _int(m.group(1))}
    if m := re.search(rf"precisely\s+({_NUM})\s*{_BED_UNIT}", query, re.I):
        return {"bedrooms": _int(m.group(1))}
    if m := re.search(rf"({_NUM})-bed(?:room)?\b", query, re.I):
        return {"bedrooms": _int(m.group(1))}
    if m := re.search(rf"({_NUM})-(?:bd|br)s?\b", query, re.I):
        return {"bedrooms": _int(m.group(1))}
    if m := re.search(rf"({_NUM})\s*{_BED_UNIT}", query, re.I):
        return {"bedrooms": _int(m.group(1))}
    if m := re.search(rf"four-bed\b", query, re.I):
        return {"bedrooms": 4}
    if m := re.search(rf"five-bed\b|5-bed\b", query, re.I):
        return {"bedrooms": 5}
    return {}


def _intended_baths(query: str) -> dict:
    if m := re.search(rf"somewhere between ({_NUM}) and ({_NUM})\s+{_BATH_UNIT}", query, re.I):
        return {"bathrooms_min": int(_n(m.group(1))), "bathrooms_max": int(_n(m.group(2)))}
    if m := re.search(rf"(\d+(?:\.\d+)?){_DASH}(\d+(?:\.\d+)?)\s+{_BATH_UNIT}", query, re.I):
        return {"bathrooms_min": int(_n(m.group(1))), "bathrooms_max": int(_n(m.group(2)))}
    if m := re.search(rf"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s+{_BATH_UNIT}", query, re.I):
        return {"bathrooms_min": int(_n(m.group(1))), "bathrooms_max": int(_n(m.group(2)))}
    if m := re.search(
        rf"(?:at most|no greater than|(?<!\bno\s)(?<!\bnot\s)less than|<)\s*({_NUM})[-\s]*{_BATH_UNIT}",
        query,
        re.I,
    ):
        return _bath_max(m.group(1))
    if m := re.search(
        rf"(?:at least|no less than|no fewer than|at cleast|>)\s*({_NUM})[-\s]*{_BATH_UNIT}",
        query,
        re.I,
    ):
        return _bath_min(m.group(1))
    if m := re.search(rf"({_NUM})\+\s*{_BATH_UNIT}", query, re.I):
        return _bath_min(m.group(1))
    if m := re.search(
        rf"(?:about|around|approximately|roughly)\s+({_NUM})\s*{_BATH_UNIT}",
        query,
        re.I,
    ):
        full = int(_n(m.group(1)))
        return {
            "bathrooms_min": full - 1 if full > 1 else full,
            "bathrooms_max": full + 1,
        }
    if m := re.search(rf"exactly\s+({_NUM})\s*{_BATH_UNIT}", query, re.I):
        return _bath_exact(m.group(1))
    if m := re.search(rf"precisely\s+({_NUM})\s*{_BATH_UNIT}", query, re.I):
        return _bath_exact(m.group(1))
    if m := re.search(rf"({_NUM})-bath(?:room)?\b", query, re.I):
        return _bath_exact(m.group(1))
    if m := re.search(rf"({_NUM})-ba\b", query, re.I):
        return _bath_exact(m.group(1))
    if m := re.search(rf"({_NUM})\s*{_BATH_UNIT}", query, re.I):
        return _bath_exact(m.group(1))
    return {}


def _sqft_n(raw, k=None) -> float:
    n = _n(raw)
    if k:
        n *= 1000
    return n


def _intended_sqft(query: str) -> dict:
    if m := re.search(
        rf"(?:between|from|range from|minimum)\s+([\d,.]+)\s+(?:and|to|maximum)\s+([\d,.]+)\s*{_SQ_UNIT}",
        query,
        re.I,
    ):
        return {"sqft_min": _int(m.group(1)), "sqft_max": _int(m.group(2))}
    if m := re.search(rf"([\d,.]+)\s*(?:to|{_DASH})\s*([\d,.]+)\s*{_SQ_UNIT}", query, re.I):
        return {"sqft_min": _int(m.group(1)), "sqft_max": _int(m.group(2))}
    if m := re.search(
        rf"more than ([\d,.]+) {_SQ_UNIT} but not over ([\d,.]+) {_SQ_UNIT}",
        query,
        re.I,
    ):
        return {"sqft_min": _int(m.group(1)), "sqft_max": _int(m.group(2))}
    if m := re.search(
        rf"(?:over|>)\s*([\d,.]+)\s*(?:sf|{_SQ_UNIT}) but (?:below|<)\s*([\d,.]+)",
        query,
        re.I,
    ):
        return {"sqft_min": _int(m.group(1)), "sqft_max": _int(m.group(2))}
    if m := re.search(
        rf"(?:at most|not over|below|under|maximum|max)\s+(?:of\s+)?([\d,.]+)\s*{_SQ_UNIT}",
        query,
        re.I,
    ):
        return {"sqft_max": _int(m.group(1))}
    if m := re.search(
        rf"(?:at least|minimum|minimum of|more than|over|above|greater than|larger than|no smaller than|somewhere above|upwards of|>)\s*([\d,.]+)\s*(?:k\s*)?{_SQ_UNIT}",
        query,
        re.I,
    ):
        return {"sqft_min": _int(m.group(1))}
    if m := re.search(rf"([\d,.]+)\+\s*{_SQ_UNIT}", query, re.I):
        return {"sqft_min": _int(m.group(1))}
    if m := re.search(
        rf"(?:around|about|approximately|roughly|near|close to|~)\s*([\d,.]+)(k)?\s*{_SQ_UNIT}",
        query,
        re.I,
    ):
        n = _sqft_n(m.group(1), m.group(2))
        a, b = _approx(n)
        return {"sqft_min": a, "sqft_max": b}
    if m := re.search(rf"([\d,.]+)-ish\s*{_SQ_UNIT}", query, re.I):
        n = _int(m.group(1))
        a, b = _approx(n)
        return {"sqft_min": a, "sqft_max": b}
    if m := re.search(rf"approximately\s+([\d.]+)k\s*{_SQ_UNIT}", query, re.I):
        n = _n(m.group(1)) * 1000
        a, b = _approx(n)
        return {"sqft_min": a, "sqft_max": b}
    if m := re.search(rf"exactly\s+([\d,.]+)\s*{_SQ_UNIT}", query, re.I):
        return {"sqft": _int(m.group(1))}
    if m := re.search(rf"precisely\s+([\d,.]+)\s*{_SQ_UNIT}", query, re.I):
        return {"sqft": _int(m.group(1))}
    if m := re.search(rf"ideally\s+([\d,.]+)\s*{_SQ_UNIT}", query, re.I):
        return {"sqft": _int(m.group(1))}
    if m := re.search(rf"([\d,.]+)\s*{_SQ_UNIT}", query, re.I):
        return {"sqft": _int(m.group(1))}
    return {}


def intended_gold(qid: str, query: str) -> dict:
    out = {}
    city = _city(query)
    if city:
        out["city"] = city
    out.update(_intended_beds(query))
    out.update(_intended_baths(query))
    out.update(_intended_sqft(query))
    out.update(_intended_price(query))
    return e(**out)

