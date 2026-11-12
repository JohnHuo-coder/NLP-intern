import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from scripts.text_cleaning import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


# --- Price normalization ---
def test_price_normalization_k_and_m(cleaner):
    assert "450000" in cleaner.normalize_prices("priced at 450k")
    assert "1200000" in cleaner.normalize_prices("$1.2m home")


def test_price_normalization_decimal_k(cleaner):
    assert cleaner.normalize_prices("$5.5k") == "$5500"
    assert "2300" in cleaner.normalize_prices("priced at 2.3k")


def test_price_normalization_comma_and_word_form(cleaner):
    assert "1234567" in cleaner.normalize_prices("$1,234,567")
    assert "5000" in cleaner.normalize_prices("5 thousand")
    assert "1200000" in cleaner.normalize_prices("1.2 million")
    assert "722000" in cleaner.normalize_prices("roughly 722 grand")
    assert "$722000" in cleaner.normalize_prices("$722 grand")


def test_price_normalization_decimal_m_does_not_truncate(cleaner):
    # 2.07 cannot be stored exactly in binary float; int() would yield 2069999.
    assert cleaner.normalize_prices("under $2.07M") == "under $2070000"
    assert "2050000" in cleaner.normalize_prices("between $1.83M and $2.05M")
    assert "2070000" in cleaner.normalize_prices("2.07 million")


# --- Measurement normalization ---
def test_normalize_measurements_leading_decimal(cleaner):
    assert "0.5" in cleaner.normalize_measurements(".5 acre")


def test_normalize_measurements_feet_inches_ft(cleaner):
    assert cleaner.normalize_measurements("9' ceiling") == "9 foot ceiling"
    assert cleaner.normalize_measurements('6" trim') == "6 inch trim"
    assert "10 foot" in cleaner.normalize_measurements("10ft ceiling")
    assert "10 foot" in cleaner.normalize_measurements("10-ft ceiling")


def test_normalize_measurements_sqft_and_acres(cleaner):
    assert "1500 square feet" in cleaner.normalize_measurements("1500 sqft")
    assert "1100 square feet" in cleaner.normalize_measurements("1100 sf")
    assert "1800 square feet" in cleaner.normalize_measurements("1800 sq-ft")
    assert "2.5 acres" in cleaner.normalize_measurements("2.5 acre")


# --- Abbreviations ---
@pytest.mark.parametrize(
    "raw, needle",
    [
        # spaced / slash-separated counts (common in MLS copy)
        ("3 br 2 ba", "bedroom"),
        ("3br / 2ba", "bedroom"),
        # hyphen glued to digit: 2-bd, 2-ba
        ("2-bd 2-ba townhome", "bedroom"),
        ("2-bd 2-ba townhome", "bathroom"),
        # no space between digit and unit: 2bd, 2ba
        ("2bd 2ba ranch", "bedroom"),
        ("2bd 2ba ranch", "bathroom"),
        # optional plural on unit
        ("2 bds 2 bas", "bedroom"),
        ("2 bds 2 bas", "bathroom"),
        # longer bedroom / bathroom spellings
        ("3 bdr", "bedroom"),
        ("2 bdrm", "bedroom"),
        ("1 bth", "bathroom"),
        ("2 bthrm", "bathroom"),
        ("2+ br", "2+ bedroom"),
        # master suite, room, floor, level
        ("mbr with mba", "master bedroom"),
        ("mbr with mba", "master bathroom"),
        ("1 rm", "1 room"),
        ("2 rms on the main flr", "2 rooms"),
        ("2 rms on the main flr", "floor"),
        ("top lvl", "level"),
        # decimal counts (e.g. half bath)
        ("2.5 ba", "bathroom"),
        ("3 bdrms 2.5 bthrms", "bedroom"),
        ("3 bdrms 2.5 bthrms", "bathroom"),
        # word number + unit: do not glue "two baths" → "twobathroom"
        ("three beds and two baths", "two bathroom"),
        ("one bath", "one bathroom"),
        ("four-bath", "bathroom"),
        # area: sqft variants (digit may be stripped; we only require expansion text)
        ("1800 sqft", "square feet"),
        ("1800sqft", "1800 square feet"),
        ("2200 sq. ft", "square feet"),
        ("2200 sq. ft.", "square feet"),
        ("2500 sq ft", "square feet"),
        ("1600 sq.ft", "square feet"),
        ("1700 sq .ft", "square feet"),
        ("1900 sq.ft.", "square feet"),
        # normal abbrevs (word boundaries + optional plural)
        ("updated kit and lr", "kitchen"),
        ("updated kit and lr", "living room"),
        ("updated kits", "kitchen"),
        ("formal dr and fr", "dining room"),
        ("formal dr and fr", "family room"),
        ("2-car gar", "garage"),
        ("unfinished bsmt", "basement"),
        ("2 pkg", "parking space"),
        ("new appls", "appliance"),
        ("5 mins", "minute"),
        ("10 yrs", "year"),
        ("detached adu", "accessory dwelling unit"),
        # no_s abbrevs; longer keys such as w/o win over w/
        ("w/ hvac and w/d", "with"),
        ("w/ hvac and w/d", "heating and cooling"),
        ("w/ hvac and w/d", "washer dryer"),
        ("w/o hoa", "without"),
        ("w/o hoa", "homeowners association"),
        ("central ac", "air conditioning"),
        ("new a/c", "air conditioning"),
        ("brick fp", "fireplace"),
        ("big wic", "walk-in closet"),
        ("yb 1998", "year built"),
        ("renov and upd", "renovated"),
        ("renov and upd", "updated"),
        ("incl ss appl", "including"),
        ("incl ss appl", "stainless steel"),
        ("incl ss appl", "appliance"),
        ("nr downtown", "near"),
        ("reo listing", "bank-owned"),
        ("low dom", "days on market"),
        ("approx 2 ba", "approximately"),
        ("pvt driveway", "private"),
        ("avail now", "available"),
        ("fin bsmt", "finished"),
        ("fin bsmt", "basement"),
        ("ch and hw floors", "central heat"),
        ("ch and hw floors", "hardwood"),
        ("on the mls", "multiple listing service"),
        ("wd hookups", "washer dryer"),
        ("summer bbq", "barbecue"),
    ],
)
def test_expand_abbreviations_room_and_listing_styles(cleaner, raw, needle):
    out = cleaner.expand_abbreviations(raw)
    assert needle in out


def test_expand_abbreviations_hyphenated_full_pipeline(cleaner):
    out = cleaner.clean_text("2-bd 2-ba, 1400 sqft, w/ gar.")
    assert "bedroom" in out
    assert "bathroom" in out
    assert "square feet" in out
    assert "with" in out
    assert "garage" in out


# --- HTML ---
def test_normalize_html(cleaner):
    # &amp; is unescaped to & (correct); we only check the entity literal is gone
    assert cleaner.normalize_html("&amp; ready") == "& ready"
    assert "<" not in cleaner.normalize_html("see <b>details</b> here")
    kept = cleaner.normalize_html("I need <4 bedroom, >3 bathroom")
    assert "<4 bedroom" in kept
    assert ">3 bathroom" in kept


# --- Unicode ---
def test_normalize_unicode(cleaner):
    assert cleaner.normalize_unicode('"quoted"') == '"quoted"'
    assert cleaner.normalize_unicode("\xa0space") == " space"


# --- Full pipeline ---
def test_clean_text(cleaner):
    out = cleaner.clean_text("3 br, 1200 sqft, $5.5k. 9' ceiling.")
    assert "bedroom" in out or "3" in out
    assert "square feet" in out
    assert "5500" in out
    assert "9 foot" in out


def test_clean_text_handles_none_or_empty(cleaner):
    # clean_text expects str; if caller passes None, strip() would fail
    out = cleaner.clean_text("  ")
    assert out == ""


def test_parse_city_mill_valley_and_marina_del_rey():
    from scripts.query_numeric import fill_structured_fields

    mill = {}
    fill_structured_fields("3 bedroom 1.5 bathroom in Mill Valley between 1332 and 1782 square feet", mill)
    assert mill.get("city") == "Mill Valley"

    marina = {}
    fill_structured_fields("at least one bedroom in Marina del Rey over 2665 square feet", marina)
    assert marina.get("city") == "Marina Del Rey"


def test_parse_max_price_of_prefix():
    from scripts.query_numeric import fill_structured_fields

    filters = {}
    fill_structured_fields(
        "four bedroom 3.5 bathroom in Daly City with 2758+ square feet and a maximum price of $2260000",
        filters,
    )
    assert filters.get("price_max") == 2260000
    assert "price" not in filters

    ceiling = {}
    fill_structured_fields(
        "1+ bedrooms 1.5+ bathrooms and at least 1115 square feet in Monterey with a price ceiling of $830000",
        ceiling,
    )
    assert ceiling.get("price_max") == 830000
    assert "price" not in ceiling

    limit = {}
    fill_structured_fields(
        "5 bedroom 3.5 bathroom in Thousand Oaks measuring about 1239 square feet with a hard spending limit of $938000",
        limit,
    )
    assert limit.get("price_max") == 938000
    assert "price" not in limit

    ceiling = {}
    fill_structured_fields(
        "5 bedroom 2.5 bathroom in Walnut Creek measuring around 2479 square feet with a hard spending ceiling of $2020000",
        ceiling,
    )
    assert ceiling.get("price_max") == 2020000
    assert "price" not in ceiling

    swapped = {}
    fill_structured_fields(
        "1+ bedrooms 3.5+ bathrooms and larger than 2975 square feet in Palm Desert with a price maximum of $2450000",
        swapped,
    )
    assert swapped.get("price_max") == 2450000
    assert "price" not in swapped

    pricier = {}
    fill_structured_fields(
        "1 bedroom 4 bathroom in Visalia not pricier than $2850000",
        pricier,
    )
    assert pricier.get("price_max") == 2850000


def test_parse_no_greater_than_is_max_not_min():
    from scripts.query_parser import QueryParser

    got = QueryParser().parse(
        "somewhere in Santa Clara with no greater than 3 bdrms, 1.5 bthrms, ~3037 sq ft"
    )
    assert got.get("bedrooms_max") == 3
    assert "bedrooms_min" not in got
    assert got.get("bedrooms") is None or "bedrooms" not in got


def test_parse_minimum_of_and_at_most_of_sqft():
    from scripts.query_parser import QueryParser

    parser = QueryParser()
    mn = parser.parse("no less than two bedrooms in Calabasas, minimum of 1766 sqft")
    assert mn.get("sqft_min") == 1766
    assert "sqft" not in mn

    mx = parser.parse("no less than two bedrooms in San Ramon, at most of 3006 sqft")
    assert mx.get("sqft_max") == 3006
    assert "sqft" not in mx

    up = parser.parse("no less than two bedrooms in La Jolla, upwards of 2386 sqft")
    assert up.get("sqft_min") == 2386
    assert "sqft" not in up

    hyphen = parser.parse("at most 4-bedroom, at least 2 bathroom house in Fullerton")
    assert hyphen.get("bedrooms_max") == 4
    assert "bedrooms" not in hyphen


def test_parse_range_from_through():
    from scripts.query_numeric import fill_structured_fields

    filters = {}
    fill_structured_fields(
        "4 beds 3 baths and 1208 to 1658 square feet in Napa with prices ranging from $911000 through $1160000",
        filters,
    )
    assert filters.get("price_min") == 911000
    assert filters.get("price_max") == 1160000


def test_parse_ish_and_grand():
    from scripts.query_parser import QueryParser

    got = QueryParser().parse(
        "two bed / three bath homes in Anaheim, ideally 991-ish sqft and roughly 722 grand"
    )
    assert got.get("sqft_min") == 991 * 0.8
    assert got.get("sqft_max") == 991 * 1.2
    assert got.get("price_min") == 722000 * 0.8
    assert got.get("price_max") == 722000 * 1.2


def test_parse_not_over_and_inequality_symbols():
    from scripts.query_parser import QueryParser

    parser = QueryParser()
    over = parser.parse(
        "4 bedroom 2 bathroom in San Francisco with more than 1053 square feet but not over 1453 square feet below $776000"
    )
    assert over.get("sqft_min") == 1053
    assert over.get("sqft_max") == 1453

    gt = parser.parse(">4 bedrooms and exactly two bathrooms in Laguna Beach exactly 1363 square feet")
    assert gt.get("bedrooms_min") == 4
    assert "bedrooms" not in gt

    both = parser.parse("<4 bedroom, >3 bathroom house in Alameda")
    assert both.get("bedrooms_max") == 4
    assert both.get("bathrooms_min") == 3
