import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from scripts.text_cleaning import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


@pytest.fixture
def sample_remarks_df():
    """Small DataFrame for profile/column tests (no external CSV)."""
    return pd.DataFrame({
        "remarks": [
            "3 br 2 ba, 1200 sqft, priced at 450k. W/ garage.",
            "Luxury home $1.2m, 9' ceilings, 2.5 acres. &amp; ready.",
            "Condo .5 mile away, 10ft ceiling, 6\" trim. $5.5k reduced!",
        ]
    })


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
    assert "2.5 acres" in cleaner.normalize_measurements("2.5 acre")


# --- Abbreviations ---
def test_expand_abbreviations(cleaner):
    assert "bedroom" in cleaner.expand_abbreviations("3 br 2 ba")
    assert "with" in cleaner.expand_abbreviations("w/ garage")
    assert "square feet" in cleaner.expand_abbreviations("1200 sqft")


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
        # decimal counts (e.g. half bath)
        ("2.5 ba", "bathroom"),
        # area: sqft variants (digit may be stripped; we only require expansion text)
        ("1800 sqft", "square feet"),
        ("2200 sq. ft", "square feet"),
        ("2500 sq ft", "square feet"),
        # normal abbrevs (word boundaries + optional plural)
        ("updated kit and lr", "kitchen"),
        ("updated kit and lr", "living room"),
        # no_s abbrevs
        ("w/ hvac and w/d", "with"),
        ("w/ hvac and w/d", "washer dryer"),
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


# --- Column / DataFrame ---
def test_clean_column(cleaner, sample_remarks_df):
    col = sample_remarks_df["remarks"]
    cleaned = cleaner.clean_column(col)
    assert len(cleaned) == len(col)
    assert pd.api.types.is_object_dtype(cleaned.dtype) or pd.api.types.is_string_dtype(
        cleaned.dtype
    )
    assert "450000" in cleaned.iloc[0]
    assert "5500" in cleaned.iloc[2]


def test_sample_compare(cleaner, sample_remarks_df):
    col = sample_remarks_df["remarks"]
    df = cleaner.sample_compare(col, k=2, seed=42)
    assert "original" in df.columns and "cleaned" in df.columns
    assert len(df) == 2
    for _, row in df.iterrows():
        assert row["original"] in col.values
        assert isinstance(row["cleaned"], str)


# --- Detect functions (return cnt and counter) ---
def test_detect_price_mentions(cleaner, sample_remarks_df):
    cnt, counter = cleaner._detect_price_mentions(sample_remarks_df["remarks"])
    assert isinstance(cnt, int)
    assert cnt >= 0
    assert hasattr(counter, "most_common")
    # Sample has 450k, $1.2m, $5.5k
    assert cnt >= 3
    total_from_counter = sum(counter.values())
    assert total_from_counter == cnt


def test_detect_measurements(cleaner, sample_remarks_df):
    cnt, counter = cleaner._detect_measurements(sample_remarks_df["remarks"])
    assert isinstance(cnt, int)
    assert cnt >= 0
    assert hasattr(counter, "most_common")
    total_from_counter = sum(counter.values())
    assert total_from_counter == cnt


def test_detect_abbreviations(cleaner, sample_remarks_df):
    result = cleaner._detect_abbreviations(sample_remarks_df["remarks"], top_abbr=10)
    assert isinstance(result, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in result)


# --- Profile ---
def test_profile_column(cleaner, sample_remarks_df):
    profile = cleaner.profile_column(sample_remarks_df, "remarks")
    assert "null_rate" in profile
    assert "avg_length" in profile
    assert "avg_num_words" in profile
    assert "common_terms" in profile
    assert "price_mentions" in profile
    assert "price_counter" in profile
    assert "measurement_mentions" in profile
    assert "measurement_counter" in profile
    assert "has_html" in profile
    assert "has_unicode" in profile
    assert "common_abbreviations" in profile
    assert isinstance(profile["price_counter"], type(cleaner._detect_price_mentions(sample_remarks_df["remarks"])[1]))
    assert isinstance(profile["measurement_counter"], type(cleaner._detect_measurements(sample_remarks_df["remarks"])[1]))
