"""Build evaluation/queries/rule_extractor_gold.json and verify against QueryParser.

Gold is the intended structured + catalog slots (not leftover-LLM nDCG).
List fields are compared as sets; group fields as lists of sets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from query_parser import QueryParser  # noqa: E402

OUT = ROOT / "evaluation" / "queries" / "rule_extractor_gold.json"


def e(**kwargs):
    return {k: v for k, v in kwargs.items() if v not in (None, [], {})}


# Primary pattern tags (40+). Most queries also mix amenity / feature / finance.
CASES = [
    # --- bedrooms ---
    {"id": 1, "patterns": ["bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom house in Irvine with a pool",
     "expect": e(city="Irvine", bedrooms=3, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 2, "patterns": ["bed_hyphen", "city_simple", "amenity"],
     "query": "2-bedroom condo in Pasadena with a garage",
     "expect": e(city="Pasadena", bedrooms=2, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 3, "patterns": ["bed_br", "city_simple", "amenity"],
     "query": "3 br home in Sacramento with a gym",
     "expect": e(city="Sacramento", bedrooms=3, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 4, "patterns": ["bed_bd", "city_simple", "amenity"],
     "query": "2bd house in Fresno with parking",
     "expect": e(city="Fresno", bedrooms=2, amenities=["parking"], amenity_groups=[["parking"]])},
    {"id": 5, "patterns": ["bed_word", "city_simple", "amenity"],
     "query": "three bedroom home in San Diego with a spa",
     "expect": e(city="San Diego", bedrooms=3, amenities=["spa"], amenity_groups=[["spa"]])},
    {"id": 6, "patterns": ["bed_plus", "city_simple", "feature"],
     "query": "3+ bedroom house in Oakland with a fireplace",
     "expect": e(city="Oakland", bedrooms_min=3, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 7, "patterns": ["bed_at_least", "city_simple", "feature"],
     "query": "at least 4 bedroom home in San Jose with hardwood flooring",
     "expect": e(city="San Jose", bedrooms_min=4, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 8, "patterns": ["bed_or_more", "city_simple", "amenity"],
     "query": "2 bedroom or more in Long Beach with a pool",
     "expect": e(city="Long Beach", bedrooms_min=2, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 9, "patterns": ["bed_under", "city_simple", "amenity"],
     "query": "under 4 bedroom in Anaheim with a garage",
     "expect": e(city="Anaheim", bedrooms_max=4, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 10, "patterns": ["bed_from_to", "city_simple", "finance"],
     "query": "from 3 to 5 bedroom in Riverside with FHA loan",
     "expect": e(city="Riverside", bedrooms_min=3, bedrooms_max=5, finance=["fha loan"], finance_groups=[["fha loan"]])},
    {"id": 11, "patterns": ["bed_between", "city_simple", "finance"],
     "query": "between 2 and 4 bedroom in Irvine with VA loan",
     "expect": e(city="Irvine", bedrooms_min=2, bedrooms_max=4, finance=["va loan"], finance_groups=[["va loan"]])},
    {"id": 12, "patterns": ["bed_dash", "city_simple", "finance"],
     "query": "3-4 bedroom house in Bakersfield with cash",
     "expect": e(city="Bakersfield", bedrooms_min=3, bedrooms_max=4, finance=["cash"], finance_groups=[["cash"]])},
    {"id": 13, "patterns": ["bed_around", "city_multi", "amenity"],
     "query": "around 3 bedroom in Santa Monica with a pool",
     "expect": e(city="Santa Monica", bedrooms_min=2, bedrooms_max=4, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 14, "patterns": ["bed_gt", "city_simple", "amenity"],
     "query": "> 3 bedroom in Los Angeles with a gym",
     "expect": e(city="Los Angeles", bedrooms_min=3, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 15, "patterns": ["bed_from_min", "city_simple", "amenity"],
     "query": "from 4 bedroom in Irvine with a garage",
     "expect": e(city="Irvine", bedrooms_min=4, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 16, "patterns": ["bed_minimum", "city_simple", "amenity"],
     "query": "minimum 3 bedroom in Pasadena with parking",
     "expect": e(city="Pasadena", bedrooms_min=3, amenities=["parking"], amenity_groups=[["parking"]])},
    {"id": 17, "patterns": ["bed_at_most", "city_simple", "feature"],
     "query": "at most 3 bedroom in Sacramento with carpet",
     "expect": e(city="Sacramento", bedrooms_max=3, features=["carpet"], feature_groups=[["carpet"]])},
    {"id": 18, "patterns": ["bed_less_than", "city_simple", "feature"],
     "query": "less than 5 bedroom in Fresno with a dishwasher",
     "expect": e(city="Fresno", bedrooms_max=5, features=["dishwasher"], feature_groups=[["dishwasher"]])},
    {"id": 19, "patterns": ["bed_below", "city_simple", "amenity"],
     "query": "below 4 bedroom in San Diego with a spa",
     "expect": e(city="San Diego", bedrooms_max=4, amenities=["spa"], amenity_groups=[["spa"]])},
    {"id": 20, "patterns": ["bed_max", "city_simple", "amenity", "feature"],
     "query": "max 4 bedroom in Oakland with an elevator",
     "expect": e(city="Oakland", bedrooms_max=4, amenities=["elevator"], amenity_groups=[["elevator"]], features=["elevator"], feature_groups=[["elevator"]])},

    # --- bathrooms ---
    {"id": 21, "patterns": ["bath_exact", "city_simple", "amenity"],
     "query": "2 bathroom condo in Irvine with a pool",
     "expect": e(city="Irvine", bathrooms=2, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 22, "patterns": ["bath_half", "city_simple", "amenity"],
     "query": "2.5 bathroom house in Sacramento with a garage",
     "expect": e(city="Sacramento", bathrooms=2, bathroom_half=1, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 23, "patterns": ["bath_plus", "city_simple", "feature"],
     "query": "2+ bathroom in Oakland with a fireplace",
     "expect": e(city="Oakland", bathrooms_min=2, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 24, "patterns": ["bath_under", "city_simple", "amenity"],
     "query": "under 3 bathroom in San Diego with a gym",
     "expect": e(city="San Diego", bathrooms_max=3, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 25, "patterns": ["bath_from_to", "city_simple", "finance"],
     "query": "from 2 to 3 bathroom in Fresno with FHA",
     "expect": e(city="Fresno", bathrooms_min=2, bathrooms_max=3, finance=["fha"], finance_groups=[["fha"]])},
    {"id": 26, "patterns": ["bath_between", "city_simple", "amenity"],
     "query": "between 1 and 2 bathroom in Long Beach with a pool",
     "expect": e(city="Long Beach", bathrooms_min=1, bathrooms_max=2, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 27, "patterns": ["bath_dash", "city_simple", "finance"],
     "query": "2-3 bathroom in Anaheim with cash",
     "expect": e(city="Anaheim", bathrooms_min=2, bathrooms_max=3, finance=["cash"], finance_groups=[["cash"]])},
    {"id": 28, "patterns": ["bath_around", "city_simple", "feature"],
     "query": "around 2 bathroom in San Jose with hardwood flooring",
     "expect": e(city="San Jose", bathrooms_min=1, bathrooms_max=3, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 29, "patterns": ["bath_word", "city_multi", "amenity"],
     "query": "two bathroom condo in Santa Monica with a spa",
     "expect": e(city="Santa Monica", bathrooms=2, amenities=["spa"], amenity_groups=[["spa"]])},
    {"id": 30, "patterns": ["bath_hyphen", "city_simple", "amenity"],
     "query": "2-bath house in Riverside with a garage",
     "expect": e(city="Riverside", bathrooms=2, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 31, "patterns": ["bath_ba", "city_simple", "amenity"],
     "query": "2 ba condo in Irvine with parking",
     "expect": e(city="Irvine", bathrooms=2, amenities=["parking"], amenity_groups=[["parking"]])},
    {"id": 32, "patterns": ["bath_full", "city_simple", "amenity"],
     "query": "2 full bathroom in Pasadena with a pool",
     "expect": e(city="Pasadena", bathrooms=2, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 33, "patterns": ["bath_or_more", "city_simple", "amenity"],
     "query": "2 bathroom or more in Oakland with a gym",
     "expect": e(city="Oakland", bathrooms_min=2, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 34, "patterns": ["bath_at_least", "city_simple", "feature"],
     "query": "at least 2 bathroom in Sacramento with a fireplace",
     "expect": e(city="Sacramento", bathrooms_min=2, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 35, "patterns": ["bath_half_plus", "city_simple", "finance"],
     "query": "2.5+ bathroom in San Jose with VA loan",
     "expect": e(city="San Jose", bathrooms_min=2, finance=["va loan"], finance_groups=[["va loan"]])},
    {"id": 36, "patterns": ["bath_half_hyphen", "city_simple", "amenity"],
     "query": "2.5-bath home in Long Beach with a patio",
     "expect": e(city="Long Beach", bathrooms=2, bathroom_half=1, amenities=["patio"], amenity_groups=[["patio"]])},

    # --- price max family ---
    {"id": 37, "patterns": ["price_under", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Irvine under $800000 with a pool",
     "expect": e(city="Irvine", bedrooms=3, price_max=800000, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 38, "patterns": ["price_below", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Pasadena below $900000 with a garage",
     "expect": e(city="Pasadena", bedrooms=3, price_max=900000, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 39, "patterns": ["price_at_most", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Sacramento at most $750000 with a fireplace",
     "expect": e(city="Sacramento", bedrooms=3, price_max=750000, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 40, "patterns": ["price_less_than", "bed_exact", "city_simple", "amenity"],
     "query": "2 bedroom in Fresno less than $650000 with a gym",
     "expect": e(city="Fresno", bedrooms=2, price_max=650000, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 41, "patterns": ["price_cheaper_than", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in San Diego cheaper than $800000 with a spa",
     "expect": e(city="San Diego", bedrooms=3, price_max=800000, amenities=["spa"], amenity_groups=[["spa"]])},
    {"id": 42, "patterns": ["price_up_to", "bed_exact", "city_simple", "finance"],
     "query": "3 bedroom in Oakland up to $700000 with FHA loan",
     "expect": e(city="Oakland", bedrooms=3, price_max=700000, finance=["fha loan"], finance_groups=[["fha loan"]])},
    {"id": 43, "patterns": ["price_max", "bed_exact", "city_simple", "amenity"],
     "query": "2 bedroom in Long Beach max $600000 with parking",
     "expect": e(city="Long Beach", bedrooms=2, price_max=600000, amenities=["parking"], amenity_groups=[["parking"]])},
    {"id": 44, "patterns": ["price_no_more_than", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Anaheim no more than $850000 with hardwood flooring",
     "expect": e(city="Anaheim", bedrooms=3, price_max=850000, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 45, "patterns": ["price_capped_at", "bed_exact", "city_simple", "finance"],
     "query": "4 bedroom in Riverside capped at $900000 with cash",
     "expect": e(city="Riverside", bedrooms=4, price_max=900000, finance=["cash"], finance_groups=[["cash"]])},
    {"id": 46, "patterns": ["price_or_less", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Bakersfield $750000 or less with a pool",
     "expect": e(city="Bakersfield", bedrooms=3, price_max=750000, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 47, "patterns": ["price_maximum", "bed_exact", "city_simple", "amenity"],
     "query": "2 bedroom in Irvine maximum $550000 with a garage",
     "expect": e(city="Irvine", bedrooms=2, price_max=550000, amenities=["garage"], amenity_groups=[["garage"]])},

    # --- price min family ---
    {"id": 48, "patterns": ["price_plus_k", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in San Jose $800k+ with a gym",
     "expect": e(city="San Jose", bedrooms=3, price_min=800000, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 49, "patterns": ["price_plus", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Los Angeles 900000+ with a fireplace",
     "expect": e(city="Los Angeles", bedrooms=3, price_min=900000, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 50, "patterns": ["price_at_least", "bed_exact", "city_simple", "finance"],
     "query": "3 bedroom in Irvine at least $500000 with VA loan",
     "expect": e(city="Irvine", bedrooms=3, price_min=500000, finance=["va loan"], finance_groups=[["va loan"]])},
    {"id": 51, "patterns": ["price_over", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Pasadena over $600000 with a pool",
     "expect": e(city="Pasadena", bedrooms=3, price_min=600000, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 52, "patterns": ["price_more_than", "bed_exact", "city_simple", "amenity"],
     "query": "2 bedroom in Sacramento more than $400000 with parking",
     "expect": e(city="Sacramento", bedrooms=2, price_min=400000, amenities=["parking"], amenity_groups=[["parking"]])},
    {"id": 53, "patterns": ["price_from_min", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Fresno from $500000 with carpet",
     "expect": e(city="Fresno", bedrooms=3, price_min=500000, features=["carpet"], feature_groups=[["carpet"]])},
    {"id": 54, "patterns": ["price_starting_from", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in San Diego starting from $700000 with a spa",
     "expect": e(city="San Diego", bedrooms=3, price_min=700000, amenities=["spa"], amenity_groups=[["spa"]])},
    {"id": 55, "patterns": ["price_no_less_than", "bed_exact", "city_simple", "finance"],
     "query": "4 bedroom in Oakland no less than $800000 with conventional loan",
     "expect": e(city="Oakland", bedrooms=4, price_min=800000, finance=["conventional loan"], finance_groups=[["conventional loan"]])},

    # --- price range / approx / formats ---
    {"id": 56, "patterns": ["price_from_to", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Long Beach from $500000 to $900000 with a garage",
     "expect": e(city="Long Beach", bedrooms=3, price_min=500000, price_max=900000, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 57, "patterns": ["price_between", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Anaheim between $400000 and $800000 with a pool",
     "expect": e(city="Anaheim", bedrooms=3, price_min=400000, price_max=800000, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 58, "patterns": ["price_dash", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Riverside $500000-$900000 with hardwood flooring",
     "expect": e(city="Riverside", bedrooms=3, price_min=500000, price_max=900000, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 59, "patterns": ["price_around", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Irvine around $500000 with a gym",
     "expect": e(city="Irvine", bedrooms=3, price_min=400000.0, price_max=600000.0, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 60, "patterns": ["price_about", "bed_exact", "city_simple", "amenity"],
     "query": "2 bedroom in Pasadena about $600000 with a patio",
     "expect": e(city="Pasadena", bedrooms=2, price_min=480000.0, price_max=720000.0, amenities=["patio"], amenity_groups=[["patio"]])},
    {"id": 61, "patterns": ["price_approximately", "bed_exact", "city_simple", "finance"],
     "query": "3 bedroom in Sacramento approximately $750000 with FHA",
     "expect": e(city="Sacramento", bedrooms=3, price_min=600000.0, price_max=900000.0, finance=["fha"], finance_groups=[["fha"]])},
    {"id": 62, "patterns": ["price_roughly", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Fresno roughly $650000 with a dishwasher",
     "expect": e(city="Fresno", bedrooms=3, price_min=520000.0, price_max=780000.0, features=["dishwasher"], feature_groups=[["dishwasher"]])},
    {"id": 63, "patterns": ["price_near", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in San Diego near $800000 with a pool",
     "expect": e(city="San Diego", bedrooms=3, price_min=640000.0, price_max=960000.0, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 64, "patterns": ["price_k", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Oakland at $750k with a garage",
     "expect": e(city="Oakland", bedrooms=3, price=750000, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 65, "patterns": ["price_million", "bed_exact", "city_multi", "amenity"],
     "query": "2 bedroom in Santa Monica at $1.5 million with a spa",
     "expect": e(city="Santa Monica", bedrooms=2, price=1500000, amenities=["spa"], amenity_groups=[["spa"]])},
    {"id": 66, "patterns": ["price_m", "bed_exact", "city_simple", "finance"],
     "query": "3 bedroom in San Jose at $1.2m with cash",
     "expect": e(city="San Jose", bedrooms=3, price=1200000, finance=["cash"], finance_groups=[["cash"]])},
    {"id": 67, "patterns": ["price_comma", "bed_exact", "city_simple", "feature"],
     "query": "4 bedroom in Los Angeles at $1,200,000 with a fireplace",
     "expect": e(city="Los Angeles", bedrooms=4, price=1200000, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 68, "patterns": ["price_exactly", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Irvine exactly $600000 with parking",
     "expect": e(city="Irvine", bedrooms=3, price=600000, amenities=["parking"], amenity_groups=[["parking"]])},
    {"id": 69, "patterns": ["price_k_range", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Pasadena from $500k to $900k with a pool",
     "expect": e(city="Pasadena", bedrooms=3, price_min=500000, price_max=900000, amenities=["pool"], amenity_groups=[["pool"]])},

    # --- sqft ---
    {"id": 70, "patterns": ["sqft_plus", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Irvine 900+ sqft with a garage",
     "expect": e(city="Irvine", bedrooms=3, sqft_min=900, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 71, "patterns": ["sqft_at_least", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Sacramento at least 1800 sqft with hardwood flooring",
     "expect": e(city="Sacramento", bedrooms=3, sqft_min=1800, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 72, "patterns": ["sqft_under", "bed_exact", "city_simple", "amenity"],
     "query": "2 bedroom in Fresno under 1200 sqft with a gym",
     "expect": e(city="Fresno", bedrooms=2, sqft_max=1200, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 73, "patterns": ["sqft_from_to", "bed_exact", "city_simple", "finance"],
     "query": "3 bedroom in San Diego from 900 to 1800 sqft with FHA loan",
     "expect": e(city="San Diego", bedrooms=3, sqft_min=900, sqft_max=1800, finance=["fha loan"], finance_groups=[["fha loan"]])},
    {"id": 74, "patterns": ["sqft_tilde", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Oakland ~ 1500 sqft with a pool",
     "expect": e(city="Oakland", bedrooms=3, sqft_min=1200.0, sqft_max=1800.0, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 75, "patterns": ["sqft_comma", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Long Beach with 1,500 sqft and a garage",
     "expect": e(city="Long Beach", bedrooms=3, sqft=1500, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 76, "patterns": ["sqft_sq_ft", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Anaheim with 1800 sq. ft and a fireplace",
     "expect": e(city="Anaheim", bedrooms=3, sqft=1800, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 77, "patterns": ["sqft_exact", "bed_exact", "city_simple", "amenity"],
     "query": "2 bedroom in Riverside at 2000 square feet with a spa",
     "expect": e(city="Riverside", bedrooms=2, sqft=2000, amenities=["spa"], amenity_groups=[["spa"]])},
    {"id": 78, "patterns": ["sqft_between", "bed_exact", "city_simple", "finance"],
     "query": "4 bedroom in San Jose between 900 and 1800 sqft with VA loan",
     "expect": e(city="San Jose", bedrooms=4, sqft_min=900, sqft_max=1800, finance=["va loan"], finance_groups=[["va loan"]])},
    {"id": 79, "patterns": ["sqft_or_more", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Los Angeles 1600 square feet or more with parking",
     "expect": e(city="Los Angeles", bedrooms=3, sqft_min=1600, amenities=["parking"], amenity_groups=[["parking"]])},
    {"id": 80, "patterns": ["sqft_dash", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Bakersfield 900-1800 sqft with carpet",
     "expect": e(city="Bakersfield", bedrooms=3, sqft_min=900, sqft_max=1800, features=["carpet"], feature_groups=[["carpet"]])},

    # --- city variants ---
    {"id": 81, "patterns": ["city_laguna", "bed_exact", "amenity"],
     "query": "3 bedroom in Laguna Beach with a pool",
     "expect": e(city="Laguna Beach", bedrooms=3, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 82, "patterns": ["city_thousand_oaks", "city_multi_nocue", "bed_exact", "feature"],
     "query": "4 bedroom Thousand Oaks with a fireplace",
     "expect": e(city="Thousand Oaks", bedrooms=4, features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 83, "patterns": ["city_multi_nocue", "city_oak_park", "bed_exact", "amenity"],
     "query": "3 bedroom Oak Park with a garage",
     "expect": e(city="Oak Park", bedrooms=3, amenities=["garage", "park"], amenity_groups=[["park"], ["garage"]])},
    {"id": 84, "patterns": ["city_cue_walnut", "bed_exact", "feature"],
     "query": "3 bedroom in Walnut with hardwood flooring",
     "expect": e(city="Walnut", bedrooms=3, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 85, "patterns": ["city_cue_orange", "bed_exact", "amenity"],
     "query": "2 bedroom near Orange with a gym",
     "expect": e(city="Orange", bedrooms=2, amenities=["gym"], amenity_groups=[["gym"]])},
    {"id": 86, "patterns": ["city_located_in", "bed_exact", "finance"],
     "query": "3 bedroom located in Irvine with FHA loan",
     "expect": e(city="Irvine", bedrooms=3, finance=["fha loan"], finance_groups=[["fha loan"]])},
    {"id": 87, "patterns": ["city_city_of", "bed_exact", "amenity"],
     "query": "2 bedroom located in the city of Pasadena with a pool",
     "expect": e(city="Pasadena", bedrooms=2, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 88, "patterns": ["city_longest", "bed_exact", "amenity"],
     "query": "3 bedroom in Walnut Creek with a spa",
     "expect": e(city="Walnut Creek", bedrooms=3, amenities=["spa"], amenity_groups=[["spa"]])},

    # --- catalog: abbrev, AND/OR, negation, mixed ---
    {"id": 89, "patterns": ["abbrev_gar", "bed_exact", "city_simple"],
     "query": "3 bedroom in Irvine with a gar",
     "expect": e(city="Irvine", bedrooms=3, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 90, "patterns": ["abbrev_wd", "bed_exact", "city_simple"],
     "query": "2 bedroom in Sacramento with w/d",
     "expect": e(city="Sacramento", bedrooms=2, amenities=["washer dryer"], amenity_groups=[["washer dryer"]], features=["dryer", "washer"], feature_groups=[["washer"], ["dryer"]])},
    {"id": 91, "patterns": ["abbrev_fp", "abbrev_wo", "neg_without", "bed_exact", "city_simple"],
     "query": "3 bedroom in Fresno with fp, w/o a pool",
     "expect": e(city="Fresno", bedrooms=3, features=["fireplace"], feature_groups=[["fireplace"]], negated_amenities=["pool"])},
    {"id": 92, "patterns": ["abbrev_pkg", "bed_exact", "city_simple"],
     "query": "2 bedroom in San Diego with pkg",
     "expect": e(city="San Diego", bedrooms=2, amenities=["parking space"], amenity_groups=[["parking space"]])},
    {"id": 93, "patterns": ["abbrev_bbq", "bed_exact", "city_simple"],
     "query": "3 bedroom in Oakland with a bbq",
     "expect": e(city="Oakland", bedrooms=3, amenities=["barbecue"], amenity_groups=[["barbecue"]])},
    {"id": 94, "patterns": ["abbrev_hw", "bed_exact", "city_simple"],
     "query": "3 bedroom in Long Beach with hw flooring",
     "expect": e(city="Long Beach", bedrooms=3, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 95, "patterns": ["abbrev_ac", "bed_exact", "city_simple"],
     "query": "2 bedroom in Anaheim with ac",
     "expect": e(city="Anaheim", bedrooms=2, amenities=["air conditioning"], amenity_groups=[["air conditioning"]], features=["air conditioning"], feature_groups=[["air conditioning"]])},
    {"id": 96, "patterns": ["or_or", "bed_exact", "city_simple"],
     "query": "3 bedroom in Irvine with a pool or gym",
     "expect": e(city="Irvine", bedrooms=3, amenities=["gym", "pool"], amenity_groups=[["pool", "gym"]])},
    {"id": 97, "patterns": ["or_slash", "bed_exact", "city_simple"],
     "query": "3 bedroom in Pasadena with pool/gym",
     "expect": e(city="Pasadena", bedrooms=3, amenities=["gym", "pool"], amenity_groups=[["pool", "gym"]])},
    {"id": 98, "patterns": ["and_and", "bed_exact", "city_simple", "feature"],
     "query": "3 bedroom in Sacramento with a pool and garage and a fireplace",
     "expect": e(city="Sacramento", bedrooms=3, amenities=["garage", "pool"], amenity_groups=[["pool"], ["garage"]], features=["fireplace"], feature_groups=[["fireplace"]])},
    {"id": 99, "patterns": ["neg_no", "neg_without_any", "neg_does_not_have", "bed_exact", "city_simple", "finance"],
     "query": "3 bedroom in San Jose with FHA loan, does not have a pool, without any garage",
     "expect": e(city="San Jose", bedrooms=3, finance=["fha loan"], finance_groups=[["fha loan"]], negated_amenities=["garage", "pool"])},
    {"id": 100, "patterns": ["mixed_aff", "or_andor", "neg_doesnt_have", "neg_finance", "bed_exact", "bath_exact", "price_under", "city_simple"],
     "query": "3 bedroom 2 bathroom in Irvine under $900000 with pool and/or gym, hardwood flooring, doesn't have a spa, no FHA",
     "expect": e(
         city="Irvine", bedrooms=3, bathrooms=2, price_max=900000,
         amenities=["gym", "pool"], amenity_groups=[["pool", "gym"]],
         features=["hardwood flooring"], feature_groups=[["hardwood flooring"]],
         negated_amenities=["spa"],
         negated_finance=["fha"],
     )},

    # --- held-out gaps: intended gold, current parser does not match ---
    {"id": 101, "hard": True, "patterns": ["hard_min_abbrev", "city_simple", "amenity"],
     "query": "min 3 bedroom in Irvine with a pool",
     "expect": e(city="Irvine", bedrooms_min=3, amenities=["pool"], amenity_groups=[["pool"]])},
    {"id": 102, "patterns": ["sqft_plus", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Irvine 1500+ sqft with a garage",
     "expect": e(city="Irvine", bedrooms=3, sqft_min=1500, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 103, "hard": True, "patterns": ["hard_city_need_cue", "bed_exact", "feature"],
     "query": "Walnut 3 bedroom with hardwood flooring",
     "expect": e(city="Walnut", bedrooms=3, features=["hardwood flooring"], feature_groups=[["hardwood flooring"]])},
    {"id": 104, "hard": True, "patterns": ["hard_one_and_a_half_bath", "bed_exact", "city_simple", "amenity"],
     "query": "3 bedroom in Irvine with one and a half bathroom and a garage",
     "expect": e(city="Irvine", bedrooms=3, bathrooms=1, bathroom_half=1, amenities=["garage"], amenity_groups=[["garage"]])},
    {"id": 105, "hard": True, "patterns": ["hard_negation_or", "bed_exact", "city_simple"],
     "query": "3 bedroom in Irvine with no pool or gym",
     "expect": e(city="Irvine", bedrooms=3, negated_amenities=["gym", "pool"])},
    {"id": 106, "hard": True, "patterns": ["hard_studio", "city_simple", "amenity"],
     "query": "studio in Irvine with a pool",
     "expect": e(city="Irvine", bedrooms=0, amenities=["pool"], amenity_groups=[["pool"]])},
]


LIST_KEYS = (
    "amenities", "features", "finance",
    "negated_amenities", "negated_features", "negated_finance",
)
GROUP_KEYS = ("amenity_groups", "feature_groups", "finance_groups")
SCALAR_SKIP = set(LIST_KEYS) | set(GROUP_KEYS)


def _norm_list(v):
    return sorted(v) if v else []


def _norm_groups(v):
    return [sorted(g) for g in (v or [])]


def compare(expect: dict, got: dict) -> list[str]:
    diffs = []
    keys = set(expect) | set(got)
    for k in sorted(keys):
        ev, gv = expect.get(k), got.get(k)
        if k in LIST_KEYS:
            if _norm_list(ev or []) != _norm_list(gv or []):
                diffs.append(f"{k}: expect {sorted(ev or [])} got {sorted(gv or [])}")
        elif k in GROUP_KEYS:
            if _norm_groups(ev) != _norm_groups(gv):
                diffs.append(f"{k}: expect {ev} got {gv}")
        else:
            if ev != gv:
                diffs.append(f"{k}: expect {ev!r} got {gv!r}")
    return diffs


def _dump_case(c: dict) -> dict:
    row = {"id": c["id"], "patterns": c["patterns"], "query": c["query"], "expect": c["expect"]}
    if c.get("hard"):
        row["hard"] = True
    return row


def main() -> int:
    ids = [c["id"] for c in CASES]
    assert ids == list(range(1, len(CASES) + 1)), ids
    hard_ids = {c["id"] for c in CASES if c.get("hard")}

    parser = QueryParser()
    unexpected = []
    expected_fail = []
    unexpected_pass = []
    n_pass = 0
    for case in CASES:
        got = parser.parse(case["query"])
        diffs = compare(case["expect"], got)
        if not diffs:
            n_pass += 1
            if case.get("hard"):
                unexpected_pass.append(case["id"])
            continue
        rec = (case["id"], case["query"], diffs, got)
        if case.get("hard"):
            expected_fail.append(rec)
        else:
            unexpected.append(rec)

    n = len(CASES)
    payload = {
        "note": (
            "Rule-extractor gold: coverage queries (1–100) plus a few held-out gaps "
            "(`hard: true`). Coverage gold matches current QueryParser.parse() after cleaner "
            "expansion. Hard-case gold is the intended slot fill, not the parser dump — "
            "those are known misses (min→minute, cue-required city, "
            "spelled half-bath, negation+OR, studio). Compare list fields as sets; group "
            "fields as lists of sets. Structured slot match and catalog match should be "
            "reported separately. Dual-taxonomy hits (elevator/ac/washer-dryer) are in both "
            "lists on coverage items."
        ),
        "queries": [_dump_case(c) for c in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({n} queries, {len(hard_ids)} hard)")
    print(f"pass {n_pass}/{n} = {100.0 * n_pass / n:.1f}%")

    if expected_fail:
        print(f"hard misses {len(expected_fail)}/{len(hard_ids)} (expected)")
        for i, q, diffs, got in expected_fail:
            print(f"\n#{i} {q}")
            for d in diffs:
                print(f"  {d}")
            print(f"  got={got}")

    if unexpected_pass:
        print(f"WARNING hard cases now pass: {unexpected_pass}")
    if unexpected:
        print(f"UNEXPECTED coverage miss {len(unexpected)}")
        for i, q, diffs, got in unexpected:
            print(f"\n#{i} {q}")
            for d in diffs:
                print(f"  {d}")
            print(f"  got={got}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
