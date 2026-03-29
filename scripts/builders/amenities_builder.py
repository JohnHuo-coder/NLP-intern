import json

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def normalize(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

# build and save amenities for easy access
with open("data/processed/taxonomy_categorized.json", "r") as f:
    taxonomy = json.load(f)
amenities = [item["term"] for item in taxonomy["terms"] if item["category"] == "amenity"]
single_word_amenities = (
    "clubhouse",
    "concierge",
    "doorman",
    "elevator",
    "balcony",
    "patio",
    "deck",
    "garage",
    "attic",
    "sauna",
    "jacuzzi",
    "spa",
    "pool",
    "gym",
    "yard",
    "garden",
    "parking",
    "driveway"
)
views = (
    "ocean view",
    "lake view",
    "river view",
    "waterfront",
    "city skyline view",
    "park view",
    "garden view",
    "mountain view",
    "forest view",
    "greenbelt view",
    "city view",
    "street view",
)

with open("data/processed/text_features_from_db.json", "r") as f:
    text_features = json.load(f)

association_amenity = set(text_features["association_amenity"])
terms_to_exclude = set([
    "pets not allowed",
    "maintenance grounds",
    "maintenance front yard",
    "electricity",
    "gas",
    "trash",
    "utilities",
    "other courts",
    "insurance",
    "water",
    "management",
    "hot water",
    "other",
    "cable tv",
    "storage",
    "call for rules",
])
association_amenity_cleaned = association_amenity - terms_to_exclude

spa_types = set(text_features["spa_feature"])
terms_to_exclude = set([
    "no permits",
    "permits",
    "bath"
])
spa_types_cleaned = spa_types - terms_to_exclude
spa_feature = [t + " spa" for t in spa_types_cleaned]

security_amenities = set(text_features["security_feature"])
terms_to_exclude = set([
    "resident manager",
    "fire sprinkler system",
    "firewalls",
    "carbon monoxide detectors",
    "fire detection system",
    "prewired",
    "smoke detectors",
    "fire rated drywall"
])

security_amenity_cleaned = security_amenities - terms_to_exclude


community_amenities = set(text_features["community_feature"])
terms_to_exclude = set([
    "storm drains",
    "horse trails",
    "ravine",
    "sidewalks",
    "street lights",
    "gated",
    "foothills",
    "mountainous",
    "lake",
    "valley",
    "suburban",
    "preserve public land",
    "curbs",
    "gutters",
    "near national forest",
    "urban",
    "military land",
    "rural"
])
community_amenity_cleaned = community_amenities - terms_to_exclude

pool_to_raw_dict = {}
pool = text_features["pool_feature"]
pool_set = set()
for i in pool:
    if i in ["no permits", "permits", "filtered"]:
        continue
    elif i.split()[-1] == "heat":
        key = f"{i}ed pool"
        pool_set.add(key)
    elif i in ["diving board", "pool cover", "pebble", "waterfall"]:
        key = f"pool with {i}"
        pool_set.add(key)
    else:
        key = f"{i} pool"
        pool_set.add(key)
    pool_to_raw_dict[key] = i


view_feature = text_features["view"]
terms_to_exclude = {
    "neighborhood",
    "water",
    "landmark",
    "rocks",
    "catalina",
}
normalize_word = {
    "trees woods": "forest",
    "creek stream": "creek",
    "park greenbelt": "park",
    "peek a boo": "partial ocean",
    "city lights": "city",
    "coastline": "coastal",
}

view_to_raw_dict = {}
view_set = set()
for v in view_feature:
    if v in terms_to_exclude:
        continue
    normalized = normalize_word.get(v, v)
    key = f"{normalized} view"
    view_set.add(key)
    view_to_raw_dict[key] = v



amenity_set = ( set(amenities) 
                | set(single_word_amenities)
                | association_amenity_cleaned
                | set(spa_feature)
                | security_amenity_cleaned
                | community_amenity_cleaned
                | pool_set
                | view_set)

set_from_db = {"association": list(association_amenity_cleaned),
                "spa": spa_feature,
                "security": list(security_amenity_cleaned),
                "community": list(community_amenity_cleaned),
                "pool": list(pool_set),
                "view": list(view_set)}

amenity_list = list(set([normalize(a) for a in amenity_set]))
amenity_list = sorted(amenity_list)
amenity_dict = {"amenities": amenity_list,
                "set_from_db": set_from_db,
                "pool2raw": pool_to_raw_dict,
                "view2raw": view_to_raw_dict}
with open("data/processed/amenities.json", "w") as f:
    json.dump(amenity_dict, f, indent = 2)