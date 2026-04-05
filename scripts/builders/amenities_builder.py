import json

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# conver plural to singular
def normalize(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

# build and save amenities for easy access
with open("data/processed/taxonomy_categorized.json", "r") as f:
    taxonomy = json.load(f)
amenities = [item["term"] for item in taxonomy["terms"] if item["category"] == "amenity"]
single_word_amenities = set([
    "backyard",
    "clubhouse",
    "concierge",
    "doorman",
    "elevator",
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
])
views = set([
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
])

# amenities that could appear in different type of amenity group
cross_amenity = set([
    "pool", "swimming pool", # too general, could appear in association amenity, community amenity, pool features
    "guard", # in association and secruity
    "spa", # in spa feature and association, spaYN is 1 is fine
    "security", # in association and security
    "hourse trail", # community and association
    "golf", "golf course", # community and association
    "dog park", "dog", # association and community
    "gated", # in community, but also a security feature
    "park" # community and assoication
])

with open("data/processed/text_features_from_db.json", "r") as f:
    text_features = json.load(f)

# contains plural like trails, horse trails, needs to be normalize. 
# dict is used for converting normalized term back to its form in db
def build_set(amenity_type, terms_to_exclude):
    amenity_set = set()
    processed_to_raw_dict = {}
    amenity_list = text_features[amenity_type]
    for i in amenity_list:
        if i in terms_to_exclude:
            continue
        key = normalize(i)
        if amenity_type == "spa_feature":
            key = key + " spa"
        amenity_set.add(key)
        processed_to_raw_dict[key] = i
    return amenity_set, processed_to_raw_dict


excluded_term_assoc = set([
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

    "pool", # common amenity across several sets
    "guard",
    "secruity",
    "horse trails",
    "park",
    "golf course",
    "dog park"

])
association_amenity_set, association_to_raw_dict = build_set("association_amenity", excluded_term_assoc)

excluded_term_spa = set([
    "no permits",
    "permits",
    "bath"
])
spa_amenity_set, spa_to_raw_dict = build_set("spa_feature", excluded_term_spa)
   

excluded_term_security = set([
    "resident manager",
    "fire sprinkler system",
    "firewalls",
    "carbon monoxide detectors",
    "fire detection system",
    "prewired",
    "smoke detectors",
    "fire rated drywall"
])

security_set, security_to_raw_dict = build_set("security_feature", excluded_term_security)

excluded_term_community = set([
    "storm drains",
    "ravine",
    "sidewalks",
    "street lights",
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
    "rural",

    "pool", # cross set
    "dog park",
    "gated",
    "golf",
    "park",
    "horse trails",
])
community_set, community_to_raw_dict = build_set("community_feature", excluded_term_community)


# pool and view has more cases to consider, handle seperately
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
                | views
                | single_word_amenities
                | cross_amenity
                | association_amenity_set
                | spa_amenity_set
                | security_set
                | community_set
                | pool_set
                | view_set)

set_from_db = {"association": list(association_amenity_set),
                "spa": list(spa_amenity_set),
                "security": list(security_set),
                "community": list(community_set),
                "pool": list(pool_set),
                "view": list(view_set)}

amenity_list = list(set([normalize(a) for a in amenity_set]))
amenity_list = sorted(amenity_list)
amenity_dict = {"amenities": amenity_list,
                "set_from_db": set_from_db,
                "association2raw": association_to_raw_dict,
                "spa2raw": spa_to_raw_dict,
                "security2raw": security_to_raw_dict,
                "community2raw": community_to_raw_dict,
                "pool2raw": pool_to_raw_dict,
                "view2raw": view_to_raw_dict}
                
with open("data/processed/amenities.json", "w") as f:
    json.dump(amenity_dict, f, indent = 2)