import json
from collections import defaultdict
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def normalize(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

# build and save amenities for easy access
with open("data/processed/taxonomy_categorized.json", "r") as f:
    taxonomy = json.load(f)

features = [normalize(item["term"]) for item in taxonomy["terms"] if item["category"] == "interior feature"]

single_word_features = (
    "island", "pantry", "range", "oven",
    "microwave", "dishwasher",
    "refrigerator", "hood",
    "tub", "shower", "vanity", "sink",
    "heating and cooling", "air conditioning", "heating",
    "fireplace", "basement"
)

with open("data/processed/text_features_from_db.json", "r") as f:
    text_features = json.load(f)

interior_feature = text_features["interior_feature"]
excluded_term_interior = set([
    "built in features",
    "tandem",
    "paneling wainscoting",
    "partially furnished",
    "furnished",
    "unfurnished",
    "unfinished walls",
    "wired for sound", 
    "wired for data"
])

normalize_feature = {
    "wood product walls": ["wood product wall", "wood paneling"],
    "entrance foyer": ["entrance foyer", "entryway", "entry hall"],
    "chair climber": ["chair climber", "chair lift"],
    "bedroom on main level": ["bedroom on main level", "bedroom on main floor", "first floor bedroom"],
    "french doors atrium doors": ["french door", "atrium door"],
    "main level primary": ["primary bedroom on main floor", "master bedroom on first floor", "first floor bedroom", "main floor bedroom"],
    "all bedrooms up": ["all bedrooms on the same floor", "bedrooms upstairs", "all bedrooms together"],
    "all bedrooms down": ["all bedrooms on the same floor", "bedrooms downstairs", "all bedrooms together"],
    "atrium": ["house with lots of natural light", "bright open space", "indoor garden space"],
    "in law floorplan": ["house for my parents", "home for elderly parents", 
                        "separate living space", "separate area", "adu", "extra unit",
                        "guest suite", "extra space for guests"],
    "living room deck attached": ["living room with deck", "living room with attached deck"]
}
interior_to_raw_dict = defaultdict(list) # 2 different raw term can be match to same normalized term. 
interior_feature_set = set()
for i in interior_feature:
    if i in excluded_term_interior:
        continue
    normalized_terms = normalize_feature.get(i, [i])
    for term in normalized_terms:
        norm_term = normalize(term)
        interior_feature_set.add(norm_term)
        interior_to_raw_dict[norm_term].append(i)
   

floor_types = text_features["flr_type"]
floor_to_raw_dict = {}
floor_set = set()
for i in floor_types:
    key1 = f"{i} floor"
    key2 = f"{i} flooring"
    floor_to_raw_dict[key1] = i
    floor_to_raw_dict[key2] = i
    floor_set.add(key1)
    floor_set.add(key2)

appl = text_features["appl"]
excluded_term_appl = set([
    "propane cooking",
    "electric cooking",
    "gas cooking",
    "built in",
    "free standing",
    "no hot water",
    "barbecue",
    "water to refrigerator",
    "counter top",
])
normalize_appl = {
    "disposal": "garbage disposal"
}
appl_to_raw_dict = {}
appl_set = set()
for i in appl:
    if i in excluded_term_appl:
        continue
    normalized = normalize_appl.get(i, i)
    normalized = normalize(normalized)
    appl_set.add(normalized)
    appl_to_raw_dict[normalized] = i
    
cooling = text_features["cooling"]
excluded_term_cooling = set([
    "dual",
    "gas", # almost always followed by heating
    "electric", # same as gas, followed by heating
    "energy star qualified equipment"
])
normalize_cooling = {
    "ductless": ["ductless AC", "ductless", "ductless cooling", "ductless system", "mini split"],
    "high efficiency": ["energy efficient", "high efficiency", "energy-saving"],
    "zoned": ["zoned cooling", "zoned heating and cooling"]
}
cooling_to_raw_dict = {}
cooling_set = set()
for raw in cooling:
    if raw in excluded_term_cooling:
        continue
    normalized_terms = normalize_cooling.get(raw, [raw])
    for term in normalized_terms:
        norm_term = normalize(term)
        cooling_set.add(norm_term)
        cooling_to_raw_dict[norm_term] = raw

heating = text_features["heating"]
excluded_term_heating = set([
    "combination",
    "gravity",
    "wood",
    "energy star qualified equipment"
])
normalize_heating = {
    "ductless": ["ductless AC", "ductless", "ductless heating", "ductless system", "mini split"],
    "natural gass": ["natural gas heating", "gas heating"],
    "high efficiency": ["energy efficient", "high efficiency", "energy-saving"],
    "zoned": ["zoned heating", "zoned heating and cooling"],
    "solar": ["solar heating", "solar energy", "solar panel"],
    "kerosene": ["kerosene heater"], # almost not used, but keep for extreme case
}
plus_heating = set([
    "radiant",
    "propane",
    "oil",
    "electric",
    "baseboard",
    "central",
])
heating_to_raw_dict = {}
heating_set = set()
for raw in heating:
    if raw in excluded_term_heating:
        continue
    if raw in plus_heating:
        key = f"{raw} heating"
        heating_to_raw_dict[key] = raw
        heating_set.add(key)
    else:
        normalized_terms = normalize_heating.get(raw, [raw])
        for term in normalized_terms:
            norm_term = normalize(term)
            heating_set.add(norm_term)
            heating_to_raw_dict[norm_term] = raw



feature_set = set(features) | set(single_word_features) | floor_set | interior_feature_set | appl_set | cooling_set | heating_set
set_from_db = {"interior": list(interior_feature_set),
                "floor": list(floor_set),
                "appl": list(appl_set),
                "cooling": list(cooling_set),
                "heating": list(heating_set)}
feature_list = list(feature_set)
feature_list = sorted(feature_list)
feature_dict = {"features": feature_list,
                "set_from_db": set_from_db,
                "interior2raw": interior_to_raw_dict,
                "floor2raw": floor_to_raw_dict,
                "appl2raw": appl_to_raw_dict,
                "cool2raw": cooling_to_raw_dict,
                "heat2raw": heating_to_raw_dict}

with open("data/processed/features.json", "w") as f:
    json.dump(feature_dict, f, indent = 2)