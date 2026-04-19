import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# compare to the amenities for amenity extraction from query, amnenities to display could be more diverse and incomplete

terms_to_exclude_dict = {
    "AssociationAmenities": set([
        "maintenance grounds",
        "maintenance front yard",
        "electricity",
        "gas",
        "trash",
        "utilities",
        "insurance",
        "water",
        "management",
        "hot water",
        "cable tv",
        "storage"
    ]),
    "InteriorFeatures": set([
        "built in features",
        "tandem",
        "unfinished walls"
    ]),
    "Appliances": set([
        "propane cooking",
        "electric cooking",
        "gas cooking",
        "built in",
        "free standing",
        "water to refrigerator",
        "counter top"
    ])

}

normalizers = {
    "PoolFeatures": {
        "electric heat": "electric heated",
        "solar heat": "solar heated",
        "propane heat": "propane heated",
        "gas heat": "gas heated"
    },
    "View": {
        "trees woods": ["tree", "wood"],
        "creek stream": ["creek", "stream"],
        "park greenbelt": ["park", "greenbelt"]
    },
    "InteriorFeatures": {
        "paneling wainscoting": ["paneling", "wainscoting"],
        "french doors atrium doors": ["french door", "atrium door"],
        "main level primary": ["primary bedroom on main floor"],
        "all bedrooms up": ["bedrooms upstairs"],
        "all bedrooms down": ["bedrooms downstairs"],
        "living room deck attached": ["living room with attached deck"]
    }

}

def normalize(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

def normalize_item(s):
    if not s or s == "None":
        return None
    # CamelCase → whitespace
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
    # lowercase
    s = s.lower().strip()
    
    return s

def normalize_terms(input_list, terms_to_exclude, normalizer):
    terms = set()
    for t in input_list:
        if t in terms_to_exclude:
            continue
        if normalizer:
            t = normalizer.get(t, t)
        if isinstance(t, list):
            for i in t:
                terms.add(i) # already normalized by hand
        else:
            terms.add(normalize(t))
    return terms

def process_result(result: dict, keys_to_process: list) -> dict:
    for key in keys_to_process:
        field = result[key]
        if not isinstance(field, str):
            result[key] = []
            continue
        items = field.split(",")
        normalized_items = list(set(filter(None, [normalize_item(x) for x in items])))
        terms_to_exclude = terms_to_exclude_dict.get(key, [])
        normalizer = normalizers.get(key)
        result[key] = normalize_terms(normalized_items, terms_to_exclude, normalizer)
    return result
