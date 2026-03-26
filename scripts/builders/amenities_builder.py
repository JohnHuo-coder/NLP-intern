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
amenity_set = set(amenities) | set(single_word_amenities)
amenity_list = [normalize(a) for a in amenity_set]
amenity_dict = {"amenities": amenity_list}
with open("data/processed/amenities.json", "w") as f:
    json.dump(amenity_dict, f, indent = 2)