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

features = [item["term"] for item in taxonomy["terms"] if item["category"] == "interior feature"]
single_word_features = (
    "granite", "marble", "quartz",
    "hardwood", "tile", "vinyl", "laminate",
    "island", "pantry", "range", "oven",
    "microwave", "dishwasher",
    "refrigerator", "hood",
    "tub", "shower", "vanity", "sink",
    "heating and cooling", "air conditioning", "heating",
    "fireplace", "lighting"
)
feature_set = set(features) | set(single_word_features)
feature_list = [normalize(a) for a in feature_set]
feature_dict = {"features": feature_list}
with open("data/processed/features.json", "w") as f:
    json.dump(feature_dict, f, indent = 2)