import re
import pandas as pd
import json
# Phrases commonly used as property amenities in listing text (longer phrases matched first).

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
features = [item["term"]+item["id"] for item in taxonomy["terms"] if item["category"] == "interior feature"]
single_word_features = (
    "granite", "marble", "quartz",
    "hardwood", "tile", "vinyl", "laminate",
    "island", "pantry", "range", "oven",
    "microwave", "dishwasher",
    "refrigerator", "hood",
    "tub", "shower", "vanity", "sink",
    "hvac", "ac", "heating",
    "fireplace", "lighting"
)
feature_set = set(features) | set(single_word_features)

def _amenity_feature_pattern(phrase):
    words = phrase.split()
    if len(words) == 1:
        return re.compile(rf"\b{re.escape(phrase)}\b", re.I)
    return re.compile(
        r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b",
        re.I,
    )


class EntityExtractor:
    def extract_bedrooms(self, text):
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bed|br|bedroom)s?',
            r'(\d+(?:\.\d+)?)bd'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return float(match.group(1))
        return None
    
    def extract_bathrooms(self, text):
        pattern = r'(\d+(?:\.\d+)?)\s*(?:bathroom|bath)s?'
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
        return None


    def extract_price(self, text):
        PRICE_LEFT_CONTEXT = [
            "price",
            "priced at",
            "listed at",
            "listing price",
            "sell for",
            "selling for",
            "asking",
            "ask",
            "cost",
            "worth"
        ]
        PRICE_RIGHT_CONTEXT = [
            "usd",
            "dollar",
            "dollars"
        ]
        leading = '|'.join(PRICE_LEFT_CONTEXT)
        back = '|'.join(PRICE_RIGHT_CONTEXT)
        NUMBER = r'(\d+(?:,\d{3})*(?:\.\d+)?)'

        patterns = [
            re.compile(rf'(?:{leading})\s*\$?\s*{NUMBER}', re.I),
            re.compile(rf'\$?\s*{NUMBER}\s*(?:{back})\b', re.I),
            re.compile(rf'\$\s*{NUMBER}', re.I)
        ]
        
        matches = []
        for p in patterns:
            m = p.findall(text)
            matches.extend(m)
        matches_unique = list(set(matches))
        if matches_unique:
            return int(matches_unique[0])
        return None

    def extract_sqft(self, text):
        pattern = r'\b(\d+)\s*square\s*feet'
        match = re.search(pattern, text, re.I)
        if match:
            return int(match.group(1))
        return None

    def extract_amenities(self, text):
        if not text:
            return [],[]
        lowered = text.lower()
        matches = []
        for phrase in sorted(amenity_set, key=len, reverse=True):
            pat = _amenity_feature_pattern(phrase)
            for m in pat.finditer(lowered):
                matches.append((m.start(), m.end(), phrase))
        if not matches:
            return [],[]
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        kept = []
        used = []
        for start, end, phrase in matches:
            if any(start < ue and end > us for us, ue in used):
                continue
            used.append((start, end))
            kept.append((start, end, phrase))
        kept.sort(key=lambda x: x[0])
        only_terms = [p for s,e, p in kept]
        return only_terms, kept

    def extract_interior_features(self, text):
        if not text:
            return [],[]
        lowered = text.lower()
        matches = []
        for phrase in sorted(feature_set, key = len, reverse = True):
            pat = _amenity_feature_pattern(phrase)
            for m in pat.finditer(lowered):
                matches.append((m.start(), m.end(), phrase))
        if not matches:
            return [],[]
        matches.sort(key = lambda x: (x[0], -(x[1]-x[0])))
        kept = []
        used = []
        for start, end, phrase in matches:
            if any(start < ue and end > us for us, ue in used):
                continue
            used.append((start, end))
            kept.append((start, end, phrase))
        kept.sort(key=lambda x: x[0])
        only_terms = [p for s, e, p in kept]
        return only_terms, kept

    def extract_all(self, text):
        amenity_terms, amenity_tuple = self.extract_amenities(text)
        feature_terms, feature_tuple = self.extract_interior_features(text)
        return {
            "bedrooms": self.extract_bedrooms(text),
            "bathrooms": self.extract_bathrooms(text),
            "price": self.extract_price(text),
            "sqft": self.extract_sqft(text),
            "amenities": amenity_terms,
            "amenities tuple": amenity_tuple,
            "interior features": feature_terms,
            "interior features tuple": feature_tuple
        }
    def extract_column(self, col):
        results = col.apply(self.extract_all)
        expanded = results.apply(pd.Series)
        return expanded