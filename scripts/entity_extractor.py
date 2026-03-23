import json
import re
from pathlib import Path

import pandas as pd
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

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9
}

class EntityExtractor:
    def extract_bedrooms(self, text):
        if not text:
            return None
        # 3 / 2: 3 bedrooms 2 bathrooms
        pattern1 = r'\b(\d+)\s*/\s*(\d+)\b'
        match = re.search(pattern1, text, re.I)
        if match:
            return int(match.group(1))
        # most common case: 2 bedrooms, 2.5 bedrooms, 2-bedroom
        pattern_main = r'\b(\d+(?:\.\d+)?)\s*(?:-\s*|\s+)(bed|bedroom|br|bd)s?\b'
        match = re.search(pattern_main, text, re.I)
        if match:
            return float(match.group(1))
        # two bedrooms, two-bedroom
        pattern2 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*-\s*|\s+)(bed|bedroom|br)s?\b'
        match = re.search(pattern2, text, re.I)
        if match:
            return _WORD_TO_NUM[match.group(1)]
        
        # adjectives between number and word: 3 spacious bedrooms
        pattern3 = r'\b(\d+(?:\.\d+)?)\s+(?:\w+\s+){0,2}(bed|bedroom|br)s?\b'
        match = re.search(pattern3, text, re.I)
        if match:
            return float(match.group(1))
        # two genrouly large bedrooms
        pattern4 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,2}(bed|bedroom|br)s?\b'
        match = re.search(pattern4, text, re.I)
        if match:
            return _WORD_TO_NUM[match.group(1)]
        return None
    
    def extract_bathrooms(self, text):
        if not text:
            return None
        # 3 / 2: 3 bedrooms 2 bathrooms
        pattern1 = r'\b\d+\s*/\s*(\d+)\b'
        match = re.search(pattern1, text, re.I)
        if match:
            return int(match.group(1))
        # 2 1/2 bathrooms, 2 1/2-bathroom
        pattern2 = r'\b(\d+)\s+1/2(?:\s*-\s*|\s+)(bath|bathroom)s?\b'
        match = re.search(pattern2, text, re.I)
        if match:
            return int(match.group(1)) + 0.5
        # 2 bathrooms, 2.5 bathrooms, 2-bathroom
        pattern3 = r'\b(\d+(?:\.\d+)?)\s*(?:-\s*|\s+)(bath|bathroom)s?\b'
        match = re.search(pattern3, text, re.I)
        if match:
            return float(match.group(1))
        # 1 full bathrooms, 2 full bathrooms
        pattern4 = r'\b(\d+)\s*full\s+(bath|bathroom)s?\b'
        match = re.search(pattern4, text, re.I)
        if match:
            return int(match.group(1))
        # two bathrooms, two-bathroom
        pattern5 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*-\s*|\s+)(bath|bathroom)s?\b'
        match = re.search(pattern5, text, re.I)
        if match:
            return _WORD_TO_NUM[match.group(1)]
        # two full bathrooms, one full bathrooms
        pattern6 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:full\s+)?(bath|bathroom)s?\b'
        match = re.search(pattern6, text, re.I)
        if match:
            return _WORD_TO_NUM[match.group(1)]
        
        # adjectives between number and word: 3 clean bathrooms
        pattern7 = r'\b(\d+(?:\.\d+)?)\s+(?:\w+\s+){0,2}(bath|bathroom)s?\b'
        match = re.search(pattern7, text, re.I)
        if match:
            return float(match.group(1))
        # two clean bathrooms
        pattern8 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,2}(bath|bathroom)s?\b'
        match = re.search(pattern8, text, re.I)
        if match:
            return _WORD_TO_NUM[match.group(1)]
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
            raw = str(matches_unique[0]).replace(",", "")
            try:
                return int(float(raw))
            except ValueError:
                return None
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

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    in_path = root / "data" / "processed" / "listing_sample_cleaned.csv"
    out_path = root / "data" / "processed" / "listing_entities_extracted.csv"

    df = pd.read_csv(in_path)
    extractor = EntityExtractor()
    result_df = extractor.extract_column(df["remarks"])
    summary_df = pd.concat([df[["L_ListingID", "remarks"]], result_df], axis=1)
    summary_df.to_csv(out_path, index=False)
    print(f"Wrote {len(summary_df)} rows to {out_path}")
