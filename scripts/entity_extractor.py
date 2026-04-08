import json
import re
from pathlib import Path
import sys
import pandas as pd

# Repo root (parent of ``scripts/``) so JSON paths work when cwd is e.g. ``notebooks/``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# extracting bedroom, bathroom, price, sqft are not useful for both query and remarks
# mainly amenity extractor is used.

def _listing_bed_bath_slash(text):
    """
    Parse ``a/b`` as beds/baths when it is not a calendar fragment.

    Skips:
    - ``M/D/YYYY`` (e.g. ``12/31/2024``, ``3/2/2024``) — the ``a/b`` before ``/year``.
    - Pairs where the second number is implausible as a bath count (e.g. ``1/15`` as Jan 15).
    """
    if not text:
        return None
    for m in re.finditer(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b", text):
        end = m.end()
        # Date: .../YYYY right after the pair
        if end < len(text) and text[end] == "/":
            ychunk = text[end + 1 : end + 5]
            if ychunk.isdigit() and len(ychunk) == 4:
                continue
        a, b = int(m.group(1)), int(m.group(2))
        if a < 1 or b < 1:
            continue
        if a > 20 or b > 20:
            continue
        # Bath count in slash form is rarely > 12; avoids many M/D pairs (e.g. 1/15).
        if b > 12:
            continue
        return a, b
    return None


def _amenity_feature_pattern(phrase):
    words = phrase.split()
    if len(words) == 1:
        return re.compile(rf"\b{re.escape(phrase)}s?\b", re.I)
    return re.compile(
        r"\b" + r"\s+".join(re.escape(w) for w in words) + r"s?\b",
        re.I,
    )

NEGATIONS = [
    "no",
    "not",
    "without",
    "does not have",
    "don't have",
    "doesn't have",
    "without any"
]

def _amenity_feature_negation_pattern(phrase):
    """Match negation cue + optional article + amenity phrase (same plural ``s?`` as positive)."""
    words = phrase.split()
    neg_alternation = "|".join(
        re.escape(n) for n in sorted(NEGATIONS, key=len, reverse=True)
    )
    optional_article = r"(?:a\s+|an\s+|the\s+)?"
    if len(words) == 1:
        body = rf"\b{re.escape(phrase)}s?\b"
    else:
        body = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"s?\b"
    return re.compile(
        rf"\b(?:{neg_alternation})\s+{optional_article}{body}",
        re.I,
    )

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

# Phrase taxonomies for listing remarks (longest first → matched in ``_extract_taxonomy_phrases``).
_FINANCING_PHRASES = tuple(
    sorted(
        {
            "down payment assistance",
            "subject to existing financing",
            "owner will carry",
            "seller financing",
            "owner financing",
            "assumable mortgage",
            "assumable loan",
            "conventional financing",
            "conventional loan",
            "lease purchase option",
            "1031 exchange",
            "third party financing",
            "financing available",
            "mortgage available",
            "investor friendly",
            "rent to own",
            "lease option",
            "land contract",
            "contract for deed",
            "jumbo loan",
            "fha approved",
            "fha eligible",
            "fha financing",
            "fha loan",
            "usda eligible",
            "usda financing",
            "usda loan",
            "va approved",
            "va financing",
            "va loan",
            "cash buyers only",
            "bridge loan",
            "hard money",
            "assumable",
            "short sale",
            "foreclosure",
            "foreclosed",
            "bank owned",
            "reo property",
            "cash only",
            "cash sale",
            "all cash",
            "subject to",
            "probate sale",
            "estate sale",
            "reo",
            "fha",
            "va",
        },
        key=len,
        reverse=True,
    )
)

_LOCATION_PHRASES = tuple(
    sorted(
        {
            "walking distance to schools",
            "easy freeway access",
            "golf course community",
            "close to public transit",
            "public transportation",
            "freeway access",
            "highway access",
            "golf course lot",
            "conservation area",
            "gated community",
            "near train station",
            "near public transit",
            "close to schools",
            "near schools",
            "walk to school",
            "downtown location",
            "suburban neighborhood",
            "backs to greenbelt",
            "backs to preserve",
            "oversized lot",
            "waterfront property",
            "waterfront home",
            "skyline view",
            "panoramic view",
            "mountain view",
            "preserve view",
            "lakefront",
            "lake front",
            "lake view",
            "lake access",
            "riverfront",
            "river front",
            "ocean view",
            "water view",
            "bay view",
            "city view",
            "cul-de-sac lot",
            "on golf course",
            "golf course",
            "greenbelt",
            "in downtown",
            "rural setting",
            "rural acreage",
            "infill location",
            "corner lot",
            "corner unit",
            "end unit",
            "private road",
            "wooded lot",
            "treed lot",
            "mature trees",
            "interior lot",
            "large lot",
            "waterfront",
            "water front",
            "hilltop",
            "hillside",
            "cul-de-sac",
            "cul de sac",
            "downtown",
            "walkable",
            "near metro",
            "near bus",
            "zero lot line",
        },
        key=len,
        reverse=True,
    )
)

_CONDITION_PHRASES = tuple(
    sorted(
        {
            "brand new construction",
            "completely renovated",
            "completely remodeled",
            "excellent condition",
            "move-in ready",
            "move in ready",
            "newly renovated",
            "newly remodeled",
            "original condition",
            "partially renovated",
            "partially updated",
            "pristine condition",
            "recently renovated",
            "recently updated",
            "turnkey property",
            "as-is sale",
            "sold as-is",
            "sold as is",
            "fixer upper",
            "fixer-upper",
            "good condition",
            "fair condition",
            "poor condition",
            "great condition",
            "handyman special",
            "like new condition",
            "needs everything",
            "never lived in",
            "new construction",
            "tear down",
            "fully renovated",
            "fully updated",
            "updated throughout",
            "new build",
            "turnkey",
            "renovated",
            "remodeled",
            "updated",
            "like new",
            "needs repairs",
            "needs some tlc",
            "needs tlc",
            "needs work",
            "vintage charm",
            "mostly original",
            "as-is",
            "as is",
            "gutted",
            "mint condition",
        },
        key=len,
        reverse=True,
    )
)


def _spell_num_from_word(word):
    """Map spelled-out digit to int; ``re.I`` can yield ``Two`` so normalize case."""
    if not word:
        return None
    return _WORD_TO_NUM.get(str(word).lower())


class EntityExtractor:
    def __init__(self):
        amenity_full_dict = self._load_amenities()
        feature_full_dict = self._load_features()
        self.amenity_full = amenity_full_dict
        self.feature_full = feature_full_dict
        self.amenities = amenity_full_dict["amenities"]
        self.features = feature_full_dict["features"]
    
    def _load_amenities(self, amenities_path=None):
        path = (
            Path(amenities_path)
            if amenities_path
            else _PROJECT_ROOT / "data" / "processed" / "amenities.json"
        )
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_features(self, features_path=None):
        path = (
            Path(features_path)
            if features_path
            else _PROJECT_ROOT / "data" / "processed" / "features.json"
        )
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    def extract_bedrooms(self, text):
        if not text:
            return None
        # 3/2 = 3 bed, 2 bath (not 12/31/2024 or 3/2/2024)
        slash = _listing_bed_bath_slash(text)
        if slash is not None:
            return slash[0]
        # most common case: 2 bedrooms, 2.5 bedrooms, 2-bedroom
        pattern_main = r'\b(\d+(?:\.\d+)?)\s*(?:-\s*|\s+)(bed|bedroom|br|bd)s?\b'
        match = re.search(pattern_main, text, re.I)
        if match:
            return float(match.group(1))
        # two bedrooms, two-bedroom
        pattern2 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*-\s*|\s+)(bed|bedroom|br)s?\b'
        match = re.search(pattern2, text, re.I)
        if match:
            w = _spell_num_from_word(match.group(1))
            if w is not None:
                return w
        
        # adjectives between number and word: 3 spacious bedrooms
        pattern3 = r'\b(\d+(?:\.\d+)?)\s+(?:\w+\s+){0,2}(bed|bedroom|br)s?\b'
        match = re.search(pattern3, text, re.I)
        if match:
            return float(match.group(1))
        # two genrouly large bedrooms
        pattern4 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,2}(bed|bedroom|br)s?\b'
        match = re.search(pattern4, text, re.I)
        if match:
            w = _spell_num_from_word(match.group(1))
            if w is not None:
                return w
        return None
    
    def extract_bathrooms(self, text):
        if not text:
            return None
        slash = _listing_bed_bath_slash(text)
        if slash is not None:
            return slash[1]
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
            w = _spell_num_from_word(match.group(1))
            if w is not None:
                return w
        # two full bathrooms, one full bathrooms
        pattern6 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:full\s+)?(bath|bathroom)s?\b'
        match = re.search(pattern6, text, re.I)
        if match:
            w = _spell_num_from_word(match.group(1))
            if w is not None:
                return w
        
        # adjectives between number and word: 3 clean bathrooms
        pattern7 = r'\b(\d+(?:\.\d+)?)\s+(?:\w+\s+){0,2}(bath|bathroom)s?\b'
        match = re.search(pattern7, text, re.I)
        if match:
            return float(match.group(1))
        # two clean bathrooms
        pattern8 = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,2}(bath|bathroom)s?\b'
        match = re.search(pattern8, text, re.I)
        if match:
            w = _spell_num_from_word(match.group(1))
            if w is not None:
                return w
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
        pattern = r'\b(\d+)(?:\s*\+)?\s*square\s*feet'
        match = re.search(pattern, text, re.I)
        if match:
            return int(match.group(1))
        return None

    def _extract_taxonomy_phrases(self, text, phrases, neg_spans=None):
        """
        Match phrases from a taxonomy (same word-boundary rules as amenities).

        - If ``neg_spans`` is a list of ``(start, end)`` (e.g. from negated amenity/feature
          detection), drops positive hits that overlap any negated span.
        - Then drops overlapping spans among positives (longer span wins at same start).
        """
        if not text:
            return [], []
        lowered = text.lower()
        neg_spans = neg_spans or []
        matches = []
        for phrase in phrases:
            pat = _amenity_feature_pattern(phrase)
            for m in pat.finditer(lowered):
                if any(
                    m.start() < ne and m.end() > ns for ns, ne in neg_spans
                ):
                    continue
                matches.append((m.start(), m.end(), phrase))
        if not matches:
            return [], []
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        kept = []
        used = []
        for start, end, phrase in matches:
            if any(start < ue and end > us for us, ue in used):
                continue
            used.append((start, end))
            kept.append((start, end, phrase))
        kept.sort(key=lambda x: x[0])
        only_terms = list({p for _, _, p in kept})
        return only_terms, kept

    def extract_financing_options(self, text):
        """Financing and sale-type cues (e.g. FHA, VA, cash, seller financing, short sale)."""
        return self._extract_taxonomy_phrases(text, _FINANCING_PHRASES)

    def extract_location_features(self, text):
        """Location-oriented phrases (views, lot type, schools, transit, downtown, etc.)."""
        return self._extract_taxonomy_phrases(text, _LOCATION_PHRASES)

    def extract_condition(self, text):
        """Condition and renovation status (e.g. move-in ready, as-is, renovated, new construction)."""
        return self._extract_taxonomy_phrases(text, _CONDITION_PHRASES)

    def extract_negated_amenities_features(self, text, type):
        """Amenities/features explicitly rejected (e.g. ``no pool``, ``without a garage``)."""
        if not text:
            return [], []
        tax = self.amenities if type == "amenity" else self.features
        lowered = text.lower()
        matches = []
        for phrase in sorted(tax, key=len, reverse=True):
            pat = _amenity_feature_negation_pattern(phrase)
            for m in pat.finditer(lowered):
                matches.append((m.start(), m.end(), phrase))
        if not matches:
            return [], []
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        kept = []
        used = []
        for start, end, phrase in matches:
            if any(start < ue and end > us for us, ue in used):
                continue
            used.append((start, end))
            kept.append((start, end, phrase))
        kept.sort(key=lambda x: x[0])
        only_terms = list({p for _, _, p in kept})
        return only_terms, kept

    def extract_amenities_features(self, text, type, neg_kept=None):
        """
        Positive amenity/features mentions. Spans overlapping ``extract_negated_amenities`` hits are dropped
        so e.g. ``no pool`` does not count as wanting ``pool``.
        """
        if not text:
            return [], []
        if neg_kept is None:
            _, neg_kept = self.extract_negated_amenities_features(text, type)
        neg_spans = [(s, e) for s, e, _ in neg_kept]
        tax = self.amenities if type == "amenity" else self.features
        phrases = sorted(tax, key=len, reverse=True)
        return self._extract_taxonomy_phrases(text, phrases, neg_spans=neg_spans)

    def extract_all(self, text):
        neg_amenity_terms, neg_amenity_kept = self.extract_negated_amenities_features(text, type = "amenity")
        amenity_terms, amenity_tuple = self.extract_amenities_features(
            text, type = "amenity", neg_kept=neg_amenity_kept
        )
        neg_feature_terms, neg_feature_kept = self.extract_negated_amenities_features(text, type = "feature")
        feature_terms, feature_tuple = self.extract_amenities_features(
            text, type = "feature", neg_kept=neg_feature_kept
        )
        financing_terms, financing_tuple = self.extract_financing_options(text)
        location_terms, location_tuple = self.extract_location_features(text)
        condition_terms, condition_tuple = self.extract_condition(text)
        return {
            "bedrooms": self.extract_bedrooms(text),
            "bathrooms": self.extract_bathrooms(text),
            "price": self.extract_price(text),
            "sqft": self.extract_sqft(text),
            "amenities": amenity_terms,
            "amenities tuple": amenity_tuple,
            "negated amenities": neg_amenity_terms,
            "negated amenities tuple": neg_amenity_kept,
            "interior features": feature_terms,
            "interior features tuple": feature_tuple,
            "negated features": neg_feature_terms,
            "negated features tuple": neg_feature_kept,
            "financing options": financing_terms,
            "financing options tuple": financing_tuple,
            "location features": location_terms,
            "location features tuple": location_tuple,
            "condition": condition_terms,
            "condition tuple": condition_tuple,
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
