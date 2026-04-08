import re
import sys
from pathlib import Path

# support in the future: house district (high school district), hoa fee range, fencing type

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from text_cleaning import TextCleaner
from entity_extractor import EntityExtractor

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def _spell_num_from_word(word):
    """Map spelled-out digit to int; ``re.I`` can yield ``Two`` so normalize case."""
    if not word:
        return None
    return _WORD_TO_NUM.get(str(word).lower())

def _split_number(x):
    whole = int(x)
    frac = x - whole
    return whole, frac

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class QueryParser:
    def __init__(self):
        self.normalizer = TextCleaner()
        self.filters = {}
        self.extractor = EntityExtractor()
        self.word_num_pattern = "|".join(_WORD_TO_NUM.keys())
        self.price_format = "\$?(\d{4,})(?!\s*(?:square feet))"

    def _clean_query(self, query):
        query = self.normalizer.clean_text(query)
        return query


    def _parse_range(self, query, cat):
        if cat == "bedrooms":
            prefix = "(\d+|" + self.word_num_pattern + ")"
            suffix = "(\d+|" + self.word_num_pattern + ")\s*(?:bed|bedroom|br|bd)s?"
            symbol_pattern = r'\b(\d+)\s*(?:-|~)\s*(\d+)\s*(?:bed|bedroom|br|bd)s?\b'
        elif cat == "bathrooms":
            prefix = "(\d+(?:\.\d+)?|" + self.word_num_pattern + ")"
            suffix = "(\d+(?:\.\d+)?|" + self.word_num_pattern + ")\s*(?:bath|bathroom)s?"
            symbol_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:-|~)\s*(\d+(?:\.\d+)?)\s*(bath|bathroom)s?\b'
        elif cat == "price":
            prefix = suffix = self.price_format
            symbol_pattern = fr'\b{prefix}\s*(?:-|~)\s*{suffix}\b'
        elif cat == "sqft":
            prefix = "(\d+(?:\.\d+)?)"
            suffix = "(\d+(?:\.\d+)?)\s*(?:square feet)"
            symbol_pattern = rf'\b{prefix}\s*(?:-|~)\s*{suffix}\b'

        to_pattern = rf'\b(?:from\s+)?{prefix}\s+(?:to)\s+{suffix}\b'
        between_pattern = rf'\b(?:between)\s+{prefix}\s+(?:and)\s+{suffix}\b'
        filter_key_max = f'{cat}_max'
        filter_key_min = f'{cat}_min'

        for p in [to_pattern, between_pattern, symbol_pattern]:
            match = re.search(p, query, re.I)
            if match :
                raw_min = match.group(1)
                raw_max = match.group(2)
                if cat == "bathrooms":
                    min_b = float(raw_min) if _is_number(raw_min) else _spell_num_from_word(raw_min) 
                    min_b, _ = _split_number(min_b)
                    max_b = float(raw_max) if _is_number(raw_max) else _spell_num_from_word(raw_max)
                    max_b, _ = _split_number(max_b)
                else:
                    min_b = int(raw_min) if _is_number(raw_min) else _spell_num_from_word(raw_min) 
                    max_b = int(raw_max) if _is_number(raw_max) else _spell_num_from_word(raw_max)
                self.filters[filter_key_min] = min_b
                self.filters[filter_key_max] = max_b
                return True
        return False

    def _parse_max(self, query, cat):
        if cat == "bedrooms":
            num = "(\d+|" + self.word_num_pattern + ")\s*(?:bed|bedroom|br|bd)s?"
            num_symbol = "(\d+)\s*(?:bed|bedroom|br|bd)s?"
        elif cat == "bathrooms":
            num = "(\d+(?:\.\d+)?|" + self.word_num_pattern + ")\s*(?:bath|bathroom)s?"
            num_symbol = "(\d+(?:\.\d+)?)\s*(?:bath|bathroom)s?"
        elif cat == "price":
            num = num_symbol = self.price_format
        elif cat == "sqft":
            num = num_symbol = "(\d+(?:\.\d+)?)\s*(?:square feet)"

        word_pattern = rf'\b(?:under|below|at most|less than|up to|maximum|max|no more than|capped at)\s+{num}\b'
        word_pattern_2 = rf'\b{num}\s+(?:or less)\b'
        symbol_pattern = rf'<\s*{num_symbol}\b'

        filter_key_max = f'{cat}_max'

        for p in [word_pattern, word_pattern_2, symbol_pattern]:
            match = re.search(p, query, re.I)
            if match :
                raw_max = match.group(1)
                if cat == "bathrooms":
                    max_b = float(raw_max) if _is_number(raw_max) else _spell_num_from_word(raw_max)
                    max_b, _ = _split_number(max_b)
                else:
                    max_b = int(raw_max) if _is_number(raw_max) else _spell_num_from_word(raw_max)
                self.filters[filter_key_max] = max_b
                return True
        return False

    def _parse_min(self, query, cat):

        if cat == "bedrooms":
            num = "(\d+|" + self.word_num_pattern + ")\s*(?:bed|bedroom|br|bd)s?"
            symbol_gt = "(\d+)\s*(?:bed|bedroom|br|bd)s?"
            symbol_plus_pattern = r'\b(\d+)\s*\+\s*(?:bed|bedroom|br|bd)s?\b'
        elif cat == "bathrooms":
            num = "(\d+(?:\.\d+)?|" + self.word_num_pattern + ")\s*(?:bath|bathroom)s?"
            symbol_gt = "(\d+(?:\.\d+)?)\s*(?:bath|bathroom)s?"
            symbol_plus_pattern = r'\b(\d+(?:\.\d+)?)\s*\+\s*(bath|bathroom)s?\b'
        elif cat == "price":
            num = symbol_gt = self.price_format
            symbol_plus_pattern = r'\b\$?(\d+)\s*\+\b'
        elif cat == "sqft":
            num = symbol_gt = "(\d+(?:\.\d+)?)\s*(?:square feet)"
            symbol_plus_pattern = r'\b(\d+(?:\.\d+)?)\s*\+\s*(?:square feet)\b'

        word_pattern = rf'\b(?:at least|above|over|more than|starting from|no less than|minimum|min)\s+{num}\b'
        word_pattern_2 = rf'\b{num}\s+(?:or more)\b'
        symbol_gt_pattern = rf'>\s*{symbol_gt}\b'

        filter_key_min = f'{cat}_min'

        for p in [word_pattern, word_pattern_2, symbol_gt_pattern, symbol_plus_pattern]:
            match = re.search(p, query, re.I)
            if match :
                raw_min = match.group(1)
                if cat == "bathrooms":
                    min_b = float(raw_min) if _is_number(raw_min) else _spell_num_from_word(raw_min)
                    min_b, _ = _split_number(min_b)
                else:
                    min_b = int(raw_min) if _is_number(raw_min) else _spell_num_from_word(raw_min)
                self.filters[filter_key_min] = min_b
                return True
        return False

    def parse_price(self, query):

        # Price patterns range
        # (from) $1000 to $2000, $1000 ~ $2000, $1000 - $2000, between a and b 
        has_range = self._parse_range(query, 'price')
        if has_range:
            return

        # Price patterns max
        # under $20000, <$200000
        max_matched = self._parse_max(query, 'price')
        
        # Price patterns min
        # above $20000, $20000+, >20000
        min_matched = self._parse_min(query, "price")
    
        
        if min_matched or max_matched:
            return 
        
        # Price pattern approx
        # ~ $20000, around 20000
        p1 = r'(?:\b(?:around|about|approximately|roughly|near)\b|~)\s*\$?(\d{4,})(?!\s*(?:square feet))'
        match = re.search(p1, query, re.I)
        if match :
            self.filters['price_min'] = int(match.group(1)) * 0.8
            self.filters['price_max'] = int(match.group(1)) * 1.2
            return 

        # Price pattern exact
        if match := re.search(r'\b(?:at|exactly)?\s+\$?(\d{4,})(?!\s*(?:square feet))\b', query, re.I):
            self.filters['price'] = int(match.group(1))

    def parse_sqft(self, query):
        # only support sqft as unit now, add conversion from arces to sqft in the future

        # sqft patterns range
        # (from) 1000 sqft to 2000 sqft, 1000 ~ 2000sqft, 1000 - 2000 sqft, between a and b sqft
        has_range = self._parse_range(query, 'sqft')
        if has_range:
            return

        # sqft patterns max
        # under 20000 sqft, < 2000 sqft
        max_matched = self._parse_max(query, 'sqft')
        
        # sqft patterns min
        # above 2000 sqft, 20000+ sqft, >2000 sqft
        min_matched = self._parse_min(query, "sqft")
    
        
        if min_matched or max_matched:
            return 
        
        # Sqft pattern approx
        # ~ 2000 sqft, around 2000 sqft
        p1 = r'(?:\b(?:around|about|approximately|roughly|near)\b|~)\s*(\d+(?:\.\d+)?)\s*(?:square feet)\b'
        match = re.search(p1, query, re.I)
        if match :
            self.filters['sqft_min'] = int(match.group(1)) * 0.8
            self.filters['sqft_max'] = int(match.group(1)) * 1.2
            return 

        # sqft pattern exact
        if match := re.search(r'\b(?:at|exactly)?\s+(\d+(?:\.\d+)?)\s*(?:square feet)\b', query, re.I):
            self.filters['sqft'] = int(match.group(1))

    def parse_bedroom(self, query):
        # Bedrooms patterns range
        # 3 to 4, two to three, 3 ~ 4, 3 - 4, between 3 and 4, between one and two
        has_range = self._parse_range(query, 'bedrooms')
        if has_range:
            return

        # Bedroom patterns max
        # less than two bedrooms, below 2 bedrooms, <5 bedrooms
        max_matched = self._parse_max(query, "bedrooms")
        
        # Bedroom patterns min
        # More than two bedrooms, above 2 bedrooms, >5 bedrooms, 5+ bedrooms
        min_matched = self._parse_min(query, 'bedrooms')

        if min_matched or max_matched:
            return 

        # bedrooms pattern approx
        # ~ 5 bedrooms, around 5 bedrooms
        p1 = r'(?:\b(?:around|about|approximately|roughly)\b|~)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bed|bedroom|br|bd)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = int(raw) if raw.isdigit() else _spell_num_from_word(raw)
            self.filters['bedrooms_min'] = num - 1 if num - 1 > 0 else num
            self.filters['bedrooms_max'] = num +1
            return 
            
        # Bedrooms pattern exact
        # 2 bedrooms, 2-bedroom, two bedrooms, two-bedroom
        p1= r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*-\s*|\s+)(bed|bedroom|br|bd)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = int(raw) if raw.isdigit() else _spell_num_from_word(raw)
            self.filters["bedrooms"] = num

    def parse_bathroom(self, query):
        # convert 2.5 bath into 2 full bath + 1 half bath, since the table only has full bath column and half bath column
        # to limit the scope, only supports the following format
        # 2.5bathrooms, 2.5 bathrooms, 2 bathrooms, 2-bathroom, 2 - bathroom, 2 full bathroom, 2full bathroom, 
        # one bathroom, one-bathroom, two full bathrooms

        # not supporting or support in the future
        # 2 full bathroom and 1 half bathroom, one full bathroom and one half bathroom
        # two and a half bathroom

        # also, only supports exact match for half bathrooms, since 4.5 to 6 bathrooms is hard to express with sql

        # Bathroom patterns range
        # 3 to 4, 3.5 to 4.5, two to three, 3 ~ 4, 3 - 4, 3.5 ~ 4.5, 3.5 - 4.5, between 3 and 4, between 3.5 and 4.5, between one and two
        has_range = self._parse_range(query, 'bathrooms')
        if has_range:
            return

        # Bathroom patterns max
        # less than two bathrooms, below 2.5 bathrooms, <5.5 bathrooms
        max_matched = self._parse_max(query, "bathrooms")
        
        # Bathroom patterns min
        # More than two bathrooms, above 2.5 bathrooms, >5.5 bthrooms, 5+ bathrooms
        min_matched = self._parse_min(query, 'bathrooms')

        if min_matched or max_matched:
            return 

        # bathrooms pattern approx
        # ~ 5 bathrooms, around 5 bathrooms
        p1 = r'(?:\b(?:around|about|approximately|roughly)\b|~)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bath|bathroom)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = float(raw) if _is_number(raw) else _spell_num_from_word(raw)
            full, half = _split_number(num)
            self.filters['bathrooms_min'] = full -1 if full - 1 > 0 else full
            self.filters['bathrooms_max'] = full +1
            return 

        # Bathrooms pattern exact
        # 2.5 bathrooms, 2-bathroom, two bathrooms, two-bathroom, 2 full bathrooms, two full bathrooms
        p1= r'\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*-\s*|\s+|\s*full\s+)(?:bath|bathroom)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = float(raw) if _is_number(raw) else _spell_num_from_word(raw)
            full, half = _split_number(num)
            if half:
                self.filters["bathroom_half"] = 1
            self.filters["bathrooms"] = full
        
        # more to add: half bathroom cases, mix of full and half
    
    def parse_city(self, query):
        match = re.search(r'\b(?:in|near|at|around|located\s+in)\s+(?:the\s+city\s+of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
        if match:
            self.filters["city"] = match.group(1)
            
    def parse_amenity(self, query):
        neg_amenity_terms, neg_amenity_kept = self.extractor.extract_negated_amenities_features(query, type = "amenity")
        amenity_terms, _ = self.extractor.extract_amenities_features(
            query, type = "amenity", neg_kept=neg_amenity_kept
        )
        if neg_amenity_terms:
            self.filters["negated_amenities"] = neg_amenity_terms
        if amenity_terms:
            self.filters["amenities"] = amenity_terms

    def parse_features(self, query):
        neg_feature_terms, neg_feature_kept = self.extractor.extract_negated_amenities_features(query, type = "feature")
        feature_terms, _ = self.extractor.extract_amenities_features(
            query, type = "feature", neg_kept=neg_feature_kept
        )
        if neg_feature_terms:
            self.filters["negated_features"] = neg_feature_terms
        if feature_terms:
            self.filters["features"] = feature_terms

    def parse(self, query):
        self.filters = {}
        query = self._clean_query(query)
        self.parse_bedroom(query)
        self.parse_bathroom(query)
        self.parse_sqft(query)
        self.parse_price(query)

        self.parse_city(query)
        self.parse_amenity(query)
        self.parse_features(query)
        

        return self.filters



    def parse_amenity_to_sql(self, conditions, params):

        amenity_full_dict = self.extractor.amenity_full
        set_from_db = amenity_full_dict["set_from_db"]
        pool_set = set_from_db["pool"]
        pool2raw = amenity_full_dict["pool2raw"]
        spa_set = set_from_db["spa"]
        spa2raw = amenity_full_dict["spa2raw"]
        view_set = set_from_db["view"]
        view2raw = amenity_full_dict["view2raw"]
        security_set = set_from_db["security"]
        security2raw = amenity_full_dict["security2raw"]
        association_set = set_from_db["association"]
        association2raw = amenity_full_dict["association2raw"]
        community_set = set_from_db["community"]
        community2raw = amenity_full_dict["community2raw"]


        # Amenities
        for amenity in self.filters.get("amenities", []):
            flag = False
            if amenity == "private pool":
                conditions.append("PoolPrivateYN = 1")
                continue
            if amenity == "attached garage":
                conditions.append("AttachedGarageYN = 1")
                continue
            if amenity == "spa":
                conditions.append("SpaYN = 1")
                continue
            if amenity == "pool" or amenity == "swimming pool":
                conditions.append(
                    "(LOWER(AssociationAmenities) LIKE %s OR LOWER(CommunityFeatures) LIKE %s OR (LOWER(PoolFeatures) != 'none' AND LOWER(PoolFeatures) IS NOT NULL) OR LOWER(L_Remarks) LIKE %s)"
                )
                no_space = amenity.replace(" ", "")
                params.extend([
                    f"%{no_space}%",
                    f"%{no_space}%",
                    f"%{no_space}%"
                ])
                continue
            if amenity == "guard":
                conditions.append(
                    "(LOWER(AssociationAmenities) LIKE %s OR LOWER(Security) LIKE %s OR LOWER(L_Remarks) LIKE %s)"
                )
                params.extend([
                    f"%{amenity}%",
                    f"%{amenity}%",
                    f"%{amenity}%"
                ])
                continue
            if amenity == "gated":
                conditions.append(
                    "(LOWER(CommunityFeatures) LIKE %s OR LOWER(Security) LIKE %s OR LOWER(L_Remarks) LIKE %s)"
                )
                params.extend([
                    f"%{amenity}%",
                    f"%{amenity}%",
                    f"%{amenity}%"
                ])
            if (amenity == "horse trail" or amenity == "golf" or amenity  == "golf course" 
                        or amenity == "dog" or amenity  == "dog park" or amenity == "park"):
                conditions.append(
                    "(LOWER(AssociationAmenities) LIKE %s OR LOWER(CommunityFeatures) LIKE %s OR LOWER(L_Remarks) LIKE %s)"
                )
                no_space = amenity.replace(" ", "")
                params.extend([
                    f"%{no_space}%",
                    f"%{no_space}%",
                    f"%{no_space}%"
                ])
                continue
            if "garage" in amenity:
                conditions.append("(GarageYN = 1 OR AttachedGarageYN = 1)")
            if "parking" in amenity:
                conditions.append("(OpenParkingSpaces >= 1 OR GarageYN = 1 OR AttachedGarageYN = 1)")
            if amenity in pool_set:
                raw = pool2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(PoolFeatures) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{amenity}%"
                ])

            elif amenity in spa_set:
                conditions.append("SpaYN = 1")
                raw = spa2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(SpaFeatures) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{amenity}%"
                ])

            elif amenity in view_set:
                conditions.append("ViewYN = 1")
                raw = view2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(View) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{amenity}%"
                ])
            
            elif amenity in security_set:
                raw = security2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(SecurityFeatures) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{amenity}%"
                ])
            
            elif amenity in association_set:
                raw = association2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(AssociationAmenities) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{amenity}%"
                ])

            elif amenity in community_set:
                raw = community2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(CommunityFeatures) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{amenity}%"
                ])
            else:
                conditions.append("LOWER(L_Remarks) LIKE %s")
                params.append(
                    f"%{amenity}%"
                )
        
        # Negated Amenities
        for amenity in self.filters.get("negated_amenities", []):
            if amenity == "private pool":
                conditions.append("PoolPrivateYN = 0")
                continue
            if amenity == "attached garage":
                conditions.append("AttachedGarageYN = 0")
                continue
            if amenity == "garage":
                conditions.append("(GarageYN = 0 AND AttachedGarageYN = 0)")
                continue
            if amenity == "spa":
                conditions.append("SpaYN = 0")
                continue
            if "parking" in amenity:
                conditions.append("(OpenParkingSpaces = 0 AND GarageYN = 0 AND AttachedGarageYN = 0)")
            if amenity in pool_set:
                raw = pool2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(PoolFeatures) NOT LIKE %s OR PoolFeatures IS NULL)")
                params.append(f"%{raw_no_space}%")

            elif amenity in spa_set:
                conditions.append("SpaYN = 0")
                raw = spa2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(SpaFeatures) NOT LIKE %s OR SpaFeatures IS NULL)")
                params.append(f"%{raw_no_space}%")

            elif amenity in view_set:
                conditions.append("ViewYN = 0")
                raw = view2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(View) NOT LIKE %s OR View IS NULL)")
                params.append(f"%{raw_no_space}%")
            
            elif amenity in security_set:
                raw = security2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(SecurityFeatures) NOT LIKE %s OR SecurityFeatures IS NULL)")
                params.append(f"%{raw_no_space}%")
            
            elif amenity in association_set:
                raw = association2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(AssociationAmenities) NOT LIKE %s OR AssociationAmenities IS NULL)")
                params.append(f"%{raw_no_space}%")

            elif amenity in community_set:
                raw = community2raw[amenity]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(CommunityFeatures) NOT LIKE %s OR CommunityFeatures IS NULL)")
                params.append(f"%{raw_no_space}%")
            else:
                conditions.append("(LOWER(L_Remarks) NOT LIKE %s OR L_Remarks IS NULL)")
                params.append(f"%{amenity.lower()}%")

    def parse_house_feature_to_sql(self, conditions, params):

        house_feature_full_dict = self.extractor.feature_full
        set_from_db = house_feature_full_dict["set_from_db"]

        interior_set = set_from_db["interior"]
        interior2raw = house_feature_full_dict["interior2raw"]
        floor_set = set_from_db["floor"]
        floor2raw = house_feature_full_dict["floor2raw"]
        appl_set = set_from_db["appl"]
        appl2raw = house_feature_full_dict["appl2raw"]
        cooling_set = set_from_db["cooling"]
        cool2raw = house_feature_full_dict["cool2raw"]
        heating_set = set_from_db["heating"]
        heat2raw = house_feature_full_dict["heat2raw"]

        for feature in self.filters.get("features", []):
            if "fireplace" in feature: # fireplace is in heating feature, but it doesn't has to be
                conditions.append("FireplaceYN = 1") 
                continue
            if feature in interior_set:
                raw_list = interior2raw[feature]
                cond = []
                for raw in raw_list:
                    raw_no_space = raw.replace(" ", "")
                    cond.append("LOWER(InteriorFeatures) LIKE %s")
                    params.append(f"%{raw_no_space}%")

                or_clause = "(" + " OR ".join(cond) + " OR LOWER(L_Remarks) LIKE %s" + ")"
                params.append(f"%{feature}%")
                conditions.append(or_clause)

            elif feature in appl_set:
                raw = appl2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Appliances) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{feature}%"
                ])

            elif feature in cooling_set:
                conditions.append("CoolingYN = 1")
                raw = cool2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Cooling) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{feature}%"
                ])
            
            elif feature in floor_set:
                raw = floor2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Flooring) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{feature}%"
                ])
            
            elif feature in heating_set:
                conditions.append("HeatingYN = 1")
                raw = heat2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Heating) LIKE %s OR LOWER(L_Remarks) LIKE %s)")
                params.extend([
                    f"%{raw_no_space}%",
                    f"%{feature}%"
                ])

            else:
                conditions.append("LOWER(L_Remarks) LIKE %s")
                params.append(f"%{feature}%")

        # negated features
        for feature in self.filters.get("negated_features", []):
            if "fireplace" in feature:
                conditions.append("FireplaceYN = 0")
                continue
            if feature in interior_set:
                raw_list = interior2raw[feature]
                cond = []
                for raw in raw_list:
                    raw_no_space = raw.replace(" ", "")
                    cond.append("(LOWER(InteriorFeatures) NOT LIKE %s OR InteriorFeatures IS NULL)")
                    params.append(f"%{raw_no_space.lower()}%")

                or_clause = "(" + " AND ".join(cond) + ")"
                conditions.append(or_clause)

            elif feature in appl_set:
                raw = appl2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Appliances) NOT LIKE %s OR Appliances IS NULL)")
                params.append(f"%{raw_no_space}%")

            elif feature in cooling_set:
                conditions.append("CoolingYN = 0")
                raw = cool2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Cooling) NOT LIKE %s OR Cooling IS NULL)")
                params.append(f"%{raw_no_space}%")
            
            elif feature in floor_set:
                raw = floor2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Flooring) NOT LIKE %s OR Flooring IS NULL)")
                params.append(f"%{raw_no_space}%")
            
            elif feature in heating_set:
                conditions.append("HeatingYN = 0")
                raw = heat2raw[feature]
                raw_no_space = raw.replace(" ", "")
                conditions.append("(LOWER(Heating) NOT LIKE %s OR Heating IS NULL)")
                params.append(f"%{raw_no_space}%")

            else:
                conditions.append("(LOWER(L_Remarks) NOT LIKE %s OR L_Remarks IS NULL)")
                params.append(f"%{feature.lower()}%")

    def to_sql(self, table = 'rets_property', filters = None):
        conditions = []
        params = []
        if filters is None:
            filters = self.filters
        
        if "price_max" in filters:
            conditions.append("L_SystemPrice <= %s")
            params.append(filters["price_max"])

        if "price_min" in filters:
            conditions.append("L_SystemPrice >= %s")
            params.append(filters["price_min"])
        
        if "price" in filters:
            conditions.append("L_SystemPrice = %s")
            params.append(filters["price"])

        if "bedrooms" in filters:
            conditions.append("L_Keyword2 = %s")
            params.append(filters["bedrooms"])

        if "bedrooms_min" in filters:
            conditions.append("L_Keyword2 >= %s")
            params.append(filters["bedrooms_min"])
        
        if "bedrooms_max" in filters:
            conditions.append("L_Keyword2 <= %s")
            params.append(filters["bedrooms_max"])

        if "bathrooms" in filters:
            conditions.append("LM_Dec_3 = %s")
            params.append(filters["bathrooms"])

        if "bathrooms_min" in filters:
            conditions.append("LM_Dec_3 >= %s")
            params.append(filters["bathrooms_min"])
        
        if "bathrooms_max" in filters:
            conditions.append("LM_Dec_3 <= %s")
            params.append(filters["bathrooms_max"])
        
        if "bathroom_half" in filters:
            conditions.append("BathroomsHalf = %s")
            params.append(filters["bathroom_half"])

        if "sqft_min" in filters:
            conditions.append("LM_Int2_3 >= %s")
            params.append(filters["sqft_min"])

        if "sqft_max" in filters:
            conditions.append("LM_Int2_3 <= %s")
            params.append(filters["sqft_max"])
        
        if "sqft" in filters:
            conditions.append("LM_Int2_3 = %s")
            params.append(filters["sqft"])

        if "city" in filters:
            conditions.append("L_City = %s")
            params.append(filters["city"])

        self.parse_amenity_to_sql(conditions, params)
        self.parse_house_feature_to_sql(conditions, params)
        

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = (
            f"""
            SELECT id,
                L_Address as address,
                L_Zip as zipcode,
                L_City as city,
                L_State as state,
                L_Keyword2 as bedrooms,
                LM_Dec_3 as bathrooms,
                L_SystemPrice as price,
                L_Remarks as remark,
                L_Photos as photos,
                Flooring as flooring,
                ViewYN,
                PoolPrivateYN,
                AttachedGarageYN,
                FireplaceYN,
                HeatingYN,
                Appliances,
                CoolingYN,
                GarageYN,
                SpaYN,
                BathroomsHalf as half_bathrooms,
                AssociationAmenities,
                StructureType,
                ArchitecturalStyle,
                Cooling,
                Heating,
                View,
                FireplaceFeatures,
                InteriorFeatures,
                PoolFeatures,
                CommunityFeatures,
                SecurityFeatures,
                SpaFeatures
            FROM {table} WHERE {where_clause} LIMIT 50""")
        return sql, params
