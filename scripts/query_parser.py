import re
import sys
from pathlib import Path

# convert 2.5 bath into 2 full bath + 1 half bath, since the table only has full bath column and half bath column

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

        word_pattern = rf'\b(?:under|below|at most|less than|up to|maximum|no more than|capped at)\s+{num}\b'
        symbol_pattern = rf'\b<\s*{num_symbol}\b'

        filter_key_max = f'{cat}_max'

        for p in [word_pattern, symbol_pattern]:
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

        word_pattern = rf'\b(?:at least|above|over|more than|starting from|no less than|minimum)\s+{num}\b'
        symbol_gt_pattern = rf'\b>\s*{symbol_gt}\b'

        filter_key_min = f'{cat}_min'

        for p in [word_pattern, symbol_gt_pattern, symbol_plus_pattern]:
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
        p1 = r'\b(?:around|about|approximately|roughly|~)\s*\$?(\d{4,})\s*(dollar|usd|dollars)?\b'
        match = re.search(p1, query, re.I)
        if match :
            self.filters['price_min'] = int(match.group(1)) * 0.8
            self.filters['price_max'] = int(match.group(1)) * 1.2
            return 

        # Price pattern exact
        if match := re.search(r'\b(?:at|exactly)?\s+\$?(\d{4,})\s*(dollar|usd|dollars)?\b', query, re.I):
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
        p1 = r'\b(?:around|about|approximately|roughly|~)\s*(\d+(?:\.\d+)?)\s*(?:square feet)\b'
        match = re.search(p1, query, re.I)
        if match :
            self.filters['sqft_min'] = int(match.group(1)) * 0.8
            self.filters['sqft_max'] = int(match.group(1)) * 1.2
            return 

        # sqft pattern exact
        if match := re.search(r'\b(?:at|exactly)\s+(\d+(?:\.\d+)?)\s*(?:square feet)\b', query, re.I):
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
        p1 = r'\b(?:around|about|approximately|roughly|~)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bed|bedroom|br|bd)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = int(raw) if raw.isdigit() else _spell_num_from_word(raw)
            self.filters['bedrooms_min'] = num * 0.8
            self.filters['bedrooms_max'] = num * 1.2
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
        p1 = r'\b(?:around|about|approximately|roughly|~)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bath|bathroom)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = float(raw) if _is_number(raw) else _spell_num_from_word(raw)
            full, half = _split_number(num)
            self.filters['bathrooms_min'] = full * 0.8
            self.filters['bathrooms_max'] = full * 1.2
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
        amenities, _ = self.extractor.extract_amenities(query)
        if amenities:
            self.filters["amenities"] = amenities
    def parse_features(self, query):
        features, _ = self.extractor.extract_amenities(query)
        if features:
            self.filters["amenities"] = features

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

    def to_sql(self, filters, table = 'rets_property'):
        conditions = []
        params = []
        
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

        # Amenities: search L_Remarks for each keyword
        for amenity in filters.get("amenities", []):
            conditions.append("LOWER(L_Remarks) LIKE %s")
            params.append(f"%{amenity.lower()}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT 50"
        return sql, params
