import re
import sys
from text_cleaning import TextCleaner


_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def _spell_num_from_word(word):
    """Map spelled-out digit to int; ``re.I`` can yield ``Two`` so normalize case."""
    if not word:
        return None
    return _WORD_TO_NUM.get(str(word).lower())

class QueryParser:
    def __init__(self):
        self.normalizer = TextCleaner()
        self.filters = {}
    def _clean_query(self, query):
        query = self.normalizer.clean_text(query)
        return query
    
    def parse_price(self, query):

        # Price patterns range
        # (from) $1000 to $2000, $1000 ~ $2000, $1000 - $2000, between a and b 
        p1 = r'\b(?:from\s+)?\$?(\d+)\s+(?:to)\s+\$?(\d+)\b'
        p2 = r'\b(?:between)\s+\$?(\d+)\s+(?:and)\s+\$?(\d+)\b'
        p3 = r'\b\$?(\d+)\s*(?:-|~)\s*\$?(\d+)\b'
        for p in [p1, p2, p3]:
            match = re.search(p, query, re.I)
            if match :
                self.filters['price_min'] = int(match.group(1))
                self.filters['price_max'] = int(match.group(2))
                return # prevent being covered by exact match

        min_matched = False
        max_matched = False

        # Price patterns max
        # under $20000, <$200000
        p1 = r'\b(?:under|below|at most|less than|up to|maximum|no more than|capped at)\s+\$?(\d+)\b'
        p2 = r'\b<\s*\$?(\d+)\b'
        for p in [p1, p2]:
            match = re.search(p, query, re.I)
            if match :
                self.filters['price_max'] = int(match.group(1))
                max_matched = True
                break
        
        # Price patterns min
        # above $20000, $20000+, >20000
        p1 = r'\b(?:at least|above|over|more than|starting from|no less than|minimum)\s+\$?(\d+)\b'
        p2 = r'\b\$?(\d+)\s*\+\b'
        p3 = r'\b>\s*\$?(\d+)\b'
        for p in [p1, p2, p3]:
            match = re.search(p, query, re.I)
            if match :
                self.filters['price_min'] = int(match.group(1))
                min_matched = True
                break
        
        if min_matched or max_matched:
            return 
        
        # Price pattern approx
        # ~ $20000, around 20000
        p1 = r'\b(?:around|about|approximately|roughly|~)\s*\$?(\d+)\b'
        match = re.search(p1, query, re.I)
        if match :
            self.filters['price_min'] = int(match.group(1)) * 0.8
            self.filters['price_max'] = int(match.group(1)) * 1.2
            return 

        # Price pattern exact
        if match := re.search(r'\b(?:at|exactly)\s+\$?(\d+)\b', query, re.I):
            self.filters['price'] = int(match.group(1))

    def parse_bedroom(self, query):
        # Bedrooms patterns range
        # 3 to 4, two to three, 3 ~ 4, 3 - 4, between 3 and 4, between one and two
        p1 = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:to)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bed|bedroom|br|bd)s?\b'
        p2 = r'\b(?:between)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:and)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bed|bedroom|br|bd)s?\b'
        p3 = r'\b(\d+)\s*(?:-|~)\s*(\d+)\s*(?:bed|bedroom|br|bd)s?\b'
        for p in [p1, p2, p3]:
            match = re.search(p, query, re.I)
            if match :
                raw_min = match.group(1)
                raw_max = match.group(2)
                min_b = int(raw_min) if raw_min.isdigit() else _spell_num_from_word(raw_min)
                max_b = int(raw_max) if raw_max.isdigit() else _spell_num_from_word(raw_max)
                self.filters['bedrooms_min'] = min_b
                self.filters['bedrooms_max'] = max_b
                return 

        min_matched = False
        max_matched = False

        # Bedroom patterns max
        # less than two bedrooms, below 2 bedrooms, <5 bedrooms
        p1 = r'\b(?:under|below|at most|less than|up to|maximum|no more than|capped at)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bed|bedroom|br|bd)s?\b'
        p2 = r'\b<\s*(\d+)\s*(?:bed|bedroom|br|bd)s?\b'
        for p in [p1, p2]:
            match = re.search(p, query, re.I)
            if match :
                raw = match.group(1)
                num = int(raw) if raw.isdigit() else _spell_num_from_word(raw)
                self.filters['bedrooms_max'] = num
                max_matched = True
                break
        
        # Bedroom patterns min
        # More than two bedrooms, above 2 bedrooms, >5 bedrooms, 5+ bedrooms
        p1 = r'\b(?:at least|above|over|more than|starting from|no less than|minimum)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bed|bedroom|br|bd)s?\b'
        p2 = r'\b(\d+)\s*\+\s*(?:bed|bedroom|br|bd)s?\b'
        p3 = r'\b>\s*(\d+)\s*(?:bed|bedroom|br|bd)s?\b'
        for p in [p1, p2, p3]:
            match = re.search(p, query, re.I)
            if match :
                raw = match.group(1)
                num = int(raw) if raw.isdigit() else _spell_num_from_word(raw)
                self.filters['bedrooms_min'] = num
                min_matched = True
                break

        if min_matched or max_matched:
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
        # Bathroom patterns range
        # 3 to 4, 3.5 to 4.5, two to three, 3 ~ 4, 3 - 4, 3.5 ~ 4.5, 3.5 - 4.5, between 3 and 4, between 3.5 and 4.5, between one and two
        p1 = r'\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:to)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(bath|bathroom)s?\b'
        p2 = r'\b(?:between)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:and)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(bath|bathroom)s?\b'
        p3 = r'\b(\d+(?:\.\d+)?)\s*(?:-|~)\s*(\d+(?:\.\d+)?)\b'
        for p in [p1, p2, p3]:
            match = re.search(p, query, re.I)
            if match :
                raw_min = match.group(1)
                raw_max = match.group(2)
                min_b = float(raw_min) if raw_min.isdigit() else _spell_num_from_word(raw_min)
                max_b = float(raw_max) if raw_max.isdigit() else _spell_num_from_word(raw_max)
                self.filters['bathrooms_min'] = min_b
                self.filters['bathrooms_max'] = max_b
                return 

        min_matched = False
        max_matched = False

        # Bathroom patterns max
        # less than two bathrooms, below 2.5 bathrooms, <5.5 bathrooms
        p1 = r'\b(?:under|below|at most|less than|up to|maximum|no more than|capped at)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(bath|bathroom)s?\b'
        p2 = r'\b<\s*(\d+(?:\.\d+)?)\s*(bath|bathroom)s?\b'
        for p in [p1, p2]:
            match = re.search(p, query, re.I)
            if match :
                raw = match.group(1)
                num = float(raw) if raw.isdigit() else _spell_num_from_word(raw)
                self.filters['bathrooms_max'] = num
                max_matched = True
                break
        
        # Bathroom patterns min
        # More than two bathrooms, above 2.5 bathrooms, >5.5 bthrooms, 5+ bathrooms
        p1 = r'\b(?:at least|above|over|more than|starting from|no less than|minimum)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(bath|bathroom)s?\b'
        p2 = r'\b(\d+(?:\.\d+)?)\s*\+\s*(bath|bathroom)s?\b'
        p3 = r'\b>\s*(\d+(?:\.\d+)?)\s*(bath|bathroom)s?\b'
        for p in [p1, p2, p3]:
            match = re.search(p, query, re.I)
            if match :
                raw = match.group(1)
                num = float(raw) if raw.isdigit() else _spell_num_from_word(raw)
                self.filters['bathrooms_min'] = num
                min_matched = True
                break

        if min_matched or max_matched:
            return 

        # Bathrooms pattern exact
        # 2.5 bathrooms, 2-bathroom, two bathrooms, two-bathroom, 
        p1= r'\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*-\s*|\s+)(bath|bathroom)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = float(raw) if raw.isdigit() else _spell_num_from_word(raw)
            self.filters["bathrooms"] = num
        
        # 2 full bathrooms, two full bathrooms
        p1= r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:full\s+)(bath|bathroom)s?\b'
        match = re.search(p1, query, re.I)
        if match :
            raw = match.group(1)
            num = float(raw) if raw.isdigit() else _spell_num_from_word(raw)
            self.filters["bathrooms"] = num
        
        # more to add: half bathroom cases, mix of full and half
    def parse_sqft(self, query):
        return
    
    def parse_city(self, query):
        return

    def parse_amenity(self, query):
        return

    def parse(self, query):
        self.filters = {}
        self.parse_bedroom(query)
        self.parse_bathroom(query)
        self.parse_price(query)
        

        return self.filters

    def to_sql(self, filters):
        conditions = []
        params = []

        if 'price_max' in filters:
            conditions.append('L_SystemPrice <= %s')
            params.append(filters['price_max'])

        if 'bedrooms' in filters:
            conditions.append('L_Keyword2 = %s')
            params.append(filters['bedrooms'])

        where_clause = ' AND '.join(conditions)
        return f"SELECT * FROM rets_property WHERE {where_clause}", params