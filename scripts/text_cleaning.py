import re
import nltk
from collections import Counter
from nltk.util import ngrams
import html
import unicodedata
from nltk.corpus import stopwords
import pandas as pd
class TextCleaner:
    def __init__(self):
        self.abbrev_map = {
            'br': 'bedroom', 'bd': 'bedroom', 'bdr': 'bedroom', 'bdrm': 'bedroom', 'bdrm': 'bedroom', 
            'ba': 'bathroom', 'bth': 'bathroom', 
            'mbr': 'master bedroom', 'mba': 'master bathroom', 'kit': 'kitchen',
            'lr': 'living room', 'dr': 'dining room', 'fr': 'family room',
            'rm': 'room', 'rms': 'rooms', 'flr': 'floor', 'lvl': 'level',
            'ac': 'air conditioning', 'ch': 'central heat', 'gar': 'garage', 'bsmt': 'basement',
            'pkg': 'parking', 'fp': 'fireplace', 'wic': 'walk-in closet',
            'sqft': 'square feet', "sq ft": "square feet", "sq. ft": "square feet",
            'w/': 'with', 'w/o': 'without', 'yr': 'year', 'yb': 'year built', 'hoa': 'homeowners association',
            'ss': 'stainless steel', 'hw': 'hardwood', 'w/d': 'washer dryer', 'wd': 'washer dryer', 'a/c': 'air conditioning',
            'hvac': 'heating and cooling', 'fin': 'finished', 'dom': 'days on market', 'mls': 'multiple listing service',
            'adu': 'accessory dwelling unit', 'reo': 'bank-owned', 'nr': 'near', 'mins': 'minutes',
            'approx': 'approximately', 'pvt': 'private', 'avail': 'available', 'incl': 'including',
            'appl': 'appliances', 'renov': 'renovated', 'upd': 'updated'
        }
        self.stop_words = set(stopwords.words("english"))
    def _extract_top_ngrams(self, col, top_n = 200):
        all_text = ' '.join(col.dropna().str.lower())
        tokens = [ t for t in nltk.word_tokenize(all_text) if re.fullmatch(r"[a-z]+", t.lower())]
        bigrams = [(a, b) for (a, b) in ngrams(tokens, 2) if a not in self.stop_words and b not in self.stop_words]
        freq = Counter(bigrams)
        top_bigrams = [{"term": " ".join(bigram), "count": count} for bigram, count in freq.most_common(top_n)]
        return top_bigrams
    def _detect_abbreviations(self, col, top_abbr = 10):
        pattern = r'(?<!\w)(' + '|'.join(map(re.escape, self.abbrev_map.keys())) + r')(?!\w)'
        counter = Counter()
        matches = col.dropna().str.findall(pattern, flags = re.I)
        for lst in matches:
            for m in lst:
                counter[m.lower()] += 1
        return counter.most_common(top_abbr)
    def _detect_html(self, col):
        """
        Detect HTML entities and HTML tags in the text column.
        """
        all_text = " ".join(col.dropna().astype(str))
        # HTML entities
        entities = re.findall(r"&[a-zA-Z0-9#]+;", all_text)
        # HTML tags
        tag_pattern = r"</?[^>]+>"
        tags = re.findall(tag_pattern, all_text)
        results = entities + tags
        cnt = len(results)
        return cnt, Counter(results)
    def _detect_unicode(self, col):
        counter = Counter(c for c in " ".join(col.dropna()) if ord(c) > 127)
        cnt = len(counter)
        return cnt, counter
    # Patterns we detect for measurements (for _detect_measurements)
    _measurement_patterns = [
        (r'\d+(?:,\d{3})*(?:\.\d+)?\s*sq\.?\s*ft\.?', 'sq ft'),
        (r'\d+(?:,\d{3})*(?:\.\d+)?\s*sqft', 'sqft'),
        (r'\d+(?:,\d{3})*(?:\.\d+)?\s*square\s+feet', 'square feet'),
        (r'\d+(?:,\d{3})*(?:\.\d+)?\s*acres?', 'acres'),
        (r'\d+(?:,\d{3})*(?:\.\d+)?\s*sq\.?\s*m(?:eters?)?', 'sq meters'),
        (r'(?<!\d)\.(\d)', 'leading_decimal'),
        (r"(\d+)'", 'feet_apostrophe'),
        (r'(\d+)"', 'inches_quote'),
        (r'(\d+)-?ft\b', 'ft_abbrev'),
    ]
    def _detect_measurements(self, col):
        """Detect measurement patterns in the text column. Returns (total_count, Counter of pattern types)."""
        counter = Counter()
        cnt = 0
        for pattern, label in self._measurement_patterns:
            matches = col.dropna().astype(str).str.findall(pattern, flags=re.I)
            for lst in matches:
                for _ in lst:
                    counter[label] += 1
                    cnt += 1
        return cnt, counter
    # Price patterns we detect (for _detect_price_mentions)
    _price_patterns = [
        (r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*k\b', '$_k'),
        (r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*m\b', '$_m'),
        (r'\b(\d+\.?\d*)\s*k\b', 'Nk'),
        (r'\b(\d+\.?\d*)\s*m\b', 'Nm'),
        (r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', '$N'),
        (r'\b(\d{1,3}(?:,\d{3})+)\b', 'N_comma'),
        (r'\b(\d+)\s*thousand\b', 'N thousand'),
        (r'\b(\d+\.?\d*)\s*million\b', 'N million'),
    ]
    def _detect_price_mentions(self, col):
        """Detect price-like patterns in the text column. Returns (total_count, Counter of pattern types)."""
        counter = Counter()
        cnt = 0
        for pattern, label in self._price_patterns:
            matches = col.dropna().astype(str).str.findall(pattern, flags=re.I)
            for lst in matches:
                for _ in lst:
                    counter[label] += 1
                    cnt += 1
        return cnt, counter
    def clean_text(self, text):
        text = self.normalize_unicode(text)
        text = self.normalize_html(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.expand_abbreviations(text)
        return text.strip()
    def normalize_html(self, text):
        text = html.unescape(text)
        text = re.sub(r'</?[^>]*>', "", text)
        return text
    def normalize_unicode(self, text):
        text = unicodedata.normalize("NFKC", text)
        replacements = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
            "\xa0": " ",
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return text
    def normalize_measurements(self, text):
        # .5 → 0.5 (leading decimal)
        text = re.sub(r'(?<!\d)\.(\d)', r'0.\1', text)
        # 9' → 9 foot, 6" → 6 inch
        text = re.sub(r"(\d+)'", r'\1 foot', text)
        text = re.sub(r'(\d+)"', r'\1 inch', text)
        # 10-ft / 10ft → 10 foot
        text = re.sub(r'(\d+)-?ft\b', r'\1 foot', text, flags=re.I)
        # Normalize comma in numbers next to units: 1,500 -> 1500
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)\s*(sq\.?\s*ft\.?|sqft|acres?)\b', lambda m: m.group(1).replace(",", "") + " " + m.group(2), text, flags=re.I)
        # sq. ft. / sq ft / sqft -> square feet
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sq\.?\s*ft\.?\b', r'\1 square feet', text, flags=re.I)
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sqft\b', r'\1 square feet', text, flags=re.I)
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*square\s+feet\b', r'\1 square feet', text, flags=re.I)
        # acres: keep as "X acres"
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s+acres?\b', r'\1 acres', text, flags=re.I)
        # sq meters -> square meters
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sq\.?\s*m(?:eters?)?\b', r'\1 square meters', text, flags=re.I)
        return text
    def expand_abbreviations(self, text):
        pattern = r'(?<!\w)(' + '|'.join(map(re.escape, self.abbrev_map.keys())) + r')(?!\w)'
        return re.sub(
            pattern,
            lambda m: self.abbrev_map[m.group(0).lower()],
            text,
            flags=re.I
        )
    def normalize_prices(self, text):
        # Remove spaces between $ and number for consistent matching
        text = re.sub(r'\$\s+', '$', text)
        # Remove commas in numbers first so $1,234k and 1,234,567 work correctly
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)\b', lambda m: m.group(0).replace(",", ""), text)
        # $450k / $5.5k / $1.2m (allow decimals)
        text = re.sub(r'\$(\d+\.?\d*)k\b', lambda m: '$' + str(int(float(m.group(1)) * 1000)), text, flags=re.I)
        text = re.sub(r'\$(\d+\.?\d*)m\b', lambda m: '$' + str(int(float(m.group(1)) * 1000000)), text, flags=re.I)
        # 450k / 5.5k → 450000 / 5500, 1.2m → 1200000 (no $)
        text = re.sub(r'\b(\d+\.?\d*)k\b', lambda m: str(int(float(m.group(1)) * 1000)), text, flags=re.I)
        text = re.sub(r'\b(\d+\.?\d*)m\b', lambda m: str(int(float(m.group(1)) * 1000000)), text, flags=re.I)
        # N thousand / N million (word form)
        text = re.sub(r'\b(\d+)\s+thousand\b', lambda m: str(int(m.group(1)) * 1000), text, flags=re.I)
        text = re.sub(r'\b(\d+\.?\d*)\s+million\b', lambda m: str(int(float(m.group(1)) * 1000000)), text, flags=re.I)
        return text
    
    def clean_column(self, col):
        cleaned_col = col.apply(self.clean_text)
        return cleaned_col
    
    def sample_compare(self, col, k = 10, seed = None):
        idx = col.sample(k, random_state = seed).index
        cleaned = self.clean_column(col)
        return (
            pd.DataFrame({
                "original": col.loc[idx],
                "cleaned": cleaned.loc[idx]
            }).reset_index(drop = True)
        )

    def profile_column(self, df, column_name, most_common_gram = 200, most_common_abbr = 10):
        """Analyze what's actually in L_Remarks"""
        html_cnt, html_results = self._detect_html(df[column_name])
        unicode_cnt, unicode_results = self._detect_unicode(df[column_name])
        price_cnt, price_counter = self._detect_price_mentions(df[column_name])
        meas_cnt, meas_counter = self._detect_measurements(df[column_name])
        return {
            'null_rate': df[column_name].isnull().mean(),
            'avg_length': df[column_name].str.len().mean(),
            'avg_num_words': df[column_name].str.split().str.len().mean(),
            'common_terms': self._extract_top_ngrams(df[column_name], most_common_gram),
            'price_mentions': price_cnt,
            'price_counter': price_counter,
            'measurement_mentions': meas_cnt,
            'measurement_counter': meas_counter,
            'has_html': html_cnt,
            'html_examples': html_results,
            'has_unicode': unicode_cnt,
            'unicode_examples': unicode_results,
            'common_abbreviations': self._detect_abbreviations(df[column_name], most_common_abbr)
        }