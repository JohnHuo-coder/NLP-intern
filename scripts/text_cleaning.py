import re
import nltk
from collections import Counter
from nltk.util import ngrams
import html
import unicodedata
class TextCleaner:
    def __init__(self):
        self.abbrev_map = {
            'br': 'bedroom', 'bdrm': 'bedroom', 'ba': 'bathroom', 'bth': 'bathroom', 
            'mbr': 'master bedroom', 'mba': 'master bathroom', 'kit': 'kitchen',
            'lr': 'living room', 'dr': 'dining room', 'fr': 'family room',
            'ac': 'air conditioning', 'ch': 'central heat', 'gar': 'garage',
            'pkg': 'parking',
            'sqft': 'square feet', "sq ft": "square feet", "sq. ft": "square feet",
            'w/': 'with', 'w/o': 'without', 
        }
    def _extract_top_ngrams(self, col, n = 2, top_n = 200):
        all_text = ' '.join(col.dropna().str.lower())
        tokens = nltk.word_tokenize(all_text)
        grams = list(ngrams(tokens, n))
        freq = Counter(grams)
        top_ngrams = [{"term": " ".join(ngram), "count": count} for ngram, count in freq.most_common(top_n)]
        return top_ngrams
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
        entity_pattern = r"&[a-zA-Z]+;"
        entities = re.findall(entity_pattern, all_text)
        # HTML tags
        tag_pattern = r"</?[^>]+>"
        tags = re.findall(tag_pattern, all_text)
        results = entities + tags
        cnt = len(results)
        return cnt, results
    def clean_text(self, text):
        text = self.normalize_unicode(text)
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
        return text
    def normalize_measurements(self, text):

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
        # 450k → 450000
        text = re.sub(r'(\d+)k', lambda m: str(int(m.group(1))*1000), text, flags=re.I)
        # 1.2m → 1200000
        text = re.sub(r'(\d+\.?\d*)m', lambda m: str(int(float(m.group(1))*1000000)), text, flags=re.I)
        return text
    def profile_column(self, df, column_name, n_gram = 2, most_common_gram = 200, most_common_abbr = 10):
        """Analyze what's actually in L_Remarks"""
        html_cnt, html_results = self._detect_html(df[column_name])
        return {
            'null_rate': df[column_name].isnull().mean(),
            'avg_length': df[column_name].str.len().mean(),
            'common_terms': self._extract_top_ngrams(df[column_name], n_gram, most_common_gram),
            'price_mentions': df[column_name].str.contains(r'\$\d').sum(),
            'has_html': html_cnt,
            'html_examples': html_results,
            'common_abbreviations': self._detect_abbreviations(df[column_name], most_common_abbr)
        }