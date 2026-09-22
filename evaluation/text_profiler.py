import re
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.util import ngrams

from scripts.text_cleaning import TextCleaner


class TextProfiler:
    """Offline column profiling for remarks / listing text. Not used on the API path."""

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

    def __init__(self, cleaner: TextCleaner | None = None):
        self.cleaner = cleaner or TextCleaner()
        self.stop_words = set(stopwords.words("english"))

    def _extract_top_ngrams(self, col, top_n=200):
        all_text = ' '.join(col.dropna().str.lower())
        tokens = [t for t in nltk.word_tokenize(all_text) if re.fullmatch(r"[a-z]+", t.lower())]
        bigrams = [(a, b) for (a, b) in ngrams(tokens, 2) if a not in self.stop_words and b not in self.stop_words]
        freq = Counter(bigrams)
        top_bigrams = [{"term": " ".join(bigram), "count": count} for bigram, count in freq.most_common(top_n)]
        return top_bigrams

    def detect_abbreviations(self, col, top_abbr=10):
        patterns = self.cleaner._abbrev_pattern()
        counter = Counter()
        for pattern in patterns:
            matches = col.dropna().str.findall(pattern, flags=re.I)
            for lst in matches:
                for m in lst:
                    if isinstance(m, tuple):
                        m = m[-1]
                    counter[m.lower()] += 1
        return counter.most_common(top_abbr)

    def detect_html(self, col):
        all_text = " ".join(col.dropna().astype(str))
        entities = re.findall(r"&[a-zA-Z0-9#]+;", all_text)
        tags = re.findall(r"</?[^>]+>", all_text)
        results = entities + tags
        return len(results), Counter(results)

    def detect_unicode(self, col):
        counter = Counter(c for c in " ".join(col.dropna()) if ord(c) > 127)
        return len(counter), counter

    def detect_measurements(self, col):
        counter = Counter()
        cnt = 0
        for pattern, label in self._measurement_patterns:
            matches = col.dropna().astype(str).str.findall(pattern, flags=re.I)
            for lst in matches:
                for _ in lst:
                    counter[label] += 1
                    cnt += 1
        return cnt, counter

    def sample_compare(self, col, k=10, seed=None):
        idx = col.sample(k, random_state=seed).index
        cleaned = self.cleaner.clean_column(col)
        return (
            pd.DataFrame({
                "original": col.loc[idx],
                "cleaned": cleaned.loc[idx]
            }).reset_index(drop=True)
        )

    @staticmethod
    def _minmax_avg(values: list[int]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "avg": None, "max": None}
        return {
            "min": int(min(values)),
            "avg": sum(values) / len(values),
            "max": int(max(values)),
        }

    @classmethod
    def _span_length_stats(cls, spans: list[str]) -> dict:
        return {
            "count": len(spans),
            "chars": cls._minmax_avg([len(s) for s in spans]),
            "words": cls._minmax_avg([len(s.split()) for s in spans]),
        }

    @staticmethod
    def _period_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.findall(r"[^.]*\.", text) if s.strip()]

    @staticmethod
    def _split_paragraphs(text: str, delimiter: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return [p.strip() for p in normalized.split(delimiter) if p.strip()]

    def _chunking_length_stats(self, col: pd.Series) -> dict:
        texts = col.dropna().astype(str).tolist()
        sentences: list[str] = []
        paragraphs_newline: list[str] = []
        paragraphs_blank: list[str] = []
        has_blank_line = False
        for text in texts:
            sentences.extend(self._period_sentences(text))
            paragraphs_newline.extend(self._split_paragraphs(text, "\n"))
            if "\n\n" in text.replace("\r\n", "\n"):
                has_blank_line = True
                paragraphs_blank.extend(self._split_paragraphs(text, "\n\n"))
        return {
            "char_length": self._minmax_avg([len(t) for t in texts]),
            "word_count": self._minmax_avg([len(t.split()) for t in texts]),
            "sentence": self._span_length_stats(sentences),
            "paragraph_newline": self._span_length_stats(paragraphs_newline),
            "paragraph_blank_line": (
                self._span_length_stats(paragraphs_blank) if has_blank_line else None
            ),
        }

    def profile_column(self, df, column_name, most_common_gram=200, most_common_abbr=10):
        html_cnt, html_results = self.detect_html(df[column_name])
        unicode_cnt, unicode_results = self.detect_unicode(df[column_name])
        meas_cnt, meas_counter = self.detect_measurements(df[column_name])
        length_stats = self._chunking_length_stats(df[column_name])
        return {
            'null_rate': df[column_name].isnull().mean(),
            **length_stats,
            'common_terms': self._extract_top_ngrams(df[column_name], most_common_gram),
            'measurement_mentions': meas_cnt,
            'measurement_counter': meas_counter,
            'has_html': html_cnt,
            'html_examples': html_results,
            'has_unicode': unicode_cnt,
            'unicode_examples': unicode_results,
            'common_abbreviations': self.detect_abbreviations(df[column_name], most_common_abbr)
        }
