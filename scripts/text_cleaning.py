import re
import html
import unicodedata
import pandas as pd
from pathlib import Path


def _scale_money(num: str, scale: int) -> str:
    return str(int(round(float(num) * scale)))


class TextCleaner:
    def __init__(self):
        # usually stick together with number: 2bd, 2ba, 2rms, 2-bd, 2-ba
        self.room_abbrev_map = {
            'br': 'bedroom', 'bd': 'bedroom', 'bdr': 'bedroom', 'bdrm': 'bedroom', 'bdrms': 'bedroom',
            'ba': 'bathroom', 'bth': 'bathroom', 'bthrm': 'bathroom', 'bthrms': 'bathroom', 'bath': 'bathroom',
            'mbr': 'master bedroom', 'mba': 'master bathroom',
            'rm': 'room', 'rms': 'rooms', 'flr': 'floor', 'lvl': 'level',
        }

        self.measurement_abbrev_map = {
            'sqft': 'square feet',
            "sq ft": "square feet",
            "sq. ft": "square feet",
            "sq. ft.": "square feet",
            "sq.ft": "square feet",
            "sq.ft.": "square feet",
            "sq .ft": "square feet"
        }

        self.normal_abbrev_map = {
            'kit': 'kitchen',
            'lr': 'living room',
            'dr': 'dining room',
            'fr': 'family room',
            'gar': 'garage',
            'bsmt': 'basement',
            'pkg': 'parking space',
            'appl': 'appliance',
            'min': 'minute',
            'yr': 'year',
            'adu': 'accessory dwelling unit'
        }
        self.no_s_abbrev = {'ac': 'air conditioning', 'a/c': 'air conditioning', 'fp': 'fireplace', 'wic': 'walk-in closet',
            'w/': 'with', 'w/o': 'without','yb': 'year built','renov': 'renovated', 'upd': 'updated',
            'incl': 'including', 'nr': 'near', 'reo': 'bank-owned', 'dom': 'days on market', 'approx': 'approximately',
            'pvt': 'private', 'avail': 'available', 'hvac': 'heating and cooling', 'fin': 'finished', 'ch': 'central heat',
            'mls': 'multiple listing service', 'hoa': 'homeowners association', 'ss': 'stainless steel', 'hw': 'hardwood',
            'w/d': 'washer dryer','wd': 'washer dryer', "bbq": "barbecue"
        }

    def _abbrev_pattern(self):
        room_units = sorted(self.room_abbrev_map.keys(), key=len, reverse=True)
        # Digit and the hyphen/space before the unit are one optional group so
        # "two baths" does not swallow the space and become "twobathroom".
        room_pattern = (
            r'\b(?:(\d+(?:\.\d+)?\+?)\s*-?\s*)?('
            + '|'.join(map(re.escape, room_units))
            + r')s?\b'
        )
        area_pattern = r'\b(?:(\d+(?:\.\d+)?\+?)\s*-?\s*)?(sq\s*\.?\s*ft\.?|sqft)\b'
        no_s_keys = sorted(self.no_s_abbrev.keys(), key=len, reverse=True)
        no_s_pattern = r'(?<!\w)(' + '|'.join(map(re.escape, no_s_keys)) + r')(?!\w)'
        normal_keys = sorted(self.normal_abbrev_map.keys(), key=len, reverse=True)
        normal_pattern = r'(?<!\w)(' + '|'.join(map(re.escape, normal_keys)) + r')s?(?!\w)'

        return [room_pattern, area_pattern, no_s_pattern, normal_pattern]

    def clean_text(self, text):
        text = self.normalize_unicode(text)
        text = self.normalize_html(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.expand_abbreviations(text)
        return text.strip()

    def normalize_html(self, text):
        text = html.unescape(text)
        # Tag names start with a letter so "<4 bedroom, >3 bath" is not stripped.
        text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
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
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)\s*(sq\.?\s*-?\s*ft\.?|sqft|sf|acres?)\b', lambda m: m.group(1).replace(",", "") + " " + m.group(2), text, flags=re.I)
        # sq. ft. / sq ft / sqft / sq-ft / sf -> square feet
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sq\.?\s*ft\.?\b', r'\1 square feet', text, flags=re.I)
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sq-ft\.?\b', r'\1 square feet', text, flags=re.I)
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sqft\b', r'\1 square feet', text, flags=re.I)
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sf\b', r'\1 square feet', text, flags=re.I)
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*square\s+feet\b', r'\1 square feet', text, flags=re.I)
        # acres: keep as "X acres"
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s+acres?\b', r'\1 acres', text, flags=re.I)
        # sq meters -> square meters
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*sq\.?\s*m(?:eters?)?\b', r'\1 square meters', text, flags=re.I)
        return text

    def expand_abbreviations(self, text):
        room_pattern, area_pattern, no_s_pattern, normal_pattern = self._abbrev_pattern()
        text = re.sub(room_pattern, lambda m: (f"{m.group(1)} {self.room_abbrev_map[m.group(2).lower()]}"
                                                if m.group(1)
                                                else self.room_abbrev_map[m.group(2).lower()]), text, flags=re.I)
        text = re.sub(area_pattern, lambda m: (f"{m.group(1)} {self.measurement_abbrev_map[m.group(2).lower()]}"
                                                if m.group(1)
                                                else self.measurement_abbrev_map[m.group(2).lower()]), text, flags=re.I)
        text = re.sub(no_s_pattern, lambda m: self.no_s_abbrev[m.group(1).lower()], text, flags=re.I)
        text = re.sub(normal_pattern, lambda m: self.normal_abbrev_map[m.group(1).lower()], text, flags=re.I)
        return text

    # TODO: dollar sign might be kept as a signal
    def normalize_prices(self, text):
        # Remove spaces between $ and number for consistent matching
        text = re.sub(r'\$\s+', '$', text)
        # Remove commas in numbers first so $1,234k and 1,234,567 work correctly
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)\b', lambda m: m.group(0).replace(",", ""), text)
        # $450k / $5.5k / $1.2m (allow decimals). round() not int():
        # float("2.07") * 1_000_000 is 2069999.999… and int() truncates to 2069999.
        text = re.sub(r'\$(\d+\.?\d*)k\b', lambda m: '$' + _scale_money(m.group(1), 1000), text, flags=re.I)
        text = re.sub(r'\$(\d+\.?\d*)m\b', lambda m: '$' + _scale_money(m.group(1), 1_000_000), text, flags=re.I)
        # 450k / 5.5k → 450000 / 5500, 1.2m → 1200000 (no $)
        text = re.sub(r'\b(\d+\.?\d*)k\b', lambda m: _scale_money(m.group(1), 1000), text, flags=re.I)
        text = re.sub(r'\b(\d+\.?\d*)m\b', lambda m: _scale_money(m.group(1), 1_000_000), text, flags=re.I)
        # 722 grand / $722 grand → 722000 (same scale as k)
        text = re.sub(r'\$(\d+\.?\d*)\s*grand\b', lambda m: '$' + _scale_money(m.group(1), 1000), text, flags=re.I)
        text = re.sub(r'\b(\d+\.?\d*)\s*grand\b', lambda m: _scale_money(m.group(1), 1000), text, flags=re.I)
        # N thousand / N million (word form)
        text = re.sub(r'\b(\d+)\s+thousand\b', lambda m: str(int(m.group(1)) * 1000), text, flags=re.I)
        text = re.sub(r'\b(\d+\.?\d*)\s+million\b', lambda m: _scale_money(m.group(1), 1_000_000), text, flags=re.I)
        return text

    def clean_column(self, col):
        cleaned_col = col.apply(self.clean_text)
        return cleaned_col


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    in_path = root / "data" / "unprocessed" / "listing_sample.csv"
    in_path_10k = root / "data" / "unprocessed" / "listing_sample_10k.csv"
    in_path_all = root / "data" / "unprocessed" / "all_listings.csv"
    out_path = root / "data" / "processed" / "listing_sample_cleaned.csv"
    out_path_10k = root / "data" / "processed" / "listing_sample_cleaned_10k.csv"
    out_path_all = root / "data" / "processed" / "all_listings_cleaned.csv"

    df = pd.read_csv(in_path)
    df_10k = pd.read_csv(in_path_10k)
    df_all = pd.read_csv(in_path_all)
    cleaner = TextCleaner()
    df["remarks"] = cleaner.clean_column(df["remarks"])
    df.to_csv(out_path, index=False)
    df_10k["remarks"] = cleaner.clean_column(df_10k["remarks"])
    df_10k.to_csv(out_path_10k, index=False)
    df_all["remarks"] = cleaner.clean_column(df_all["remarks"])
    df_all.to_csv(out_path_all, index=False)
