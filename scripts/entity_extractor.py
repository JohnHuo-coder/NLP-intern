import json
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEGATIONS = [
    "no",
    "not",
    "without",
    "does not have",
    "don't have",
    "doesn't have",
    "without any",
]

_OR_GAP = re.compile(
    r"""
    ^[\s,]*
    (?:
        /+
        | \|
        | and/or
        | or(?:\s+(?:a|an|the))?
    )
    [\s,]*$
    """,
    re.I | re.X,
)


def _build_pattern(phrase):
    words = phrase.split()
    if len(words) == 1:
        return re.compile(rf"\b{re.escape(phrase)}s?\b", re.I)
    return re.compile(
        r"\b" + r"\s+".join(re.escape(w) for w in words) + r"s?\b",
        re.I,
    )


def _build_negation_pattern(phrase):
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


def group_spans_by_connector(text, kept):
    """Split kept ``(start, end, phrase)`` hits into AND groups; OR lives inside a group.

    Looks at the gap *between* consecutive hits (not a per-term prefix group):
    ``/``, ``or``, ``and/or`` → same OR group; anything else (``and``, ``with``,
    comma, other words) → new AND group.
    """
    if not kept:
        return []
    lowered = text.lower()
    ordered = sorted(kept, key=lambda x: x[0])
    groups = [[ordered[0][2]]]
    for i in range(1, len(ordered)):
        prev_end = ordered[i - 1][1]
        start, _, phrase = ordered[i]
        gap = lowered[prev_end:start]
        if _OR_GAP.match(gap.strip()) or (
            "/" in gap and re.fullmatch(r"[\s/,|]*", gap)
        ):
            if phrase not in groups[-1]:
                groups[-1].append(phrase)
        else:
            groups.append([phrase])
    return groups


def _load_json(filename: str, override=None):
    path = Path(override) if override else _PROJECT_ROOT / "data" / "processed" / filename
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class EntityExtractor:
    def __init__(self):
        self.amenity_full = _load_json("amenities.json")
        self.feature_full = _load_json("features.json")
        self.finance_full = _load_json("finance.json")
        self.amenities = self.amenity_full["amenities"]
        self.features = self.feature_full["features"]
        self.finance = self.finance_full["finance"]

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
            pat = _build_pattern(phrase)
            for m in pat.finditer(lowered):
                if any(m.start() < ne and m.end() > ns for ns, ne in neg_spans):
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

    def _extract_negated_phrases(self, text, phrases):
        """Negation cue + optional article + taxonomy phrase (``no FHA``, ``without a garage``)."""
        if not text:
            return [], []
        lowered = text.lower()
        matches = []
        for phrase in sorted(phrases, key=len, reverse=True):
            pat = _build_negation_pattern(phrase)
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

    def extract_financing_options(self, text, neg_kept=None):
        """Financing and sale-type cues (e.g. FHA, VA, cash, seller financing, short sale)."""
        if not text:
            return [], []
        if neg_kept is None:
            _, neg_kept = self.extract_negated_financing_options(text)
        neg_spans = [(s, e) for s, e, _ in neg_kept]
        phrases = sorted(self.finance, key=len, reverse=True)
        return self._extract_taxonomy_phrases(text, phrases, neg_spans=neg_spans)

    def extract_negated_financing_options(self, text):
        """Financing cues explicitly rejected (e.g. ``no FHA``, ``without VA``)."""
        return self._extract_negated_phrases(text, self.finance)

    def extract_negated_amenities_features(self, text, type):
        """Amenities/features explicitly rejected (e.g. ``no pool``, ``without a garage``)."""
        tax = self.amenities if type == "amenity" else self.features
        return self._extract_negated_phrases(text, tax)

    def extract_amenities_features(self, text, type, neg_kept=None):
        """
        Positive amenity/features mentions. Spans overlapping negated hits are dropped
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
