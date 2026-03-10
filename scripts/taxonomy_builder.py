import nltk
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
import pandas as pd
import json
import re

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


df = pd.read_csv("data/processed/listing_sample.csv")
# Extract bigrams from remarks
all_text = ' '.join(df["remarks"].dropna().str.lower())
# tokens = [ t for t in nltk.word_tokenize(all_text) if re.match(r"[a-z]+", t) and t not in stop_words]
# bigrams = list(ngrams(tokens, 2))
tokens = [ t for t in nltk.word_tokenize(all_text) if re.fullmatch(r"[a-z]+", t.lower())]
bigrams = [(a, b) for (a, b) in ngrams(tokens, 2) if a not in stop_words and b not in stop_words]
freq = Counter(bigrams)
# Top 200 bigrams become taxonomy seed
top_bigrams = [{"term": " ".join(bigram), "count": count} for bigram, count in freq.most_common(1000)]
with open("data/processed/taxonomy.json", 'w') as f:
    json.dump(top_bigrams, f, indent = 2)
for bigram, count in freq.most_common(200):
    print(f"{' '.join(bigram)}: {count}")