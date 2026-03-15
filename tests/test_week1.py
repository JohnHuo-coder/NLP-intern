import pytest
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import re
import json
import pandas as pd
def test_taxonomy_loaded():
    with open('data/processed/taxonomy.json') as f:
        tax = json.load(f)
    assert len(tax['terms']) >= 200
    assert all('id' in t and 'term' in t for t in tax['terms'])

def test_sample_data_quality():
    df = pd.read_csv('data/processed/listing_sample.csv')
    assert len(df) >= 500
    assert df['remarks'].str.len().min() > 50

def text_taxonomy_coverage():
    df = pd.read_csv("data/processed/listing_sample.csv")
    with open("data/processed/taxonomy.json", 'r') as f:
        data = json.load(f)
    terms = set([item["term"] for item in data])
    stop_words = set(stopwords.words("english"))

    bigram_sizes = []
    text_length = 0
    term_count = 0
    for text in df["remarks"]:
        tokens = [ t for t in nltk.word_tokenize(text) if re.match(r"[a-z]+", t.lower())]
        bigrams = [" ".join(bg) for bg in ngrams(tokens, 2) if bg[0] not in stop_words and bg[1] not in stop_words]
        bigram_sizes.append(len(bigrams))
        for b in bigrams:
            if b in terms:
                term_count +=1
        text_length += len(bigrams)
    coverage = term_count / text_length * 100
    assert coverage > 30