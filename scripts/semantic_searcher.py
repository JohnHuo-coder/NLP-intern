from pathlib import Path

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pandas as pd
import time

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LISTINGS = _PROJECT_ROOT / "data" / "processed" / "listing_sample_cleaned_10k.csv"


class SemanticSearcher:
    def __init__(self, listings_path=None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # dim 384
        self.index = None
        path = self._resolve_listings_path(listings_path)
        self.listings = self._load_listings(path)
        self.remarks = self._extract_remarks()
        self._build_index()
        self.bm25 = self._build_BM_corpus()

    @staticmethod
    def _resolve_listings_path(listings_path):
        if listings_path is None:
            return _DEFAULT_LISTINGS
        p = Path(listings_path)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    def _load_listings(self, path):
        df = pd.read_csv(path)
        df = df[["L_ListingID", "remarks"]]
        df = df[df["remarks"].notna()]
        return df
    
    def _extract_remarks(self):
        remarks = self.listings["remarks"].to_list()
        return remarks

    def embed_remarks(self, batch_size = 64):
        print(f"Encoding {len(self.remarks)} listings...")

        embeddings = self.model.encode(
            self.remarks,
            batch_size = batch_size,
            show_progress_bar = True,
            convert_to_numpy = True
        )

        embeddings = embeddings.astype("float32")
        
        faiss.normalize_L2(embeddings)

        return embeddings

    def _build_index(self):
        # Build FAISS index
        embeddings = self.embed_remarks()
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) # Inner product + normalized embd to mimic cosine sim
        self.index.add(embeddings)

    def _build_BM_corpus(self):
        corpus = [remark.split() for remark in self.remarks]
        bm25 = BM25Okapi(corpus)
        return bm25

    def search_emb(self, query, top_k=5):
        start = time.time()
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)
        end = time.time()
        latency_ms = (end - start) * 1000
        results = [(self.remarks[i], scores[0][j]) for j, i in enumerate(indices[0])]
        ids = [(self.listings.iloc[i]["L_ListingID"], scores[0][j]) for j, i in enumerate(indices[0])]
        return results, ids, latency_ms
    
    def search_bm25(self, query, top_k = 5):
        query_tokens = query.split()
        start = time.time()
        scores = self.bm25.get_scores(query_tokens)
        top_k_idx = np.argsort(scores)[::-1][:top_k]
        end = time.time()
        latency_ms = (end-start) * 1000
        results = [(self.remarks[i], scores[i]) for i in top_k_idx]
        ids = [(self.listings.iloc[i]["L_ListingID"], scores[i]) for i in top_k_idx]
        return results, ids, latency_ms