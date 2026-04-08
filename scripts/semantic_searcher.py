from pathlib import Path

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pandas as pd
import time


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LISTINGS = _PROJECT_ROOT / "data" / "processed" / "listing_sample_cleaned_10k.csv"
_INDEX_PATH = _PROJECT_ROOT / "data" / "processed" / "index.faiss"

def _chunk_text(text, max_chars=480, overlap=80):
    """
    Split listing text into overlapping chunks by **whole words** (whitespace-split),
    growing each chunk until adding the next word would exceed ``max_chars``.
    The next chunk starts with a suffix of the previous chunk of up to ``overlap``
    characters (still whole words), so BM25 / embeddings never split a token.

    A single word longer than ``max_chars`` is hard-truncated (rare in remarks).
    """
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    joined = " ".join(words)
    if len(joined) <= max_chars:
        return [joined]

    chunks = []
    start = 0
    n = len(words)
    while start < n:
        w0 = words[start]
        if len(w0) > max_chars:
            chunks.append(w0[:max_chars])
            start += 1
            continue

        char_len = 0
        end = start
        while end < n:
            w = words[end]
            if len(w) > max_chars:
                break
            add = len(w) + (1 if end > start else 0)
            if char_len + add > max_chars and end > start:
                break
            char_len += add
            end += 1

        if end > start:
            chunks.append(" ".join(words[start:end]))
        if end >= n:
            break

        if overlap <= 0:
            start = end
            continue

        ov_len = 0
        new_start = end
        for j in range(end - 1, start - 1, -1):
            piece = len(words[j]) + (1 if ov_len > 0 else 0)
            if ov_len + piece > overlap:
                break
            ov_len += piece
            new_start = j

        # If overlap would reuse the whole chunk (e.g. single-word chunk), advance.
        if new_start > start and new_start < end:
            start = new_start
        else:
            start = end

    return chunks if chunks else [joined[:max_chars]]

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class SemanticSearcher:
    def __init__(
        self,
        listings_path=None,
        *,
        max_chunk_chars=480,
        chunk_overlap=80,
    ):
        self.model = MODEL  # dim 384
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        self.index = None
        path = self._resolve_listings_path(listings_path)
        self.listings = self._load_listings(path)
        self.remarks = self._extract_remarks()
        self._build_chunk_tables()
        self._initialize_index()
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
        return self.listings["remarks"].to_list()

    def _build_chunk_tables(self):
        """One listing -> many chunks; record mapping chunk row -> listing row index."""
        self.chunk_texts = []
        self.chunk_listing_idx = []
        for li, remark in enumerate(self.remarks):
            parts = _chunk_text(
                remark, self.max_chunk_chars, self.chunk_overlap
            )
            for ch in parts:
                self.chunk_texts.append(ch)
                self.chunk_listing_idx.append(li)
        self.chunk_listing_idx = np.asarray(self.chunk_listing_idx, dtype=np.int32)
        self._n_chunks = len(self.chunk_texts)

    def embed_chunks(self, batch_size=64):
        print(f"Encoding {self._n_chunks} chunks from {len(self.remarks)} listings...")

        embeddings = self.model.encode(
            self.chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _initialize_index(self):
        # build and store for the first time, avoid building in future times
        if _INDEX_PATH.exists():
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(str(_INDEX_PATH))
            return
        print("Building FAISS index from scratch...")
        embeddings = self.embed_chunks()
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print("Saving FAISS index...")
        faiss.write_index(self.index, str(_INDEX_PATH))

    def _build_BM_corpus(self):
        corpus = [chunk.split() for chunk in self.chunk_texts]
        return BM25Okapi(corpus)

    def _chunk_search_depth(self, top_k):
        """How many chunk hits to retrieve before aggregating to listings.
           Try to have tok_k number of listings from these chunks
        """
        return min(
            self._n_chunks,
            max(100, top_k * 25),
        )

    def _aggregate_chunks_to_listings(self, chunk_indices, chunk_scores):
        """
        From retrieved chunk hits, take the max score per listing and remember which
        chunk achieved it (among hits only).

        Returns:
            ranked: [(listing_row_idx, best_score), ...] sorted by score desc
            best_chunk_idx: listing_row_idx -> global chunk index into chunk_texts
        """
        best_score = {}
        best_chunk_idx = {}
        for i, cidx in enumerate(chunk_indices):
            if cidx < 0:
                continue
            cidx = int(cidx)
            s = float(chunk_scores[i])
            li = int(self.chunk_listing_idx[cidx])
            if li not in best_score or s > best_score[li]:
                best_score[li] = s
                best_chunk_idx[li] = cidx
        ranked = sorted(best_score.items(), key=lambda x: -x[1])
        return ranked, best_chunk_idx

    def search_emb(self, query, top_k=5, return_chunks=False):
        start = time.time()
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        k = self._chunk_search_depth(top_k)
        scores, indices = self.index.search(query_emb, k)
        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(
            indices[0], scores[0]
        )
        ranked = ranked[:top_k]

        end = time.time()
        latency_ms = (end - start) * 1000

        results = []
        ids = []
        for li, sc in ranked:
            remark = self.remarks[li]
            lid = self.listings.iloc[li]["L_ListingID"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((lid, sc))
        return results, ids, latency_ms

    def search_bm25(self, query, top_k=5, return_chunks=False):
        query_tokens = query.split()
        start = time.time()
        scores = self.bm25.get_scores(query_tokens)
        k = self._chunk_search_depth(top_k)
        top_chunk_idx = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_chunk_idx]
        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(
            top_chunk_idx, top_scores
        )
        ranked = ranked[:top_k]
        end = time.time()
        latency_ms = (end - start) * 1000

        results = []
        ids = []
        for li, sc in ranked:
            remark = self.remarks[li]
            lid = self.listings.iloc[li]["L_ListingID"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((lid, sc))
        return results, ids, latency_ms
