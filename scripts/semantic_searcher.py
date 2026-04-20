from pathlib import Path

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pandas as pd
import time


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LISTINGS_PATH = _PROJECT_ROOT / "data" / "processed" / "all_listings_cleaned.csv"
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
        listings_path=_DEFAULT_LISTINGS_PATH,
        *,
        max_chunk_chars=480,
        chunk_overlap=80,
    ):
        self.model = MODEL  # dim 384
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.listings = self._load_listings(listings_path)
        self.remarks = self._extract_remarks()
        self._build_chunk_tables()
        self._initialize_index()
        self.bm25 = self._build_BM_corpus()

    def _load_listings(self, path):
        df = pd.read_csv(path)
        df = df[["id", "remarks"]]
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

    def _min_max_normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.full_like(x, 0.5)
        return (x - lo) / (hi - lo)

    def _chunk_embedding_scores(self, query_emb, chunk_indices, search_idx, search_scores):
        """IP scores from the FAISS hit list, or reconstruct + dot for other chunks."""
        hit = {}
        for cidx, s in zip(search_idx, search_scores):
            if cidx < 0:
                continue
            hit[int(cidx)] = float(s)
        q = query_emb.reshape(-1).astype(np.float64)
        out = []
        for cidx in chunk_indices:
            ci = int(cidx)
            if ci in hit:
                out.append(hit[ci])
                continue
            vec = np.ascontiguousarray(
                self.index.reconstruct(ci).reshape(-1), dtype=np.float32
            )
            faiss.normalize_L2(vec.reshape(1, -1))
            out.append(float(np.dot(q, vec.astype(np.float64))))
        return np.asarray(out, dtype=np.float64)

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
            lid = self.listings.iloc[li]["id"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((lid, sc))
        return results, ids, latency_ms

    def search_hybrid(self, query, top_k=5, return_chunks=False, alpha=0.5):
        """
        Fuse dense retrieval (embedding / FAISS inner product) with BM25.

        ``alpha`` weights the normalized embedding score; ``1 - alpha`` weights
        normalized BM25. Each score type is min--max normalized over the **union**
        of top-``k`` chunks from both retrievers (``k`` from ``_chunk_search_depth``).

        For ``alpha`` near 0 or 1, delegates to pure BM25 / pure embedding so
        candidate sets match those single-channel searches.
        """
        if alpha >= 1.0 - 1e-9:
            return self.search_emb(query, top_k, return_chunks)
        if alpha <= 1e-9:
            return self.search_bm25(query, top_k, return_chunks)

        start = time.time()
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)

        k = self._chunk_search_depth(top_k)
        emb_scores, emb_idx = self.index.search(query_emb, k)

        query_tokens = query.split()
        bm_full = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float64)
        bm_order = np.argsort(bm_full)[::-1][:k]
        bm_top_idx = bm_order.astype(np.int64)

        cand = np.unique(np.concatenate([emb_idx[0].astype(np.int64), bm_top_idx]))

        emb_raw = self._chunk_embedding_scores(
            query_emb, cand, emb_idx[0], emb_scores[0]
        )
        bm_raw = bm_full[cand]

        emb_n = self._min_max_normalize(emb_raw)
        bm_n = self._min_max_normalize(bm_raw)
        fused = alpha * emb_n + (1.0 - alpha) * bm_n

        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(cand, fused)
        ranked = ranked[:top_k]

        end = time.time()
        latency_ms = (end - start) * 1000

        results = []
        ids = []
        for li, sc in ranked:
            remark = self.remarks[li]
            lid = self.listings.iloc[li]["id"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((lid, sc))
        return results, ids, latency_ms

    def _listing_ids_to_rows(self, listing_ids):
        """Map business listing ``id`` values to dataframe row indices."""
        if isinstance(listing_ids, (int, np.integer)):
            id_set = {int(listing_ids)}
        else:
            id_set = {int(x) for x in listing_ids}
        mask = self.listings["id"].isin(id_set)
        return np.flatnonzero(mask.to_numpy()).astype(np.int32)

    def _chunk_indices_for_listing_rows(self, listing_rows):
        """Chunk indices whose ``chunk_listing_idx`` is in ``listing_rows``."""
        if listing_rows.size == 0:
            return np.array([], dtype=np.int64)
        m = np.isin(self.chunk_listing_idx, listing_rows.astype(np.int32, copy=False))
        return np.flatnonzero(m).astype(np.int64)

    def _subset_chunk_depth(self, top_k, n_chunks_allowed):
        if n_chunks_allowed <= 0:
            return 0
        return min(n_chunks_allowed, max(100, top_k * 25))

    def _faiss_search_among_chunks(self, query_emb, allowed_chunks, k):
        """
        Inner-product search restricted to ``allowed_chunks`` (global chunk ids).
        Brute-forces over the subset via reconstruct + GEMV (no full-index scan).
        """
        allowed_chunks = np.asarray(allowed_chunks, dtype=np.int64).reshape(-1)
        n = int(allowed_chunks.size)
        if n == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.full(0, -1, dtype=np.int64),
            )
        k_eff = min(int(k), n)
        q = np.ascontiguousarray(query_emb.reshape(-1), dtype=np.float32)
        dim = int(q.shape[0])
        vecs = np.empty((n, dim), dtype=np.float32)
        for i, c in enumerate(allowed_chunks):
            vecs[i] = self.index.reconstruct(int(c))
        sims = vecs @ q
        if k_eff >= n:
            order = np.argsort(-sims)
            return sims[order].astype(np.float32), allowed_chunks[order]
        pick = np.argpartition(-sims, k_eff - 1)[:k_eff]
        best = pick[np.argsort(-sims[pick])]
        return sims[best].astype(np.float32), allowed_chunks[best]

    def _search_emb_for_listings(self, query, listing_ids, top_k, return_chunks):
        rows = self._listing_ids_to_rows(listing_ids)
        allowed = self._chunk_indices_for_listing_rows(rows)
        if allowed.size == 0:
            return [], [], 0.0

        start = time.time()
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        depth = self._subset_chunk_depth(top_k, allowed.size)
        scores, indices = self._faiss_search_among_chunks(query_emb, allowed, depth)
        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(indices, scores)
        ranked = ranked[:top_k]
        end = time.time()
        latency_ms = (end - start) * 1000

        results = []
        ids = []
        for li, sc in ranked:
            remark = self.remarks[li]
            lid = self.listings.iloc[li]["id"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((lid, sc))
        return results, ids, latency_ms

    def _search_bm25_for_listings(self, query, listing_ids, top_k, return_chunks):
        rows = self._listing_ids_to_rows(listing_ids)
        allowed = self._chunk_indices_for_listing_rows(rows)
        if allowed.size == 0:
            return [], [], 0.0

        query_tokens = query.split()
        start = time.time()
        bm_full = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float64)
        depth = self._subset_chunk_depth(top_k, allowed.size)
        sub = bm_full[allowed]
        order = np.argsort(sub)[::-1][:depth]
        top_chunk_idx = allowed[order]
        top_scores = bm_full[top_chunk_idx]
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
            lid = self.listings.iloc[li]["id"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((lid, sc))
        return results, ids, latency_ms

    def search_hybrid_for_listings(
        self,
        query,
        listing_ids,
        top_k=5,
        return_chunks=False,
        alpha=0.5,
    ):
        """
        Hybrid search over **only** chunks belonging to the given listing id(s).

        ``listing_ids`` may be a single int or an iterable of ids matching
        ``listings['id']``. Embedding and BM25 candidates are restricted to the
        union of chunks for those listings; fusion matches ``search_hybrid``.
        """
        if alpha >= 1.0 - 1e-9:
            return self._search_emb_for_listings(
                query, listing_ids, top_k, return_chunks
            )
        if alpha <= 1e-9:
            return self._search_bm25_for_listings(
                query, listing_ids, top_k, return_chunks
            )

        rows = self._listing_ids_to_rows(listing_ids)
        allowed = self._chunk_indices_for_listing_rows(rows)
        if allowed.size == 0:
            return [], [], 0.0

        start = time.time()
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)

        depth = self._subset_chunk_depth(top_k, allowed.size)
        emb_scores, emb_idx = self._faiss_search_among_chunks(
            query_emb, allowed, depth
        )

        query_tokens = query.split()
        bm_full = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float64)
        sub_bm = bm_full[allowed]
        bm_order = np.argsort(sub_bm)[::-1][:depth]
        bm_top_idx = allowed[bm_order]

        cand = np.unique(
            np.concatenate([emb_idx.astype(np.int64), bm_top_idx])
        )

        emb_raw = self._chunk_embedding_scores(
            query_emb, cand, emb_idx, emb_scores
        )
        bm_raw = bm_full[cand]

        emb_n = self._min_max_normalize(emb_raw)
        bm_n = self._min_max_normalize(bm_raw)
        fused = alpha * emb_n + (1.0 - alpha) * bm_n

        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(cand, fused)
        ranked = ranked[:top_k]

        end = time.time()
        latency_ms = (end - start) * 1000

        results = []
        ids_out = []
        for li, sc in ranked:
            remark = self.remarks[li]
            lid = self.listings.iloc[li]["id"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids_out.append((lid, sc))
        return results, ids_out, latency_ms

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
            lid = self.listings.iloc[li]["id"]
            if return_chunks:
                cidx = chunk_for_listing[li]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((lid, sc))
        return results, ids, latency_ms
