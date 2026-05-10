from pathlib import Path

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pandas as pd
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LISTINGS_PATH = _PROJECT_ROOT / "data" / "processed" / "all_listings_cleaned.csv"
_INDEX_PATH = _PROJECT_ROOT / "data" / "processed" / "index.faiss"

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
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size =  max_chunk_chars,
            chunk_overlap = chunk_overlap,
            separators = ["\n\n", "\n", " ", ""]
        )
        self.index = None
        self.listings = self._load_listings(listings_path)
        self._row_by_listing_id = {
            int(lid): i for i, lid in enumerate(self.listings["id"].tolist())
        } # easier to locate the row with listing_id to get the remark
        self.remarks = self._extract_remarks()
        self._build_chunk_tables_with_langchain()
        self._initialize_index()
        self.bm25 = self._build_BM_corpus()

    def _load_listings(self, path):
        df = pd.read_csv(path)
        df = df[["id", "remarks"]]
        df = df[df["remarks"].notna()]
        return df

    def _extract_remarks(self):
        return self.listings["remarks"].to_list()

    def _build_chunk_tables_with_langchain(self):
        base_docs = []
        for _, row in self.listings.iterrows():
            base_docs.append(
                Document(
                    page_content = str(row["remarks"]),
                    metadata = {"listing_id": int(row["id"])}
                )
            )
        split_docs = self.splitter.split_documents(base_docs)
        self.chunk_docs = split_docs
        self.chunk_texts = [d.page_content for d in split_docs]
        self.chunk_listing_ids = np.asarray(
            [int(d.metadata["listing_id"]) for d in split_docs], dtype=np.int64
        )
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
            if int(self.index.ntotal) == int(self._n_chunks):
                return
            print("FAISS size mismatch with current chunks. Rebuilding index...")
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
        """
        calculate the embedding approach score for result from both embedding approach and 
        bm25 approach
        chunk_indices: the index for most relavant text chunks, from bm25 and embedding search
                       (tok_k from bm25 + top_k from embedding)
        search_idx, search_score: result from embedding approach, we don't need to calculate their 
                                  score again
        return cos_sim in the order of chunk_indices
        """
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
            ranked: [(listing_id, best_score), ...] sorted by score desc
            best_chunk_idx: listing_id -> global chunk index into chunk_texts
        """
        best_score = {}
        best_chunk_idx = {}
        for i, cidx in enumerate(chunk_indices):
            if cidx < 0:
                continue
            cidx = int(cidx)
            s = float(chunk_scores[i])
            li = int(self.chunk_docs[cidx].metadata["listing_id"])
            if li not in best_score or s > best_score[li]:
                best_score[li] = s # listing id to best score
                best_chunk_idx[li] = cidx # listing id to best chunk idx
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
        for lid, sc in ranked:
            row_idx = self._row_by_listing_id[int(lid)]
            remark = self.remarks[row_idx]
            if return_chunks:
                cidx = chunk_for_listing[int(lid)]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((int(lid), sc))
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
        # bm score for each chunk
        bm_full = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float64)
        # top_k score index
        bm_order = np.argsort(bm_full)[::-1][:k]
        bm_top_idx = bm_order.astype(np.int64)

        # [top_k embd index| top_k bm25 index]
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
        for lid, sc in ranked:
            row_idx = self._row_by_listing_id[int(lid)]
            remark = self.remarks[row_idx]
            if return_chunks:
                cidx = chunk_for_listing[int(lid)]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((int(lid), sc))
        return results, ids, latency_ms

    def _chunk_indices_for_listing_id(self, listing_ids):
        """Chunk indices belonging to the given listing id"""

        if isinstance(listing_ids, (int, np.integer)):
            id_set = {int(listing_ids)}
        else:
            id_set = {int(x) for x in listing_ids}
        if not id_set:
            return np.array([], dtype=np.int64)
        ids = np.array(sorted(id_set), dtype=np.int64) 
        m = np.isin(self.chunk_listing_ids, ids)
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
        allowed = self._chunk_indices_for_listing_id(listing_ids)
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
        for lid, sc in ranked:
            row_idx = self._row_by_listing_id[int(lid)]
            remark = self.remarks[row_idx]
            if return_chunks:
                cidx = chunk_for_listing[int(lid)]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((int(lid), sc))
        return results, ids, latency_ms

    def _search_bm25_for_listings(self, query, listing_ids, top_k, return_chunks):
        allowed = self._chunk_indices_for_listing_id(listing_ids)
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
        for lid, sc in ranked:
            row_idx = self._row_by_listing_id[int(lid)]
            remark = self.remarks[row_idx]
            if return_chunks:
                cidx = chunk_for_listing[int(lid)]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((int(lid), sc))
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

        allowed = self._chunk_indices_for_listing_id(listing_ids)
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
        for lid, sc in ranked:
            row_idx = self._row_by_listing_id[int(lid)]
            remark = self.remarks[row_idx]
            if return_chunks:
                cidx = chunk_for_listing[int(lid)]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids_out.append((int(lid), sc))
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
        for lid, sc in ranked:
            row_idx = self._row_by_listing_id[int(lid)]
            remark = self.remarks[row_idx]
            if return_chunks:
                cidx = chunk_for_listing[int(lid)]
                chunk_text = self.chunk_texts[cidx]
                results.append((remark, sc, chunk_text))
            else:
                results.append((remark, sc))
            ids.append((int(lid), sc))
        return results, ids, latency_ms
