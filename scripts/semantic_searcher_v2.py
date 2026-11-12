from pathlib import Path

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pandas as pd
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import Any, List, Dict, Tuple, Protocol


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LISTINGS_PATH = _PROJECT_ROOT / "data" / "processed" / "all_listings_cleaned.csv"
_INDEX_PATH = _PROJECT_ROOT / "data" / "processed" / "index.faiss"

MODEL = SentenceTransformer("all-MiniLM-L6-v2")



class ListingChunker:
    def __init__(
        self,
        listings_path=_DEFAULT_LISTINGS_PATH,
        *,
        max_chunk_chars=480,
        chunk_overlap=80,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size =  max_chunk_chars,
            chunk_overlap = chunk_overlap,
            separators = ["\n\n", "\n", " ", ""]
        )
        self.listings = self._load_listings(listings_path)

    def _load_listings(self, path):
        df = pd.read_csv(path)
        df = df[["id", "remarks"]]
        df = df[df["remarks"].notna()]
        return df

    def build_chunk_tables(self):
        base_docs = []
        for _, row in self.listings.iterrows():
            base_docs.append(
                Document(
                    page_content = str(row["remarks"]),
                    metadata = {"listing_id": int(row["id"])}
                )
            )
        split_docs = self.splitter.split_documents(base_docs)
        chunk_texts = [d.page_content for d in split_docs]
        chunk_listing_ids = np.asarray(
            [int(d.metadata["listing_id"]) for d in split_docs], dtype=np.int64
        )
        return chunk_texts, chunk_listing_ids


class VectorIndex:
    def __init__(
        self,
        listings,
        chunk_texts, 
        chunk_listing_ids
    ):
        self.model = MODEL  # dim 384
        self.index = None
        self.listings = listings
        self._row_by_listing_id = {
            int(lid): i for i, lid in enumerate(self.listings["id"].tolist())
        } # easier to locate the row with listing_id to get the remark
        self.chunk_texts = chunk_texts
        self.chunk_listing_ids = chunk_listing_ids
        self._n_chunks = len(chunk_texts)
        self._initialize_index()

    def _embed_chunks(self, batch_size=64):
        print(f"Encoding {self._n_chunks} chunks from {len(self.listings)} listings...")

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
        embeddings = self._embed_chunks()
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print("Saving FAISS index...")
        faiss.write_index(self.index, str(_INDEX_PATH))

    def _chunk_search_depth(self, top_k):
        """How many chunk hits to retrieve before aggregating to listings.
           Try to have top_k number of listings from these chunks
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
            li = int(self.chunk_listing_ids[cidx])
            if li not in best_score or s > best_score[li]:
                best_score[li] = s # listing id to best score
                best_chunk_idx[li] = cidx # listing id to best chunk idx
        ranked = sorted(
            best_score.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked, best_chunk_idx

    def search(self, query, top_k=5, return_chunks=False):
        # start = time.time()
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        k = self._chunk_search_depth(top_k)
        scores, indices = self.index.search(query_emb, k)
        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(
            indices[0], scores[0]
        )
        ranked = ranked[:top_k]

        # end = time.time()
        # latency_ms = (end - start) * 1000

        results = []
        for lid, sc in ranked:
            lid = int(lid)
            row_idx = self._row_by_listing_id[lid]
            remark = self.listings.iloc[row_idx]["remarks"]
            
            cidx = chunk_for_listing[lid]
            chunk_text = self.chunk_texts[cidx] if return_chunks else None
            results.append((lid, remark, sc, chunk_text))
            
        return results

    def _chunk_indices_for_listing_id(self, listing_ids):
        """Chunk indices belonging to the given listing id"""

        if isinstance(listing_ids, (int, np.integer)):
            id_set = {int(listing_ids)}
        else:
            id_set = {int(x) for x in listing_ids}
        if not id_set:
            return np.array([], dtype=np.int64)
        allowed_listing_ids = np.array(sorted(id_set), dtype=np.int64)
        chunk_belongs_to_allowed = np.isin(self.chunk_listing_ids, allowed_listing_ids)
        return np.flatnonzero(chunk_belongs_to_allowed).astype(np.int64)

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
        q = np.ascontiguousarray(query_emb.reshape(-1), dtype=np.float32)
        dim = int(q.shape[0])
        vecs = np.empty((n, dim), dtype=np.float32)
        for i, c in enumerate(allowed_chunks):
            vecs[i] = self.index.reconstruct(int(c))
        sims = vecs @ q
        # k is already min(n, max(100, top_k*25)); k == n is fine (full set, then sort)
        pick = np.argpartition(-sims, k - 1)[:k]
        best = pick[np.argsort(-sims[pick])]
        return sims[best].astype(np.float32), allowed_chunks[best]

    def search_subset(self, query, listing_ids, top_k, return_chunks):
        allowed = self._chunk_indices_for_listing_id(listing_ids)
        if allowed.size == 0:
            return []

        # start = time.time()
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        depth = self._subset_chunk_depth(top_k, allowed.size)
        scores, indices = self._faiss_search_among_chunks(query_emb, allowed, depth)
        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(indices, scores)
        ranked = ranked[:top_k]
        # end = time.time()
        # latency_ms = (end - start) * 1000

        results = []
        for lid, sc in ranked:
            lid = int(lid)
            row_idx = self._row_by_listing_id[lid]
            remark = self.listings.iloc[row_idx]["remarks"]
            
            cidx = chunk_for_listing[lid]
            chunk_text = self.chunk_texts[cidx] if return_chunks else None
            results.append((lid, remark, sc, chunk_text))
        return results

    

class BM25Index:
    def __init__(
        self,
        listings,
        chunk_texts, 
        chunk_listing_ids
    ):
        self.listings = listings
        self._row_by_listing_id = {
            int(lid): i for i, lid in enumerate(self.listings["id"].tolist())
        } # easier to locate the row with listing_id to get the remark
        self.chunk_texts = chunk_texts
        self.chunk_listing_ids = chunk_listing_ids
        self._n_chunks = len(chunk_texts)

        self.bm25 = self._build_BM_corpus()

    def _build_BM_corpus(self):
        corpus = [chunk.split() for chunk in self.chunk_texts]
        return BM25Okapi(corpus)
    
    def _chunk_search_depth(self, top_k):
        """How many chunk hits to retrieve before aggregating to listings.
           Try to have top_k number of listings from these chunks
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
            li = int(self.chunk_listing_ids[cidx])
            if li not in best_score or s > best_score[li]:
                best_score[li] = s # listing id to best score
                best_chunk_idx[li] = cidx # listing id to best chunk idx
        ranked = sorted(
            best_score.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked, best_chunk_idx

    def search(self, query, top_k=5, return_chunks=False):
        query_tokens = query.split()
        # start = time.time()
        scores = self.bm25.get_scores(query_tokens)
        k = self._chunk_search_depth(top_k)
        top_chunk_idx = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_chunk_idx]
        ranked, chunk_for_listing = self._aggregate_chunks_to_listings(
            top_chunk_idx, top_scores
        )
        ranked = ranked[:top_k]
        # end = time.time()
        # latency_ms = (end - start) * 1000

        results = []
        for lid, sc in ranked:
            lid = int(lid)
            row_idx = self._row_by_listing_id[lid]
            remark = self.listings.iloc[row_idx]["remarks"]

            cidx = chunk_for_listing[lid]
            chunk_text = self.chunk_texts[cidx] if return_chunks else None
            results.append((lid, remark, sc, chunk_text))
        return results
    
    def _subset_chunk_depth(self, top_k, n_chunks_allowed):
        if n_chunks_allowed <= 0:
            return 0
        return min(n_chunks_allowed, max(100, top_k * 25))
    
    def _chunk_indices_for_listing_id(self, listing_ids):
        """Chunk indices belonging to the given listing id"""

        if isinstance(listing_ids, (int, np.integer)):
            id_set = {int(listing_ids)}
        else:
            id_set = {int(x) for x in listing_ids}
        if not id_set:
            return np.array([], dtype=np.int64)
        allowed_listing_ids = np.array(sorted(id_set), dtype=np.int64)
        chunk_belongs_to_allowed = np.isin(self.chunk_listing_ids, allowed_listing_ids)
        return np.flatnonzero(chunk_belongs_to_allowed).astype(np.int64)
    
    def search_subset(self, query, listing_ids, top_k, return_chunks):
        allowed = self._chunk_indices_for_listing_id(listing_ids)
        if allowed.size == 0:
            return []

        query_tokens = query.split()
        # start = time.time()
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
        # end = time.time()
        # latency_ms = (end - start) * 1000

        results = []
        for lid, sc in ranked:
            lid = int(lid)
            row_idx = self._row_by_listing_id[lid]
            remark = self.listings.iloc[row_idx]["remarks"]
            
            cidx = chunk_for_listing[lid]
            chunk_text = self.chunk_texts[cidx] if return_chunks else None
            results.append((lid, remark, sc, chunk_text))
        return results


class SearchIndex(Protocol):

    def search(
        self, query: Any, top_k: int, return_chunks: bool
    ) -> List[Tuple]: ...

    def search_subset(
        self, query: Any, listing_ids: List, top_k: int, return_chunks: bool
    ) -> List[Tuple]: ...


class Retriever:
    def __init__(self, *indexes: SearchIndex):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    def search(
        self, 
        query_text: str, 
        k: int = 5,
        k_rrf: int = 60, 
        return_chunks = False, 
        listing_ids = None
    ) -> List[Tuple]:
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")
        
        pool_k = max(k * 25, 50)

        if listing_ids is not None:
            all_results = [
                index.search_subset(query_text, listing_ids, pool_k, return_chunks) for index in self._indexes
            ]
        else:
            all_results = [
                index.search(query_text, pool_k, return_chunks) for index in self._indexes
            ]

        def calc_rrf_score(ranks: List[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))

        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (lid, remark, score, chunk_text) in enumerate(results):
                if lid not in doc_ranks:
                    doc_ranks[lid] = {
                        "remark": remark,
                        "ranks": [float("inf")] * len(self._indexes),
                        "chunks": [None] * len(self._indexes)
                    }
                doc_ranks[lid]["ranks"][idx] = rank + 1
                if chunk_text is not None:
                    doc_ranks[lid]["chunks"][idx] = chunk_text

        scored_docs: List[Tuple[Dict[str, Any], float]] = [
            (lid, ranks["remark"], calc_rrf_score(ranks["ranks"]), ranks["chunks"])
            for lid, ranks in doc_ranks.items()
        ]

        filtered_docs = [
            (lid, remark, score, chunk_texts) for lid, remark, score, chunk_texts in scored_docs if score > 0
        ]

        filtered_docs.sort(key=lambda x: x[2], reverse=True)
        out = filtered_docs[:k]
        if not return_chunks:
            out = [(lid, remark, s) for lid, remark, s, _ in out]
        return out


def build_retriever(listings_path=_DEFAULT_LISTINGS_PATH) -> Retriever:
    """Dense + BM25 indexes over the same listing chunks, fused by RRF."""
    chunker = ListingChunker(listings_path)
    chunk_texts, chunk_listing_ids = chunker.build_chunk_tables()
    vector_index = VectorIndex(chunker.listings, chunk_texts, chunk_listing_ids)
    bm25_index = BM25Index(chunker.listings, chunk_texts, chunk_listing_ids)
    return Retriever(vector_index, bm25_index)


