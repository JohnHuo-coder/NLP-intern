from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class SemanticSearcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.listings = None

    def build_index(self, remarks_list):
        print(f"Encoding {len(remarks_list)} listings...")
        embeddings = self.model.encode(remarks_list)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) # Inner product for cosine sim
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.listings = remarks_list

    def search(self, query, top_k=10):
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)
        results = [(self.listings[i], scores[0][j]) for j, i in
        enumerate(indices[0])]
        return results