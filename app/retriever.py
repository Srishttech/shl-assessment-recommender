import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import CACHE_DIR, EMBEDDING_MODEL_NAME


class Retriever:
    """
    Wraps the SentenceTransformer + FAISS pipeline from the notebook.
    Same embedding model, same IndexFlatIP + normalize_L2 approach.

    Change from notebook: embeddings + index are cached to disk (CACHE_DIR)
    so that a server restart does not re-embed all 377 catalog items every
    time (avoids slow cold starts + repeated unauthenticated HF Hub calls).
    """

    def __init__(self, catalog, texts, cache_dir: str = CACHE_DIR):
        self.catalog = catalog
        self.texts = texts
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = self._build_or_load_index()

    def _build_or_load_index(self):
        emb_path = os.path.join(self.cache_dir, "embeddings.npy")
        index_path = os.path.join(self.cache_dir, "faiss.index")

        if os.path.exists(emb_path) and os.path.exists(index_path):
            return faiss.read_index(index_path)

        embeddings = self.embed_model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        np.save(emb_path, embeddings)
        faiss.write_index(index, index_path)
        return index

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.catalog[idx]
            results.append(
                {
                    "name": item.get("name", ""),
                    "url": item.get("link", ""),
                    "score": float(score),
                    "description": item.get("description", ""),
                    "job_levels": item.get("job_levels", []),
                    "keys": item.get("keys", []),
                    # BUGFIX: these four were missing from the notebook's
                    # retrieve() output, even though the comparison prompt
                    # needs duration + remote to fill its table.
                    "duration": item.get("duration", ""),
                    "remote": item.get("remote", ""),
                    "adaptive": item.get("adaptive", ""),
                    "languages": item.get("languages", []),
                }
            )
        return results
