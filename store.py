"""
ChromaDB vector store with local sentence-transformers embeddings.

Orient phase of OODA: index ingested documents so the system can
rapidly locate the most relevant fragments for any query.
"""

from typing import List, Dict
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class VectorStore:
    def __init__(self, db_path: str) -> None:
        self._client = chromadb.PersistentClient(path=db_path)
        # DefaultEmbeddingFunction runs all-MiniLM-L6-v2 via ONNX — no PyTorch needed
        self._col = self._client.get_or_create_collection(
            name="military_docs",
            embedding_function=DefaultEmbeddingFunction(),
        )

    # ------------------------------------------------------------------ write

    def add(self, documents: List[Dict]) -> None:
        """Add a batch of chunk dicts; silently skips already-present IDs."""
        if not documents:
            return
        ids = [f"{d['source']}::{d['chunk_id']}" for d in documents]
        existing = set(self._col.get(ids=ids)["ids"])
        new = [d for d, i in zip(documents, ids) if i not in existing]
        if not new:
            return
        self._col.add(
            ids=[f"{d['source']}::{d['chunk_id']}" for d in new],
            documents=[d["text"] for d in new],
            metadatas=[{"source": d["source"], "url": d["url"]} for d in new],
        )

    # ------------------------------------------------------------------ read

    def query(self, text: str, top_k: int) -> List[Dict]:
        """Return top-k most relevant chunks for the query string."""
        results = self._col.query(query_texts=[text], n_results=top_k)
        return [
            {"text": doc, "source": meta["source"], "url": meta["url"]}
            for doc, meta in zip(
                results["documents"][0], results["metadatas"][0]
            )
        ]

    def indexed_sources(self) -> List[str]:
        """Return distinct source names already in the store."""
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return list({m["source"] for m in all_meta})

    def count(self) -> int:
        return self._col.count()
