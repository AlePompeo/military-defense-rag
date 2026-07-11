"""
ChromaDB vector store with hybrid (dense + BM25) retrieval, optional
cross-encoder reranking, and a disk-backed query result cache.

Orient phase of OODA: index ingested documents so the system can
rapidly locate the most relevant fragments for any query.
"""

import hashlib
import math
import re
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.api.collection_configuration import (
    CreateCollectionConfiguration,
    CreateHNSWConfiguration,
)
from diskcache import Cache
from rank_bm25 import BM25Okapi

from embedder import Embedder, Reranker

_TOKEN_RE = re.compile(r"\w+")
_RRF_K = 60  # standard Reciprocal Rank Fusion constant
_MIN_RERANK_PROB = 0.05  # drop candidates the cross-encoder is confident are irrelevant

# Domain acronyms expanded into the query text before search — helps both BM25
# keyword overlap and dense-embedding disambiguation on doctrine-heavy jargon.
_ACRONYMS = {
    "ooda": "Observe Orient Decide Act",
    "ipb": "Intelligence Preparation of the Battlefield",
    "fm": "Field Manual",
    "atp": "Army Techniques Publication",
    "adp": "Army Doctrine Publication",
    "mcdp": "Marine Corps Doctrine Publication",
    "ajp": "Allied Joint Publication",
    "c2": "Command and Control",
    "isr": "Intelligence Surveillance and Reconnaissance",
    "ew": "Electronic Warfare",
    "roe": "Rules of Engagement",
}


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _expand_acronyms(text: str) -> str:
    found = {w for w in _tokenize(text) if w in _ACRONYMS}
    if not found:
        return text
    expansions = "; ".join(f"{w.upper()} ({_ACRONYMS[w]})" for w in sorted(found))
    return f"{text} [{expansions}]"


def _rrf_fuse(rankings: List[List[str]]) -> Dict[str, float]:
    """Combine multiple ranked id lists via Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return scores


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize to [0, 1] so relevance is on the same scale as
    cosine similarity for MMR's weighted trade-off."""
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _mmr_reorder(
    ids: List[str],
    relevance: Dict[str, float],
    embeddings: Dict[str, np.ndarray],
    lambda_: float,
) -> List[str]:
    """Maximal Marginal Relevance: greedily reorder candidates to balance
    relevance against redundancy with already-selected results, so near-
    duplicate passages (common across overlapping doctrine texts) don't
    crowd out other relevant context."""
    if len(ids) <= 1:
        return ids
    remaining = list(ids)
    selected: List[str] = []
    while remaining:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(
                remaining,
                key=lambda i: lambda_ * relevance[i]
                - (1 - lambda_) * max(_cosine(embeddings[i], embeddings[s]) for s in selected),
            )
        selected.append(best)
        remaining.remove(best)
    return selected


class VectorStore:
    def __init__(
        self,
        db_path: str,
        embedder: Embedder,
        reranker: Reranker,
        query_cache_dir: Optional[str] = None,
        query_cache_ttl: int = 3600,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 100,
        hnsw_m: int = 16,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._client = chromadb.PersistentClient(path=db_path)
        # New collection name: embeddings are 768-dim (bge-base) vs. the old
        # 384-dim MiniLM index, which is a different vector space and can't
        # be reused in place. Re-run `ingest` to populate this collection.
        self._col = self._client.get_or_create_collection(
            name="military_docs_v2",
            configuration=CreateCollectionConfiguration(
                embedding_function=None,
                hnsw=CreateHNSWConfiguration(
                    space="cosine",
                    ef_construction=hnsw_ef_construction,
                    ef_search=hnsw_ef_search,
                    max_neighbors=hnsw_m,
                ),
            ),
        )
        self._query_cache = Cache(query_cache_dir) if query_cache_dir else None
        self._query_cache_ttl = query_cache_ttl
        self._corpus_version = self._query_cache.get("__corpus_version__", 0) if self._query_cache else 0
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []

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

        texts = [d["text"] for d in new]
        embeddings = self._embedder.embed_documents(texts)
        self._col.add(
            ids=[f"{d['source']}::{d['chunk_id']}" for d in new],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": d["source"],
                    "url": d["url"],
                    "parent_id": d["parent_id"],
                    "parent_text": d["parent_text"],
                    **({"section": d["section"]} if d.get("section") else {}),
                    **({"table_json": d["table_json"]} if d.get("table_json") else {}),
                }
                for d in new
            ],
        )
        self._bm25 = None  # in-memory BM25 index is stale, rebuilt lazily
        self._bump_corpus_version()

    # ------------------------------------------------------------------ read

    def query(
        self,
        text: str,
        top_k: int,
        candidates: int = 20,
        use_reranker: bool = True,
        mmr_lambda: float = 0.7,
        dense_query_override: Optional[str] = None,
    ) -> List[Dict]:
        """Hybrid retrieval: dense (bge) + sparse (BM25) fused via RRF,
        optionally reranked by a cross-encoder (with a confidence-based
        cutoff dropping candidates it scores as clearly irrelevant), then
        diversified via MMR and expanded to parent context (small-to-big).
        Results are cached until the next `add()`.

        `dense_query_override`, if given (e.g. a HyDE-generated hypothetical
        passage), steers only the dense/embedding search — BM25, reranking,
        and citations still use the real user query `text`.
        """
        expanded = _expand_acronyms(text)
        dense_query = _expand_acronyms(dense_query_override) if dense_query_override else expanded
        cache_key = self._cache_key(expanded, dense_query, top_k, candidates, use_reranker, mmr_lambda)
        if self._query_cache is not None:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                return cached

        dense_ids = self._dense_search(dense_query, candidates)
        bm25_ids = self._bm25_search(expanded, candidates)
        fused = _rrf_fuse([dense_ids, bm25_ids])
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:candidates]
        if not ranked_ids:
            return []

        rows = self._col.get(ids=ranked_ids, include=["documents", "metadatas", "embeddings"])
        by_id = {
            i: (doc, meta, emb)
            for i, doc, meta, emb in zip(rows["ids"], rows["documents"], rows["metadatas"], rows["embeddings"])
        }
        ranked_ids = [i for i in ranked_ids if i in by_id]

        if use_reranker and ranked_ids:
            candidate_texts = [by_id[i][0] for i in ranked_ids]
            raw_scores = self._reranker.rerank(expanded, candidate_texts)
            relevance = {i: 1.0 / (1.0 + math.exp(-s)) for i, s in zip(ranked_ids, raw_scores)}
            ranked_ids = sorted(ranked_ids, key=lambda i: relevance[i], reverse=True)
            confident = [i for i in ranked_ids if relevance[i] >= _MIN_RERANK_PROB]
            ranked_ids = confident or ranked_ids[:1]  # always keep the best match, even if weak
        else:
            relevance = _normalize({i: fused[i] for i in ranked_ids})

        embeddings = {i: by_id[i][2] for i in ranked_ids}
        ranked_ids = _mmr_reorder(ranked_ids, relevance, embeddings, mmr_lambda)

        results: List[Dict] = []
        seen_parents = set()
        for doc_id in ranked_ids:
            doc, meta, _ = by_id[doc_id]
            parent_id = meta["parent_id"]
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            results.append({
                "text": meta["parent_text"] or doc,
                "source": meta["source"],
                "url": meta["url"],
                "section": meta.get("section"),
            })
            if len(results) >= top_k:
                break

        if self._query_cache is not None:
            self._query_cache.set(cache_key, results, expire=self._query_cache_ttl)
        return results

    def indexed_sources(self) -> List[str]:
        """Return distinct source names already in the store."""
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return list({m["source"] for m in all_meta})

    def count(self) -> int:
        return self._col.count()

    # ------------------------------------------------------------------ internals

    def _dense_search(self, text: str, n: int) -> List[str]:
        q_emb = self._embedder.embed_query(text)
        res = self._col.query(query_embeddings=[q_emb], n_results=n)
        return res["ids"][0]

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        rows = self._col.get(include=["documents"])
        self._bm25_ids = rows["ids"]
        tokenized = [_tokenize(doc) for doc in rows["documents"]]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _bm25_search(self, text: str, n: int) -> List[str]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self._bm25_ids[i] for i in ranked]

    def _cache_key(
        self,
        text: str,
        dense_query: str,
        top_k: int,
        candidates: int,
        use_reranker: bool,
        mmr_lambda: float,
    ) -> str:
        raw = f"v{self._corpus_version}:{top_k}:{candidates}:{use_reranker}:{mmr_lambda}:{text}:{dense_query}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _bump_corpus_version(self) -> None:
        self._corpus_version += 1
        if self._query_cache is not None:
            self._query_cache.set("__corpus_version__", self._corpus_version)
