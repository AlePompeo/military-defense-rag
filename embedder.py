"""
Embedding + reranking via fastembed (ONNX-only — no torch/GPU required).

BGE models are asymmetric: documents and queries need different instruction
prefixes, so `embed_documents` and `embed_query` map to fastembed's separate
`embed` / `query_embed` calls. Both are backed by a persistent disk cache
keyed on (model name, text) so identical chunks/queries are never re-embedded.
"""

import hashlib
from typing import Callable, Iterable, List

from diskcache import Cache
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


def _key(model_name: str, text: str) -> str:
    return f"{model_name}:{hashlib.sha256(text.encode()).hexdigest()}"


class Embedder:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name, lazy_load=True)
        self._cache = Cache(cache_dir)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, self._model.embed)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], self._model.query_embed)[0]

    def _embed(self, texts: List[str], fn: Callable[[Iterable[str]], Iterable]) -> List[List[float]]:
        keys = [_key(self._model_name, t) for t in texts]
        result: List[List[float] | None] = [None] * len(texts)
        to_compute: dict[str, str] = {}
        for i, k in enumerate(keys):
            cached = self._cache.get(k)
            if cached is not None:
                result[i] = cached
            else:
                to_compute.setdefault(k, texts[i])

        if to_compute:
            comp_keys = list(to_compute.keys())
            fresh = list(fn(list(to_compute.values())))
            fresh_map = {k: vec.tolist() for k, vec in zip(comp_keys, fresh)}
            for k, vec in fresh_map.items():
                self._cache.set(k, vec)
            for i, k in enumerate(keys):
                if result[i] is None:
                    result[i] = fresh_map[k]

        return result  # type: ignore[return-value]


class Reranker:
    def __init__(self, model_name: str) -> None:
        self._model = TextCrossEncoder(model_name=model_name, lazy_load=True)

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        return list(self._model.rerank(query, documents))
