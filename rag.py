"""
RAG pipeline: Decide + Act phases of OODA.

Given a query, retrieve the most relevant document chunks (Decide),
then generate a grounded answer via LM Studio (Act).
"""

import hashlib
import re
from typing import List, Dict, Optional
import httpx
from diskcache import Cache
from openai import OpenAI, APIConnectionError
from store import VectorStore


_SYSTEM = (
    "You are a military and defense analyst assistant. "
    "Answer using ONLY the numbered context passages provided. "
    "Cite each passage you draw from using its bracketed number, e.g. [1]. "
    "If the context does not contain enough information to answer, say so explicitly "
    "rather than speculating."
)

_HYDE_SYSTEM = (
    "Write a short, plausible passage (2-4 sentences) that could appear in a "
    "military or defense doctrine document and would directly answer the "
    "question. State it as fact, in the style of a doctrine excerpt. Do not "
    "mention that this is hypothetical, add caveats, or ask for clarification "
    "— output only the passage."
)


class RAG:
    def __init__(
        self,
        store: VectorStore,
        lm_studio_url: str,
        model: str,
        top_k: int,
        candidates: int,
        use_reranker: bool,
        mmr_lambda: float = 0.7,
        use_hyde: bool = False,
        hyde_cache_dir: Optional[str] = None,
        generation_cache_dir: Optional[str] = None,
    ) -> None:
        self._store = store
        self._client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")
        self._model = model
        self._top_k = top_k
        self._candidates = candidates
        self._use_reranker = use_reranker
        self._mmr_lambda = mmr_lambda
        self._use_hyde = use_hyde
        self._hyde_cache = Cache(hyde_cache_dir) if hyde_cache_dir else None
        self._generation_cache = Cache(generation_cache_dir) if generation_cache_dir else None
        self._lm_studio_url = lm_studio_url
        self._check_lm_studio()

    def _check_lm_studio(self) -> None:
        try:
            httpx.get(self._lm_studio_url.rstrip("/") + "/models", timeout=3)
        except httpx.ConnectError:
            raise SystemExit(
                f"Cannot reach LM Studio at {self._lm_studio_url}\n"
                "Start LM Studio, load a model, and enable the local server (default port 1234)."
            )

    def query(
        self,
        question: str,
        use_reranker: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
    ) -> str:
        reranker_on = self._use_reranker if use_reranker is None else use_reranker
        hyde_on = self._use_hyde if use_hyde is None else use_hyde
        dense_override = self._hyde_passage(question) if hyde_on else None
        chunks = self._store.query(
            question, self._top_k, self._candidates, reranker_on, self._mmr_lambda, dense_override
        )
        context = self._build_context(chunks)

        cache_key = self._generation_cache_key(question, context)
        if self._generation_cache is not None:
            cached = self._generation_cache.get(cache_key)
            if cached is not None:
                return cached

        answer = self._generate(question, context)
        result = self._flag_unsupported_citations(answer, len(chunks))
        if self._generation_cache is not None:
            self._generation_cache.set(cache_key, result)
        return result

    # ---------------------------------------------------------------- helpers

    def _generation_cache_key(self, question: str, context: str) -> str:
        """Cache-augmented generation: the final answer is keyed on the
        literal (model, retrieved context, question) triple, so it never
        needs manual invalidation — if retrieval returns different context
        (new documents, different settings), the key changes and the cache
        simply misses."""
        raw = f"{self._model}:{context}:{question}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _hyde_passage(self, question: str) -> str:
        """HyDE: generate a hypothetical doctrine-style passage that would
        answer the question, and use it (instead of the raw question) to
        steer dense retrieval — often closer in style to indexed passages
        than a short question is. Cached, since the query cache in `store.py`
        only helps once this LLM call has already happened."""
        key = hashlib.sha256(f"{self._model}:{question}".encode()).hexdigest()
        if self._hyde_cache is not None:
            cached = self._hyde_cache.get(key)
            if cached is not None:
                return cached
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        passage = response.choices[0].message.content
        if self._hyde_cache is not None:
            self._hyde_cache.set(key, passage)
        return passage

    def _build_context(self, chunks: List[Dict]) -> str:
        parts = [
            f"[{i}] Source: {c['source']}\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        ]
        return "\n\n---\n\n".join(parts)

    def _flag_unsupported_citations(self, answer: str, n_chunks: int) -> str:
        """Grounding check: flag citation numbers the model invented that
        fall outside the actual [1..n_chunks] context range it was given."""
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        out_of_range = sorted(n for n in cited if n < 1 or n > n_chunks)
        if out_of_range:
            answer += (
                f"\n\n[warning: cites passage(s) {out_of_range} which are outside "
                f"the retrieved context range 1-{n_chunks} — possible hallucination]"
            )
        return answer

    def _generate(self, question: str, context: str) -> str:
        user_msg = f"Context passages:\n\n{context}\n\nQuestion: {question}"
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
