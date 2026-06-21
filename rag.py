"""
RAG pipeline: Decide + Act phases of OODA.

Given a query, retrieve the most relevant document chunks (Decide),
then generate a grounded answer via LM Studio (Act).
"""

from typing import List, Dict
import httpx
from openai import OpenAI, APIConnectionError
from store import VectorStore


_SYSTEM = (
    "You are a military and defense analyst assistant. "
    "Answer using ONLY the numbered context passages provided. "
    "Cite each passage you draw from using its bracketed number, e.g. [1]. "
    "If the context does not contain enough information to answer, say so explicitly "
    "rather than speculating."
)


class RAG:
    def __init__(
        self,
        store: VectorStore,
        lm_studio_url: str,
        model: str,
        top_k: int,
    ) -> None:
        self._store = store
        self._client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")
        self._model = model
        self._top_k = top_k
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

    def query(self, question: str) -> str:
        chunks = self._store.query(question, self._top_k)
        context = self._build_context(chunks)
        return self._generate(question, context)

    # ---------------------------------------------------------------- helpers

    def _build_context(self, chunks: List[Dict]) -> str:
        parts = [
            f"[{i}] Source: {c['source']}\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        ]
        return "\n\n---\n\n".join(parts)

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
