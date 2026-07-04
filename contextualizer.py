"""
Optional per-chunk contextual summaries via the local LLM (adapted from
Anthropic's "Contextual Retrieval"), used to situate a chunk within its
surrounding section before embedding/indexing.

Uses the parent block as context rather than the whole source document —
most locally-run models have far smaller context windows than what the
original technique assumes, and a 100+ page field manual won't fit regardless
of GPU. This costs one local LLM call per chunk at ingest time: negligible on
a dedicated GPU (roughly hours for this corpus), impractical CPU-only (days).
Opt-in only (`ingest --contextualize`); results are disk-cached so an
interrupted ingest never redoes already-summarized chunks.
"""

import hashlib

import httpx
from diskcache import Cache
from openai import OpenAI

_SYSTEM = (
    "You produce a single short sentence (max 30 words) situating a passage "
    "within its surrounding section, for use as a search index annotation. "
    "State only what the passage is about and how it relates to the section "
    "context. Do not answer questions, summarize the whole section, or add "
    "commentary — output only the situating sentence, nothing else."
)


class Contextualizer:
    def __init__(self, lm_studio_url: str, model: str, cache_dir: str) -> None:
        self._client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")
        self._model = model
        self._cache = Cache(cache_dir)
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

    def contextualize(self, section_text: str, chunk_text: str) -> str:
        key = self._key(section_text, chunk_text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        summary = self._generate(section_text, chunk_text)
        self._cache.set(key, summary)
        return summary

    def _generate(self, section_text: str, chunk_text: str) -> str:
        user_msg = f"Section:\n{section_text}\n\nPassage to situate:\n{chunk_text}"
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()

    def _key(self, section_text: str, chunk_text: str) -> str:
        raw = f"{self._model}:{section_text}:{chunk_text}"
        return hashlib.sha256(raw.encode()).hexdigest()
