"""
Optional per-image LLM captioning (`ingest --describe-images`) via LM
Studio's vision-capable chat completions endpoint (OpenAI-style
`image_url` content parts).

Requires a vision-capable model loaded in LM Studio (e.g. LLaVA, Qwen2-VL,
MiniCPM-V) — typically 3B+ parameters, heavier than the Phi-3 Mini text
model this project recommends for 3.8GB-RAM hardware (see README.md). One
LLM call per extracted image at ingest time; results are disk-cached so an
interrupted ingest never re-captions an already-processed image.
"""

import base64
import hashlib

import httpx
from diskcache import Cache
from openai import OpenAI

_SYSTEM = (
    "Describe this image in 2-4 sentences for a document search index. "
    "If it is a chart, graph, map, or diagram, state its type and describe "
    "the axis labels, legend entries, and the key trend or data points "
    "shown. Be factual and concise; do not speculate beyond what is "
    "visibly labeled — output only the description."
)


class VisionCaptioner:
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
                "Start LM Studio, load a vision-capable model, and enable the local server."
            )

    def caption(self, image_bytes: bytes) -> str:
        key = hashlib.sha256(f"{self._model}:".encode() + image_bytes).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        b64 = base64.b64encode(image_bytes).decode("ascii")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )
        description = response.choices[0].message.content.strip()
        self._cache.set(key, description)
        return description
