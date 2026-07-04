import os

# LM Studio local server (OpenAI-compatible API)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

# ChromaDB persistent storage
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")

# Embedding model — fastembed (ONNX-only, no torch/GPU required).
# BAAI/bge-base-en-v1.5: 768-dim, ~210MB quantized, asymmetric query/document
# embeddings (query_embed applies the BGE query instruction prefix automatically).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "./data/embed_cache")

# Reranking — cross-encoder applied to the fused dense+BM25 candidates before
# the final top-k is sent to the LLM. Lazily loaded, so it costs nothing when unused.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "1") != "0"

# Query result cache (post dense+BM25+rerank). Auto-invalidated whenever `ingest`
# adds new chunks; also expires on its own after QUERY_CACHE_TTL seconds.
QUERY_CACHE_PATH = os.getenv("QUERY_CACHE_PATH", "./data/query_cache")
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", str(60 * 60)))  # seconds

# HNSW index tuning (cosine space — fastembed's BGE output is L2-normalized)
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
HNSW_M = int(os.getenv("HNSW_M", "16"))

# Chunking — hierarchical: small "child" chunks are embedded and retrieved,
# larger "parent" blocks are what actually gets sent to the LLM (small-to-big retrieval).
CHUNK_SIZE = 900          # characters per child chunk
CHUNK_OVERLAP = 150       # overlap between consecutive child chunks
PARENT_CHUNK_SIZE = 2700  # characters per parent block (~3x child)

# Retrieval
TOP_K = 6                  # final chunks sent to the LLM
RETRIEVAL_CANDIDATES = 20  # candidates considered before RRF fusion + reranking
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))  # 1.0 = pure relevance, 0.0 = pure diversity

# HyDE (Hypothetical Document Embeddings) — opt-in query rewriting: an extra
# local LLM call per query generates a hypothetical passage, which steers the
# dense search instead of the raw question. Off by default: it doubles LLM
# calls per query, on top of what's already the dominant latency cost here.
USE_HYDE = os.getenv("USE_HYDE", "0") == "1"
HYDE_CACHE_PATH = os.getenv("HYDE_CACHE_PATH", "./data/hyde_cache")

# Contextual retrieval (opt-in, `ingest --contextualize`) — an LLM-generated
# per-chunk summary prepended before embedding. One local LLM call per chunk
# at ingest time: impractical on CPU-only hardware (multi-day for this
# corpus), reasonable on a dedicated GPU (roughly hours). Off by default.
CONTEXTUALIZE_CACHE_PATH = os.getenv("CONTEXTUALIZE_CACHE_PATH", "./data/contextualize_cache")
