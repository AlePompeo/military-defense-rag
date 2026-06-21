import os

# LM Studio local server (OpenAI-compatible API)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

# ChromaDB persistent storage
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")

# Chunking
CHUNK_SIZE = 900       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks

# Retrieval
TOP_K = 6              # chunks returned per query
