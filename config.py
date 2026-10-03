# Central config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.1-flash-lite"
TEMPERATURE = 0.3
RERANK_TOP_N = 4

# Extra
DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"
RERANK_MODEL = "BAAI/bge-reranker-base"
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
MEMORY_K = 5
