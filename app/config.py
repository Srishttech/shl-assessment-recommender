import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Gemini

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Embeddings
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Catalog
CATALOG_PATH = os.environ.get("CATALOG_PATH", "data/shl_catalog_fixed.json")

# Retrieval
TOP_K_RECOMMEND = int(os.environ.get("TOP_K_RECOMMEND", 5))
TOP_K_COMPARE = int(os.environ.get("TOP_K_COMPARE", 3))

# Cache dir for FAISS index + embeddings (avoids recompute on every restart)
CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
