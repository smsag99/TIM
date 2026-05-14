"""Central configuration. Loads secrets from a .env file.

The API key is never hardcoded. Copy .env.example to .env and put
your key there. .env is gitignored.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
VECTOR_STORE_DIR = PROJECT_ROOT / ".chroma"

# --- Models ---
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

# --- Retrieval settings ---
# Logs are line-oriented, so we chunk by a number of lines rather
# than by characters. Each chunk keeps a few lines together so the
# model sees a small window of context, not isolated lines.
LINES_PER_CHUNK = 8
LINE_OVERLAP = 2
RETRIEVER_K = 6


def get_google_api_key() -> str:
    """Return the Google API key, or fail with a clear message."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key or key == "your-google-api-key-here":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env "
            "and add your key."
        )
    return key
