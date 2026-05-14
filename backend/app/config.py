"""Backend configuration. All secrets come from environment variables.

Never hardcode keys. Locally, use a .env file (gitignored). In
production (Render/Railway), set these in the host's dashboard.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BACKEND_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = BACKEND_ROOT / "data" / "logs"

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

LINES_PER_CHUNK = 8
LINE_OVERLAP = 2
RETRIEVER_K = 6

# Collection name for the pgvector table.
COLLECTION_NAME = "network_logs"

# How many refused/failed events from one IP before it is "flagged".
FLAG_THRESHOLD = 4


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value or value.startswith("your-"):
        raise RuntimeError(
            f"{name} is not set. Add it to .env (local) or to the "
            f"host's environment variables (production)."
        )
    return value


def get_google_api_key() -> str:
    return _require("GOOGLE_API_KEY")


def get_database_url() -> str:
    # Supabase connection string, e.g.
    # postgresql+psycopg://postgres:PASSWORD@HOST:5432/postgres
    return _require("DATABASE_URL")
