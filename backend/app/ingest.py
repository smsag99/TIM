"""Chunk the log files and load them into Supabase pgvector.

Run once after setting env vars:
    python -m app.ingest
"""

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from app import config


def load_log_chunks():
    """Read .log files and split into overlapping line windows."""
    documents = []
    log_files = sorted(config.LOGS_DIR.glob("*.log"))
    if not log_files:
        raise RuntimeError(f"No .log files found in {config.LOGS_DIR}.")

    step = config.LINES_PER_CHUNK - config.LINE_OVERLAP
    for path in log_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for start in range(0, len(lines), step):
            window = lines[start:start + config.LINES_PER_CHUNK]
            if not window:
                continue
            documents.append(Document(
                page_content="\n".join(window),
                metadata={
                    "source_name": path.name,
                    "line_start": start + 1,
                    "line_end": start + len(window),
                },
            ))
    return documents


def build_vector_store():
    api_key = config.get_google_api_key()
    db_url = config.get_database_url()

    documents = load_log_chunks()
    print(f"Built {len(documents)} chunk(s) from the log files.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    # pre_delete_collection=True keeps re-ingestion idempotent.
    PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=db_url,
        pre_delete_collection=True,
    )
    print(f"Loaded into Supabase pgvector collection "
          f"'{config.COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_vector_store()
