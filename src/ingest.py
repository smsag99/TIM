"""Ingestion: load log files, chunk them by line groups, embed into Chroma.

Logs are line-oriented data, so instead of splitting by characters we
group consecutive lines into small overlapping windows. Each window
becomes one searchable chunk, tagged with which log file it came from
and which line range it covers.

Run this once (or whenever the logs change) before querying:

    python -m src.ingest
"""

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from src import config


def load_log_chunks():
    """Read every .log file and split it into overlapping line windows.

    Returns a list of LangChain Documents. Each Document is a small
    block of consecutive log lines, with metadata recording the source
    file name and the line range, so an answer can be traced back to
    the exact place in the logs.
    """
    documents = []

    log_files = sorted(config.LOGS_DIR.glob("*.log"))
    if not log_files:
        raise RuntimeError(f"No .log files found in {config.LOGS_DIR}.")

    step = config.LINES_PER_CHUNK - config.LINE_OVERLAP

    for path in log_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        # Slide a window of LINES_PER_CHUNK lines across the file.
        for start in range(0, len(lines), step):
            window = lines[start:start + config.LINES_PER_CHUNK]
            if not window:
                continue
            chunk_text = "\n".join(window)
            documents.append(Document(
                page_content=chunk_text,
                metadata={
                    "source_name": path.name,
                    "line_start": start + 1,
                    "line_end": start + len(window),
                },
            ))

    return documents


def build_vector_store():
    """Chunk the log files and persist them into a Chroma store."""
    api_key = config.get_google_api_key()

    documents = load_log_chunks()
    print(f"Built {len(documents)} chunk(s) from the log files.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(config.VECTOR_STORE_DIR),
    )
    print(f"Vector store written to {config.VECTOR_STORE_DIR}")
    return vector_store


if __name__ == "__main__":
    build_vector_store()
