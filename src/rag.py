"""The RAG pipeline for log question-answering.

One query mode: ask() takes a plain-language question, retrieves the
most relevant chunks of log lines from the vector store, and passes
them to the LLM as grounding context. The model answers based on the
retrieved log lines rather than on its own assumptions, and the
answer reports which log files and line ranges it drew from.
"""

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from src import config


def _load_vector_store():
    """Open the persisted Chroma store. Assumes ingest.py has been run."""
    api_key = config.get_google_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    if not config.VECTOR_STORE_DIR.exists():
        raise RuntimeError(
            "Vector store not found. Run 'python -m src.ingest' first."
        )
    return Chroma(
        persist_directory=str(config.VECTOR_STORE_DIR),
        embedding_function=embeddings,
    )


def _get_llm():
    api_key = config.get_google_api_key()
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )


def _format_context(docs):
    """Turn retrieved log chunks into a readable, source-labelled block."""
    parts = []
    for doc in docs:
        name = doc.metadata.get("source_name", "unknown")
        start = doc.metadata.get("line_start", "?")
        end = doc.metadata.get("line_end", "?")
        parts.append(f"[{name} lines {start}-{end}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


ASK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that helps engineers understand network "
     "and service logs. Answer the question using ONLY the log lines "
     "in the context below. Be concise and factual. If you point out "
     "a pattern (for example repeated failures, a spike in latency, "
     "or many refused connections), explain briefly what in the logs "
     "shows it. If the context does not contain enough information to "
     "answer, say so clearly instead of guessing. When relevant, "
     "mention which log file the evidence came from."),
    ("human", "Log context:\n{context}\n\nQuestion: {question}"),
])


def ask(question: str) -> dict:
    """Answer a plain-language question over the indexed log data."""
    store = _load_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
    docs = retriever.invoke(question)

    llm = _get_llm()
    chain = ASK_PROMPT | llm
    answer = chain.invoke({
        "context": _format_context(docs),
        "question": question,
    })

    # Build a readable list of which sources were used.
    sources = []
    for d in docs:
        name = d.metadata.get("source_name", "unknown")
        start = d.metadata.get("line_start", "?")
        end = d.metadata.get("line_end", "?")
        sources.append(f"{name}:{start}-{end}")

    return {
        "answer": answer.content,
        "sources": sources,
    }
