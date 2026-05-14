"""RAG pipeline: retrieve relevant log chunks from pgvector, ground the LLM."""

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate

from app import config

_store = None
_llm = None


def _get_store():
    global _store
    if _store is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.get_google_api_key(),
        )
        _store = PGVector(
            embeddings=embeddings,
            collection_name=config.COLLECTION_NAME,
            connection=config.get_database_url(),
        )
    return _store


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            google_api_key=config.get_google_api_key(),
            temperature=0.2,
        )
    return _llm


def _format_context(docs):
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(
            f"[{m.get('source_name','?')} "
            f"lines {m.get('line_start','?')}-{m.get('line_end','?')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


ASK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that helps engineers understand network "
     "and service logs. Answer using ONLY the log lines in the "
     "context. Be concise and factual. If you point out a pattern "
     "(repeated failures, latency spike, many refused connections), "
     "briefly explain what in the logs shows it. If the context does "
     "not contain enough information, say so instead of guessing. "
     "Mention which log file the evidence came from."),
    ("human", "Log context:\n{context}\n\nQuestion: {question}"),
])


def ask(question: str) -> dict:
    store = _get_store()
    docs = store.similarity_search(question, k=config.RETRIEVER_K)

    chain = ASK_PROMPT | _get_llm()
    answer = chain.invoke({
        "context": _format_context(docs),
        "question": question,
    })

    sources = [
        f"{d.metadata.get('source_name','?')}:"
        f"{d.metadata.get('line_start','?')}-"
        f"{d.metadata.get('line_end','?')}"
        for d in docs
    ]
    return {"answer": answer.content, "sources": sources}
