"""RAG pipeline: retrieve relevant log chunks from pgvector, ground the LLM.

Each answer also includes a 'suggested_action': a recommendation for
what a human operator could consider next. It is phrased as advice
only — the system never performs the action and never claims to have
performed one. A human stays in the loop.
"""

import json

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
     "and service logs.\n"
     "Use ONLY the log lines in the context. Be concise and factual. "
     "If you point out a pattern (repeated failures, latency spike, "
     "many refused connections), briefly explain what in the logs "
     "shows it. If the context does not contain enough information, "
     "say so instead of guessing.\n\n"
     "Respond with a JSON object and nothing else, with exactly two "
     "string fields:\n"
     '  "answer": your factual answer to the question, mentioning '
     "which log file the evidence came from.\n"
     '  "suggested_action": one short recommendation for what a '
     "human operator could consider doing next.\n\n"
     "IMPORTANT rules for suggested_action:\n"
     "- It is ADVICE ONLY. Start it with a word like 'Recommend' or "
     "'Consider'.\n"
     "- Never claim an action was taken or completed. You do not "
     "perform actions; a human decides.\n"
     "- Do not recommend automatic blocking or banning. Recommend "
     "review, flagging, watchlisting, opening an incident, or "
     "investigating instead.\n"
     "- If nothing notable is found, say no action is needed."),
    ("human", "Log context:\n{context}\n\nQuestion: {question}"),
])


def _parse_response(raw: str) -> dict:
    """Parse the model's JSON. Fall back gracefully if it is not clean."""
    text = raw.strip()
    # Strip code fences if the model added them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        return {
            "answer": str(data.get("answer", "")).strip(),
            "suggested_action": str(
                data.get("suggested_action", "")
            ).strip(),
        }
    except (json.JSONDecodeError, AttributeError):
        # If parsing fails, return the raw text as the answer and no
        # action, rather than crashing.
        return {"answer": raw.strip(), "suggested_action": ""}


def ask(question: str) -> dict:
    store = _get_store()
    docs = store.similarity_search(question, k=config.RETRIEVER_K)

    chain = ASK_PROMPT | _get_llm()
    response = chain.invoke({
        "context": _format_context(docs),
        "question": question,
    })

    parsed = _parse_response(response.content)

    sources = [
        f"{d.metadata.get('source_name','?')}:"
        f"{d.metadata.get('line_start','?')}-"
        f"{d.metadata.get('line_end','?')}"
        for d in docs
    ]
    return {
        "answer": parsed["answer"],
        "suggested_action": parsed["suggested_action"],
        "sources": sources,
    }
