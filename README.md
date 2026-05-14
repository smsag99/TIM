# Network Log RAG Assistant

A small retrieval-augmented generation (RAG) tool that lets you ask
plain-language questions about network and service log data.

## Why this project

Network and service operations produce large volumes of log data:
connection records, request latencies, error events. The information
needed to understand an incident is usually *in* the logs — but
finding it means scrolling, grepping, and knowing what to look for.

This project is a small step toward making that data easier to
query: instead of searching logs by hand, you ask a question in
plain language and get an answer grounded in the actual log lines.
It applies a standard AI pattern — retrieval-augmented generation —
to operational log data, using Python, LangChain, and a vector
database. Working with structured operational data and making it
queryable is a data-engineering task; this project applies an LLM
layer on top of it.

## What it does

The tool indexes a small set of sample log files:

- **`connections.log`** — network connection records (source IP,
  destination, port, status, bytes).
- **`services.log`** — application/service logs for a provisioning
  API and an element manager (request latencies, results, errors).

You then ask questions in plain language, for example:

```
python -m src.main ask "were there any refused connections, and from where?"
python -m src.main ask "did the provisioning service have any problems?"
python -m src.main ask "which source IP generated the most failed attempts?"
python -m src.main ask "Were there any refused connections, and from where?" 
python -m src.main ask "Did the provisioning service have problems?" 
python -m src.main ask "Which IP made the most failed attempts?"
python -m src.main ask "What happened around 09:09?"
python -m src.main ask "Were there any errors in the service log?"
python -m src.main ask "Summarize the connection activity."ر
```

The tool retrieves the most relevant chunks of log lines from the
vector store and passes them to the language model as grounding
context. The model answers based on those retrieved lines — not on
its own assumptions — and the answer reports which log files and
line ranges the evidence came from, so every answer is traceable.

## Architecture

```
data/logs/      sample log files (connection + service logs)
       |
       v
src/ingest.py   read logs -> group into line windows ->
                embed -> Chroma vector store
       |
       v
src/rag.py      retrieve relevant log chunks -> ground the LLM prompt
       |
       v
src/main.py     command-line interface (ask)
```

- **Chunking:** logs are line-oriented, so they are split into small
  overlapping windows of consecutive lines rather than by character
  count. Each chunk keeps its source file and line range as metadata.
- **Embeddings & LLM:** Google Generative AI (`gemini-embedding-001`,
  `gemini-2.5-flash`).
- **Vector store:** Chroma, persisted locally.
- **Orchestration:** LangChain.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY

# 3. Build the vector store (run once)
python -m src.ingest

# 4. Ask questions
python -m src.main ask "were there any failed connections?"
```

The API key is read from `.env` (which is gitignored) and is never
hardcoded.


