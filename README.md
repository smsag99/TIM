# Network Log RAG Assistant — Web App

A web app that lets you ask plain-language questions about network
and service logs, with answers grounded in the actual log lines. It
also flags source IPs with a high number of failed connection events.

Built with FastAPI, LangChain, Google Generative AI, and Supabase
pgvector as the vector store. Frontend is a single static page.

## Why this project

Network and service operations generate large volumes of log data.
The information needed to understand an incident is in the logs, but
finding it means scrolling and grepping. This project applies
retrieval-augmented generation (RAG) to operational log data: you
ask a question in plain language, the system retrieves the most
relevant log lines and uses them to ground the model's answer.

A second, rule-based feature scans the connection logs and flags
source IPs with many failed events (refused / timeout / slow). This
is detection only — it flags, it does not block.

## Architecture

```
                 ┌──────────────────────────┐
  Browser ─────► │  Frontend (static page)  │   hosted on Vercel
                 └────────────┬─────────────┘
                              │  HTTP (/ask, /flagged)
                 ┌────────────▼─────────────┐
                 │  Backend — FastAPI       │   hosted on Render
                 │  app/main.py             │
                 │   ├── rag.py  (ask)      │
                 │   └── flagging.py (flag) │
                 └────────────┬─────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   ┌────────────────────┐         ┌────────────────────────┐
   │ Supabase pgvector  │         │ Google Generative AI   │
   │ (log chunk vectors)│         │ (embeddings + LLM)     │
   └────────────────────┘         └────────────────────────┘
```

- **Chunking:** logs are line-oriented, so `ingest.py` splits them
  into small overlapping windows of consecutive lines. Each chunk
  keeps its source file and line range as metadata.
- **Retrieval:** a question is embedded and the closest chunks are
  pulled from pgvector (top-K semantic search).
- **Grounding:** retrieved chunks are passed to the LLM as context;
  the model answers from them and reports which lines it used.
- **Flagging:** `flagging.py` parses the raw log lines directly
  (deterministic counting — no LLM needed).

## Project layout

```
backend/
  app/
    config.py     env vars, settings
    ingest.py     chunk logs -> embed -> Supabase pgvector
    rag.py        retrieval + LLM (the /ask logic)
    flagging.py   rule-based high-failure IP detection
    main.py       FastAPI app (/ask, /flagged)
  data/logs/      sample log files
  requirements.txt
  render.yaml     Render deploy config
  .env.example
frontend/
  index.html      single-page UI
```

## Run locally

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: add GOOGLE_API_KEY and your Supabase DATABASE_URL
```

Enable the pgvector extension in Supabase once (SQL editor):

```sql
create extension if not exists vector;
```

Then load the logs and start the server:

```bash
python -m app.ingest          # one-time: builds the vector store
uvicorn app.main:app --reload # starts on http://localhost:8000
```

### 2. Frontend

`frontend/index.html` is a static file. For local use, `API_BASE`
at the top of the `<script>` is already set to `http://localhost:8000`.
Just open the file in a browser, or serve it:

```bash
cd frontend
python -m http.server 5500    # then open http://localhost:5500
```

## Deploy

### Backend on Render

1. Push this repo to GitHub.
2. On Render: New > Web Service > connect the repo.
3. Render reads `backend/render.yaml`. Confirm root dir is `backend`.
4. Add environment variables as secrets: `GOOGLE_API_KEY`,
   `DATABASE_URL`, and `ALLOWED_ORIGINS` (set this to your Vercel URL
   once you have it).
5. Deploy. After the first deploy, run the ingest step once — either
   from the Render Shell (`python -m app.ingest`) or locally pointed
   at the same Supabase database.

### Frontend on Vercel

1. In `frontend/index.html`, set `API_BASE` to your Render backend
   URL (e.g. `https://network-log-rag-backend.onrender.com`).
2. On Vercel: New Project > import the repo > set the root directory
   to `frontend`. No build step — it is a static page.
3. Deploy. Copy the Vercel URL back into the backend's
   `ALLOWED_ORIGINS` env var and redeploy the backend.

## Security notes

- API keys and the database URL are read from environment variables
  only — never hardcoded, never committed (`.env` is gitignored).
- `ALLOWED_ORIGINS` should be set to the specific frontend URL in
  production, not `*`.
- The `/ask` endpoint limits question length. For a public
  deployment you would also want rate limiting to protect your API
  quota — see "limitations" below.

## Scope and honest limitations

This is a prototype, not a production system:

- The log sample is small and synthetic.
- Retrieval is purely semantic (top-K) — there is no time-range or
  field-based filtering yet, so questions needing an exact total
  across all logs are not reliably answered.
- There is no authentication or rate limiting on the public
  endpoints yet.
- There is no evaluation harness measuring answer quality.

Natural next steps: structured parsing of log fields (so questions
about specific IPs / ports / time ranges can be filtered precisely),
a larger and real log corpus, support for streaming logs,
authentication and rate limiting, and an evaluation set.
