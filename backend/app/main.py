"""FastAPI backend for the Network Log RAG Assistant.

Endpoints:
  GET  /              health check
  POST /ask           ask a plain-language question over the logs
  GET  /flagged       list source IPs with high failed-event counts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import rag, flagging

app = FastAPI(title="Network Log RAG Assistant")

# CORS: the frontend (Vercel) calls this from the browser.
# Set ALLOWED_ORIGINS in production to your Vercel URL.
import os
_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    suggested_action: str
    sources: list[str]


@app.get("/")
def health():
    return {"status": "ok", "service": "network-log-rag"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = rag.ask(req.question)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flagged")
def flagged():
    try:
        return {"flagged": flagging.get_flagged_ips()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
