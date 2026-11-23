"""
FastAPI REST API wrapper for AgentMemory.
Provides endpoints for adding messages, retrieving relevant context, and querying stats.
"""

import os
import sys
import re
import secrets

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Any
from agentmemory import AgentMemory, PersistenceLayer
import os

app = FastAPI(
    title="AgentMemory API",
    description="Topic-based memory management service for AI agents",
    version="1.1.0"
)

class MessageRequest(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, max_length=10_000, description="Text content of message")

class MessageResponse(BaseModel):
    role: str
    content: str

class ContextResponse(BaseModel):
    session_id: str
    query: str
    context: List[MessageResponse]
    chunks_count: int

class StatsResponse(BaseModel):
    session: str
    total_chunks: int
    total_messages: int
    persisted_sessions: List[str]
    chunks: List[Dict[str, Any]]


def validate_session_id(session_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", session_id):
        raise HTTPException(
            status_code=422,
            detail="session_id must be 1-128 letters, numbers, hyphens, or underscores",
        )
    return session_id


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    configured_key = os.getenv("AGENTMEMORY_API_KEY")
    if configured_key and not secrets.compare_digest(x_api_key or "", configured_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_memory(session_id: str) -> AgentMemory:
    return AgentMemory(
        session_id=session_id,
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        auto_summarize=env_flag("AGENTMEMORY_AUTO_SUMMARIZE"),
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AgentMemory"}


@app.post("/sessions/{session_id}/messages", dependencies=[Depends(require_api_key)])
def add_message(session_id: str, msg: MessageRequest):
    """Add a user or assistant message to session memory."""
    session_id = validate_session_id(session_id)
    if not msg.content.strip():
        raise HTTPException(status_code=422, detail="content must not be blank")
    mem = get_memory(session_id)
    mem.add_message(msg.role, msg.content)
    return {"status": "added", "session_id": session_id, "role": msg.role}


@app.get("/sessions/{session_id}/context", response_model=ContextResponse, dependencies=[Depends(require_api_key)])
def get_context(session_id: str, query: str = Query(..., min_length=1, max_length=10_000, description="Current user query to match relevant context against")):
    """Retrieve relevant conversation context for the specified query."""
    session_id = validate_session_id(session_id)
    if not query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    mem = get_memory(session_id)
    ctx = mem.get_context(query)
    stats = mem.stats()
    return ContextResponse(
        session_id=session_id,
        query=query,
        context=[MessageResponse(role=m["role"], content=m["content"]) for m in ctx],
        chunks_count=stats["total_chunks"]
    )


@app.get("/sessions/{session_id}/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
def get_stats(session_id: str):
    """Get chunk statistics for a session."""
    session_id = validate_session_id(session_id)
    mem = get_memory(session_id)
    return mem.stats()


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
def clear_session(session_id: str):
    """Clear memory chunks for a session from both Redis and SQLite."""
    session_id = validate_session_id(session_id)
    mem = get_memory(session_id)
    mem.clear()
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions", dependencies=[Depends(require_api_key)])
def list_sessions():
    """List all persisted session IDs."""
    db_path = os.getenv("AGENTMEMORY_DB_PATH", "agentmemory.db")
    db = PersistenceLayer("global", db_path=db_path)
    return {"sessions": db.list_sessions()}
