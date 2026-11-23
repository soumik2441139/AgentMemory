# AgentMemory 🧠

Lightweight memory management for AI agents. Organizes conversations into topic-based chunks so your LLM gets relevant context — not just the last N messages.

## The Problem

Every LLM has a context window limit. When conversations get long, most solutions just truncate. AgentMemory organizes.

```
Message arrives → Extract keywords (NLTK) → Classify:
  • Small talk?     → General Chat chunk
  • Matches topic?  → Update existing chunk (TF-IDF cosine similarity)
  • New topic?      → Create new chunk

On retrieval: LLM gets only the chunks relevant to the current question.
```

## Features

- **Topic-aware chunking** — messages grouped by topic, not just recency
- **TF-IDF similarity** — finds the right chunk even when wording differs
- **SQLite persistence** — memory survives Redis restarts
- **LLM summarization** — compresses old chunks to save tokens
- **Chunk drift prevention** — matches only against user messages, not LLM responses
- **Works with any OpenAI-compatible API** — Groq, DeepSeek, Ollama, etc.

## Version History

### Unreleased — Reliability and deployment hardening

- Per-session Redis locks to avoid conflicting writes from concurrent API workers
- Atomic SQLite upserts with a unique session/chunk constraint
- Batched TF-IDF scoring instead of rebuilding a vectorizer for every chunk
- Optional API-key protection, validated request input, and configurable API summarization
- Windows-safe console output and expanded tests

### v1.1 — Current tagged release

- Pip-installable `agentmemory` package
- LLM-powered summarization for large topic chunks
- SQLite persistence across Redis restarts
- Initial unit-test suite covering memory, similarity, persistence, and summarization

### v1.0 — Persistent topic memory

- SQLite backup and restore for topic chunks
- Cleaner package structure and entry point

### v0.3 — Smarter retrieval

- TF-IDF cosine similarity for topic matching
- User-only similarity matching to prevent assistant-response drift

### v0.2 — Topic memory

- NLTK-powered keyword extraction and topic clustering
- Dynamic topic-chunk updates for follow-up questions
- General-chat isolation and relevance-based context retrieval

### v0.1 — Basic hot/cold memory

- Two-tier Redis memory: a recent-message hot tier and overflow cold tier
- Simple context injection for agent conversations

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/soumik2441139/AgentMemory.git
cd AgentMemory
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# .\venv\Scripts\Activate.ps1  # Windows
pip install -e .
python3 -c "import nltk; nltk.download('stopwords')"
```

### 2. Set up Redis

```bash
# Linux
sudo service redis-server start

# Mac
brew services start redis

# Windows
Start-Process "C:\Program Files\Redis\redis-server.exe"
```

Verify: `redis-cli ping` → `PONG`

### 3. Configure

Create a `.env` file with any supported provider key (`agent.py` auto-detects in the order OpenAI → Groq → DeepSeek):

```ini
# Set at least one LLM provider key:
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
# GROQ_API_KEY=your_groq_key_here
# DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional Redis & persistence configuration:
REDIS_HOST=localhost
REDIS_PORT=6379
AGENTMEMORY_DB_PATH=agentmemory.db

# Optional API security & summarization:
AGENTMEMORY_API_KEY=a-long-random-secret
AGENTMEMORY_AUTO_SUMMARIZE=true
```

### 4. Run

```bash
python agent.py
```

Commands inside the chat:
- `stats` — see memory chunks and message counts
- `quit` — exit

## Library Usage

```python
from agentmemory import AgentMemory

memory = AgentMemory("session-1", auto_summarize=True)

# Store messages
memory.add_message("user", "Tell me about Redis pub/sub")
memory.add_message("assistant", "Redis pub/sub is a messaging pattern...")

# Later — retrieve relevant context
context = memory.get_context("How does Redis pub/sub handle failures?")
# Returns only chunks relevant to Redis, formatted for the LLM

# Check what's stored
memory.stats()
# {'session': 'session-1', 'total_chunks': 2, 'total_messages': 5, ...}

# Clear everything
memory.clear()
```

## How It Works

### Storage Model

Memory is stored in **chunks** — each chunk is a topic container:

```json
{
  "id": 1,
  "title": "Redis Pubsub",
  "type": "topic",
  "keywords": ["redis", "pubsub", "messaging"],
  "messages": [
    {"role": "user", "content": "Tell me about Redis pub/sub", "timestamp": 1234.5}
  ],
  "created_at": 1234.5,
  "updated_at": 1234.5
}
```

### Retrieval Pipeline

1. Extract keywords from the current message (NLTK stopword removal)
2. Compute TF-IDF cosine similarity against each chunk's **user-only** messages
3. Return the top 2 most relevant chunks (last 4 messages each)
4. Always include last 2 messages from the General Chat chunk
5. When `auto_summarize=True` and a chunk exceeds 6 messages, the LLM summarizes it into a single system message

### Dual Storage

- **Redis** — fast in-memory access during conversations
- **SQLite** — persistent backup; on startup, if Redis is empty, chunks restore from SQLite automatically

## Benchmarks & Evaluation

### Recall Accuracy on Synthetic Long Conversations (3 scenarios, 58–88 turns)

We evaluated 5 retrieval strategies on 3 synthetic conversations where a fact planted at Turn 3 is followed by 50–80 turns of unrelated topics:

| Strategy | Recall (3 scenarios) | Avg Tokens | Avg Context Msgs |
| :--- | :---: | :---: | :---: |
| **Last-10 Messages** | 0.0% (0/3) | 89 | 10.0 |
| **Last-20 Messages** | 0.0% (0/3) | 174 | 20.0 |
| **Keyword Chunking** (v0.2) | 66.7% (2/3) | 35 | 4.0 |
| **AgentMemory TF-IDF** (v1.1) | **66.7% (2/3)** | **14** | **1.3** |
| **Char N-Gram TF-IDF** (`char_wb` 1–3) | **100.0% (3/3)** | **58** | **6.0** |

> **What this shows (and its limits):**
> - Last-N truncation misses all 3 scenarios because 50+ distractor turns push Turn 3 out of the window.
> - Word-level TF-IDF and keyword matching both hit the 2 scenarios that share vocabulary between fact and query (`database_port`, `favorite_editor`), with AgentMemory retrieving fewer irrelevant messages (14 vs 35 avg tokens).
> - Both word-level methods miss `dog_name` (`"golden retriever ... peanut butter"` vs `"dog's name and favorite treat"`) because there is zero word overlap (`"named"` vs `"name"` only matches subword character n-grams). Semantic matching without subword overlap requires dense embeddings.

Run the evaluation yourself:
```bash
python eval/run_eval.py
```

### In-Memory Retrieval & Scoring Latency

Measures `get_context()` TF-IDF scoring and chunk-selection overhead against an in-memory mock storage backend (excludes network round-trip to a live Redis instance):

| Stored Chunks | p50 Latency | p95 Latency | Max Latency |
| :--- | :---: | :---: | :---: |
| **10 chunks** | 8.2 ms | 106.7 ms* | 248.5 ms* |
| **50 chunks** | 6.3 ms | 13.2 ms | 13.4 ms |
| **100 chunks** | 9.8 ms | 13.9 ms | 14.7 ms |
| **500 chunks** | 27.1 ms | 36.8 ms | 39.0 ms |
| **1,000 chunks** | 54.9 ms | 67.9 ms | 72.8 ms |

*\*First-iteration cold start on the 10-chunk run includes initial scikit-learn warm-up.*

Run the latency benchmarks:
```bash
python benchmarks/bench_latency.py
```

## REST API & Docker Deployment

Run AgentMemory as an independent microservice with Redis in a single command:

```bash
docker compose up --build
```

API documentation will be available at `http://localhost:8000/docs`.
When `AGENTMEMORY_API_KEY` is set, send it as the `X-API-Key` header with every request.
Set `AGENTMEMORY_AUTO_SUMMARIZE=true` to enable LLM summarization for API sessions.

### Key Endpoints:
- `POST /sessions/{id}/messages` — Append message (`{"role": "user", "content": "..."}`)
- `GET /sessions/{id}/context?query=...` — Retrieve relevant conversation context
- `GET /sessions/{id}/stats` — Inspect active memory chunks
- `DELETE /sessions/{id}` — Clear session from Redis and SQLite

## Tech Stack

- **Python 3.8+**
- **Redis** — real-time memory storage
- **SQLite** — persistent backup
- **NLTK** — keyword extraction
- **scikit-learn** — TF-IDF vectorization and cosine similarity
- **Groq / OpenAI API** — LLM calls (agent + summarization)

## Project Structure

```
agentmemory/
├── __init__.py        # Package exports
├── memory.py          # Core: topic chunking + TF-IDF + persistence + summarization
├── memory_basic.py    # Simple hot/cold memory (no topic awareness)
├── similarity.py      # TF-IDF cosine similarity matching
├── persistence.py     # SQLite persistence layer
└── summarizer.py      # LLM-powered chunk summarization
agent.py               # Interactive chat agent
tests/                 # Automated unit tests
api/                   # FastAPI microservice
eval/                  # Context retrieval evaluation harness
benchmarks/            # Latency benchmarking suite
```

## Running Tests

```bash
python -m unittest discover tests -v
```

## Why Not LangChain?

LangChain memory requires the entire LangChain framework. AgentMemory is a standalone drop-in library — works with any LLM API in 5 lines of code.

## License

MIT
