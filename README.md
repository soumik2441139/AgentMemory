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

### Unreleased — v1.2 Retrieval & Precision Hardening

- Split TF-IDF roles: batched word-level TF-IDF for topic assignment (`threshold=0.12`) and stopword-filtered character n-gram TF-IDF (`char_wb` 4–5, `threshold=0.15`) for context retrieval
- Bounded per-session LRU cache (`max_cached_indexes=128`) for warm context retrieval
- Short and long-conversation no-match precision checks in the evaluation harness
- Separate warm-read and alternating write/read chat turn latency benchmarks
- Per-session Redis locks to avoid conflicting writes from concurrent API workers
- Atomic SQLite upserts with a unique session/chunk constraint
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

### 2. Set up Redis (Optional — Automatic In-Memory + SQLite Fallback)

If a Redis server is running, AgentMemory uses it for active multi-worker session state. If Redis is not installed or not running, AgentMemory **automatically falls back to an in-memory store backed by SQLite** (`agentmemory.db`) so it works out of the box with zero setup.

```bash
# Linux
sudo service redis-server start

# Mac
brew services start redis

# Windows
Start-Process "C:\Program Files\Redis\redis-server.exe"
```

Verify (if using Redis): `redis-cli ping` → `PONG`

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
2. Assign incoming user messages to topics using batched word-level TF-IDF cosine similarity against each candidate chunk's **user-only** messages
3. Rank chunks on `get_context()` using cached character n-gram TF-IDF (`char_wb`, 4–5 chars) over stopword-filtered user messages (`threshold > 0.15`)
4. Return the top 2 most relevant chunks (last 4 messages each)
5. Always include last 2 messages from the General Chat chunk
6. When `auto_summarize=True` and a chunk exceeds 6 messages, the LLM summarizes it into a single system message

### Dual Storage

- **Redis (with automatic in-memory fallback)** — fast in-memory access during conversations; if no Redis server is reachable, falls back transparently to a thread-safe in-memory store
- **SQLite** — persistent backup; on startup, if the active store is empty, chunks restore from SQLite automatically

## Benchmarks & Evaluation

### Recall & No-Match Precision on Synthetic Long Conversations (3 scenarios, 58–88 turns)

We evaluated 5 retrieval strategies on 3 synthetic conversations where a fact planted at Turn 3 is followed by 50–80 turns of unrelated topics, plus 6 no-match precision queries (3 short, 3 against the full 58–88 turn noisy sessions):

| Strategy | Recall (3 scenarios) | No-Match Precision | Avg Tokens | Avg Context Msgs |
| :--- | :---: | :---: | :---: | :---: |
| **Last-10 Messages** | 0.0% (0/3) | — | 89 | 10.0 |
| **Last-20 Messages** | 0.0% (0/3) | — | 174 | 20.0 |
| **Keyword Chunking** (v0.2) | 66.7% (2/3) | — | 35 | 4.0 |
| **AgentMemory TF-IDF** (v1.2) | **100.0% (3/3)** | **100.0% (6/6)** | **20** | **2.0** |
| **Char N-Gram TF-IDF** (`char_wb` 1–3) | **100.0% (3/3)** | — | **56** | **6.0** |

> **What this shows (and its limits):**
> - Last-N truncation misses all 3 scenarios because 50+ distractor turns push Turn 3 out of the window.
> - Splitting TF-IDF roles — word-level TF-IDF for chunk formation and stopword-filtered `char_wb` (4–5) for retrieval — lets AgentMemory match morphological variants like `"named"` vs `"name"` in `dog_name` without merging unrelated topics or leaking false-positive chunks (20 vs 56 avg tokens, 6/6 no-match precision).
> - Both `char_wb` methods are still lexical/subword matchers: true zero-overlap synonyms (e.g., `"dog"` ↔ `"golden retriever"` with no shared stem) require dense embeddings.

Run the evaluation yourself:
```bash
python eval/run_eval.py
```

### In-Memory Retrieval & Turn Latency

Measures TF-IDF scoring and chunk-selection overhead against an in-memory mock storage backend (excludes network round-trip to a live Redis instance):

**1. Warm Read-Only Retrieval (`get_context` with cached per-session index):**

| Stored Chunks | p50 Latency | p95 Latency | Max Latency |
| :--- | :---: | :---: | :---: |
| **10 chunks** | 1.6 ms | 3.2 ms | 4.1 ms |
| **50 chunks** | 1.9 ms | 2.8 ms | 3.2 ms |
| **100 chunks** | 2.3 ms | 3.9 ms | 4.1 ms |
| **500 chunks** | 7.4 ms | 10.4 ms | 10.9 ms |
| **1,000 chunks** | 17.3 ms | 24.7 ms | 26.9 ms |

**2. Alternating Chat Turn (`add_message("user")` → `get_context()` → `add_message("assistant")`):**

Each new user topic message updates a chunk and invalidates that session's cached retrieval index, so this measures full per-turn cost:

| Stored Chunks | p50 Turn | p95 Turn | Max Turn |
| :--- | :---: | :---: | :---: |
| **10 chunks** | 6.6 ms | 12.3 ms | 12.7 ms |
| **100 chunks** | 17.5 ms | 21.2 ms | 21.3 ms |
| **1,000 chunks** | 134.6 ms | 176.1 ms | 178.0 ms |

> **Known architectural trade-offs:**
> - The session retrieval cache accelerates repeated reads between writes, while alternating chat turns refit the retrieval index after each user write.
> - When `auto_summarize=False`, chunks reaching `MAX_CHUNK_SIZE` (8 messages) roll over into a new chunk for the same topic.
> - Full chunk lists are rewritten in Redis on save; granular Redis hash storage and optional Redis-free local mode are planned for a separate storage slice.

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

## Design Philosophy & Where This Fits

AgentMemory started from a practical failure mode: while running local coding agents (such as OpenClaw) with smaller-context models like `gpt-4.1-mini`, long multi-topic sessions and large workspace instructions quickly overflowed the context window or lost early facts.

Instead of reaching directly for an external vector database or a 500 MB PyTorch embedding model, this project explores **how far deterministic, CPU-only lexical and subword retrieval can go** when paired with write-time topic organization:

- **Write-time topic segmentation vs. flat message logs:** Rather than indexing every chat turn in isolation (where short replies like `"16379"` or `"yes"` lose meaning), messages are clustered into evolving topic chunks as the conversation happens, matching only against user turns so verbose assistant replies do not cause topic drift.
- **Role-split TF-IDF (Precision on write, Recall on read):** Word-level TF-IDF keeps unrelated topics from merging during chunk assignment, while stopword-filtered character n-grams (`char_wb` 4–5) recover morphological variants (`"name"` ↔ `"named"`) at retrieval time without leaking false-positive chunks.
- **When to use AgentMemory vs. Vector / Hybrid Systems:**
  - **Use AgentMemory** when you want a zero-GPU, dependency-light drop-in library or sidecar service for small/fast models (Groq, DeepSeek, `gpt-4o-mini`), deterministic recall of exact technical terms (ports, config keys, error codes), and sub-20 ms local CPU retrieval with zero per-query embedding cost.
  - **Use hybrid BM25 + dense embeddings (`pgvector`, `SQLite FTS5 + embeddings`, `mem0`)** when your workload requires zero-overlap semantic synonym matching (e.g., `"automobile"` ↔ `"car"`) or structured entity/contradiction tracking across thousands of sessions.

## License

MIT
