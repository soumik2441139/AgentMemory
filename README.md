# AgentMemory 🧠

Lightweight AI agent memory management using Redis. Solves the context overflow problem by organizing conversations into intelligent topic-based chunks instead of flat message lists.

## The Problem

Every LLM has a context window limit. When conversations get long:
- Important early context gets lost
- Agents forget what was discussed
- Performance degrades or crashes entirely

Most solutions just truncate. AgentMemory organizes.

## How It Works
```
Message arrives
      ↓
Extract keywords (NLTK)
      ↓
Small talk? → General Chat chunk
      ↓
Matches existing topic? → Update that chunk
      ↓
New topic? → Create new chunk
      ↓
LLM gets only relevant chunks as context
```

## Versions

### v0.1 - Basic Hot/Cold Memory
- Two tier Redis storage
- Hot tier: recent N messages
- Cold tier: overflow storage
- Simple context injection

### v0.2 - Topic Memory (current)
- NLTK-powered keyword extraction
- Automatic topic detection and clustering
- Dynamic chunk merging — same topic discussed again updates existing chunk
- General chat detection — greetings don't pollute topic chunks
- Relevance-based context retrieval

### v0.3 - Coming Soon
- TF-IDF cosine similarity for smarter chunk matching
- LLM-powered chunk summarization
- SQLite persistence across Redis restarts

## Installation
```bash
git clone https://github.com/soumik2441139/AgentMemory.git
cd AgentMemory
python3 -m venv venv
source venv/bin/activate
pip install redis openai python-dotenv nltk
python3 -c "import nltk; nltk.download('stopwords')"
```

## Setup

Create `.env` file:
```
GROQ_API_KEY=your_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
```

Make sure Redis is running:
```bash
sudo service redis-server start
redis-cli ping  # should return PONG
```

## Usage

### v0.1 - Basic Memory
```python
from memory import AgentMemory

memory = AgentMemory("session-1", hot_limit=20)
memory.add_message("user", "hello")
context = memory.get_context()
memory.stats()
```

### v0.2 - Topic Memory
```python
from topic_memory import TopicMemory

memory = TopicMemory("session-1")
memory.add_message("user", "tell me about Redis")
context = memory.get_context("Redis question")
memory.stats()
```

Run the agent:
```bash
# v0.1
python3 agent.py

# v0.2
python3 agent_v2.py
```

Commands inside chat:
- `stats` — see memory chunks
- `quit` — exit

## Tech Stack

- Python 3.x
- Redis — hot memory storage
- NLTK — keyword extraction and stopwords
- Groq API — LLM (OpenAI compatible)
- Any OpenAI-compatible API works

## Why Not LangChain?

LangChain memory requires the entire LangChain framework. AgentMemory is a standalone drop-in library — works with any LLM API in 5 lines of code.

## Roadmap

- [ ] TF-IDF cosine similarity (v0.3)
- [ ] LLM summarization of old chunks (v0.3)  
- [ ] SQLite persistence (v0.3)
- [ ] pip package (v1.0)
- [ ] Multi-language support (v1.0)


