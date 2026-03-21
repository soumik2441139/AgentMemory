# AgentMemory v0.2 - Topic Memory Design

## Problem
Current v0.1 stores messages in a flat Redis list.
When conversations get long, LLM loses early context.
Cold tier exists but is never retrieved.

## Solution - Topic-Aware Chunking

### Core Idea
- Group related messages into "chunks" by topic
- Each chunk has: id, title, summary, keywords, messages
- When new message arrives - check if it belongs to existing chunk
- If yes - update that chunk (don't create duplicate)
- If no - create new chunk with new topic

### Data Structure
chunk = {
    "id": 1,
    "title": "topic name",
    "summary": "compressed summary of discussion",
    "keywords": ["keyword1", "keyword2"],
    "messages": [...],
    "created_at": timestamp,
    "updated_at": timestamp
}

### Retrieval
- When LLM needs context
- Find chunks whose keywords match current message
- Inject relevant chunk summaries + recent messages
- LLM gets focused context without token explosion

## Why This Works
- Same topic discussed again = hits existing chunk
- New topic = new chunk created
- LLM always gets relevant context, not just recent context
