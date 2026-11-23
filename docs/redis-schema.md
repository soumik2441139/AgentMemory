# AgentMemory — Redis Data Model & Architecture Specification

This document details the Redis key schema, storage structures, serialization formats, TTL policies, and eviction behavior used by AgentMemory.

---

## 1. Key Schema & Types

| Key Pattern | Redis Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `session:{session_id}:chunks_v1` | `LIST` | Ordered list of JSON-serialized conversation chunks | `['{"id": 1, "title": "Redis", ...}', ...]` |
| `session:{session_id}:counter_v1` | `STRING` (Integer) | Atomic auto-incrementing integer sequence for chunk IDs | `"3"` |
| `session:{session_id}:hot` *(basic)* | `LIST` | Recent messages in hot FIFO window | `['{"role": "user", ...}']` |
| `session:{session_id}:cold` *(basic)* | `LIST` | Overflow messages evicted from hot tier | `['{"role": "user", ...}']` |

---

## 2. Chunk Data Structure (JSON)

Every entry in `session:{session_id}:chunks_v1` deserializes to the following JSON payload:

```json
{
  "id": 1,
  "title": "Redis Pubsub Caching",
  "type": "topic",
  "keywords": ["redis", "pubsub", "caching", "cluster"],
  "messages": [
    {
      "role": "user",
      "content": "Tell me about Redis pubsub and caching",
      "timestamp": 1700000000.123
    },
    {
      "role": "assistant",
      "content": "Redis pub/sub is an asynchronous message broker model...",
      "timestamp": 1700000001.456
    }
  ],
  "created_at": 1700000000.123,
  "updated_at": 1700000001.456,
  "summarized": false
}
```

### Chunk Types
- **`general`**: Small-talk or introductory pleasantries (e.g. "hi", "how are you"). Isolated so greetings do not dilute domain topic similarity.
- **`topic`**: Clustered domain conversation (e.g. "Database", "Kubernetes", "Recipes").

---

## 3. Eviction & TTL Policies

1. **In-Chunk Message Cap**:
   - Default `MAX_CHUNK_SIZE = 8` messages per topic chunk.
   - Once a chunk reaches 6 messages, `maybe_summarize()` condenses earlier messages into a compact system summary: `[SUMMARY OF EARLIER CONVERSATION]: ...` preserving the 2 most recent messages for conversational flow.
2. **Session Expiry (Optional TTL)**:
   - Sessions can have an active TTL set via `r.expire(chunks_key, ttl_seconds)` to auto-reclaim memory for idle sessions.
3. **Redis Maxmemory Eviction**:
   - Recommended Redis configuration:
     ```conf
     maxmemory 512mb
     maxmemory-policy volatile-lru
     ```

---

## 4. Concurrency & Atomicity

In multi-threaded or multi-worker environments:
- Reading chunks via `LRANGE`, modifying in Python, and rewriting via `DELETE` + `RPUSH` presents a classic Time-of-Check to Time-of-Use (TOCTOU) race condition.
- **Mitigation**:
  1. Use Redis `WATCH` / `MULTI` transaction pipeline to ensure chunk modification fails and retries if modified by a concurrent worker.
  2. Or manage chunks using Redis Hashes (`HSET session:{id}:chunks {chunk_id} {json}`) for $O(1)$ individual chunk updates rather than rewriting the entire list.
