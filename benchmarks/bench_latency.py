"""
Latency benchmarks measuring get_context() and add_message() execution time
across varying numbers of stored chunks (10, 100, 1000).
"""

import time
import statistics
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentmemory.memory import AgentMemory

class MockStorage:
    def __init__(self):
        self.lists = {}
        self.counters = {}
    def llen(self, k): return len(self.lists.get(k, []))
    def lrange(self, k, s, e):
        items = self.lists.get(k, [])
        return items[s:] if e == -1 else items[s:e+1]
    def rpush(self, k, v):
        self.lists.setdefault(k, []).append(v)
        return len(self.lists[k])
    def delete(self, *keys):
        for k in keys: self.lists.pop(k, None)
    def incr(self, k):
        self.counters[k] = self.counters.get(k, 0) + 1
        return self.counters[k]

class DummyDB:
    def load_chunks(self): return []
    def save_chunk(self, chunk): pass
    def delete_session(self, s): pass
    def list_sessions(self): return []


def setup_memory_with_chunks(n_chunks):
    mem = AgentMemory.__new__(AgentMemory)
    mem.r = MockStorage()
    mem.session_id = f"bench-{n_chunks}"
    mem.chunks_key = f"session:bench-{n_chunks}:chunks"
    mem.chunk_counter_key = f"session:bench-{n_chunks}:counter"
    mem.db = DummyDB()
    mem.auto_summarize = False

    topics = [
        "Kubernetes container orchestration and pod scaling",
        "PostgreSQL relational database indexing and replication",
        "Machine learning transformer attention layers",
        "Redis in-memory caching and distributed lock mechanisms",
        "React frontend components state management hooks",
        "Rust ownership borrowing lifetimes concurrency",
        "Docker multi-stage builds image optimization",
        "Kafka distributed event streaming partition rebalancing",
        "FastAPI asynchronous web framework serialization",
        "Linux kernel system calls memory management"
    ]

    chunks = []
    for i in range(n_chunks):
        topic = topics[i % len(topics)]
        chunk = {
            "id": i + 1,
            "title": topic.split()[0].title(),
            "type": "topic",
            "keywords": topic.lower().split()[:5],
            "messages": [
                {"role": "user", "content": f"Discussion #{i} about {topic} with specific architecture details.", "timestamp": time.time()},
                {"role": "assistant", "content": f"Acknowledged topic {topic} for session index {i}.", "timestamp": time.time()}
            ],
            "created_at": time.time(),
            "updated_at": time.time(),
            "summarized": False
        }
        mem.r.rpush(mem.chunks_key, json.dumps(chunk))

    return mem


def run_benchmarks():
    scales = [10, 50, 100, 500, 1000]
    queries = [
        "How do I configure Redis cluster replication?",
        "What is the best way to optimize PostgreSQL indices?",
        "Explain transformer self-attention mechanism in neural networks."
    ]

    print("=" * 65)
    print("⚡ AGENTMEMORY RETRIEVAL LATENCY BENCHMARKS")
    print("=" * 65)
    print(f"{'Chunks Stored':<15} | {'p50 Latency':<14} | {'p95 Latency':<14} | {'Max Latency'}")
    print("-" * 65)

    for n in scales:
        mem = setup_memory_with_chunks(n)
        timings = []

        # Run 5 query retrievals to measure distribution
        for _ in range(5):
            for q in queries:
                t0 = time.perf_counter()
                _ = mem.get_context(q)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)  # milliseconds

        p50 = statistics.median(timings)
        p95 = statistics.quantiles(timings, n=10)[8] if len(timings) >= 10 else max(timings)
        max_lat = max(timings)

        print(f"{n:<15} | {p50:>7.2f} ms     | {p95:>7.2f} ms     | {max_lat:>7.2f} ms", flush=True)

    print("=" * 65)


if __name__ == "__main__":
    run_benchmarks()
