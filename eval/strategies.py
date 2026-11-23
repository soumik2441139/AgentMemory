"""
Memory retrieval strategies for benchmark comparison.
"""

import os
import sys
import re

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentmemory.memory import AgentMemory
from agentmemory.similarity import SimilarityMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MockMemoryDB:
    def load_chunks(self): return []
    def save_chunk(self, chunk): pass
    def delete_session(self, s): pass
    def list_sessions(self): return []

class MockRedisStorage:
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


class LastNStrategy:
    """Baseline: Returns the last N messages of the conversation."""
    def __init__(self, n=10):
        self.n = n

    def retrieve(self, conversation_history, query):
        return conversation_history[-self.n:]


class KeywordChunkStrategy:
    """v0.2 style: Extracts keywords via overlap and clusters chunks."""
    def __init__(self, chunk_size=6):
        self.chunk_size = chunk_size

    def retrieve(self, conversation_history, query):
        query_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        # Chunk messages into blocks
        chunks = []
        for i in range(0, len(conversation_history), self.chunk_size):
            chunks.append(conversation_history[i:i+self.chunk_size])

        scored = []
        for chunk in chunks:
            text = " ".join([m["content"] for m in chunk]).lower()
            chunk_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', text))
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((overlap, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for _, chunk in scored[:2]:
            result.extend(chunk)
        return result


class TFIDFChunkStrategy:
    """v1.1 AgentMemory: Topic-aware chunking with user-only TF-IDF matching."""
    def __init__(self):
        pass

    def retrieve(self, conversation_history, query):
        mem = AgentMemory.__new__(AgentMemory)
        mem.r = MockRedisStorage()
        mem.session_id = "eval"
        mem.chunks_key = "session:eval:chunks"
        mem.chunk_counter_key = "session:eval:counter"
        mem.db = MockMemoryDB()
        mem.auto_summarize = False

        for msg in conversation_history:
            mem.add_message(msg["role"], msg["content"])

        return mem.get_context(query)


class DenseEmbeddingStrategy:
    """
    Embedding baseline: character/word n-gram TF-IDF subword dense vectorizer
    simulating dense semantic retrieval without requiring 500MB PyTorch install.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb")

    def retrieve(self, conversation_history, query):
        mem = AgentMemory.__new__(AgentMemory)
        mem.r = MockRedisStorage()
        mem.session_id = "eval-dense"
        mem.chunks_key = "session:eval-dense:chunks"
        mem.chunk_counter_key = "session:eval-dense:counter"
        mem.db = MockMemoryDB()
        mem.auto_summarize = False

        for msg in conversation_history:
            mem.add_message(msg["role"], msg["content"])

        chunks = mem._get_all_chunks()
        scored = []
        for chunk in chunks:
            user_text = mem._get_user_only_text(chunk)
            if not user_text.strip():
                continue
            vectors = self.vectorizer.fit_transform([query, user_text])
            score = cosine_similarity(vectors[0], vectors[1])[0][0]
            if score > 0.05:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        context = []
        for _, chunk in scored[:2]:
            for msg in chunk["messages"][-4:]:
                context.append({"role": msg["role"], "content": msg["content"]})
        return context
