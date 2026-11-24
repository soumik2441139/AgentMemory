import unittest
import time
from agentmemory import SimilarityMatcher
from agentmemory.memory import AgentMemory

class MockRedis:
    def __init__(self):
        self.data = {}
        self.lists = {}
        self.counters = {}

    def llen(self, key):
        return len(self.lists.get(key, []))

    def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        if end == -1:
            return items[start:]
        return items[start:end+1]

    def rpush(self, key, val):
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].append(val)
        return len(self.lists[key])

    def delete(self, *keys):
        for k in keys:
            self.lists.pop(k, None)
            self.data.pop(k, None)
            self.counters.pop(k, None)

    def incr(self, key):
        val = self.counters.get(key, 0) + 1
        self.counters[key] = val
        return val

    def set(self, key, val):
        self.data[key] = val

    def get(self, key):
        return self.data.get(key)


class TestChunkingLogic(unittest.TestCase):
    def setUp(self):
        # Instantiate AgentMemory with isolated SQLite and MockRedis
        self.mem = AgentMemory.__new__(AgentMemory)
        self.mem.r = MockRedis()
        self.mem.session_id = "test-session"
        self.mem.chunks_key = "session:test-session:chunks_v1"
        self.mem.chunk_counter_key = "session:test-session:counter_v1"
        self.mem.auto_summarize = False

        class DummyDB:
            def load_chunks(self): return []
            def save_chunk(self, chunk): pass
            def delete_session(self, s): pass
            def list_sessions(self): return []

        self.mem.db = DummyDB()

    def test_small_talk_goes_to_general(self):
        self.mem.add_message("user", "hi there")
        chunks = self.mem._get_all_chunks()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "general")
        self.assertEqual(chunks[0]["title"], "General Chat")
        self.assertEqual(chunks[0]["messages"][0]["content"], "hi there")

    def test_topic_creates_new_chunk(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        self.assertEqual(len(topic_chunks), 1)
        self.assertIn("redis", [k.lower() for k in topic_chunks[0]["keywords"]])

    def test_same_topic_updates_existing_chunk(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        self.mem.add_message("user", "How does Redis handle cluster replication?")
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        self.assertEqual(len(topic_chunks), 1, "Should merge into existing Redis chunk rather than duplicate")
        self.assertEqual(len(topic_chunks[0]["messages"]), 2)

    def test_new_topic_creates_second_chunk(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        self.mem.add_message("user", "What is the best way to train deep learning neural networks?")
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        self.assertEqual(len(topic_chunks), 2, "Distinct topics must form distinct chunks")

    def test_mixed_redis_and_transformer_message_does_not_merge_topics(self):
        self.mem.add_message(
            "user",
            "Redis pubsub caching replication persistence cluster sentinel streams "
            "transactions eviction and distributed locks are separate concerns.",
        )
        self.mem.add_message(
            "user",
            "Can Redis store transformer attention embeddings for my deep learning pipeline",
        )
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        self.assertEqual(len(topic_chunks), 2)

    def test_mixed_redis_and_sourdough_message_does_not_merge_topics(self):
        self.mem.add_message(
            "user",
            "Redis pubsub caching replication persistence cluster sentinel streams "
            "transactions eviction and distributed locks are separate concerns.",
        )
        self.mem.add_message(
            "user",
            "Tell me about sourdough bread starter hydration and Redis",
        )
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        self.assertEqual(len(topic_chunks), 2)

    def test_assistant_reply_routes_to_current_topic(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        self.mem.add_message("assistant", "Redis pubsub is an asynchronous messaging model.")
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        self.assertEqual(len(topic_chunks[0]["messages"]), 2)
        self.assertEqual(topic_chunks[0]["messages"][1]["role"], "assistant")

    def test_assistant_reply_routes_to_general_when_no_topics(self):
        self.mem.add_message("user", "Hello there!")
        self.mem.add_message("assistant", "Hello! How can I help you today?")
        chunks = self.mem._get_all_chunks()
        general_chunks = [c for c in chunks if c.get("type") == "general"]
        self.assertEqual(len(general_chunks), 1)
        self.assertEqual(len(general_chunks[0]["messages"]), 2)
        self.assertEqual(general_chunks[0]["messages"][1]["role"], "assistant")

    def test_assistant_reply_to_small_talk_does_not_pollute_previous_topic(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        self.mem.add_message("assistant", "Redis pubsub is an asynchronous messaging model.")
        # User follows up with small talk greeting/thanks
        self.mem.add_message("user", "thanks!")
        # Long assistant reply to small talk must go to General Chat, not Redis topic
        self.mem.add_message("assistant", "You are very welcome! Let me know if you need anything else today.")
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        general_chunks = [c for c in chunks if c.get("type") == "general"]
        self.assertEqual(len(topic_chunks[0]["messages"]), 2)
        self.assertEqual(len(general_chunks[0]["messages"]), 2)

    def test_short_assistant_reply_stays_in_active_topic(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        self.mem.add_message("assistant", "Sure!")
        chunks = self.mem._get_all_chunks()
        topic_chunks = [c for c in chunks if c.get("type") == "topic"]
        general_chunks = [c for c in chunks if c.get("type") == "general"]
        self.assertEqual(len(topic_chunks[0]["messages"]), 2)
        self.assertEqual(len(general_chunks), 0)

    def test_context_retrieval_prioritizes_matching_topic(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        self.mem.add_message("user", "What is the best way to train deep learning neural networks?")
        context = self.mem.get_context("How do I configure Redis memory limits?")
        contents = " ".join([m["content"] for m in context])
        self.assertIn("Redis", contents)

    def test_context_retrieval_excludes_unrelated_topics(self):
        self.mem.add_message("user", "Tell me about Redis pubsub and caching")
        self.mem.add_message("user", "How do transformer attention layers work in deep learning?")
        context = self.mem.get_context("Explain quantum entanglement in particle physics")
        self.assertEqual(context, [])

    def test_summarized_chunk_remains_retrievable_by_early_fact(self):
        self.mem.add_message("user", "Our staging Redis cluster runs on port 16379 with TLS.")
        self.mem.add_message("assistant", "Noted the staging Redis port.")
        # Simulate maybe_summarize() compressing early messages and leaving only
        # recent follow-ups as raw user messages.
        chunks = self.mem._get_all_chunks()
        topic_chunk = next(c for c in chunks if c.get("type") == "topic")
        topic_chunk["summarized"] = True
        topic_chunk["messages"] = [
            {
                "role": "system",
                "content": "[SUMMARY OF EARLIER CONVERSATION]: User's staging Redis cluster runs on port 16379 with TLS.",
                "timestamp": time.time(),
                "summarized": True,
            },
            {
                "role": "user",
                "content": "How should we monitor replication lag?",
                "timestamp": time.time(),
            },
            {
                "role": "assistant",
                "content": "Track master_repl_offset and replica_repl_offset.",
                "timestamp": time.time(),
            },
        ]
        self.mem._save_chunk(topic_chunk)

        context = self.mem.get_context("Which port does our staging Redis cluster use?")
        combined = " ".join(m["content"] for m in context)
        self.assertIn("16379", combined)


if __name__ == "__main__":
    unittest.main()
