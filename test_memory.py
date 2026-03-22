import unittest
import time
from memory_v1 import AgentMemory as BasicMemory
from similarity import SimilarityMatcher
from persistence import PersistenceLayer
from summarizer import maybe_summarize

class TestBasicMemory(unittest.TestCase):

    def setUp(self):
        self.memory = BasicMemory("test-unit", hot_limit=5)
        self.memory.clear()

    def test_add_message(self):
        self.memory.add_message("user", "hello")
        context = self.memory.get_context()
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["role"], "user")
        self.assertEqual(context[0]["content"], "hello")

    def test_hot_cold_eviction(self):
        for i in range(7):
            self.memory.add_message("user", f"message {i}")
        stats = self.memory.stats()
        self.assertEqual(stats["hot_messages"], 5)
        self.assertEqual(stats["cold_messages"], 2)

    def test_context_strips_timestamp(self):
        self.memory.add_message("user", "test message")
        context = self.memory.get_context()
        for msg in context:
            self.assertNotIn("timestamp", msg)
            self.assertIn("role", msg)
            self.assertIn("content", msg)


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.matcher = SimilarityMatcher(threshold=0.1)

    def test_same_topic_matches(self):
        score = self.matcher.score(
            "tell me about Redis databases",
            "Redis is great for caching"
        )
        self.assertGreater(score, 0.1)

    def test_different_topic_no_match(self):
        score = self.matcher.score(
            "tell me about Redis",
            "what is the weather today"
        )
        self.assertLess(score, 0.1)

    def test_empty_string(self):
        score = self.matcher.score("", "some text")
        self.assertEqual(score, 0.0)

    def test_find_best_match(self):
        chunks = [
            {
                "id": 1,
                "title": "Redis",
                "type": "topic",
                "messages": [
                    {"role": "user", "content": "tell me about Redis databases"}
                ]
            },
            {
                "id": 2,
                "title": "Weather",
                "type": "topic",
                "messages": [
                    {"role": "user", "content": "what is the weather today"}
                ]
            }
        ]
        best, score = self.matcher.find_best_match("Redis caching", chunks)
        self.assertIsNotNone(best)
        self.assertEqual(best["title"], "Redis")


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.db = PersistenceLayer("test-persist", db_path="test.db")
        self.db.delete_session("test-persist")

    def test_save_and_load(self):
        chunk = {
            "id": 1,
            "title": "Test Chunk",
            "type": "topic",
            "keywords": ["test", "redis"],
            "messages": [{"role": "user", "content": "hello", "timestamp": 1234}],
            "created_at": 1234.0,
            "updated_at": 1234.0
        }
        self.db.save_chunk(chunk)
        loaded = self.db.load_chunks()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "Test Chunk")
        self.assertEqual(loaded[0]["keywords"], ["test", "redis"])

    def test_update_chunk(self):
        chunk = {
            "id": 1,
            "title": "Original",
            "type": "topic",
            "keywords": [],
            "messages": [],
            "created_at": 1234.0,
            "updated_at": 1234.0
        }
        self.db.save_chunk(chunk)
        chunk["title"] = "Updated"
        self.db.save_chunk(chunk)
        loaded = self.db.load_chunks()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "Updated")

    def test_list_sessions(self):
        chunk = {
            "id": 1, "title": "T", "type": "topic",
            "keywords": [], "messages": [],
            "created_at": 1.0, "updated_at": 1.0
        }
        self.db.save_chunk(chunk)
        sessions = self.db.list_sessions()
        self.assertIn("test-persist", sessions)


class TestSummarizer(unittest.TestCase):

    def test_no_summarize_small_chunk(self):
        chunk = {
            "id": 1, "title": "Test", "type": "topic",
            "summarized": False,
            "messages": [
                {"role": "user", "content": "hello", "timestamp": 1.0}
            ]
        }
        result = maybe_summarize(chunk)
        self.assertEqual(len(result["messages"]), 1)

    def test_summarize_large_chunk(self):
        chunk = {
            "id": 1, "title": "Python", "type": "topic",
            "summarized": False,
            "messages": [
                {"role": "user", "content": f"python question {i}", "timestamp": float(i)}
                for i in range(7)
            ]
        }
        result = maybe_summarize(chunk)
        self.assertLess(len(result["messages"]), 7)
        self.assertEqual(result["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main(verbosity=2)
