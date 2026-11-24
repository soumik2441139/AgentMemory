import unittest
from unittest.mock import patch, MagicMock
import time
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redis
from agentmemory import AgentMemory, BasicMemory, SimilarityMatcher, PersistenceLayer, maybe_summarize


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

    def test_scores_matches_individual_scoring(self):
        documents = ["Redis caching and replication", "weather forecast today", ""]
        scores = self.matcher.scores("How does Redis replication work?", documents)
        self.assertEqual(len(scores), len(documents))
        self.assertGreater(scores[0], 0.1)
        self.assertEqual(scores[2], 0.0)

    def test_subword_tfidf_matches_related_word_forms(self):
        # Use the production retrieval configuration from agentmemory.memory.
        from agentmemory.memory import retrieval_matcher
        matcher = SimilarityMatcher(**retrieval_matcher.vectorizer_options)
        score = matcher.score(
            "What is my dog's name and favorite treat?",
            "My golden retriever is named Barnaby and he loves peanut butter.",
        )
        self.assertGreater(score, 0.0)

    def test_cached_scores_reuse_index_until_documents_change(self):
        documents = ["Redis caching and replication", "weather forecast today"]
        first = self.matcher.cached_scores("session-1", "How does Redis replication work?", documents)
        cached_index = self.matcher._indexes["session-1"]
        second = self.matcher.cached_scores("session-1", "How does Redis caching work?", documents)

        self.assertGreater(first[0], 0.1)
        self.assertGreater(second[0], 0.1)
        self.assertIs(self.matcher._indexes["session-1"], cached_index)

        self.matcher.cached_scores(
            "session-1",
            "How does Redis replication work?",
            documents + ["Python typing annotations"],
        )
        self.assertIsNot(self.matcher._indexes["session-1"], cached_index)

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
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = PersistenceLayer(
            "test-persist", db_path=os.path.join(self.temp_dir.name, "test.db")
        )
        self.db.delete_session("test-persist")

    def tearDown(self):
        self.temp_dir.cleanup()

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

    @patch("agentmemory.summarizer._get_summarizer_client")
    def test_summarize_large_chunk(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Concise summary of python discussion."))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = (mock_client, "gpt-4o-mini")

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
        self.assertIn("Concise summary", result["messages"][0]["content"])


class TestRedisOptionalFallback(unittest.TestCase):

    def test_agentmemory_works_EndToEnd_without_redis_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "fallback.db")
            # Port 1 is guaranteed unreachable so the in-memory fallback engages.
            memory = AgentMemory("session-fallback", db_path=db_path, host="127.0.0.1", port=1)
            memory.clear()

            memory.add_message("user", "Tell me about Redis pub/sub")
            memory.add_message("assistant", "Redis pub/sub is a messaging pattern...")

            context = memory.get_context("How does Redis pub/sub handle failures?")
            self.assertTrue(any("Redis pub/sub" in m["content"] for m in context))

            stats = memory.stats()
            self.assertEqual(stats["session"], "session-fallback")
            self.assertGreaterEqual(stats["total_chunks"], 1)
            self.assertEqual(stats["total_messages"], 2)

            memory.clear()
            self.assertEqual(memory.stats()["total_chunks"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
