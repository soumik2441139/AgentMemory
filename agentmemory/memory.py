import os
import redis
import json
import time
import re
from contextlib import nullcontext
from dotenv import load_dotenv

load_dotenv()

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    }

from .similarity import SimilarityMatcher
from .persistence import PersistenceLayer
from .summarizer import maybe_summarize

matcher = SimilarityMatcher(threshold=0.12)
retrieval_matcher = SimilarityMatcher(
    analyzer="char_wb",
    ngram_range=(4, 5),
)
MAX_CHUNK_SIZE = 8
# Four-character n-grams retain useful variants such as ``name`` / ``named``
# while dropping many accidental three-character overlaps.
RETRIEVAL_THRESHOLD = 0.15

class AgentMemory:
    def __init__(self, session_id, db_path=None, host=None, port=None, auto_summarize=False):
        redis_host = host or os.getenv("REDIS_HOST", "localhost")
        redis_port = int(port or os.getenv("REDIS_PORT", 6379))
        resolved_db_path = db_path or os.getenv("AGENTMEMORY_DB_PATH", "agentmemory.db")
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.session_id = session_id
        self.chunks_key = f"session:{session_id}:chunks_v1"
        self.chunk_counter_key = f"session:{session_id}:counter_v1"
        self.last_chunk_key = f"session:{session_id}:last_chunk_v1"
        self.lock_key = f"session:{session_id}:lock_v1"
        self.auto_summarize = auto_summarize
        self.db = PersistenceLayer(session_id, resolved_db_path)

        # On startup load from SQLite into Redis
        with self._session_lock():
            self._restore_from_db()

    def _session_lock(self):
        """Serialize mutations for a session across API workers."""
        if not hasattr(self.r, "lock"):
            return nullcontext()
        return self.r.lock(self.lock_key, timeout=60, blocking_timeout=10)

    def _restore_from_db(self):
        # If Redis is empty but SQLite has data — restore it
        if self.r.llen(self.chunks_key) == 0:
            chunks = self.db.load_chunks()
            if chunks:
                for chunk in chunks:
                    self.r.rpush(self.chunks_key, json.dumps(chunk))
                # Restore counter
                max_id = max(c["id"] for c in chunks)
                self.r.set(self.chunk_counter_key, max_id)
                print(f"Restored {len(chunks)} chunks from SQLite")

    def _extract_keywords(self, text):
        text = re.sub(r'[*#@_`\[\](){}\"\'<>]', ' ', text)
        words = text.lower().split()
        seen = set()
        keywords = []
        for w in words:
            cleaned = w.strip(".,?!;:")
            if (
                cleaned not in STOPWORDS
                and len(cleaned) > 3
                and cleaned.isalpha()
                and cleaned not in seen
            ):
                seen.add(cleaned)
                keywords.append(cleaned)
                if len(keywords) >= 10:
                    break
        return keywords

    def _is_small_talk(self, text, keywords):
        return len(text.split()) < 4 or len(keywords) == 0

    def _find_or_create_general(self):
        chunks = self._get_all_chunks()
        for chunk in chunks:
            if chunk.get("type") == "general":
                return chunk
        counter = self.r.incr(self.chunk_counter_key)
        return {
            "id": int(counter),
            "title": "General Chat",
            "type": "general",
            "keywords": [],
            "messages": [],
            "created_at": time.time(),
            "updated_at": time.time()
        }

    def _get_all_chunks(self):
        raw = self.r.lrange(self.chunks_key, 0, -1)
        return [json.loads(c) for c in raw]

    def _save_chunk(self, chunk):
        if getattr(self, "auto_summarize", False) and chunk.get("type") == "topic":
            chunk = maybe_summarize(chunk)
        
        chunks = self._get_all_chunks()
        updated = False
        for i, c in enumerate(chunks):
            if c["id"] == chunk["id"]:
                chunks[i] = chunk
                updated = True
                break
        if not updated:
            chunks.append(chunk)

        serialized = [json.dumps(c) for c in chunks]
        if hasattr(self.r, "pipeline"):
            pipe = self.r.pipeline(transaction=True)
            pipe.delete(self.chunks_key)
            for item in serialized:
                pipe.rpush(self.chunks_key, item)
            pipe.execute()
        else:
            self.r.delete(self.chunks_key)
            for item in serialized:
                self.r.rpush(self.chunks_key, item)
        # Also persist to SQLite
        self.db.save_chunk(chunk)

    def _generate_title(self, keywords):
        meaningful = [k for k in keywords if len(k) > 4][:3]
        if not meaningful:
            meaningful = keywords[:3]
        return " ".join(meaningful).title()

    def _get_user_only_text(self, chunk):
        user_msgs = [
            m["content"] for m in chunk["messages"]
            if m["role"] == "user"
        ]
        # When a chunk has been summarized, older user messages were compressed
        # into a summary message. Include the chunk's preserved user keywords
        # and summary text so summarized facts remain discoverable.
        if chunk.get("summarized"):
            keywords = " ".join(chunk.get("keywords", []))
            summaries = " ".join(
                m["content"].replace("[SUMMARY OF EARLIER CONVERSATION]:", "").strip()
                for m in chunk["messages"]
                if m.get("summarized")
            )
            return " ".join(part for part in [keywords, summaries] + user_msgs if part)
        return " ".join(user_msgs)

    def _strip_stopwords(self, text):
        return " ".join(
            w for w in text.lower().split()
            if w.strip(".,?!;:\"'()[]{}") not in STOPWORDS
        )

    def add_message(self, role, content):
        with self._session_lock():
            self._add_message(role, content)

    def _set_last_chunk_id(self, chunk_id):
        self._last_chunk_id = chunk_id
        if hasattr(self, "last_chunk_key") and hasattr(self.r, "set"):
            self.r.set(self.last_chunk_key, chunk_id)

    def _get_last_chunk_id(self):
        if hasattr(self, "last_chunk_key") and hasattr(self.r, "get"):
            val = self.r.get(self.last_chunk_key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return getattr(self, "_last_chunk_id", None)

    def _add_message(self, role, content):
        if role == "assistant":
            chunks = self._get_all_chunks()
            last_id = self._get_last_chunk_id()
            target_chunk = None
            if last_id is not None:
                for c in chunks:
                    if c["id"] == last_id:
                        target_chunk = c
                        break
            if target_chunk is None and chunks:
                target_chunk = max(chunks, key=lambda c: c.get("updated_at", 0))
            if target_chunk is None:
                target_chunk = self._find_or_create_general()

            target_name = target_chunk.get("title", "General Chat")
            target_chunk["messages"].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
            target_chunk["updated_at"] = time.time()
            self._save_chunk(target_chunk)
            print(f"  Assistant -> '{target_name}'")
            return

        keywords = self._extract_keywords(content)

        if self._is_small_talk(content, keywords):
            general = self._find_or_create_general()
            general["messages"].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
            general["updated_at"] = time.time()
            self._save_chunk(general)
            self._set_last_chunk_id(general["id"])
            print("  General chat")
            return

        chunks = self._get_all_chunks()
        candidates = []
        candidate_texts = []
        for chunk in chunks:
            if chunk.get("type") == "general":
                continue
            if len(chunk["messages"]) >= MAX_CHUNK_SIZE:
                continue
            chunk_text = self._get_user_only_text(chunk)
            if not chunk_text.strip():
                continue
            candidates.append(chunk)
            candidate_texts.append(chunk_text)

        best_chunk = None
        best_score = 0
        for chunk, score in zip(
            candidates,
            # Assignment compares a new message with a threshold.  Fit the
            # vectorizer with the message as well as the candidate chunks so
            # this retains the v1.1 score scale used to tune that threshold.
            matcher.scores(content, candidate_texts),
        ):
            if score > matcher.threshold and score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk:
            best_chunk["messages"].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
            merged_keywords = list(best_chunk["keywords"])
            for kw in keywords:
                if kw not in merged_keywords:
                    merged_keywords.append(kw)
            best_chunk["keywords"] = merged_keywords[:15]
            best_chunk["updated_at"] = time.time()
            self._save_chunk(best_chunk)
            self._set_last_chunk_id(best_chunk["id"])
            print(f"  Updated: '{best_chunk['title']}' (similarity: {best_score:.3f})")
        else:
            counter = self.r.incr(self.chunk_counter_key)
            title = self._generate_title(keywords)
            new_chunk = {
                "id": int(counter),
                "title": title,
                "type": "topic",
                "keywords": keywords,
                "messages": [{
                    "role": role,
                    "content": content,
                    "timestamp": time.time()
                }],
                "created_at": time.time(),
                "updated_at": time.time()
            }
            self._save_chunk(new_chunk)
            self._set_last_chunk_id(new_chunk["id"])
            print(f"  New topic: '{title}'")

    def get_context(self, current_message):
        chunks = self._get_all_chunks()
        context = []

        for chunk in chunks:
            if chunk.get("type") == "general":
                for msg in chunk["messages"][-2:]:
                    context.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        candidates = []
        candidate_texts = []
        for chunk in chunks:
            if chunk.get("type") == "general":
                continue
            chunk_text = self._strip_stopwords(self._get_user_only_text(chunk))
            if not chunk_text.strip():
                continue
            candidates.append(chunk)
            candidate_texts.append(chunk_text)

        scored = []
        query_text = self._strip_stopwords(current_message)
        for chunk, score in zip(
            candidates,
            retrieval_matcher.cached_scores(self.session_id, query_text, candidate_texts),
        ):
            if score > RETRIEVAL_THRESHOLD:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, chunk in scored[:2]:
            msgs = chunk["messages"]
            selected = msgs[-4:]
            if len(msgs) > 4 and msgs[0].get("summarized"):
                selected = [msgs[0]] + selected
            for msg in selected:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return context

    def clear(self):
        with self._session_lock():
            self.r.delete(self.chunks_key)
            self.r.delete(self.chunk_counter_key)
            if hasattr(self, "last_chunk_key"):
                self.r.delete(self.last_chunk_key)
            self._last_chunk_id = None
            self.db.delete_session(self.session_id)
            retrieval_matcher.clear_cached_index(self.session_id)
        print("Memory cleared from Redis and SQLite.")

    def stats(self):
        chunks = self._get_all_chunks()
        total_messages = sum(len(c["messages"]) for c in chunks)
        sessions = self.db.list_sessions()
        return {
            "session": self.session_id,
            "total_chunks": len(chunks),
            "total_messages": total_messages,
            "persisted_sessions": sessions,
            "chunks": [
                {
                    "id": c["id"],
                    "title": c["title"],
                    "type": c.get("type", "topic"),
                    "messages": len(c["messages"]),
                    "keywords": c["keywords"][:5]
                }
                for c in chunks
            ]
        }
