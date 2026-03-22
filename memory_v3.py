import redis
import json
import time
import re
from nltk.corpus import stopwords
from similarity import SimilarityMatcher

STOPWORDS = set(stopwords.words('english'))
matcher = SimilarityMatcher(threshold=0.15)
MAX_CHUNK_SIZE = 8

class TopicMemory:
    def __init__(self, session_id):
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.session_id = session_id
        self.chunks_key = f"session:{session_id}:chunks_v3"
        self.chunk_counter_key = f"session:{session_id}:counter_v3"

    def _extract_keywords(self, text):
        text = re.sub(r'[*#@_`\[\](){}\"\'<>]', ' ', text)
        words = text.lower().split()
        keywords = [
            w.strip(".,?!;:")
            for w in words
            if w.strip(".,?!;:") not in STOPWORDS
            and len(w.strip(".,?!;:")) > 3
            and w.strip(".,?!;:").isalpha()
        ]
        return list(set(keywords))[:10]

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
        chunks = self._get_all_chunks()
        updated = False
        for i, c in enumerate(chunks):
            if c["id"] == chunk["id"]:
                chunks[i] = chunk
                updated = True
                break
        if not updated:
            chunks.append(chunk)
        self.r.delete(self.chunks_key)
        for c in chunks:
            self.r.rpush(self.chunks_key, json.dumps(c))

    def _generate_title(self, keywords):
        meaningful = [k for k in keywords if len(k) > 4][:3]
        if not meaningful:
            meaningful = keywords[:3]
        return " ".join(meaningful).title()

    def _get_user_only_text(self, chunk):
        # Only use USER messages for similarity matching
        # Prevents LLM responses from polluting chunk matching
        user_msgs = [
            m["content"] for m in chunk["messages"]
            if m["role"] == "user"
        ]
        return " ".join(user_msgs)

    def add_message(self, role, content):
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
            print(f"  💬 General chat")
            return

        # Only match against USER messages in chunks
        # Skip matching for assistant responses
        if role == "assistant":
            # Find most recently updated topic chunk
            chunks = self._get_all_chunks()
            topic_chunks = [c for c in chunks if c.get("type") == "topic"]
            if topic_chunks:
                latest = max(topic_chunks, key=lambda c: c["updated_at"])
                latest["messages"].append({
                    "role": role,
                    "content": content,
                    "timestamp": time.time()
                })
                latest["updated_at"] = time.time()
                self._save_chunk(latest)
                print(f"  🤖 Assistant → '{latest['title']}'")
            return

        # For user messages — use TF-IDF on USER-only text
        chunks = self._get_all_chunks()
        best_chunk = None
        best_score = 0

        for chunk in chunks:
            if chunk.get("type") == "general":
                continue
            # Skip chunks that are too large
            if len(chunk["messages"]) >= MAX_CHUNK_SIZE:
                continue
            chunk_text = self._get_user_only_text(chunk)
            if not chunk_text.strip():
                continue
            score = matcher.score(content, chunk_text)
            if score > 0.15 and score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk:
            best_chunk["messages"].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
            best_chunk["keywords"] = list(
                set(best_chunk["keywords"] + keywords)
            )[:15]
            best_chunk["updated_at"] = time.time()
            self._save_chunk(best_chunk)
            print(f"  📌 Updated: '{best_chunk['title']}' (similarity: {best_score:.3f})")
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
            print(f"  🆕 New topic: '{title}'")

    def get_context(self, current_message):
        chunks = self._get_all_chunks()
        context = []

        # Always include recent general chat
        for chunk in chunks:
            if chunk.get("type") == "general":
                for msg in chunk["messages"][-2:]:
                    context.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # Find relevant chunks using TF-IDF on USER messages only
        scored = []
        for chunk in chunks:
            if chunk.get("type") == "general":
                continue
            chunk_text = self._get_user_only_text(chunk)
            if not chunk_text.strip():
                continue
            score = matcher.score(current_message, chunk_text)
            if score > 0.1:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, chunk in scored[:2]:
            for msg in chunk["messages"][-4:]:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return context

    def clear(self):
        self.r.delete(self.chunks_key)
        self.r.delete(self.chunk_counter_key)
        print("🗑️ Memory cleared!")

    def stats(self):
        chunks = self._get_all_chunks()
        total_messages = sum(len(c["messages"]) for c in chunks)
        return {
            "session": self.session_id,
            "total_chunks": len(chunks),
            "total_messages": total_messages,
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
