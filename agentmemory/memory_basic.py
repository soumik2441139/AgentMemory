import os
import json
import time
from dotenv import load_dotenv
from .memory import connect_redis_or_fallback

load_dotenv()

class AgentMemory:
    def __init__(self, session_id, hot_limit=20, host=None, port=None):
        redis_host = host or os.getenv("REDIS_HOST", "localhost")
        redis_port = int(port or os.getenv("REDIS_PORT", 6379))
        self.r = connect_redis_or_fallback(redis_host, redis_port)
        self.session_id = session_id
        self.hot_limit = hot_limit
        self.hot_key = f"session:{session_id}:hot"
        self.cold_key = f"session:{session_id}:cold"

    def add_message(self, role, content):
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.r.rpush(self.hot_key, message)
        if self.r.llen(self.hot_key) > self.hot_limit:
            oldest = self.r.lpop(self.hot_key)
            self.r.rpush(self.cold_key, oldest)

    def get_context(self):
        messages = self.r.lrange(self.hot_key, 0, -1)
        parsed = [json.loads(m) for m in messages]
        return [{"role": m["role"], "content": m["content"]} for m in parsed]

    def clear(self):
        self.r.delete(self.hot_key)
        self.r.delete(self.cold_key)

    def stats(self):
        return {
            "session": self.session_id,
            "hot_messages": self.r.llen(self.hot_key),
            "cold_messages": self.r.llen(self.cold_key)
        }
