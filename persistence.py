import sqlite3
import json
import time

class PersistenceLayer:
    def __init__(self, session_id, db_path="agentmemory.db"):
        self.session_id = session_id
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                title TEXT,
                type TEXT,
                keywords TEXT,
                messages TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def save_chunk(self, chunk):
        conn = sqlite3.connect(self.db_path)
        existing = conn.execute(
            "SELECT id FROM chunks WHERE session_id=? AND chunk_id=?",
            (self.session_id, chunk["id"])
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE chunks SET
                    title=?, type=?, keywords=?,
                    messages=?, updated_at=?
                WHERE session_id=? AND chunk_id=?
            """, (
                chunk["title"],
                chunk.get("type", "topic"),
                json.dumps(chunk["keywords"]),
                json.dumps(chunk["messages"]),
                chunk["updated_at"],
                self.session_id,
                chunk["id"]
            ))
        else:
            conn.execute("""
                INSERT INTO chunks
                (session_id, chunk_id, title, type, keywords, messages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                chunk["id"],
                chunk["title"],
                chunk.get("type", "topic"),
                json.dumps(chunk["keywords"]),
                json.dumps(chunk["messages"]),
                chunk["created_at"],
                chunk["updated_at"]
            ))

        conn.commit()
        conn.close()

    def load_chunks(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT chunk_id, title, type, keywords, messages, created_at, updated_at FROM chunks WHERE session_id=? ORDER BY chunk_id",
            (self.session_id,)
        ).fetchall()
        conn.close()

        chunks = []
        for row in rows:
            chunks.append({
                "id": row[0],
                "title": row[1],
                "type": row[2],
                "keywords": json.loads(row[3]),
                "messages": json.loads(row[4]),
                "created_at": row[5],
                "updated_at": row[6]
            })
        return chunks

    def delete_session(self, session_id):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM chunks WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
        print(f"🗑️ Session {session_id} deleted from SQLite")

    def list_sessions(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM chunks"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
