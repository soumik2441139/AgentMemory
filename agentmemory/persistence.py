import sqlite3
import json
import time

class PersistenceLayer:
    def __init__(self, session_id, db_path="agentmemory.db"):
        self.session_id = session_id
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            with conn:
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
                        updated_at REAL,
                        UNIQUE(session_id, chunk_id)
                    )
                """)
                # Upgrade databases created before the uniqueness constraint.
                try:
                    conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "idx_chunks_session_chunk ON chunks(session_id, chunk_id)"
                    )
                except sqlite3.IntegrityError:
                    conn.execute("""
                        DELETE FROM chunks
                        WHERE id NOT IN (
                            SELECT MAX(id) FROM chunks GROUP BY session_id, chunk_id
                        )
                    """)
                    conn.execute(
                        "CREATE UNIQUE INDEX idx_chunks_session_chunk "
                        "ON chunks(session_id, chunk_id)"
                    )
        finally:
            conn.close()

    def save_chunk(self, chunk):
        conn = self._connect()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO chunks
                    (session_id, chunk_id, title, type, keywords, messages, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, chunk_id) DO UPDATE SET
                        title=excluded.title,
                        type=excluded.type,
                        keywords=excluded.keywords,
                        messages=excluded.messages,
                        updated_at=excluded.updated_at
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
        finally:
            conn.close()

    def load_chunks(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT chunk_id, title, type, keywords, messages, created_at, updated_at FROM chunks WHERE session_id=? ORDER BY chunk_id",
                (self.session_id,)
            ).fetchall()
        finally:
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
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM chunks WHERE session_id=?", (session_id,))
        finally:
            conn.close()
        print(f"Session {session_id} deleted from SQLite")

    def list_sessions(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM chunks"
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]
