import os
import psycopg2
import psycopg2.extras
from datetime import datetime


class Database:
    def __init__(self):
        self.url = os.environ.get("DATABASE_URL", "")
        if not self.url:
            print("[database] WARNING: DATABASE_URL not set - using in-memory fallback (chat history will not persist)")
            self.url = None
            self._memory = {}
            return
        try:
            self._ensure_tables()
        except Exception as e:
            print(f"[database] WARNING: Failed to connect to DB {e} - using in-memory fallback")
            self.url = None
            self._memory = {}

    def _conn(self):
        if not self.url:
            raise RuntimeError("DATABASE_URL not set - in-memory mode")
        return psycopg2.connect(self.url)

    def _ensure_tables(self):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                        role VARCHAR(10) NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                conn.commit()
        finally:
            conn.close()

    def create_conversation(self) -> str:
        if not self.url:
            import uuid

            cid = str(uuid.uuid4())
            self._memory[cid] = []
            return cid
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO conversations DEFAULT VALUES RETURNING id;")
                conv_id = cur.fetchone()[0]
                conn.commit()
                return str(conv_id)
        finally:
            conn.close()

    def get_history(self, conversation_id: str, limit: int = 20) -> list[dict]:
        if not self.url:
            return self._memory.get(conversation_id, [])[-limit:]
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at ASC LIMIT %s",
                    (conversation_id, limit),
                )
                rows = cur.fetchall()
                return list(rows)
        finally:
            conn.close()

    def save_message(self, conversation_id: str, role: str, content: str):
        if not self.url:
            if conversation_id not in self._memory:
                self._memory[conversation_id] = []
            self._memory[conversation_id].append({"role": role, "content": content})
            return
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                    (conversation_id, role, content),
                )
                conn.commit()
        finally:
            conn.close()

    def delete_conversation(self, conversation_id: str):
        if not self.url:
            self._memory.pop(conversation_id, None)
            return
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
                conn.commit()
        finally:
            conn.close()
