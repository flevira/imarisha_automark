import sqlite3
from pathlib import Path

DB_PATH = Path("imarisha_automark.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS extractions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file TEXT,
        page INTEGER,
        student_id TEXT,
        student_id_all TEXT,
        student_id_qr TEXT,
        student_id_text TEXT,
        student_id_handwritten TEXT,
        student_id_source TEXT,
        id_conflict INTEGER,
        sheet_type TEXT,
        exam_id TEXT,
        test_id TEXT,
        section TEXT,
        question_id TEXT,
        item_number INTEGER,
        answer TEXT,
        matching_pair TEXT,
        manual_marking INTEGER,
        needs_review INTEGER,
        status TEXT DEFAULT 'pending',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
