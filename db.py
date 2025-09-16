import sqlite3
from datetime import datetime
from typing import Optional
from .embeddings import bytes_to_emb
import numpy as np

DB = 'faces.db'


def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        embedding BLOB,
        image BLOB,
        created_at TEXT
    )
    ''')
    conn.commit()
    conn.close()


def insert_face(name: str, emb_blob: bytes, image_bytes: bytes):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute('INSERT INTO faces (name, embedding, image, created_at) VALUES (?, ?, ?, ?)',
              (name, emb_blob, image_bytes, now))
    conn.commit()
    rowid = c.lastrowid
    conn.close()
    return rowid


def find_best_match(emb: np.ndarray, top_k=1, threshold=0.4) -> Optional[dict]:
    # emb expected normalized float32
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('SELECT id, name, embedding FROM faces')
    best = None
    best_score = -1.0
    for row in c.fetchall():
        id_, name, emb_blob = row
        db_emb = bytes_to_emb(emb_blob)
        # ensure normalized
        # cosine similarity
        score = float(np.dot(emb / np.linalg.norm(emb), db_emb / np.linalg.norm(db_emb)))
        if score > best_score:
            best_score = score
            best = {'id': id_, 'name': name, 'score': score}
    conn.close()
    if best and best['score'] >= threshold:
        return best
    return None