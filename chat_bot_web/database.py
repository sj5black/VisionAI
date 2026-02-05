"""
1:1 대화방 메시지 영구 저장 (Serena 제외, 유저 간 메시지만 저장)
회원가입/로그인: 아이디(username) + 비밀번호, 닉네임
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_WEB = Path(__file__).resolve().parent
DB_PATH = ROOT_WEB / "chat.db"
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db() -> None:
    """DB 테이블 생성. 스키마 변경 시 기존 users 테이블 마이그레이션."""
    conn = _get_conn()
    # 스키마 변경 시 users 테이블 재생성
    try:
        cur = conn.execute("PRAGMA table_info(users)")
        cols = [r[1] for r in cur.fetchall()]
        if "username" not in cols:
            conn.execute("DROP TABLE IF EXISTS dm_messages")
            conn.execute("DROP TABLE IF EXISTS dm_rooms")
            conn.execute("DROP TABLE IF EXISTS users")
    except sqlite3.OperationalError:
        pass

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

        CREATE TABLE IF NOT EXISTS dm_rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user1_id INTEGER NOT NULL,
            user2_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user1_id, user2_id)
        );
        CREATE INDEX IF NOT EXISTS idx_dm_rooms_users ON dm_rooms(user1_id, user2_id);

        CREATE TABLE IF NOT EXISTS dm_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER NOT NULL,
            sender_id INTEGER NOT NULL,
            text TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (room_id) REFERENCES dm_rooms(id),
            FOREIGN KEY (sender_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_dm_messages_room ON dm_messages(room_id);
    """)
    conn.commit()


def create_user(username: str, name: str, password_hash: str) -> int:
    """회원가입. user id 반환. 아이디 중복 시 에러."""
    conn = _get_conn()
    username = (username or "").strip()[:50]
    name = (name or "").strip()[:100]
    if not username or not name or not password_hash:
        raise ValueError("아이디, 닉네임, 비밀번호를 입력해 주세요.")
    if len(username) < 2:
        raise ValueError("아이디는 2자 이상이어야 합니다.")
    try:
        cur = conn.execute(
            "INSERT INTO users (username, name, password_hash) VALUES (?, ?, ?)",
            (username, name, password_hash)
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("이미 사용 중인 아이디입니다.")


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """아이디로 유저 조회."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id, username, name, password_hash FROM users WHERE username = ?",
        ((username or "").strip(),)
    )
    row = cur.fetchone()
    return dict(row) if row else None


def search_users_by_username(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """아이디 부분 일치 검색 (1:1 대화 상대 찾기)."""
    conn = _get_conn()
    q = (query or "").strip()
    if not q:
        return []
    cur = conn.execute(
        "SELECT id, username, name FROM users WHERE username LIKE ? ORDER BY username LIMIT ?",
        ("%" + q + "%", limit)
    )
    return [dict(r) for r in cur.fetchall()]


def update_user_name(user_id: int, name: str) -> None:
    """유저 닉네임 변경."""
    conn = _get_conn()
    conn.execute("UPDATE users SET name = ? WHERE id = ?", ((name or "").strip()[:100], user_id))
    conn.commit()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """user_id로 유저 조회."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id, username, name FROM users WHERE id = ?",
        (user_id,)
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_or_create_dm_room(user1_id: int, user2_id: int) -> int:
    """1:1 방 조회 또는 생성."""
    a, b = min(user1_id, user2_id), max(user1_id, user2_id)
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id FROM dm_rooms WHERE user1_id = ? AND user2_id = ?",
        (a, b)
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO dm_rooms (user1_id, user2_id) VALUES (?, ?)",
        (a, b)
    )
    conn.commit()
    return cur.lastrowid


def get_dm_room(room_id: int, current_user_id: int) -> Optional[Dict[str, Any]]:
    """1:1 방 조회."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id, user1_id, user2_id FROM dm_rooms WHERE id = ?",
        (room_id,)
    )
    row = cur.fetchone()
    if not row:
        return None
    other_id = row["user2_id"] if row["user1_id"] == current_user_id else row["user1_id"]
    other = get_user_by_id(other_id)
    if not other:
        return None
    return {"id": row["id"], "other_user": other}


def list_dm_rooms(user_id: int) -> List[Dict[str, Any]]:
    """유저의 1:1 대화방 목록."""
    conn = _get_conn()
    cur = conn.execute("""
        SELECT r.id, r.user1_id, r.user2_id,
               (SELECT text FROM dm_messages m WHERE m.room_id = r.id ORDER BY m.id DESC LIMIT 1) as last_text,
               (SELECT created_at FROM dm_messages m WHERE m.room_id = r.id ORDER BY m.id DESC LIMIT 1) as last_at
        FROM dm_rooms r
        WHERE r.user1_id = ? OR r.user2_id = ?
        ORDER BY COALESCE(last_at, '1970-01-01') DESC, r.id DESC
    """, (user_id, user_id))
    rows = cur.fetchall()
    result = []
    for row in rows:
        other_id = row["user2_id"] if row["user1_id"] == user_id else row["user1_id"]
        other = get_user_by_id(other_id)
        if other:
            result.append({
                "id": row["id"],
                "other_user": other,
                "last_text": row["last_text"],
                "last_at": row["last_at"],
            })
    return result


def save_dm_message(room_id: int, sender_id: int, text: str, image_url: Optional[str] = None) -> int:
    """1:1 메시지 저장."""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO dm_messages (room_id, sender_id, text, image_url) VALUES (?, ?, ?, ?)",
        (room_id, sender_id, text or "", image_url or "")
    )
    conn.commit()
    return cur.lastrowid


def get_dm_messages(room_id: int, current_user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    """1:1 방 메시지 목록."""
    room = get_dm_room(room_id, current_user_id)
    if not room:
        return []
    conn = _get_conn()
    cur = conn.execute("""
        SELECT m.id, m.room_id, m.sender_id, m.text, m.image_url, m.created_at, u.name as sender_name
        FROM dm_messages m
        JOIN users u ON m.sender_id = u.id
        WHERE m.room_id = ?
        ORDER BY m.id ASC
        LIMIT ?
    """, (room_id, limit))
    rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "message_id": str(r["id"]),
            "sender_id": r["sender_id"],
            "nickname": r["sender_name"],
            "text": r["text"] or "",
            "image_url": r["image_url"],
            "created_at": r["created_at"],
            "is_me": r["sender_id"] == current_user_id,
        }
        for r in rows
    ]
