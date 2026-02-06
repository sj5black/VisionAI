"""
1:1 대화방 메시지 영구 저장 (Serena 제외, 유저 간 메시지만 저장)
- 대화하는 두 user id 쌍(user1_id, user2_id)으로 row를 특정하고,
  해당 row의 messages 컬럼에 전체 대화내용을 JSON 리스트로 저장.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
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
            messages TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user1_id, user2_id)
        );
        CREATE INDEX IF NOT EXISTS idx_dm_rooms_users ON dm_rooms(user1_id, user2_id);
    """)
    # 기존 dm_rooms에 messages, updated_at 컬럼이 없으면 추가 (마이그레이션)
    try:
        cur = conn.execute("PRAGMA table_info(dm_rooms)")
        cols = [r[1] for r in cur.fetchall()]
        
        # messages 컬럼 추가
        if "messages" not in cols:
            conn.execute("ALTER TABLE dm_rooms ADD COLUMN messages TEXT NOT NULL DEFAULT '[]'")
            # 기존 dm_messages 데이터가 있으면 JSON으로 이전 후 삭제
            try:
                cur = conn.execute(
                    "SELECT room_id, sender_id, text, image_url, created_at FROM dm_messages ORDER BY id ASC"
                )
                by_room: Dict[int, List[Dict[str, Any]]] = {}
                for r in cur.fetchall():
                    rid = r[0]
                    if rid not in by_room:
                        by_room[rid] = []
                    by_room[rid].append({
                        "id": len(by_room[rid]) + 1,
                        "sender_id": r[1],
                        "text": r[2] or "",
                        "image_url": r[3],
                        "created_at": r[4] or "",
                    })
                for rid, msgs in by_room.items():
                    conn.execute(
                        "UPDATE dm_rooms SET messages = ? WHERE id = ?",
                        (json.dumps(msgs, ensure_ascii=False), rid),
                    )
            except sqlite3.OperationalError:
                pass
            conn.execute("DROP TABLE IF EXISTS dm_messages")
        
        # updated_at 컬럼 추가 (messages와 별도로 체크)
        # SQLite는 ALTER TABLE에서 DEFAULT CURRENT_TIMESTAMP를 지원하지 않으므로 DEFAULT 없이 추가
        cur = conn.execute("PRAGMA table_info(dm_rooms)")
        cols = [r[1] for r in cur.fetchall()]
        if "updated_at" not in cols:
            conn.execute("ALTER TABLE dm_rooms ADD COLUMN updated_at TIMESTAMP")
            conn.execute("UPDATE dm_rooms SET updated_at = created_at WHERE updated_at IS NULL")
        
        conn.commit()
    except sqlite3.OperationalError:
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


def delete_user(user_id: int) -> bool:
    """회원탈퇴: 해당 유저의 dm_rooms(1:1 대화방) 삭제 후 users에서 삭제."""
    conn = _get_conn()
    conn.execute("DELETE FROM dm_rooms WHERE user1_id = ? OR user2_id = ?", (user_id, user_id))
    cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return cur.rowcount > 0


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
    """유저의 1:1 대화방 목록. messages JSON에서 마지막 메시지로 last_text, last_at 계산."""
    conn = _get_conn()
    cur = conn.execute("""
        SELECT r.id, r.user1_id, r.user2_id, r.messages
        FROM dm_rooms r
        WHERE r.user1_id = ? OR r.user2_id = ?
        ORDER BY r.updated_at DESC, r.id DESC
    """, (user_id, user_id))
    rows = cur.fetchall()
    result = []
    for row in rows:
        other_id = row["user2_id"] if row["user1_id"] == user_id else row["user1_id"]
        other = get_user_by_id(other_id)
        if not other:
            continue
        last_text = None
        last_at = None
        try:
            msgs = json.loads(row["messages"] or "[]")
            if msgs:
                last = msgs[-1]
                last_text = last.get("text") or ("(사진)" if last.get("image_url") else None)
                last_at = last.get("created_at")
        except (json.JSONDecodeError, TypeError):
            pass
        result.append({
            "id": row["id"],
            "other_user": other,
            "last_text": last_text,
            "last_at": last_at,
        })
    return result


def save_dm_message(room_id: int, sender_id: int, text: str, image_url: Optional[str] = None) -> int:
    """1:1 메시지 저장. 해당 room의 messages JSON 리스트에 한 건 추가."""
    conn = _get_conn()
    cur = conn.execute("SELECT messages FROM dm_rooms WHERE id = ?", (room_id,))
    row = cur.fetchone()
    if not row:
        return 0
    try:
        messages = json.loads(row["messages"] or "[]")
    except (json.JSONDecodeError, TypeError):
        messages = []
    msg_id = max((m.get("id") or 0) for m in messages) + 1 if messages else 1
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    new_msg = {
        "id": msg_id,
        "sender_id": sender_id,
        "text": text or "",
        "image_url": image_url,
        "created_at": created_at,
    }
    messages.append(new_msg)
    conn.execute(
        "UPDATE dm_rooms SET messages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(messages, ensure_ascii=False), room_id),
    )
    conn.commit()
    return msg_id


def get_dm_messages(room_id: int, current_user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    """1:1 방 메시지 목록. 해당 room의 messages JSON에서 읽어옴."""
    room = get_dm_room(room_id, current_user_id)
    if not room:
        return []
    conn = _get_conn()
    cur = conn.execute("SELECT messages FROM dm_rooms WHERE id = ?", (room_id,))
    row = cur.fetchone()
    if not row:
        return []
    try:
        messages = json.loads(row["messages"] or "[]")
    except (json.JSONDecodeError, TypeError):
        messages = []
    messages = messages[-limit:] if len(messages) > limit else messages
    result = []
    for m in messages:
        sender_id = m.get("sender_id")
        sender = get_user_by_id(sender_id)
        sender_name = (sender or {}).get("name") or ""
        result.append({
            "id": m.get("id"),
            "message_id": str(m.get("id", "")),
            "sender_id": sender_id,
            "nickname": sender_name,
            "text": m.get("text") or "",
            "image_url": m.get("image_url"),
            "created_at": m.get("created_at", ""),
            "is_me": sender_id == current_user_id,
        })
    return result


def update_dm_message(room_id: int, msg_id: int, sender_id: int, text: Optional[str] = None, image_url: Optional[str] = None) -> bool:
    """1:1 방 메시지 수정. text/image_url 중 전달된 것만 갱신. 본인 메시지만 수정 가능."""
    conn = _get_conn()
    cur = conn.execute("SELECT messages FROM dm_rooms WHERE id = ?", (room_id,))
    row = cur.fetchone()
    if not row:
        return False
    try:
        messages = json.loads(row["messages"] or "[]")
    except (json.JSONDecodeError, TypeError):
        return False
    for m in messages:
        if m.get("id") == msg_id and m.get("sender_id") == sender_id:
            if text is not None:
                m["text"] = text
            if image_url is not None:
                m["image_url"] = image_url
            conn.execute(
                "UPDATE dm_rooms SET messages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(messages, ensure_ascii=False), room_id),
            )
            conn.commit()
            return True
    return False


def delete_dm_message(room_id: int, msg_id: int, sender_id: int) -> bool:
    """1:1 방 메시지 삭제. 본인 메시지만 삭제 가능."""
    conn = _get_conn()
    cur = conn.execute("SELECT messages FROM dm_rooms WHERE id = ?", (room_id,))
    row = cur.fetchone()
    if not row:
        return False
    try:
        messages = json.loads(row["messages"] or "[]")
    except (json.JSONDecodeError, TypeError):
        return False
    new_messages = [m for m in messages if not (m.get("id") == msg_id and m.get("sender_id") == sender_id)]
    if len(new_messages) == len(messages):
        return False
    conn.execute(
        "UPDATE dm_rooms SET messages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(new_messages, ensure_ascii=False), room_id),
    )
    conn.commit()
    return True
