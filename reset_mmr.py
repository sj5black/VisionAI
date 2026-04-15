#!/usr/bin/env python3
"""
모든 회원의 체스 MMR을 초기화하는 스크립트.

- mmr_rating: 650
- mmr_games: 0
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "chat_bot_web" / "chat.db"


def reset_mmr(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        # users 테이블에 컬럼이 없을 수도 있으니, 있다면 초기화하는 방식으로 처리
        cur.execute("PRAGMA table_info(users)")
        cols = {row[1] for row in cur.fetchall()}

        updates = []
        if "mmr_rating" in cols:
            updates.append("mmr_rating = 650")
        if "mmr_games" in cols:
            updates.append("mmr_games = 0")

        if not updates:
            print("users 테이블에 mmr 관련 컬럼이 없습니다. 아무 작업도 수행하지 않습니다.")
            return

        sql = "UPDATE users SET " + ", ".join(updates)
        cur.execute(sql)
        conn.commit()
        print(f"모든 유저 MMR 초기화 완료: {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    reset_mmr(DB_PATH)