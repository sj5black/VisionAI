#!/usr/bin/env python3
"""
dm_rooms 테이블에서 지정한 id의 1:1 대화방을 삭제하는 스크립트.

사용법:
  python -m chat_bot_web.scripts.delete_dm_room <room_id>
  또는
  cd chat_bot_web && python scripts/delete_dm_room.py <room_id>

예: Teddy2 등 모바일 접속 사용자의 특정 dm_room 삭제
  python -m chat_bot_web.scripts.delete_dm_room 3
"""
import sys
from pathlib import Path

# chat_bot_web 기준으로 DB 경로 결정
ROOT_WEB = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_WEB / "chat.db"


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python -m chat_bot_web.scripts.delete_dm_room <room_id>")
        print("예: python -m chat_bot_web.scripts.delete_dm_room 3")
        sys.exit(1)

    try:
        room_id = int(sys.argv[1])
    except ValueError:
        print("오류: room_id는 정수여야 합니다.")
        sys.exit(1)

    if not DB_PATH.exists():
        print(f"오류: DB 파일을 찾을 수 없습니다: {DB_PATH}")
        sys.exit(1)

    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute("SELECT id, user1_id, user2_id FROM dm_rooms WHERE id = ?", (room_id,))
    row = cur.fetchone()
    if not row:
        print(f"dm_room id={room_id} 인 행이 없습니다.")
        conn.close()
        sys.exit(1)

    conn.execute("DELETE FROM dm_rooms WHERE id = ?", (room_id,))
    conn.commit()
    conn.close()
    print(f"dm_room id={room_id} (user1_id={row[1]}, user2_id={row[2]}) 를 삭제했습니다.")


if __name__ == "__main__":
    main()
