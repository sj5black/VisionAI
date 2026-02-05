"""
회원가입/로그인 인증 (아이디 + 비밀번호, 닉네임)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fastapi import Request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_session_secret() -> str:
    return os.getenv("SESSION_SECRET") or os.getenv("SECRET_KEY") or "visionai-chat-secret-change-in-production"


def hash_password(password: str) -> str:
    """비밀번호 해시."""
    import bcrypt
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """비밀번호 검증."""
    import bcrypt
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """세션에서 현재 로그인 유저 반환."""
    return request.session.get("user")


# AccessToken(WS 토큰) 유효기간: 7일
ACCESS_TOKEN_SECONDS = 7 * 24 * 60 * 60


def create_ws_token(user: Dict[str, Any]) -> str:
    """WebSocket 인증용 토큰 (7일 유효)."""
    from itsdangerous import URLSafeTimedSerializer
    secret = get_session_secret()
    s = URLSafeTimedSerializer(secret)
    payload = {
        "user_id": user["id"],
        "username": user.get("username"),
        "name": user["name"],
        "exp": int(time.time()) + ACCESS_TOKEN_SECONDS,
    }
    return s.dumps(payload)


def verify_ws_token(token: str) -> Optional[Dict[str, Any]]:
    """WebSocket 토큰 검증."""
    from itsdangerous import URLSafeTimedSerializer, BadSignature
    secret = get_session_secret()
    s = URLSafeTimedSerializer(secret)
    try:
        payload = s.loads(token, max_age=ACCESS_TOKEN_SECONDS)
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except (BadSignature, Exception):
        return None
