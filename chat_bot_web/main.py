"""
영어 채팅 챗봇 웹 서비스
- FastAPI 기반 채팅 UI
- 175.197.131.234:8004 에서 접속
- 회원가입/로그인, 1:1 대화방 지원
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import requests
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# english_chat_bot 로직 재사용
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess

from english_chat_bot import (
    get_client,
    parse_ai_response,
    CONVERSATION_SITUATIONS,
    SYSTEM_PROMPT,
    KOREAN_LABEL,
)

from . import auth as auth_module
from . import database as db

# 상황 제목 -> 한국어 설명 (대화 주제)
SITUATION_KO = {
    "Job interview": "취업 면접",
    "At a restaurant": "레스토랑",
    "Travel": "여행 (공항에서)",
    "Shopping": "쇼핑 (매장)",
    "First day at work": "첫 출근",
    "At a party": "파티",
    "Doctor's visit": "병원 진료",
    "Hotel check-in": "호텔 체크인",
    "Coffee shop": "커피숍",
    "Booking a flight": "항공권 예약",
}

ROOT_WEB = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT_WEB / "templates"
STATIC_DIR = ROOT_WEB / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# 허용 이미지 타입 (모바일 호환: heic, jpg 등 포함)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/heic", "image/pjpeg"}

app = FastAPI(title="English Chat Bot - 영어 대화 연습")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 세션 (RefreshToken 역할: 30일 유지 → 창 껐다 켜도 자동 로그인)
REFRESH_TOKEN_DAYS = 30
app.add_middleware(
    SessionMiddleware,
    secret_key=auth_module.get_session_secret(),
    max_age=REFRESH_TOKEN_DAYS * 24 * 60 * 60,
)

# DB 초기화
db.init_db()

# 세션별 대화 저장 (in-memory)
_conversations: Dict[str, Dict[str, Any]] = {}

# 멀티 채팅방: (websocket, user_info) 목록. user_info = {id, name, email}
_room_connections: List[tuple[WebSocket, Dict[str, Any]]] = []
_room_connections_lock = asyncio.Lock()
# 메시지 읽음 상태: {message_id: {"sender_ws": WebSocket, "read_by": Set[str], "total_participants": int}}
_message_read_status: Dict[str, Dict[str, Any]] = {}
# 메시지 저장 (수정/삭제용): {message_id: {"sender_nickname": str, "text": str}}
_room_messages: Dict[str, Dict[str, Any]] = {}
# IP별 마지막 닉네임 (재접속 시 자동 입력용)
_ip_nickname: Dict[str, str] = {}
# 최근 대화 (Serena 초대용, 최대 10개): [{"nickname": str, "text": str}, ...]
_room_message_history: List[Dict[str, str]] = []
# Serena가 초대된 상태인지 (초대 후 계속 대화 참여)
_serena_invited: bool = False
# Serena 자동 응답 예약 태스크 (디바운스용)
_serena_pending_task: Optional[asyncio.Task] = None

# 체스 게임: 단일 방당 1게임 (Serena vs 유저 또는 유저 vs 유저)
# {"board": chess.Board, "white_player": str, "black_player": str, "mode": "serena"|"pvp", "status": str, "white_captured": List[str], "black_captured": List[str]}
_chess_game: Optional[Dict[str, Any]] = None
_chess_pending_task: Optional[asyncio.Task] = None

# 1:1 DM: {room_id: [(websocket, user_info), ...]}
_dm_connections: Dict[int, List[tuple[WebSocket, Dict[str, Any]]]] = {}
_dm_connections_lock = asyncio.Lock()
# DM 메시지 읽음 상태: {(room_id, message_id): {"sender_id": int, "sender_ws": WebSocket, "read": bool}}
_dm_message_read_status: Dict[tuple[int, str], Dict[str, Any]] = {}


def _get_captured_piece(board: chess.Board, move: chess.Move) -> Optional[str]:
    """캡처된 기물 기호 반환 (push 전에 호출)."""
    if not board.is_capture(move):
        return None
    p = board.piece_at(move.to_square)
    if p is None and board.is_en_passant(move):
        p = chess.Piece(chess.PAWN, not board.turn)
    return p.symbol() if p else None


class StartResponse(BaseModel):
    conversation_id: str
    situation: str
    situation_title: str
    situation_display: str  # "Job interview (취업 면접)" 형식
    first_message: str
    first_korean: Optional[str] = None
    messages: List[Dict[str, str]]


class ChatRequest(BaseModel):
    conversation_id: str
    user_message: str


class StartRequest(BaseModel):
    situation_index: Optional[int] = None  # None 또는 -1이면 랜덤


class ChatResponse(BaseModel):
    correction: Optional[str]
    score: Optional[int]
    reply: str
    korean_reply: Optional[str] = None
    raw_reply: Optional[str] = None


def _fetch_link_preview(url: str) -> Optional[Dict[str, str]]:
    """URL에서 og 태그 추출 (동기, 스레드에서 실행)."""
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0 (compatible; LinkPreview/1.0)"})
        r.raise_for_status()
        html = r.text
        def _extract(prop: str) -> Optional[str]:
            for tag in ("og:", "twitter:"):
                m = re.search(rf'<meta[^>]+(?:property|name)=["\'](?:{re.escape(tag)}{re.escape(prop)})["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
                if m:
                    return m.group(1).strip()
                m = re.search(rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:{re.escape(tag)}{re.escape(prop)})["\']', html, re.I)
                if m:
                    return m.group(1).strip()
            return None
        title = _extract("title") or _extract("site_name")
        desc = _extract("description")
        image = _extract("image")
        if image and image.startswith("/") and not image.startswith("//"):
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            image = base + image
        if not title and not image:
            return None
        return {"url": url, "title": (title or "")[:200], "description": (desc or "")[:300], "image": (image or "")[:1000]}
    except Exception:
        return None


async def _fetch_and_broadcast_link_preview(message_id: str, url: str) -> None:
    """백그라운드에서 링크 미리보기 조회 후 브로드캐스트."""
    loop = asyncio.get_event_loop()
    preview = await loop.run_in_executor(None, _fetch_link_preview, url)
    if preview:
        await _room_broadcast({"type": "link_preview", "message_id": message_id, "preview": preview}, exclude_ws=None)


SERENA_SYSTEM = """You are Serena, a native English speaker from California who is participating in an international group chat. You are friendly and outgoing, but you ONLY speak English - you don't understand or speak Korean at all.

Style & register (important):
- Use relatively advanced, natural English: idiomatic phrases, phrasal verbs, and vocabulary that educated native speakers use in casual conversation (e.g. "no cap", "lowkey", "it hits different", "that's a stretch", "I'm not gonna lie", "on the flip side", "at the end of the day", "throw shade", "spill the tea" where it fits).
- Vary your sentence structure: use subordinate clauses, relative clauses, and the occasional inversion or fronting when it sounds natural (e.g. "What I'm saying is...", "The thing is,...", "Not that I'm complaining, but...").
- Sound like a real Californian in a group chat: mix in slang and colloquialisms naturally—don't overdo it, but do use expressions real natives use (e.g. "totally", "for real", "kinda", "gonna", "gotta", "pretty much", "no biggie", "my bad", "that's on me").

Rules:
- You are a native English speaker. Respond ONLY in English, no matter what language others use.
- When someone's message contains Korean (mixed with English or alone), naturally work into your reply the English equivalent for that Korean. Keep it brief and friendly so it helps them practice English.
- If the message is mostly or only Korean and you can't infer meaning, say something like "I didn't catch that - could you try in English?" and encourage them to use English.
- When the last user message is in English but has grammar mistakes or unnatural phrasing, briefly and kindly add feedback: suggest the correct or more natural way. Keep it to one short phrase so your reply stays conversational; then continue the conversation naturally.
- Be conversational and natural—react, comment, ask questions, or add to the discussion.
- Keep your replies to 2–4 sentences so you have room for richer expression.
- Don't mention that you are AI. You're just a regular person who happens to only speak English.
- Stay in character as a friendly, English-speaking person joining the chat.
- Do NOT use Unicode emoji (e.g. 😀 👍 ❤️). Instead, occasionally use simple text emoticons like lol, :), :(, T-T, ^^, haha when it fits naturally."""


SEOUL_TZ = timezone(timedelta(hours=9))


def _seoul_time_str() -> str:
    """현재 서울(UTC+9) 기준 24시 형식 시간 반환 (HH:MM)."""
    return datetime.now(SEOUL_TZ).strftime("%H:%M")


def _utc_datetime_str_to_seoul_hhmm(utc_str: str) -> str:
    """DB 등에 저장된 UTC 시각 문자열('YYYY-MM-DD HH:MM:SS')을 서울(KST) HH:MM으로 변환.
    다른 PC/서버에서 저장된 메시지도 서울 기준으로 표시하기 위함."""
    if not utc_str or len(utc_str) < 16:
        return _seoul_time_str()
    try:
        # naive UTC로 파싱 후 timezone 붙여서 서울로 변환
        dt = datetime.strptime(utc_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        seoul = dt.astimezone(SEOUL_TZ)
        return seoul.strftime("%H:%M")
    except (ValueError, TypeError):
        return _seoul_time_str()


def _contains_korean(text: str) -> bool:
    """문자열에 한글이 포함되어 있는지 여부."""
    if not text or not isinstance(text, str):
        return False
    for c in text:
        if "\uAC00" <= c <= "\uD7A3" or "\u1100" <= c <= "\u11FF" or "\u3130" <= c <= "\u318F":
            return True
    return False


def _last_user_message_has_korean(recent_messages: List[Dict[str, str]]) -> bool:
    """Serena가 아닌 유저의 마지막 메시지(방금 보낸 쿼리)에 한글이 포함된 경우만 True."""
    for m in reversed(recent_messages):
        nick = (m.get("nickname") or "").strip()
        if nick.lower() == "serena":
            continue
        return _contains_korean(m.get("text") or "")
    return False


def _call_serena(recent_messages: List[Dict[str, str]]) -> Optional[str]:
    """OpenAI로 Serena 응답 생성 (동기). gpt-5-mini/nano는 Responses API 사용."""
    try:
        client = get_client()
        model = os.getenv("OPENAI_MODEL")
        if not model:
            print("❌ Serena error: OPENAI_MODEL not set in .env")
            return None
        
        conv = "\n".join(f"{m['nickname']}: {m['text']}" for m in recent_messages)
        if not conv.strip():
            conv = "(No recent messages)"
        
        # 유저의 마지막 쿼리에 한글이 포함된 경우에만 한국어→영어 지도 문구 추가
        korean_hint = _last_user_message_has_korean(recent_messages)
        if korean_hint:
            user_instruction = "Respond in English. The last user message contains Korean—briefly give the English equivalent for that Korean in a natural, friendly way."
        else:
            user_instruction = "Respond in English only. Do NOT provide Korean-to-English translation or language guidance in this reply. If the last user message in English has grammar or naturalness issues, briefly and kindly suggest a correction or more natural phrasing (e.g. one short tip), then continue the conversation."
        
        # gpt-5-mini, gpt-5-nano는 Responses API 사용
        use_responses_api = any(x in model.lower() for x in ['gpt-5-mini', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.2'])
        
        if use_responses_api:
            # Responses API (최신 모델용)
            print(f"🔄 Using Responses API for model: {model}")
            input_text = f"{SERENA_SYSTEM}\n\nRecent chat:\n{conv}\n\n{user_instruction}"
            resp = client.responses.create(
                model=model,
                input=input_text,
                max_output_tokens=300,
            )
            if hasattr(resp, 'output_text') and resp.output_text:
                return resp.output_text.strip()
            elif hasattr(resp, 'output') and resp.output:
                # output이 list인 경우
                for item in resp.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                return content_item.text.strip()
        else:
            # Chat Completions API (기존 모델용)
            print(f"💬 Using Chat Completions API for model: {model}")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SERENA_SYSTEM},
                    {"role": "user", "content": f"Recent chat messages:\n{conv}\n\n{user_instruction}"},
                ],
                max_completion_tokens=300,
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Serena OpenAI error: {e}")
        import traceback
        traceback.print_exc()
    return None


SERENA_RESPONSE_TIMEOUT = 30  # Serena 응답 타임아웃 (초)

CHESS_SYSTEM = """You are Serena playing chess (black pieces). You ONLY reply with ONE valid UCI move from the legal moves list. Format: 4 chars like e7e5 or g8f6. No other text."""


def _call_serena_chess(fen: str, legal_moves: Optional[List[str]] = None) -> Optional[str]:
    """OpenAI로 Serena 체스 수 반환 (동기). UCI 형식 4자 (e.g., e7e5). 실패 시 None."""
    try:
        client = get_client()
        model = os.getenv("OPENAI_CHESS_MODEL") or os.getenv("OPENAI_MODEL")
        if not model:
            return None
        model = model.strip()
        use_responses_api = any(x in model.lower() for x in ['gpt-5-mini', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.2'])
        legal_str = ", ".join(legal_moves[:30]) if legal_moves else "(compute from FEN)"
        user_content = f"Board FEN: {fen}\nLegal black moves (pick ONE): {legal_str}\nReply with exactly 4 chars (e.g. e7e5)."
        if use_responses_api:
            resp = client.responses.create(model=model, input=f"{CHESS_SYSTEM}\n\n{user_content}", max_output_tokens=20)
            text = (resp.output_text if hasattr(resp, 'output_text') else None) or ""
            if hasattr(resp, 'output') and resp.output is not None and not text:
                for item in (resp.output or []):
                    if item is None:
                        continue
                    content = getattr(item, 'content', None)
                    if content is None:
                        continue
                    items = content if isinstance(content, (list, tuple)) else [content]
                    for c in items:
                        if c is not None:
                            t = getattr(c, 'text', None)
                            if t:
                                text = str(t)
                                break
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CHESS_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=20,
            )
            text = (resp.choices[0].message.content or "").strip()
        s = (text or "").strip().lower()
        m = re.search(r'([a-h][1-8][a-h][1-8])', s)
        return m.group(1) if m else None
    except Exception as e:
        print(f"Serena chess error: {e}")
    return None


def _pick_random_legal_move(board: chess.Board) -> Optional[str]:
    """합법적인 수 중 랜덤 선택 (AI 실패 시 폴백)."""
    moves = list(board.legal_moves)
    if not moves:
        return None
    m = secrets.choice(moves)
    return m.uci()


async def _play_serena_chess_move() -> None:
    """Serena 차례일 때 AI 수 두고 브로드캐스트. AI 실패 시 랜덤 합법 수 사용."""
    global _chess_game, _chess_pending_task
    if not _chess_game or _chess_game.get("status") != "active":
        return
    board: chess.Board = _chess_game["board"]
    if board.turn != chess.BLACK:
        return
    fen = board.fen()
    legal_uci = [m.uci() for m in board.legal_moves]
    if not legal_uci:
        return

    def _try_ai() -> Optional[str]:
        return _call_serena_chess(fen, legal_uci)

    loop = asyncio.get_event_loop()
    uci: Optional[str] = None
    try:
        uci = await asyncio.wait_for(
            loop.run_in_executor(None, _try_ai),
            timeout=SERENA_RESPONSE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print("Serena chess: AI timeout, using random move")
        uci = _pick_random_legal_move(board)
    _chess_pending_task = None

    if not uci:
        uci = _pick_random_legal_move(board)
        if uci:
            print("Serena chess: AI returned invalid/empty, using random move")
    if not uci or not _chess_game or _chess_game.get("status") != "active":
        return
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            uci = _pick_random_legal_move(board)
            if not uci:
                return
            move = chess.Move.from_uci(uci)
        cap = _get_captured_piece(board, move)
        if cap:
            _chess_game.setdefault("black_captured", []).append(cap)
        board.push(move)
        uci = move.uci()
        status = "active"
        if board.is_checkmate():
            status = "checkmate_white"
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            status = "draw"
        _chess_game["status"] = status
        await _room_broadcast({
            "type": "chess_state",
            "fen": board.fen(),
            "last_move": uci,
            "status": status,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "in_check": board.is_check(),
            "white_captured": _chess_game.get("white_captured", []),
            "black_captured": _chess_game.get("black_captured", []),
            "white_player": _chess_game.get("white_player"),
            "black_player": _chess_game.get("black_player"),
            "mode": _chess_game.get("mode", "serena"),
        }, exclude_ws=None)
    except (ValueError, chess.InvalidMoveError):
        uci_fb = _pick_random_legal_move(board)
        if uci_fb and _chess_game and _chess_game.get("status") == "active":
            try:
                move = chess.Move.from_uci(uci_fb)
                board.push(move)
                status = "active"
                if board.is_checkmate():
                    status = "checkmate_white"
                elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                    status = "draw"
                _chess_game["status"] = status
                await _room_broadcast({
                    "type": "chess_state",
                    "fen": board.fen(),
                    "last_move": uci_fb,
                    "status": status,
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "in_check": board.is_check(),
                    "white_captured": _chess_game.get("white_captured", []),
                    "black_captured": _chess_game.get("black_captured", []),
                    "white_player": _chess_game.get("white_player"),
                    "black_player": _chess_game.get("black_player"),
                    "mode": _chess_game.get("mode", "serena"),
                }, exclude_ws=None)
            except Exception:
                pass


async def _invite_serena_and_broadcast() -> None:
    """Serena 응답 생성 후 브로드캐스트. 실패 시 시스템 메시지로 알림."""
    global _serena_invited
    try:
        recent = list(_room_message_history)[-10:] if _room_message_history else []
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, _call_serena, recent)
        if not reply:
            await _room_broadcast({
                "type": "system",
                "message": "Serena를 불러올 수 없습니다. 잠시 후 다시 시도해 주세요."
            }, exclude_ws=None)
            return
        was_invited = _serena_invited
        _serena_invited = True
        
        # 첫 초대 시 입장 안내
        if not was_invited:
            await _room_broadcast({"type": "system", "message": "Serena님이 입장했습니다."}, exclude_ws=None)
            participants = _build_participants_list()
            await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
            await _room_broadcast({"type": "serena_status", "present": True}, exclude_ws=None)
        
        message_id = str(uuid4())
        await _room_broadcast({
            "type": "chat",
            "message_id": message_id,
            "nickname": "Serena",
            "text": reply,
            "unread_count": 0,
            "timestamp": _seoul_time_str(),
        }, exclude_ws=None)
        # 모든 클라이언트에 Serena 상태 갱신 (초대한 사람 외 다른 참여자도 강퇴 버튼 표시)
        await _room_broadcast({"type": "serena_status", "present": True}, exclude_ws=None)
        _room_message_history.append({"nickname": "Serena", "text": reply})
        if len(_room_message_history) > 10:
            _room_message_history.pop(0)
    except Exception as e:
        print(f"Serena invite error (non-fatal): {e}")
        try:
            await _room_broadcast({
                "type": "system",
                "message": "Serena를 불러올 수 없습니다. 잠시 후 다시 시도해 주세요."
            }, exclude_ws=None)
        except Exception:
            pass


def _schedule_serena_response() -> None:
    """사용자 메시지 후 Serena가 자연스럽게 응답하도록 디바운스 예약 (3초 후)."""
    global _serena_pending_task

    async def _delayed_serena() -> None:
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            return
        await _invite_serena_and_broadcast()

    if _serena_pending_task and not _serena_pending_task.done():
        _serena_pending_task.cancel()
    _serena_pending_task = asyncio.create_task(_delayed_serena())


def _build_participants_list() -> List[Dict[str, Any]]:
    """참여자 목록 (user_id, username 포함, Serena는 user_id=null)."""
    out = []
    for _, u in _room_connections:
        out.append({
            "user_id": u["id"],
            "username": u.get("username"),
            "name": u["name"],
            "avatar_url": u.get("avatar_url") or None,
        })
    if _serena_invited:
        out.append({"user_id": None, "name": "Serena", "avatar_url": "/serena.png"})
    return out


async def _room_broadcast(message: Dict[str, Any], exclude_ws: Optional[WebSocket] = None) -> None:
    """멀티 채팅방 전체에 메시지 브로드캐스트."""
    text = json.dumps(message, ensure_ascii=False)
    dead = []
    for ws, u in _room_connections:
        if ws is exclude_ws:
            continue
        try:
            await ws.send_text(text)
        except Exception as e:
            print(f"Broadcast error to {u.get('name', '?')}: {e}")
            dead.append((ws, u))
    for ws, u in dead:
        _room_connections[:] = [(w, usr) for w, usr in _room_connections if w is not ws]
        print(f"Removed dead connection: {u.get('name', '?')}")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    """랜딩: AI와 채팅 / 사용자들과 채팅 선택."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request) -> Any:
    """AI 영어 채팅 페이지."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/room", response_class=HTMLResponse)
async def room_page(request: Request) -> Any:
    """멀티 채팅방 페이지. 로그인하면 채팅 이용 가능."""
    return templates.TemplateResponse("room.html", {"request": request})


@app.post("/api/auth/signup")
async def api_auth_signup(request: Request):
    """회원가입. body: {username, name, password, password_confirm}"""
    try:
        body = await request.json()
        username = (body.get("username") or "").strip()
        name = (body.get("name") or "").strip()
        password = body.get("password") or ""
        password_confirm = body.get("password_confirm") or ""
        if not username or not name or not password:
            raise HTTPException(status_code=400, detail="아이디, 닉네임, 비밀번호를 모두 입력해 주세요.")
        if len(username) < 2:
            raise HTTPException(status_code=400, detail="아이디는 2자 이상이어야 합니다.")
        if len(password) < 4:
            raise HTTPException(status_code=400, detail="비밀번호는 4자 이상이어야 합니다.")
        if password != password_confirm:
            raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")
        password_hash = auth_module.hash_password(password)
        user_id = db.create_user(username, name, password_hash)
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=500, detail="가입 처리 중 오류")
        request.session["user"] = {
            "id": user["id"],
            "username": user["username"],
            "name": user["name"],
        }
        return {"ok": True, "user": request.session["user"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def api_auth_login(request: Request):
    """로그인. body: {username, password}"""
    try:
        body = await request.json()
        username = (body.get("username") or "").strip()
        password = body.get("password") or ""
        if not username or not password:
            raise HTTPException(status_code=400, detail="아이디와 비밀번호를 입력해 주세요.")
        user = db.get_user_by_username(username)
        if not user or not auth_module.verify_password(password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
        request.session["user"] = {
            "id": user["id"],
            "username": user["username"],
            "name": user["name"],
        }
        return {"ok": True, "user": request.session["user"]}
    except HTTPException:
        raise


@app.get("/api/auth/logout")
async def auth_logout(request: Request):
    """로그아웃."""
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)


@app.post("/api/auth/withdraw")
async def api_auth_withdraw(request: Request):
    """회원탈퇴. 로그인 상태에서만 가능. 탈퇴 후 세션 삭제 및 홈으로 리다이렉트."""
    user = await auth_module.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    try:
        body = await request.json() or {}
    except Exception:
        body = {}
    password = (body.get("password") or "").strip()
    if not password:
        raise HTTPException(status_code=400, detail="비밀번호를 입력해 주세요.")
    db_user = db.get_user_by_username(user.get("username") or "")
    if not db_user or not auth_module.verify_password(password, db_user["password_hash"]):
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")
    if not db.delete_user(user["id"]):
        raise HTTPException(status_code=500, detail="탈퇴 처리에 실패했습니다.")
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)


@app.get("/api/auth/me")
async def api_auth_me(request: Request):
    """현재 로그인 유저."""
    user = await auth_module.get_current_user(request)
    if not user:
        return {"logged_in": False}
    return {"logged_in": True, "user": user}


@app.get("/api/auth/ws-token")
async def api_auth_ws_token(request: Request):
    """WebSocket 인증용 토큰 (5분 유효)."""
    user = await auth_module.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    token = auth_module.create_ws_token(user)
    return {"token": token}


@app.get("/api/room/saved-nickname")
def api_room_saved_nickname(request: Request) -> Any:
    """요청 IP에 저장된 닉네임이 있으면 반환 (구버전 호환)."""
    client_host = request.client.host if request.client else None
    nickname = _ip_nickname.get(client_host) if client_host else None
    return {"nickname": nickname}


@app.get("/api/users/search")
async def api_users_search(request: Request, q: str = ""):
    """아이디로 유저 검색 (1:1 대화 상대 찾기)."""
    user = await auth_module.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    q = (q or "").strip()
    if len(q) < 1:
        return {"users": []}
    users = db.search_users_by_username(q, limit=10)
    return {"users": [{"id": u["id"], "username": u["username"], "name": u["name"]} for u in users]}


@app.get("/api/dm/rooms")
async def api_dm_rooms(request: Request):
    """내 1:1 대화방 목록."""
    user = await auth_module.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    rooms = db.list_dm_rooms(user["id"])
    return {"rooms": rooms}


@app.get("/api/dm/rooms/{room_id}")
async def api_dm_room_get(room_id: int, request: Request):
    """1:1 방 정보 조회 (참여자만)."""
    user = await auth_module.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    room = db.get_dm_room(room_id, user["id"])
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"room": room}


@app.post("/api/dm/rooms/create")
async def api_dm_create(request: Request):
    """1:1 대화방 생성 또는 기존 방 반환. body: {other_user_id: int} 또는 {other_username: str}"""
    user = await auth_module.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        body = await request.json() or {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body")
    other_id = None
    if "other_user_id" in body and body["other_user_id"] is not None:
        try:
            other_id = int(body["other_user_id"])
        except (TypeError, ValueError):
            pass
    if other_id is None and "other_username" in body and body["other_username"]:
        other = db.get_user_by_username((body["other_username"] or "").strip())
        if other:
            other_id = other["id"]
    if not other_id:
        raise HTTPException(status_code=400, detail="대화 상대(아이디 또는 사용자)를 지정해 주세요.")
    if other_id == user["id"]:
        raise HTTPException(status_code=400, detail="자기 자신과는 1:1 대화를 할 수 없습니다.")
    other = db.get_user_by_id(other_id)
    if not other:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    room_id = db.get_or_create_dm_room(user["id"], other_id)
    room = db.get_dm_room(room_id, user["id"])
    return {"room": room}


@app.get("/api/dm/rooms/{room_id}/messages")
async def api_dm_messages(room_id: int, request: Request):
    """1:1 방 메시지 목록."""
    user = await auth_module.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    messages = db.get_dm_messages(room_id, user["id"])
    return {"messages": messages}


def _get_image_ext(content_type: Optional[str], filename: Optional[str]) -> str:
    """content_type 또는 filename에서 이미지 확장자 결정 (모바일 호환)."""
    ct_map = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/pjpeg": ".jpg", "image/png": ".png",
              "image/gif": ".gif", "image/webp": ".webp", "image/heic": ".heic"}
    ext = ct_map.get((content_type or "").lower().split(";")[0].strip(), None)
    if ext:
        return ext
    fn = (filename or "").lower()
    for e in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"]:
        if fn.endswith(e):
            return e
    return ".jpg"


@app.post("/api/room/upload")
async def api_room_upload(file: UploadFile = File(...)) -> Any:
    """멀티채팅 이미지 업로드. 최대 5MB. 모바일 호환."""
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    ext_ok = (file.filename or "").lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"))
    is_image = (
        content_type.startswith("image/")
        or content_type in ALLOWED_IMAGE_TYPES
        or (ext_ok and content_type in ("", "application/octet-stream"))
    )
    if not is_image:
        raise HTTPException(status_code=400, detail="허용 형식: JPEG, PNG, GIF, WebP, HEIC")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="이미지 크기는 5MB 이하여야 합니다.")
    ext = _get_image_ext(file.content_type, file.filename)
    name = f"{uuid4().hex}{ext}"
    path = UPLOADS_DIR / name
    path.write_bytes(content)
    return {"url": f"/static/uploads/{name}"}


@app.get("/room-bg.png")
def room_background_image() -> FileResponse:
    """멀티 채팅방 배경 이미지 (chat_bot_web/back.png)."""
    path = ROOT_WEB / "back.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Background image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/serena.png")
def serena_avatar() -> FileResponse:
    """Serena AI 프로필 이미지 (chat_bot_web/Serena.png)."""
    path = ROOT_WEB / "Serena.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Serena image not found")
    return FileResponse(path, media_type="image/png")


@app.websocket("/ws/room")
async def ws_room(websocket: WebSocket) -> None:
    """멀티 채팅방 WebSocket.
    첫 메시지: {"type":"join","ws_token":"..."}
    이후:
    - {"type":"chat","text":"..."}
    - {"type":"read","message_id":"..."}
    - {"type":"edit","message_id":"...","text":"..."}  (발신자만)
    - {"type":"delete","message_id":"..."}            (발신자만)
    """
    global _serena_invited, _serena_pending_task, _chess_game, _chess_pending_task
    await websocket.accept()
    user_info: Optional[Dict[str, Any]] = None
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        if data.get("type") != "join":
            await websocket.send_text(json.dumps({"type": "error", "message": "Invalid join message."}, ensure_ascii=False))
            await websocket.close()
            return
        ws_token = (data.get("ws_token") or "").strip()
        user_info = auth_module.verify_ws_token(ws_token)
        if not user_info:
            await websocket.send_text(json.dumps({"type": "error", "message": "로그인이 만료되었습니다. 새로고침 후 다시 시도해 주세요."}, ensure_ascii=False))
            await websocket.close()
            return
        # DB에서 user 조회
        db_user = db.get_user_by_id(user_info.get("user_id"))
        if not db_user:
            await websocket.send_text(json.dumps({"type": "error", "message": "사용자를 찾을 수 없습니다."}, ensure_ascii=False))
            await websocket.close()
            return
        user_info = {
            "id": db_user["id"],
            "username": db_user.get("username"),
            "name": db_user["name"],
        }
        nickname = user_info["name"]
        user_id = user_info["id"]
        
        # 같은 user_id의 기존 연결을 제거하고 새 연결 추가 (로그아웃 후 재로그인 시 중복 방지)
        async with _room_connections_lock:
            _room_connections[:] = [(ws, u) for ws, u in _room_connections if u.get("id") != user_id]
            _room_connections.append((websocket, user_info))
            participants = _build_participants_list()
        
        await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=websocket)
        await websocket.send_text(json.dumps({"type": "participants", "list": participants}, ensure_ascii=False))
        await websocket.send_text(json.dumps({"type": "serena_status", "present": _serena_invited}, ensure_ascii=False))
        if _chess_game and _chess_game.get("status"):
            b = _chess_game["board"]
            lm = b.peek().uci() if len(b.move_stack) > 0 else None
            await websocket.send_text(json.dumps({
                "type": "chess_state",
                "fen": b.fen(),
                "white_player": _chess_game.get("white_player"),
                "black_player": _chess_game.get("black_player"),
                "mode": _chess_game.get("mode", "serena"),
                "status": _chess_game.get("status"),
                "turn": "white" if b.turn == chess.WHITE else "black",
                "last_move": lm,
                "in_check": b.is_check(),
                "white_captured": _chess_game.get("white_captured", []),
                "black_captured": _chess_game.get("black_captured", []),
            }, ensure_ascii=False))
        await _room_broadcast({"type": "system", "message": f"{nickname}님이 입장했습니다."}, exclude_ws=None)
        
        # 메시지 수신 루프
        while True:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                
                if data.get("type") == "chat":
                    text = (data.get("text") or "").strip()[:2000]
                    image_url = (data.get("image_url") or "").strip()[:500]
                    if not text and not image_url:
                        continue
                    message_id = str(uuid4())
                    total_participants = len(_room_connections)
                    _room_messages[message_id] = {"sender_nickname": nickname, "text": text, "image_url": image_url or None}
                    # 읽음 상태 초기화 (발신자는 자동 읽음 처리)
                    _message_read_status[message_id] = {
                        "sender_ws": websocket,
                        "sender_nickname": nickname,
                        "read_by": {nickname},
                        "total_participants": total_participants,
                    }
                    payload = {
                        "type": "chat",
                        "message_id": message_id,
                        "nickname": nickname,
                        "text": text,
                        "unread_count": total_participants - 1,
                        "timestamp": _seoul_time_str(),
                    }
                    if image_url:
                        payload["image_url"] = image_url
                    await _room_broadcast(payload, exclude_ws=None)

                    # Serena 초대용 최근 대화 저장 (최대 10개)
                    _room_message_history.append({"nickname": nickname, "text": text})
                    if len(_room_message_history) > 10:
                        _room_message_history.pop(0)

                    # Serena가 초대된 상태면 사용자 메시지 후 자동 응답 예약 (디바운스 3초)
                    if _serena_invited and nickname != "Serena":
                        _schedule_serena_response()

                    # 링크 미리보기: 텍스트에 URL이 있으면 백그라운드에서 og 태그 조회
                    url_match = re.search(r"https?://[^\s<>\"']+", text)
                    if url_match:
                        found_url = url_match.group(0).rstrip(".,;:!?)")
                        asyncio.create_task(_fetch_and_broadcast_link_preview(message_id, found_url))

                    # 읽음 상태 메모리 관리: 100개 이상이면 오래된 것 삭제
                    if len(_message_read_status) > 100:
                        old_ids = list(_message_read_status.keys())[:-50]  # 오래된 50개만 남기고 삭제
                        for old_id in old_ids:
                            _message_read_status.pop(old_id, None)
                
                elif data.get("type") == "read" and (data.get("message_id") or "").strip():
                    message_id = data["message_id"]
                    if message_id in _message_read_status:
                        status = _message_read_status[message_id]
                        status["read_by"].add(nickname)
                        unread_count = status["total_participants"] - len(status["read_by"])
                        # 발신자에게 읽음 업데이트 전송
                        sender_ws = status["sender_ws"]
                        if sender_ws in [ws for ws, _ in _room_connections]:
                            try:
                                await sender_ws.send_text(json.dumps({
                                    "type": "read_update",
                                    "message_id": message_id,
                                    "unread_count": unread_count,
                                }, ensure_ascii=False))
                            except Exception:
                                pass
                        # 모두 읽었으면 상태 삭제
                        if unread_count == 0:
                            del _message_read_status[message_id]

                elif data.get("type") == "edit" and (data.get("message_id") or "").strip():
                    message_id = data["message_id"]
                    new_text = (data.get("text") or "").strip()[:2000]
                    if not new_text:
                        continue
                    msg = _room_messages.get(message_id)
                    if not msg:
                        continue
                    if msg.get("sender_nickname") != nickname:
                        continue
                    msg["text"] = new_text
                    await _room_broadcast({
                        "type": "edit",
                        "message_id": message_id,
                        "text": new_text,
                    }, exclude_ws=None)

                elif data.get("type") == "delete" and (data.get("message_id") or "").strip():
                    message_id = data["message_id"]
                    msg = _room_messages.get(message_id)
                    if not msg:
                        continue
                    if msg.get("sender_nickname") != nickname:
                        continue
                    _room_messages.pop(message_id, None)
                    _message_read_status.pop(message_id, None)
                    await _room_broadcast({
                        "type": "delete",
                        "message_id": message_id,
                    }, exclude_ws=None)

                elif data.get("type") == "rename" and (data.get("nickname") or "").strip():
                    new_nick = (data["nickname"] or "").strip()[:100]
                    if 2 <= len(new_nick):
                        old_nick = nickname
                        nickname = new_nick
                        for i, (w, u) in enumerate(_room_connections):
                            if w is websocket:
                                new_user = {**u, "name": new_nick}
                                _room_connections[i] = (websocket, new_user)
                                db.update_user_name(u.get("id"), new_nick)
                                break
                        participants = _build_participants_list()
                        await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
                        await _room_broadcast({"type": "system", "message": f"{old_nick}님이 닉네임을 {new_nick}(으)로 변경했습니다."}, exclude_ws=None)

                elif data.get("type") == "invite_serena":
                    asyncio.create_task(_invite_serena_and_broadcast())

                elif data.get("type") == "kick_serena":
                    if _serena_invited:
                        _serena_invited = False
                        if _serena_pending_task and not _serena_pending_task.done():
                            _serena_pending_task.cancel()
                            _serena_pending_task = None
                        if _chess_pending_task and not _chess_pending_task.done():
                            _chess_pending_task.cancel()
                            _chess_pending_task = None
                        _chess_game = None
                        await _room_broadcast({"type": "chess_state", "fen": None, "status": None}, exclude_ws=None)
                        await _room_broadcast({"type": "system", "message": "Serena님이 퇴장했습니다."}, exclude_ws=None)
                        participants = _build_participants_list()
                        await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
                        await _room_broadcast({"type": "serena_status", "present": False}, exclude_ws=None)

                elif data.get("type") == "chess_start":
                    if not _serena_invited:
                        continue
                    if _chess_pending_task and not _chess_pending_task.done():
                        _chess_pending_task.cancel()
                        _chess_pending_task = None
                    board = chess.Board()
                    _chess_game = {
                        "board": board,
                        "white_player": nickname,
                        "black_player": "Serena",
                        "mode": "serena",
                        "status": "active",
                        "white_captured": [],
                        "black_captured": [],
                    }
                    await _room_broadcast({
                        "type": "chess_state",
                        "fen": board.fen(),
                        "white_player": nickname,
                        "black_player": "Serena",
                        "mode": "serena",
                        "status": "active",
                        "turn": "white",
                        "last_move": None,
                        "in_check": False,
                        "white_captured": [],
                        "black_captured": [],
                    }, exclude_ws=None)

                elif data.get("type") == "chess_start_pvp" and (data.get("opponent") or "").strip():
                    opp = (data["opponent"] or "").strip()[:32]
                    if opp == nickname:
                        continue
                    if not any(u.get("name") == opp for _, u in _room_connections):
                        continue
                    if _chess_pending_task and not _chess_pending_task.done():
                        _chess_pending_task.cancel()
                        _chess_pending_task = None
                    board = chess.Board()
                    _chess_game = {
                        "board": board,
                        "white_player": nickname,
                        "black_player": opp,
                        "mode": "pvp",
                        "status": "active",
                        "white_captured": [],
                        "black_captured": [],
                    }
                    await _room_broadcast({
                        "type": "chess_state",
                        "fen": board.fen(),
                        "white_player": nickname,
                        "black_player": opp,
                        "mode": "pvp",
                        "status": "active",
                        "turn": "white",
                        "last_move": None,
                        "in_check": False,
                        "white_captured": [],
                        "black_captured": [],
                    }, exclude_ws=None)

                elif data.get("type") == "chess_move" and (data.get("uci") or "").strip():
                    uci = (data["uci"] or "").strip().lower()[:10]
                    if not _chess_game or _chess_game.get("status") != "active":
                        continue
                    board: chess.Board = _chess_game["board"]
                    mode = _chess_game.get("mode", "serena")
                    white_p = _chess_game.get("white_player")
                    black_p = _chess_game.get("black_player")
                    if board.turn == chess.WHITE:
                        if nickname != white_p:
                            continue
                    else:
                        if mode == "serena":
                            continue
                        if nickname != black_p:
                            continue
                    try:
                        move = chess.Move.from_uci(uci)
                        if move not in board.legal_moves:
                            continue
                        captured = _get_captured_piece(board, move)
                        if captured:
                            if board.turn == chess.WHITE:
                                _chess_game.setdefault("white_captured", []).append(captured)
                            else:
                                _chess_game.setdefault("black_captured", []).append(captured)
                        board.push(move)
                        status = "active"
                        if board.is_checkmate():
                            status = "checkmate_black"
                        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                            status = "draw"
                        _chess_game["status"] = status
                        await _room_broadcast({
                            "type": "chess_state",
                            "fen": board.fen(),
                            "last_move": uci,
                            "status": status,
                            "turn": "white" if board.turn == chess.WHITE else "black",
                            "in_check": board.is_check(),
                            "white_captured": _chess_game.get("white_captured", []),
                            "black_captured": _chess_game.get("black_captured", []),
                            "white_player": white_p,
                            "black_player": black_p,
                            "mode": mode,
                        }, exclude_ws=None)
                        if status == "active" and board.turn == chess.BLACK and mode == "serena":
                            _chess_pending_task = asyncio.create_task(_play_serena_chess_move())
                    except (ValueError, chess.InvalidMoveError):
                        pass

                elif data.get("type") == "chess_resign":
                    if not _chess_game or _chess_game.get("status") != "active":
                        continue
                    if _chess_pending_task and not _chess_pending_task.done():
                        _chess_pending_task.cancel()
                        _chess_pending_task = None
                    if nickname == _chess_game.get("white_player"):
                        status = "resign_white"
                    elif nickname == _chess_game.get("black_player"):
                        status = "resign_black"
                    else:
                        continue
                    _chess_game["status"] = status
                    await _room_broadcast({
                        "type": "chess_state",
                        "fen": _chess_game["board"].fen(),
                        "status": status,
                        "in_check": False,
                        "white_captured": _chess_game.get("white_captured", []),
                        "black_captured": _chess_game.get("black_captured", []),
                    }, exclude_ws=None)
            
            except json.JSONDecodeError:
                # JSON 파싱 에러만 무시하고 계속 진행
                continue
            except WebSocketDisconnect:
                # 연결 종료 시 루프 탈출 (재진입 시 무한 로그 방지)
                raise
            except Exception as e:
                # 연결 불가/수신 불가 등은 루프 탈출
                if "receive" in str(e).lower() or "closed" in str(e).lower() or "connection" in str(e).lower():
                    raise
                print(f"WebSocket message error for {nickname}: {e}")
                continue
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket connection error for {nickname}: {e}")
    finally:
        if user_info is not None:
            async with _room_connections_lock:
                _room_connections[:] = [(w, u) for w, u in _room_connections if w is not websocket]
            # 퇴장한 유저가 읽지 않은 메시지의 읽음 상태 업데이트
            for message_id, status in list(_message_read_status.items()):
                if websocket == status["sender_ws"]:
                    # 발신자가 퇴장한 경우 메시지 상태 삭제
                    del _message_read_status[message_id]
                else:
                    # 퇴장한 유저가 수신자인 경우 total_participants 감소
                    if nickname not in status["read_by"]:
                        status["total_participants"] -= 1
                    unread_count = status["total_participants"] - len(status["read_by"])
                    # 발신자에게 업데이트 전송
                    sender_ws = status["sender_ws"]
                    if sender_ws in [ws for ws, _ in _room_connections]:
                        try:
                            await sender_ws.send_text(json.dumps({
                                "type": "read_update",
                                "message_id": message_id,
                                "unread_count": unread_count,
                            }, ensure_ascii=False))
                        except Exception:
                            pass
                    if unread_count == 0:
                        del _message_read_status[message_id]
                        _room_messages.pop(message_id, None)
            async with _room_connections_lock:
                participants = _build_participants_list()
            await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
            await _room_broadcast({"type": "system", "message": f"{nickname}님이 퇴장했습니다."}, exclude_ws=None)


@app.websocket("/ws/dm")
async def ws_dm(websocket: WebSocket) -> None:
    """1:1 대화방 WebSocket. 첫 메시지: {"type":"join","ws_token":"...","room_id":123}"""
    await websocket.accept()
    user_info: Optional[Dict[str, Any]] = None
    room_id: Optional[int] = None
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        if data.get("type") != "join":
            await websocket.send_text(json.dumps({"type": "error", "message": "Invalid join"}, ensure_ascii=False))
            await websocket.close()
            return
        user_info = auth_module.verify_ws_token((data.get("ws_token") or "").strip())
        if not user_info:
            await websocket.send_text(json.dumps({"type": "error", "message": "로그인 만료"}, ensure_ascii=False))
            await websocket.close()
            return
        user_id = user_info.get("user_id")
        if not user_id:
            await websocket.send_text(json.dumps({"type": "error", "message": "Invalid token"}, ensure_ascii=False))
            await websocket.close()
            return
        room_id = int(data.get("room_id") or 0)
        room = db.get_dm_room(room_id, user_id)
        if not room:
            await websocket.send_text(json.dumps({"type": "error", "message": "대화방을 찾을 수 없습니다."}, ensure_ascii=False))
            await websocket.close()
            return
        # 기존 메시지 전송 (저장 시각은 UTC이므로 서울 시간으로 변환해 전송)
        messages = db.get_dm_messages(room_id, user_id)
        for m in messages:
            raw = m.get("created_at") or ""
            ts = _utc_datetime_str_to_seoul_hhmm(raw)
            msg_id = m.get("id")
            message_id = f"{room_id}-{msg_id}"
            is_me = m.get("is_me", False)
            payload = {
                "type": "chat",
                "message_id": message_id,
                "nickname": m["nickname"],
                "text": m.get("text", ""),
                "image_url": m.get("image_url"),
                "timestamp": ts,
                "is_me": is_me,
                "is_history": True,
            }
            # 내가 보낸 메시지 중 상대가 아직 읽지 않은 것은 unread_count 전달 (배지는 수신자 확인 시에만 사라짐)
            if is_me:
                status = _dm_message_read_status.get((room_id, message_id))
                if status and not status.get("read"):
                    payload["unread_count"] = 1
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        # 접속 등록 (같은 user_id의 기존 연결 제거하고 추가)
        db_user = db.get_user_by_id(user_id)
        nickname = (db_user and db_user.get("name")) or user_info.get("name") or "User"
        async with _dm_connections_lock:
            if room_id not in _dm_connections:
                _dm_connections[room_id] = []
            _dm_connections[room_id] = [(ws, u) for ws, u in _dm_connections[room_id] if u.get("id") != user_id]
            _dm_connections[room_id].append((websocket, {"id": user_id, "name": nickname}))
        # 재입장 시: 내가 보낸 미읽음 메시지의 sender_ws를 이 연결로 갱신 (read_update 수신 가능하도록)
        for key, status in list(_dm_message_read_status.items()):
            if key[0] == room_id and status.get("sender_id") == user_id and not status.get("read"):
                status["sender_ws"] = websocket

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("type") == "chat":
                text = (data.get("text") or "").strip()[:2000]
                image_url = (data.get("image_url") or "").strip()[:500]
                if not text and not image_url:
                    continue
                # DB 저장 (Serena 없음 - 1:1은 유저 간만)
                msg_id = db.save_dm_message(room_id, user_id, text, image_url or None)
                ts = _seoul_time_str()
                message_id = f"{room_id}-{msg_id}"
                payload = {
                    "type": "chat",
                    "message_id": message_id,
                    "nickname": nickname,
                    "text": text,
                    "timestamp": ts,
                }
                if image_url:
                    payload["image_url"] = image_url
                
                # 같은 방의 다른 참여자에게 전송
                room_obj = db.get_dm_room(room_id, user_id)
                recipient_id = room_obj["other_user"]["id"] if room_obj else None
                recipient_in_dm = False
                
                # 수신자가 DM 채팅창을 열고 있는지 확인
                for ws, u in _dm_connections.get(room_id, []):
                    if ws is not websocket and u.get("id") == recipient_id:
                        recipient_in_dm = True
                        break
                
                # 읽음 상태 초기화 (수신자가 채팅창 열고 있으면 읽음 처리)
                unread_count = 0 if recipient_in_dm else 1
                _dm_message_read_status[(room_id, message_id)] = {
                    "sender_id": user_id,
                    "sender_ws": websocket,
                    "read": recipient_in_dm,
                }
                
                # 발신자와 수신자에게 메시지 전송
                for ws, u in _dm_connections.get(room_id, []):
                    if ws is not websocket:
                        try:
                            p = {**payload, "is_me": False}
                            await ws.send_text(json.dumps(p, ensure_ascii=False))
                        except Exception:
                            pass
                    else:
                        try:
                            p = {**payload, "is_me": True, "unread_count": unread_count}
                            await websocket.send_text(json.dumps(p, ensure_ascii=False))
                        except Exception:
                            pass
                # 수신자가 DM 탭을 열지 않았으면 멀티방 WebSocket으로 new_dm 알림
                if recipient_id and not recipient_in_dm:
                    sender_info = db.get_user_by_id(user_id)
                    for ws_multi, u_multi in _room_connections:
                        if u_multi.get("id") == recipient_id:
                            try:
                                await ws_multi.send_text(json.dumps({
                                    "type": "new_dm",
                                    "room_id": room_id,
                                    "other_user": sender_info or {"id": user_id, "name": nickname, "username": None},
                                    "preview": (text or "")[:50] or "(사진)",
                                }, ensure_ascii=False))
                            except Exception:
                                pass
                            break
            
            elif data.get("type") == "read" and (data.get("message_id") or "").strip():
                # DM 메시지 읽음 처리 (개별 메시지)
                message_id = data["message_id"]
                key = (room_id, message_id)
                if key in _dm_message_read_status:
                    status = _dm_message_read_status[key]
                    status["read"] = True
                    sender_ws = status["sender_ws"]
                    if sender_ws and sender_ws in [w for w, _ in _dm_connections.get(room_id, [])]:
                        try:
                            await sender_ws.send_text(json.dumps({
                                "type": "read_update",
                                "message_id": message_id,
                                "unread_count": 0,
                            }, ensure_ascii=False))
                        except Exception:
                            pass
                    del _dm_message_read_status[key]
            
            elif data.get("type") == "viewed_room":
                # 상대가 해당 1:1 채팅 화면을 활성화했을 때만: 이 방에서 나에게 온 모든 미읽음 메시지를 읽음 처리
                to_delete = []
                for key, status in _dm_message_read_status.items():
                    if key[0] != room_id:
                        continue
                    if status.get("sender_id") == user_id:
                        continue
                    status["read"] = True
                    sender_ws = status.get("sender_ws")
                    if sender_ws and sender_ws in [w for w, _ in _dm_connections.get(room_id, [])]:
                        try:
                            await sender_ws.send_text(json.dumps({
                                "type": "read_update",
                                "message_id": key[1],
                                "unread_count": 0,
                            }, ensure_ascii=False))
                        except Exception:
                            pass
                    to_delete.append(key)
                for k in to_delete:
                    _dm_message_read_status.pop(k, None)
            
            elif data.get("type") == "edit" and (data.get("message_id") or "").strip():
                # DM 메시지 수정 (본인 메시지만)
                message_id_str = data["message_id"]
                new_text = (data.get("text") or "").strip()[:2000]
                parts = message_id_str.split("-", 1)
                if len(parts) != 2:
                    continue
                try:
                    msg_id = int(parts[1])
                except ValueError:
                    continue
                if db.update_dm_message(room_id, msg_id, user_id, text=new_text):
                    for ws, _ in _dm_connections.get(room_id, []):
                        try:
                            await ws.send_text(json.dumps({
                                "type": "edit",
                                "message_id": message_id_str,
                                "text": new_text,
                            }, ensure_ascii=False))
                        except Exception:
                            pass
            
            elif data.get("type") == "delete" and (data.get("message_id") or "").strip():
                # DM 메시지 삭제 (본인 메시지만)
                message_id_str = data["message_id"]
                parts = message_id_str.split("-", 1)
                if len(parts) != 2:
                    continue
                try:
                    msg_id = int(parts[1])
                except ValueError:
                    continue
                if db.delete_dm_message(room_id, msg_id, user_id):
                    for ws, _ in _dm_connections.get(room_id, []):
                        try:
                            await ws.send_text(json.dumps({
                                "type": "delete",
                                "message_id": message_id_str,
                            }, ensure_ascii=False))
                        except Exception:
                            pass
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"DM WebSocket error: {e}")
    finally:
        if room_id and room_id in _dm_connections:
            async with _dm_connections_lock:
                _dm_connections[room_id] = [(w, u) for w, u in _dm_connections[room_id] if w is not websocket]
                if not _dm_connections[room_id]:
                    del _dm_connections[room_id]
        # 발신자 퇴장 시: 읽음 상태는 유지하고 sender_ws만 None으로 (상대가 확인했을 때만 배지 사라지므로)
        for v in _dm_message_read_status.values():
            if v.get("sender_ws") == websocket:
                v["sender_ws"] = None


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/situations")
def api_situations() -> List[Dict[str, Any]]:
    """대화 상황 목록 반환."""
    return [
        {"index": i, "title": s.split(":")[0].strip() if ":" in s else s, "title_ko": SITUATION_KO.get(s.split(":")[0].strip(), "")}
        for i, s in enumerate(CONVERSATION_SITUATIONS)
    ]


@app.post("/api/start")
def api_start(body: Optional[StartRequest] = None) -> StartResponse:
    """새 대화 시작. situation_index가 있으면 해당 상황, 없으면 랜덤."""
    try:
        client = get_client()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    idx = body.situation_index if body and body.situation_index is not None and body.situation_index >= 0 else None
    if idx is not None and 0 <= idx < len(CONVERSATION_SITUATIONS):
        situation = CONVERSATION_SITUATIONS[idx]
    else:
        situation = secrets.choice(CONVERSATION_SITUATIONS)
    situation_title = situation.split(":")[0].strip() if ":" in situation else situation
    situation_ko = SITUATION_KO.get(situation_title, situation_title)
    situation_display = f"{situation_title} ({situation_ko})"

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise HTTPException(status_code=500, detail="OPENAI_MODEL not configured")
    
    system_content = f"{SYSTEM_PROMPT}\n\nCurrent situation:\n{situation}"
    
    # gpt-5-mini/nano는 Responses API 사용
    use_responses_api = any(x in model.lower() for x in ['gpt-5-mini', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.2'])
    
    if use_responses_api:
        # Responses API
        resp = client.responses.create(
            model=model,
            input=system_content,
            max_output_tokens=500,
        )
        if hasattr(resp, 'output_text'):
            first_reply = resp.output_text.strip()
        else:
            # output이 list인 경우
            first_reply = ""
            for item in resp.output:
                if hasattr(item, 'content'):
                    for c in item.content:
                        if hasattr(c, 'text'):
                            first_reply = c.text.strip()
                            break
    else:
        # Chat Completions API
        messages = [{"role": "system", "content": system_content}]
        first = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=300,
        )
        first_reply = first.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": first_reply})
    first_message = first_reply
    first_korean = None
    if KOREAN_LABEL in first_reply:
        idx = first_reply.index(KOREAN_LABEL)
        first_message = first_reply[:idx].strip()
        first_korean = first_reply[idx + len(KOREAN_LABEL):].strip()
    
    # 대화 히스토리 저장 (Responses API는 텍스트 기반으로 변환 필요)
    if use_responses_api:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": first_reply}
        ]
    # Chat Completions API는 이미 messages에 추가됨

    conv_id = uuid4().hex
    _conversations[conv_id] = {
        "situation": situation,
        "situation_title": situation_title,
        "messages": messages,
        "use_responses_api": use_responses_api,  # API 타입 저장
    }

    return StartResponse(
        conversation_id=conv_id,
        situation=situation,
        situation_title=situation_title,
        situation_display=situation_display,
        first_message=first_message,
        first_korean=first_korean,
        messages=[{"role": "assistant", "content": first_reply}],
    )


@app.post("/api/chat")
def api_chat(body: ChatRequest) -> ChatResponse:
    """사용자 메시지 전송 후 교정/점수/다음 질문 반환."""
    conv = _conversations.get(body.conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msg = (body.user_message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    if user_msg.lower() in ("quit", "exit", "q"):
        return ChatResponse(
            correction=None,
            score=None,
            reply="대화를 종료합니다. 새 대화를 시작하려면 '새 대화' 버튼을 누르세요.",
            raw_reply=None,
        )

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise HTTPException(status_code=500, detail="OPENAI_MODEL not configured")
    
    try:
        client = get_client()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages = conv["messages"]
    messages.append({"role": "user", "content": user_msg})
    
    use_responses_api = conv.get("use_responses_api", False)
    
    if use_responses_api:
        # Responses API: 대화 히스토리를 텍스트로 변환
        conversation_text = "\n".join([
            f"{'System' if m['role'] == 'system' else 'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
            for m in messages
        ])
        resp = client.responses.create(
            model=model,
            input=conversation_text,
            max_output_tokens=1000,
        )
        if hasattr(resp, 'output_text'):
            ai_text = resp.output_text.strip()
        else:
            ai_text = ""
            for item in resp.output:
                if hasattr(item, 'content'):
                    for c in item.content:
                        if hasattr(c, 'text'):
                            ai_text = c.text.strip()
                            break
    else:
        # Chat Completions API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=650,
        )
        ai_text = response.choices[0].message.content.strip()
    
    messages.append({"role": "assistant", "content": ai_text})

    correction, score, reply, korean_reply = parse_ai_response(ai_text)

    if reply is None:
        reply = ai_text

    return ChatResponse(
        correction=correction,
        score=score,
        reply=reply or ai_text,
        korean_reply=korean_reply,
        raw_reply=ai_text if (correction is None and score is None and not reply) else None,
    )
