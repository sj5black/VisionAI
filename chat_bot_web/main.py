"""
영어 채팅 챗봇 웹 서비스
- FastAPI 기반 채팅 UI
- 175.197.131.234:8004 에서 접속
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# english_chat_bot 로직 재사용
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from english_chat_bot import (
    get_client,
    parse_ai_response,
    CONVERSATION_SITUATIONS,
    SYSTEM_PROMPT,
    KOREAN_LABEL,
)

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
# WebSocket은 static 마운트 전에 등록 (프록시/경로 매칭 우선)

# 세션별 대화 저장 (in-memory)
_conversations: Dict[str, Dict[str, Any]] = {}

# 멀티 채팅방: (websocket, nickname) 목록
_room_connections: List[tuple[WebSocket, str]] = []
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

Rules:
- You are a native English speaker. Respond ONLY in English, no matter what language others use.
- When someone's message contains Korean (mixed with English or alone), naturally work into your reply the English equivalent for that Korean. For example: "By the way, '맛있어' is 'it's delicious' in English :)" or "Just so you know, that word means ~ in English." Keep it brief and friendly so it helps them practice English.
- If the message is mostly or only Korean and you can't infer meaning, say something like "I didn't catch that - could you try in English?" and encourage them to use English.
- Be conversational and natural—react, comment, ask questions, or add to the discussion.
- Keep your replies short (1–3 sentences). If you add a Korean→English tip, one short phrase is enough.
- Don't mention that you are AI. You're just a regular person who happens to only speak English.
- Stay in character as a friendly, English-speaking person joining the chat.
- Do NOT use Unicode emoji (e.g. 😀 👍 ❤️). Instead, occasionally use simple text emoticons like lol, :), :(, T-T, ^^, haha when it fits naturally."""


def _call_serena(recent_messages: List[Dict[str, str]]) -> Optional[str]:
    """OpenAI로 Serena 응답 생성 (동기)."""
    try:
        client = get_client()
        conv = "\n".join(f"{m['nickname']}: {m['text']}" for m in recent_messages)
        if not conv.strip():
            conv = "(No recent messages)"
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SERENA_SYSTEM},
                {"role": "user", "content": f"Recent chat messages (some may contain Korean):\n{conv}\n\nRespond in English. If any message has Korean in it, briefly give the English equivalent for that Korean (e.g. 'that word means X in English') in a natural, friendly way."},
            ],
            max_completion_tokens=150,
        )
        if resp.choices and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Serena OpenAI error: {e}")
    return None


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
            participants = [n for _, n in _room_connections] + ["Serena"]
            await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
            await _room_broadcast({"type": "serena_status", "present": True}, exclude_ws=None)
        
        message_id = str(uuid4())
        await _room_broadcast({
            "type": "chat",
            "message_id": message_id,
            "nickname": "Serena",
            "text": reply,
            "unread_count": 0,
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


async def _room_broadcast(message: Dict[str, Any], exclude_ws: Optional[WebSocket] = None) -> None:
    """멀티 채팅방 전체에 메시지 브로드캐스트."""
    text = json.dumps(message, ensure_ascii=False)
    dead = []
    for ws, nick in _room_connections:
        if ws is exclude_ws:
            continue
        try:
            await ws.send_text(text)
        except Exception as e:
            print(f"Broadcast error to {nick}: {e}")
            dead.append((ws, nick))
    # 죽은 연결 정리
    for ws, nick in dead:
        _room_connections[:] = [(w, n) for w, n in _room_connections if w is not ws]
        print(f"Removed dead connection: {nick}")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    """랜딩: AI와 채팅 / 사용자들과 채팅 선택."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request) -> Any:
    """AI 영어 채팅 페이지."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/room", response_class=HTMLResponse)
def room_page(request: Request) -> Any:
    """멀티 채팅방 페이지 (닉네임 입력 후 유저 간 실시간 채팅)."""
    return templates.TemplateResponse("room.html", {"request": request})


@app.get("/api/room/saved-nickname")
def api_room_saved_nickname(request: Request) -> Any:
    """요청 IP에 저장된 닉네임이 있으면 반환 (멀티채팅 재접속 시 자동 입장용)."""
    client_host = request.client.host if request.client else None
    nickname = _ip_nickname.get(client_host) if client_host else None
    return {"nickname": nickname}


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
    첫 메시지: {"type":"join","nickname":"..."}
    이후:
    - {"type":"chat","text":"..."}
    - {"type":"read","message_id":"..."}
    - {"type":"edit","message_id":"...","text":"..."}  (발신자만)
    - {"type":"delete","message_id":"..."}            (발신자만)
    """
    global _serena_invited, _serena_pending_task
    await websocket.accept()
    nickname: Optional[str] = None
    try:
        # 첫 메시지: join + nickname
        raw = await websocket.receive_text()
        data = json.loads(raw)
        if data.get("type") != "join" or not (data.get("nickname") or "").strip():
            await websocket.send_text(json.dumps({"type": "error", "message": "닉네임을 입력해 주세요."}, ensure_ascii=False))
            await websocket.close()
            return
        nickname = (data["nickname"] or "").strip()[:32]
        client_host = websocket.client.host if websocket.client else None
        if client_host:
            _ip_nickname[client_host] = nickname
        _room_connections.append((websocket, nickname))
        participants = [n for _, n in _room_connections]
        if _serena_invited:
            participants.append("Serena")
        await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=websocket)
        await websocket.send_text(json.dumps({"type": "participants", "list": participants}, ensure_ascii=False))
        await websocket.send_text(json.dumps({"type": "serena_status", "present": _serena_invited}, ensure_ascii=False))
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
                    new_nick = (data["nickname"] or "").strip()[:32]
                    if 2 <= len(new_nick) <= 32:
                        old_nick = nickname
                        nickname = new_nick
                        if client_host:
                            _ip_nickname[client_host] = new_nick
                        for i, (w, n) in enumerate(_room_connections):
                            if w is websocket:
                                _room_connections[i] = (websocket, new_nick)
                                break
                        participants = [n for _, n in _room_connections]
                        if _serena_invited:
                            participants.append("Serena")
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
                        await _room_broadcast({"type": "system", "message": "Serena님이 퇴장했습니다."}, exclude_ws=None)
                        participants = [n for _, n in _room_connections]
                        await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
                        await _room_broadcast({"type": "serena_status", "present": False}, exclude_ws=None)
            
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
        if nickname is not None:
            _room_connections[:] = [(w, n) for w, n in _room_connections if w is not websocket]
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
            participants = [n for _, n in _room_connections]
            if _serena_invited:
                participants.append("Serena")
            await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
            await _room_broadcast({"type": "system", "message": f"{nickname}님이 퇴장했습니다."}, exclude_ws=None)


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

    system_content = f"{SYSTEM_PROMPT}\n\nCurrent situation:\n{situation}"
    messages = [
        {"role": "system", "content": system_content},
    ]

    first = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=messages,
        max_completion_tokens=300,
    )
    first_reply = first.choices[0].message.content.strip()
    first_message = first_reply
    first_korean = None
    if KOREAN_LABEL in first_reply:
        idx = first_reply.index(KOREAN_LABEL)
        first_message = first_reply[:idx].strip()
        first_korean = first_reply[idx + len(KOREAN_LABEL):].strip()
    messages.append({"role": "assistant", "content": first_reply})

    conv_id = uuid4().hex
    _conversations[conv_id] = {
        "situation": situation,
        "situation_title": situation_title,
        "messages": messages,
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

    try:
        client = get_client()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages = conv["messages"]
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
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
