"""
영어 채팅 챗봇 웹 서비스
- FastAPI 기반 채팅 UI
- 175.197.131.234:8004 에서 접속
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
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

app = FastAPI(title="English Chat Bot - 영어 대화 연습")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# WebSocket은 static 마운트 전에 등록 (프록시/경로 매칭 우선)

# 세션별 대화 저장 (in-memory)
_conversations: Dict[str, Dict[str, Any]] = {}

# 멀티 채팅방: (websocket, nickname) 목록
_room_connections: List[tuple[WebSocket, str]] = []


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


async def _room_broadcast(message: Dict[str, Any], exclude_ws: Optional[WebSocket] = None) -> None:
    """멀티 채팅방 전체에 메시지 브로드캐스트."""
    text = json.dumps(message, ensure_ascii=False)
    dead = []
    for ws, _ in _room_connections:
        if ws is exclude_ws:
            continue
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _room_connections[:] = [(w, n) for w, n in _room_connections if w is not ws]


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


@app.websocket("/ws/room")
async def ws_room(websocket: WebSocket) -> None:
    """멀티 채팅방 WebSocket. 첫 메시지: {"type":"join","nickname":"..."} 이후 {"type":"chat","text":"..."}."""
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
        _room_connections.append((websocket, nickname))
        participants = [n for _, n in _room_connections]
        await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=websocket)
        await websocket.send_text(json.dumps({"type": "participants", "list": participants}, ensure_ascii=False))
        await _room_broadcast({"type": "system", "message": f"{nickname}님이 입장했습니다."}, exclude_ws=None)
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("type") == "chat" and (data.get("text") or "").strip():
                text = (data["text"] or "").strip()[:2000]
                await _room_broadcast({"type": "chat", "nickname": nickname, "text": text}, exclude_ws=None)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if nickname is not None:
            _room_connections[:] = [(w, n) for w, n in _room_connections if w is not websocket]
            participants = [n for _, n in _room_connections]
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
