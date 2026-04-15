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
import random
import re
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import requests
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# english_chat_bot 로직 재사용
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# .env를 프로젝트 루트에서 명시 로드 (Serena 초대 등 OPENAI_* 사용)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import chess

try:
    import chess.engine
except ImportError:
    chess.engine = None  # type: ignore

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
# 모든 유저 MMR을 650으로 초기화 (RESET_MMR=1 일 때만 1회 실행, 완료 후 환경변수 제거)
if os.getenv("RESET_MMR") == "1":
    n = db.reset_all_mmr(650, 0)
    print(f"MMR 초기화 완료: {n}명 → 650")

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
# 선수당 10분 + 매 수 5초 증가
CHESS_INITIAL_SECONDS = 15 * 60  # 15분
CHESS_INCREMENT_SECONDS = 10
# {"board", "white_player", "black_player", "mode", "status", "white_captured", "black_captured", "white_time_remaining", "black_time_remaining", "turn_started_at"}
_chess_game: Optional[Dict[str, Any]] = None
_chess_pending_task: Optional[asyncio.Task] = None

# Stockfish 엔진 (mode "stockfish"일 때만 사용). asyncio 네이티브 API 사용.
_stockfish_protocol: Any = None  # chess.engine.UciProtocol (open 상태일 때만 non-None)
# 환경변수로 지정된 경로 우선, 없으면 일반적인 경로 순서로 시도
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "").strip()
STOCKFISH_CANDIDATE_PATHS = (
    [STOCKFISH_PATH] if STOCKFISH_PATH else []
) + ["stockfish", "/usr/bin/stockfish", "/usr/games/stockfish"]
# 난이도: (표시명, Elo, SkillLevel, Depth, blunder_rate, random_move_prob, use_clock)
# 인간 Elo에 가까운 체감을 위해 블런더율·후보수 랜덤 선택 사용
STOCKFISH_LEVELS = [
    # (표시명, Elo, SkillLevel, Depth, blunder_rate, random_move_prob, use_clock)
    ("Bronze IV", 200, 0, 1, 0.10, 0.30, False),
    ("Bronze III", 275, 0, 1, 0.095, 0.284, False),
    ("Bronze II", 350, 1, 2, 0.09, 0.267, False),
    ("Bronze I", 425, 2, 2, 0.085, 0.25, False),

    ("Silver IV", 500, 2, 2, 0.08, 0.235, False),
    ("Silver III", 575, 3, 3, 0.075, 0.22, False),
    ("Silver II", 650, 3, 3, 0.07, 0.20, False),
    ("Silver I", 725, 4, 4, 0.065, 0.19, False),

    ("Gold IV", 800, 4, 4, 0.06, 0.18, False),
    ("Gold III", 875, 5, 4, 0.055, 0.17, False),
    ("Gold II", 950, 5, 5, 0.05, 0.16, False),
    ("Gold I", 1025, 6, 5, 0.045, 0.15, False),

    ("Platinum IV", 1100, 7, 6, 0.04, 0.14, False),
    ("Platinum III", 1175, 8, 6, 0.038, 0.13, False),
    ("Platinum II", 1250, 9, 7, 0.035, 0.12, False),
    ("Platinum I", 1325, 10, 7, 0.032, 0.11, False),

    ("Diamond IV", 1400, 11, 8, 0.03, 0.10, True),
    ("Diamond III", 1500, 12, 9, 0.025, 0.09, True),
    ("Diamond II", 1600, 13, 10, 0.02, 0.08, True),
    ("Diamond I", 1700, 14, 11, 0.018, 0.07, True),

    ("Master", 2000, 17, 13, 0.012, 0.05, True),
    ("GrandMaster", 2500, 20, 16, 0.006, 0.03, True),
    ("Challenger", 3000, 20, 20, 0.0, 0.0, True),
]


def _elo_to_think_range(elo: int) -> tuple[float, float]:
    """Elo 구간별 인간 평균 생각 시간(초) 범위."""
    e = int(elo)
    # if e <= 300:
    #     return 0.5, 4.0
    # if e <= 500:
    #     return 0.5, 5.0
    # if e <= 700:
    #     return 0.5, 7.0
    # if e <= 900:
    #     return 0.5, 8.0
    # if e <= 1100:
    #     return 0.5, 10.0
    # if e <= 1300:
    #     return 0.5, 12.0
    # if e <= 1500:
    #     return 0.5, 14.0
    # if e <= 1700:
    #     return 1.0, 16.0
    # if e <= 1900:
    #     return 1.0, 18.0
    if e <= 300:
        return 0.5, 2.0
    if e <= 500:
        return 0.5, 2.0
    if e <= 700:
        return 0.5, 2.0
    if e <= 900:
        return 0.5, 2.0
    if e <= 1100:
        return 0.5, 2.0
    if e <= 1300:
        return 0.5, 2.0
    if e <= 1500:
        return 0.5, 2.0
    if e <= 1700:
        return 0.0, 2.0
    if e <= 1900:
        return 0.0, 2.0
    if e >= 2800:
        return 0.0, 2.0
    return 0.0, 2.0


def _mmr_to_tier(mmr: Optional[int]) -> str:
    """MMR 구간별 등급명 반환 (STOCKFISH_LEVELS Elo 구간과 동일)."""
    if mmr is None:
        return ""
    m = int(mmr)
    name = STOCKFISH_LEVELS[0][0]
    for t in STOCKFISH_LEVELS:
        if m >= t[1]:
            name = t[0]
    return name


# 오목 게임: 15x15, 0=빈칸 1=흑 2=백. 흑 선공.
# {"board": List[List[int]], "black_player": str, "white_player": str, "mode": "serena"|"pvp", "status": str, "turn": "black"|"white"}
GOMOKU_SIZE = 15
_gomoku_game: Optional[Dict[str, Any]] = None
_gomoku_pending_task: Optional[asyncio.Task] = None

# 1:1 DM: {room_id: [(websocket, user_info), ...]}
_dm_connections: Dict[int, List[tuple[WebSocket, Dict[str, Any]]]] = {}
_dm_connections_lock = asyncio.Lock()
# DM 메시지 읽음 상태: {(room_id, message_id): {"sender_id": int, "sender_ws": WebSocket, "read": bool}}
_dm_message_read_status: Dict[tuple[int, str], Dict[str, Any]] = {}


def _apply_chess_mmr_if_needed(game: Dict[str, Any]) -> None:
    """체스 게임 종료 시 MMR 업데이트 (PvP: 양쪽, Stockfish: 사람만, 해당 단계 Elo 기준)."""
    if not game or game.get("mmr_applied"):
        return
    status = game.get("status")
    terminal_statuses = {
        "checkmate_white",
        "checkmate_black",
        "time_loss_white",
        "time_loss_black",
        "resign_white",
        "resign_black",
        "draw",
    }
    if status not in terminal_statuses:
        return

    mode = game.get("mode", "pvp")
    white_user_id = game.get("white_user_id")
    black_user_id = game.get("black_user_id")

    # --- Stockfish: 사람 vs 해당 단계 Elo, 사람 MMR만 변동 ---
    if mode == "stockfish":
        if game.get("stockfish_casual"):
            game["mmr_applied"] = True
            game["white_mmr_delta"] = None
            game["black_mmr_delta"] = None
            return
        stockfish_elo = int(game.get("stockfish_elo") or 1500)
        human_user_id = None
        human_side = None
        if white_user_id and game.get("white_player") != "Stockfish":
            human_user_id = white_user_id
            human_side = "white"
        elif black_user_id and game.get("black_player") != "Stockfish":
            human_user_id = black_user_id
            human_side = "black"
        if not human_user_id:
            return
        hr, hg = db.get_user_mmr(int(human_user_id))
        if status in ("checkmate_black", "time_loss_black", "resign_black"):
            w_human = 1.0 if human_side == "white" else 0.0
        elif status in ("checkmate_white", "time_loss_white", "resign_white"):
            w_human = 1.0 if human_side == "black" else 0.0
        else:
            w_human = 0.5
        def expected_score(r_self: float, r_opp: float) -> float:
            return 1.0 / (1.0 + 10.0 ** ((r_opp - r_self) / 400.0))
        e_human = expected_score(hr, float(stockfish_elo))
        K = 40
        new_hr = int(round(hr + K * (w_human - e_human)))
        db.update_user_mmr(int(human_user_id), new_hr, hg + 1)
        game["mmr_applied"] = True
        if human_side == "white":
            game["white_mmr_delta"] = new_hr - hr
            game["black_mmr_delta"] = None
        else:
            game["white_mmr_delta"] = None
            game["black_mmr_delta"] = new_hr - hr
        return

    # --- PvP: 양쪽 MMR 변동 ---
    if mode != "pvp" or not white_user_id or not black_user_id:
        return

    if status in ("checkmate_black", "time_loss_black", "resign_black"):
        w_white = 1.0
        w_black = 0.0
    elif status in ("checkmate_white", "time_loss_white", "resign_white"):
        w_white = 0.0
        w_black = 1.0
    else:
        w_white = 0.5
        w_black = 0.5

    ra, ga = db.get_user_mmr(int(white_user_id))
    rb, gb = db.get_user_mmr(int(black_user_id))

    def expected_score(r_self: float, r_opp: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_opp - r_self) / 400.0))
    ea = expected_score(ra, rb)
    eb = expected_score(rb, ra)
    K = 40
    new_ra = int(round(ra + K * (w_white - ea)))
    new_rb = int(round(rb + K * (w_black - eb)))

    db.update_user_mmr(int(white_user_id), new_ra, ga + 1)
    db.update_user_mmr(int(black_user_id), new_rb, gb + 1)
    game["mmr_applied"] = True
    game["white_mmr_delta"] = new_ra - ra
    game["black_mmr_delta"] = new_rb - rb


def _chess_state_payload(game: Dict[str, Any]) -> Dict[str, Any]:
    """체스 상태 브로드캐스트용 페이로드 (시간 포함)."""
    board: chess.Board = game["board"]
    white_user_id = game.get("white_user_id")
    black_user_id = game.get("black_user_id")
    white_rating = db.get_user_mmr(int(white_user_id))[0] if white_user_id else None
    black_rating = db.get_user_mmr(int(black_user_id))[0] if black_user_id else None
    payload = {
        "type": "chess_state",
        "fen": board.fen(),
        "white_player": game.get("white_player"),
        "black_player": game.get("black_player"),
        "white_rating": white_rating,
        "black_rating": black_rating,
        "white_tier": _mmr_to_tier(white_rating),
        "black_tier": _mmr_to_tier(black_rating),
        "mode": game.get("mode", "pvp"),
        "status": game.get("status", "active"),
        "paused": bool(game.get("paused")),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "last_move": board.peek().uci() if len(board.move_stack) > 0 else None,
        "in_check": board.is_check(),
        "white_captured": game.get("white_captured", []),
        "black_captured": game.get("black_captured", []),
        "white_time": max(0.0, game.get("white_time_remaining", CHESS_INITIAL_SECONDS)),
        "black_time": max(0.0, game.get("black_time_remaining", CHESS_INITIAL_SECONDS)),
        "turn_started_at": game.get("turn_started_at") or time.time(),
    }
    if game.get("mode") == "stockfish":
        payload["stockfish_casual"] = bool(game.get("stockfish_casual"))
    if game.get("white_mmr_delta") is not None:
        payload["white_mmr_delta"] = game["white_mmr_delta"]
    if game.get("black_mmr_delta") is not None:
        payload["black_mmr_delta"] = game["black_mmr_delta"]
    mh_uci = [m.uci() for m in board.move_stack]
    payload["move_history"] = mh_uci
    _bsan = chess.Board()
    mh_san: List[str] = []
    for _m in board.move_stack:
        mh_san.append(_bsan.san(_m))
        _bsan.push(_m)
    payload["move_history_san"] = mh_san
    return payload


# 기보 재생: 종료된 대국만 (진행 중 active 제외)
CHESS_REPLAY_TERMINAL_STATUSES = frozenset({
    "checkmate_white",
    "checkmate_black",
    "time_loss_white",
    "time_loss_black",
    "resign_white",
    "resign_black",
    "draw",
})


def _chess_position_at_ply(ucis: List[str], ply: int) -> Dict[str, Any]:
    """초기 국면에서 UCI 수열 앞 ply개만 적용한 FEN·잡은 말·마지막 수."""
    b = chess.Board()
    white_cap: List[str] = []
    black_cap: List[str] = []
    last_uci: Optional[str] = None
    n = max(0, min(int(ply), len(ucis)))
    for i in range(n):
        mv = chess.Move.from_uci(ucis[i])
        cap = _get_captured_piece(b, mv)
        if cap:
            if b.turn == chess.WHITE:
                white_cap.append(cap)
            else:
                black_cap.append(cap)
        b.push(mv)
        last_uci = ucis[i]
    return {
        "fen": b.fen(),
        "white_captured": white_cap,
        "black_captured": black_cap,
        "last_move": last_uci,
        "turn": "white" if b.turn == chess.WHITE else "black",
        "in_check": b.is_check(),
    }


def _get_captured_piece(board: chess.Board, move: chess.Move) -> Optional[str]:
    """캡처된 기물 기호 반환 (push 전에 호출). 앙파상 포함."""
    if not board.is_capture(move):
        return None
    p = board.piece_at(move.to_square)
    if p is None and board.is_en_passant(move):
        p = chess.Piece(chess.PAWN, not board.turn)
    return p.symbol() if p else None


def _normalize_uci_promotion(board: chess.Board, uci: str) -> str:
    """폰 프로모션: 5번째 글자가 q/r/b/n이면 그대로, 4자만 오면 퀸(q) 기본."""
    uci = (uci or "").strip().lower()[:10]
    if len(uci) < 4:
        return uci
    try:
        from_sq = chess.parse_square(uci[:2])
        to_sq = chess.parse_square(uci[2:4])
    except ValueError:
        return uci
    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.PAWN:
        return uci[:4]
    to_rank = chess.square_rank(to_sq)
    from_rank = chess.square_rank(from_sq)
    is_promotion = (piece.color == chess.WHITE and from_rank == 6 and to_rank == 7) or (
        piece.color == chess.BLACK and from_rank == 1 and to_rank == 0
    )
    if not is_promotion:
        return uci[:4]
    if len(uci) >= 5 and uci[4] in "qrbn":
        return uci[:5]
    return uci[:4] + "q"


def _gomoku_empty_board() -> List[List[int]]:
    """15x15 빈 오목판 (0=빈칸, 1=흑, 2=백)."""
    return [[0] * GOMOKU_SIZE for _ in range(GOMOKU_SIZE)]


def _gomoku_line_info(
    board: List[List[int]], r: int, c: int, color: int, dr: int, dc: int
) -> tuple[int, bool, bool]:
    """(r,c)를 포함해 (dr,dc) 방향으로 연속된 color 개수, 왼쪽/오른쪽 끝 개방 여부."""
    count = 1
    # 양의 방향
    step = 1
    while True:
        nr, nc = r + dr * step, c + dc * step
        if 0 <= nr < GOMOKU_SIZE and 0 <= nc < GOMOKU_SIZE and board[nr][nc] == color:
            count += 1
            step += 1
        else:
            right_open = (
                not (0 <= nr < GOMOKU_SIZE and 0 <= nc < GOMOKU_SIZE)
                or board[nr][nc] == 0
            )
            break
    # 음의 방향
    step = 1
    while True:
        nr, nc = r - dr * step, c - dc * step
        if 0 <= nr < GOMOKU_SIZE and 0 <= nc < GOMOKU_SIZE and board[nr][nc] == color:
            count += 1
            step += 1
        else:
            left_open = (
                not (0 <= nr < GOMOKU_SIZE and 0 <= nc < GOMOKU_SIZE)
                or board[nr][nc] == 0
            )
            break
    return (count, left_open, right_open)


def _gomoku_check_win(board: List[List[int]], r: int, c: int, color: int) -> bool:
    """(r,c)에 color 돌을 둔 뒤 5목 달성 여부."""
    for dr, dc in (0, 1), (1, 0), (1, 1), (1, -1):
        count = 1
        for sign in (1, -1):
            for step in range(1, 5):
                nr, nc = r + sign * dr * step, c + sign * dc * step
                if 0 <= nr < GOMOKU_SIZE and 0 <= nc < GOMOKU_SIZE and board[nr][nc] == color:
                    count += 1
                else:
                    break
        if count >= 5:
            return True
    return False


def _gomoku_renju_forbidden_black(board: List[List[int]], r: int, c: int) -> bool:
    """렌주 룰: 흑이 (r,c)에 두면 금수인지. 금수면 True."""
    color = 1
    open_fours = 0
    open_threes = 0
    for dr, dc in (0, 1), (1, 0), (1, 1), (1, -1):
        cnt, left_open, right_open = _gomoku_line_info(board, r, c, color, dr, dc)
        if cnt >= 6:
            return True  # 장목(overline) 금수
        if cnt == 4 and (left_open or right_open):
            open_fours += 1
        if cnt == 3 and left_open and right_open:
            open_threes += 1
    if open_fours >= 2:
        return True  # 쌍사 금수
    if open_threes >= 2:
        return True  # 쌍삼 금수
    return False


GOMOKU_SYSTEM = """You are Serena, an expert Gomoku (Renju rules) player. You play WHITE (second player) on a 15x15 board (rows/cols 0-14).

**RENJU RULES:**
- Win: Get exactly 5 stones in a row (horizontal/vertical/diagonal)
- BLACK (opponent) has forbidden moves: overline (6+ in a row), double-three, double-four
- WHITE (you) has NO forbidden moves - you can freely make double-three, double-four, overline

**WINNING STRATEGY (Priority Order):**
1. **WIN NOW**: If you can make 5-in-a-row this turn, do it immediately

2. **BLOCK OPPONENT'S GUARANTEED WIN**: If opponent has an open four (4 stones with both ends open, guaranteed win next turn), BLOCK IT

3. **PREVENT OPPONENT'S FORK (CRITICAL)**: 
   - Scan all empty cells to find positions where BLACK could create TWO open threes simultaneously (double open-three fork)
   - If such a position exists, OCCUPY IT IMMEDIATELY to prevent the fork
   - This is a preventive defense - stop the threat before it happens

4. **BLOCK OPPONENT'S OPEN THREE**:
   - If BLACK has an open three (3 stones with both ends open in horizontal/vertical/diagonal), you MUST block one end
   - Open three becomes open four next turn, which forces you to defend
   - Check ALL directions: horizontal (—), vertical (|), diagonal (/ and \)

5. **CREATE OPEN FOUR**: Make 4-in-a-row with at least one open end - forces opponent to block, gives you tempo

6. **CREATE YOUR FORK (DOUBLE-THREAT)**: Make a move that creates TWO threats simultaneously (fork) - opponent can only block one

7. **BUILD OPEN THREE**: Make 3-in-a-row with both ends open - can become open four next turn

8. **CONTROL CENTER**: Occupy or control the center area (rows/cols 5-9) for maximum influence

9. **EXTEND YOUR LINES**: Build connected stones that can extend in multiple directions

**TACTICAL PATTERNS:**
- Look for "4-3" patterns (open four + open three = guaranteed win)
- Create "3-3" patterns (two open threes crossing) - you can do this (BLACK cannot)
- Force opponent into defensive positions
- Connect your stones to maximize future options

**DEFENSIVE PRIORITIES (CRITICAL - CHECK THESE FIRST):**
⚠️ **OPPONENT'S OPEN THREE DETECTION:**
   - Scan the board for BLACK's open three patterns: B B B with both ends empty (. B B B .)
   - Check 4 directions: horizontal (row), vertical (col), diagonal-down (\), diagonal-up (/)
   - If found, place your stone at one of the open ends to block it

⚠️ **OPPONENT'S FORK PREVENTION:**
   - For each empty cell, simulate: "If BLACK plays here, would it create 2+ open threes?"
   - If YES, that cell is a FORK POSITION - you must occupy it NOW
   - This prevents BLACK from creating an unstoppable double threat

**THINKING PROCESS:**
1. Check if you can win immediately (your 5-in-a-row)
2. **DEFENSE FIRST**: Check BLACK's open fours → open threes → potential fork positions
3. **OFFENSE SECOND**: Look for your open four opportunities, fork creation
4. Evaluate strategic positioning (center control, line extensions)
5. Calculate 2-3 moves ahead for both players
6. Choose the move with highest strategic value

You MUST reply with ONLY two integers separated by space: row col (0-14), e.g. "7 7" for center. Think carefully, then output only the coordinates."""


def _gomoku_board_to_str(board: List[List[int]]) -> str:
    """보드를 LLM에 줄 문자열로 (B=흑 W=백 . =빈칸)."""
    lines = []
    for r in range(GOMOKU_SIZE):
        row = []
        for c in range(GOMOKU_SIZE):
            v = board[r][c]
            row.append("B" if v == 1 else "W" if v == 2 else ".")
        lines.append("".join(row))
    return "\n".join(lines)


def _call_serena_gomoku(board: List[List[int]]) -> Optional[tuple[int, int]]:
    """OPENAI_CHESS_MODEL로 Serena 오목 수 반환 (동기). (row, col) 0-14. 실패 시 None."""
    try:
        client = get_client()
        model = os.getenv("OPENAI_CHESS_MODEL") or os.getenv("OPENAI_MODEL")
        if not model:
            return None
        model = model.strip()
        empty = [(i, j) for i in range(GOMOKU_SIZE) for j in range(GOMOKU_SIZE) if board[i][j] == 0]
        if not empty:
            return None
        board_str = _gomoku_board_to_str(board)
        legal_str = ", ".join(f"{r},{c}" for r, c in empty[:40])
        # 최근 수 정보 추가
        last_black = None
        for r in range(GOMOKU_SIZE):
            for c in range(GOMOKU_SIZE):
                if board[r][c] == 1:
                    last_black = (r, c)
        last_move_info = f"Last BLACK move: {last_black[0]} {last_black[1]}" if last_black else "No BLACK moves yet"
        
        user_content = f"""Current board state (B=BLACK/opponent, W=WHITE/you, .=empty), row 0 at top:
{board_str}

{last_move_info}

Legal empty cells (first 40): {legal_str}

**MANDATORY ANALYSIS CHECKLIST (in order):**

🏆 **1. WIN CHECK**: Can you make 5-in-a-row right now?

🛡️ **2. CRITICAL DEFENSE - BLACK'S OPEN FOUR**: 
   - Scan for BLACK's 4-in-a-row with open end(s): . B B B B . or B B B B .
   - If found → BLOCK IT IMMEDIATELY

🚨 **3. CRITICAL DEFENSE - BLACK'S OPEN THREE**:
   - Scan ALL 4 directions (horizontal, vertical, diagonal \, diagonal /)
   - Look for BLACK's open three pattern: . B B B . (both ends empty)
   - If found → BLOCK one end (place stone at one of the dots)
   - Example patterns to detect:
     * Horizontal: . B B B .
     * Vertical: . (top) B B B . (bottom)
     * Diagonal: . B B B . (in any diagonal line)

⚠️ **4. PREVENT BLACK'S FORK (DOUBLE OPEN-THREE)**:
   - For each empty cell, simulate: "If BLACK plays here, would it create 2 or more open threes?"
   - Check if that position connects multiple BLACK stones to form multiple threats
   - If ANY cell allows BLACK to make a fork → OCCUPY that cell NOW
   - This prevents BLACK from creating an unstoppable attack

⚔️ **5. OFFENSE - YOUR OPPORTUNITIES**:
   - Can you create an open four?
   - Can you create YOUR fork (two threats)?
   - Can you build an open three?

📍 **6. STRATEGIC POSITIONING**: Center control, line extensions

Reply with ONLY two numbers: row col (0-14), e.g. "7 7"."""
        use_responses_api = any(x in model.lower() for x in ['gpt-5-mini', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.2'])
        text = ""
        if use_responses_api:
            try:
                resp = client.responses.create(
                    model=model,
                    input=f"{GOMOKU_SYSTEM}\n\n{user_content}",
                    max_output_tokens=200,
                )
                text = (getattr(resp, "output_text", None) or "") or ""
                if not text and getattr(resp, "output", None):
                    for item in resp.output or []:
                        content = getattr(item, "content", None)
                        for c in (content if isinstance(content, (list, tuple)) else [content] or []):
                            if c and getattr(c, "text", None):
                                text = str(c.text)
                                break
                        if text:
                            break
            except Exception:
                use_responses_api = False
        if not use_responses_api or not text.strip():
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GOMOKU_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=200,
            )
            text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\b(\d{1,2})\s+(\d{1,2})\b", text)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            if 0 <= r < GOMOKU_SIZE and 0 <= c < GOMOKU_SIZE and board[r][c] == 0:
                return (r, c)
    except Exception as e:
        print(f"Serena gomoku error: {e}")
    return None


def _gomoku_serena_move(board: List[List[int]]) -> Optional[tuple[int, int]]:
    """Serena(백) 수: LLM 실패 시 빈 칸 중 랜덤."""
    move = _call_serena_gomoku(board)
    if move is not None:
        return move
    empty = [(i, j) for i in range(GOMOKU_SIZE) for j in range(GOMOKU_SIZE) if board[i][j] == 0]
    if not empty:
        return None
    return secrets.choice(empty)


async def _play_gomoku_serena_move() -> None:
    """오목 Serena(백) 차례일 때 OPENAI_CHESS_MODEL로 한 수 두고 브로드캐스트."""
    global _gomoku_game, _gomoku_pending_task
    if not _gomoku_game or _gomoku_game.get("status") != "active":
        return
    if _gomoku_game.get("turn") != "white":
        return
    board = _gomoku_game["board"]
    empty = [(i, j) for i in range(GOMOKU_SIZE) for j in range(GOMOKU_SIZE) if board[i][j] == 0]
    if not empty:
        return
    loop = asyncio.get_event_loop()
    move = None
    try:
        move = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _gomoku_serena_move(board)),
            timeout=SERENA_GOMOKU_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print("Serena gomoku: AI timeout (60s), using random move")
        move = secrets.choice(empty)
    if not move:
        return
    _gomoku_pending_task = None
    r, c = move
    board[r][c] = 2
    status = "active"
    if _gomoku_check_win(board, r, c, 2):
        status = "win_white"
    else:
        empty_count = sum(1 for row in board for cell in row if cell == 0)
        if empty_count == 0:
            status = "draw"
        else:
            _gomoku_game["turn"] = "black"
    _gomoku_game["status"] = status
    _gomoku_game["last_move"] = [r, c]
    await _room_broadcast({
        "type": "gomoku_state",
        "board": [row[:] for row in board],
        "turn": _gomoku_game.get("turn", "black"),
        "status": status,
        "black_player": _gomoku_game.get("black_player"),
        "white_player": _gomoku_game.get("white_player"),
        "mode": _gomoku_game.get("mode", "serena"),
        "last_move": [r, c],
    }, exclude_ws=None)


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
        
        # gpt-5-mini, gpt-5-nano는 Responses API 사용 (실패 시 Chat Completions 폴백)
        use_responses_api = any(x in model.lower() for x in ['gpt-5-mini', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.2'])
        user_content = f"Recent chat messages:\n{conv}\n\n{user_instruction}"

        if use_responses_api:
            try:
                print(f"🔄 Using Responses API for model: {model}")
                input_text = f"{SERENA_SYSTEM}\n\nRecent chat:\n{conv}\n\n{user_instruction}"
                resp = client.responses.create(
                    model=model,
                    input=input_text,
                    max_output_tokens=300,
                )
                if hasattr(resp, 'output_text') and resp.output_text:
                    return resp.output_text.strip()
                if hasattr(resp, 'output') and resp.output:
                    for item in resp.output:
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    return content_item.text.strip()
            except Exception as resp_err:
                print(f"⚠ Serena Responses API failed, falling back to Chat Completions: {resp_err}")
                use_responses_api = False

        if not use_responses_api:
            # Chat Completions API (기본 또는 폴백)
            print(f"💬 Using Chat Completions API for model: {model}")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SERENA_SYSTEM},
                    {"role": "user", "content": user_content},
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


SERENA_RESPONSE_TIMEOUT = 30  # Serena 채팅 응답 타임아웃 (초)
SERENA_GOMOKU_TIMEOUT = 60  # Serena 오목 AI 타임아웃 (초) - 전략적 사고 시간


def _pick_random_legal_move(board: chess.Board) -> Optional[str]:
    """합법적인 수 중 랜덤 선택 (AI 실패 시 폴백)."""
    moves = list(board.legal_moves)
    if not moves:
        return None
    m = secrets.choice(moves)
    return m.uci()


async def _open_stockfish_async(skill_level: int, elo: int) -> Optional[str]:
    """Stockfish 엔진을 asyncio 네이티브 API로 열고 Skill Level + UCI_Elo 설정.
    성공 시 None 반환, 실패 시 오류 메시지 반환."""
    global _stockfish_protocol
    if not chess.engine:
        return "chess.engine 모듈을 사용할 수 없습니다."
    await _close_stockfish_async()
    skill_level = max(0, min(20, skill_level))
    # 이 서버에 설치된 Stockfish는 UCI_Elo 범위를 [1350, 2850]로 요구.
    # GrandMaster(elo>=3000)는 UCI_Elo 제한 없이 Skill Level/Depth만 사용한다.
    elo_raw = int(elo)
    use_elo_limit = elo_raw < 3000
    if use_elo_limit:
        elo = max(1350, min(2850, elo_raw))
    else:
        elo = elo_raw
    last_err: Optional[str] = None
    for path in STOCKFISH_CANDIDATE_PATHS:
        if not path:
            continue
        try:
            _, protocol = await chess.engine.popen_uci(path)
            if use_elo_limit:
                await protocol.configure({
                    "UCI_LimitStrength": True,
                    "UCI_Elo": elo,
                    "Skill Level": skill_level,
                })
            else:
                await protocol.configure({
                    "UCI_LimitStrength": False,
                    "Skill Level": skill_level,
                })
            _stockfish_protocol = protocol
            print(f"Stockfish started: {path}, Skill Level={skill_level}, "
                  f"{'UCI_Elo='+str(elo) if use_elo_limit else 'UCI_Elo disabled'}")
            return None
        except FileNotFoundError:
            last_err = f"실행 파일을 찾을 수 없습니다: {path}"
            print(f"Stockfish not found at {path}")
        except Exception as e:
            last_err = str(e)
            print(f"Stockfish open failed at {path}: {e}")
    return last_err or "Stockfish를 찾을 수 없습니다."


async def _close_stockfish_async() -> None:
    """Stockfish 엔진 async 종료. pending 수 계산 태스크도 함께 취소."""
    global _stockfish_protocol, _chess_pending_task
    if _chess_pending_task and not _chess_pending_task.done():
        _chess_pending_task.cancel()
        try:
            await _chess_pending_task
        except asyncio.CancelledError:
            pass
        _chess_pending_task = None
    if _stockfish_protocol is not None:
        try:
            await _stockfish_protocol.quit()
        except Exception as e:
            print(f"Stockfish close error: {e}")
        _stockfish_protocol = None


async def _play_stockfish_chess_move() -> None:
    """Stockfish 차례일 때 asyncio 네이티브 API로 엔진이 수 두고 브로드캐스트."""
    global _chess_game, _chess_pending_task, _stockfish_protocol
    if not _chess_game or _chess_game.get("status") != "active":
        return
    if _chess_game.get("mode") != "stockfish":
        return
    if _chess_game.get("paused"):
        return
    if _stockfish_protocol is None:
        return
    board: chess.Board = _chess_game["board"]
    engine_color = chess.WHITE if _chess_game.get("stockfish_color") == "white" else chess.BLACK
    if board.turn != engine_color:
        return
    legal_uci = [m.uci() for m in board.legal_moves]
    if not legal_uci:
        return

    depth = _chess_game.get("stockfish_depth", 5)
    use_clock = _chess_game.get("stockfish_use_clock", False)
    blunder_rate = _chess_game.get("stockfish_blunder_rate", 0.0)
    random_move_prob = _chess_game.get("stockfish_random_move_prob", 0.0)
    think_min = float(_chess_game.get("stockfish_think_min", 0.0))
    think_max = float(_chess_game.get("stockfish_think_max", 0.0))

    now = time.time()
    turn_started = _chess_game.get("turn_started_at") or now
    elapsed = max(0.0, now - turn_started)
    # 엔진/상대 남은 시간 계산
    white_time = _chess_game.get("white_time_remaining", CHESS_INITIAL_SECONDS)
    black_time = _chess_game.get("black_time_remaining", CHESS_INITIAL_SECONDS)
    if engine_color == chess.WHITE:
        engine_time_key = "white_time_remaining"
        opp_time_key = "black_time_remaining"
        engine_captured_key = "white_captured"
        time_loss_status = "time_loss_white"
        engine_remaining = max(0.01, white_time - elapsed)
        opp_remaining = max(0.01, black_time)
        white_clock = engine_remaining
        black_clock = opp_remaining
    else:
        engine_time_key = "black_time_remaining"
        opp_time_key = "white_time_remaining"
        engine_captured_key = "black_captured"
        time_loss_status = "time_loss_black"
        engine_remaining = max(0.01, black_time - elapsed)
        opp_remaining = max(0.01, white_time)
        white_clock = opp_remaining
        black_clock = engine_remaining
    inc = float(CHESS_INCREMENT_SECONDS)

    if use_clock:
        limit = chess.engine.Limit(
            depth=depth,
            white_clock=white_clock,
            black_clock=black_clock,
            white_inc=inc,
            black_inc=inc,
        )
    else:
        limit = chess.engine.Limit(depth=depth)

    timeout_sec = min(180, max(60, engine_remaining + 20)) if use_clock else 30.0
    uci: Optional[str] = None
    # 기본 생각 시간 범위 (Elo 기반)
    eff_think_min = think_min
    eff_think_max = think_max
    try:
        # multipv 분석으로 상위 3수 후보 확보 → 블런더율/후보수 랜덤으로 인간 Elo에 가깝게 선택
        analysis_list = await asyncio.wait_for(
            _stockfish_protocol.analyse(board.copy(), limit, multipv=3),
            timeout=timeout_sec,
        )
        top_moves: List[chess.Move] = []
        evals_cp: List[float] = []
        if isinstance(analysis_list, list):
            for i, info in enumerate(analysis_list):
                if i >= 3:
                    break
                pv = info.get("pv") if isinstance(info, dict) else getattr(info, "pv", None)
                if pv and len(pv) > 0:
                    m = pv[0] if isinstance(pv[0], chess.Move) else chess.Move.from_uci(str(pv[0]))
                    if m in board.legal_moves:
                        top_moves.append(m)
                # 점수 추출 (cp 기반, 엔진 입장에서의 평가값)
                sc = info.get("score") if isinstance(info, dict) else getattr(info, "score", None)
                if sc is not None:
                    try:
                        pov_sc = sc.pov(engine_color) if hasattr(sc, "pov") else sc
                        cp = getattr(pov_sc, "cp", None)
                        if cp is not None:
                            evals_cp.append(float(cp))
                    except Exception:
                        pass
        else:
            pv = analysis_list.get("pv") if isinstance(analysis_list, dict) else getattr(analysis_list, "pv", None)
            if pv and len(pv) > 0:
                m = pv[0] if isinstance(pv[0], chess.Move) else chess.Move.from_uci(str(pv[0]))
                if m in board.legal_moves:
                    top_moves.append(m)
            sc = analysis_list.get("score") if isinstance(analysis_list, dict) else getattr(analysis_list, "score", None)
            if sc is not None:
                try:
                    pov_sc = sc.pov(engine_color) if hasattr(sc, "pov") else sc
                    cp = getattr(pov_sc, "cp", None)
                    if cp is not None:
                        evals_cp.append(float(cp))
                except Exception:
                    pass

        # 포지션 난이도에 따라 생각 시간 조정
        if eff_think_max > 0 and eff_think_max >= eff_think_min and evals_cp:
            # 가장 좋은 수 기준 평가값(폰 단위)
            best_cp = evals_cp[0]
            best_eval_pawns = abs(best_cp) / 100.0
            factor = 1.0
            # 접전(평형)에 가까울수록 더 오래 생각
            if best_eval_pawns < 0.5:
                factor = 1.0
            elif best_eval_pawns < 1.5:
                factor = 0.7
            elif best_eval_pawns < 3.0:
                factor = 0.4
            else:
                factor = 0.2

            # 상위 후보수들의 평가가 비슷하면(선택지 많음) 추가로 약간 더 오래 생각
            if len(evals_cp) >= 2:
                spread_pawns = (max(evals_cp) - min(evals_cp)) / 100.0
                if spread_pawns < 0.4:
                    factor *= 1.1

            eff_think_min = max(0.0, eff_think_min * factor)
            eff_think_max = max(eff_think_min, eff_think_max * factor)

        if top_moves:
            if random.random() < blunder_rate:
                uci = _pick_random_legal_move(board)
            else:
                # 가중치: 1수 (1 - random_move_prob), 2수·3수에 나머지 분배 (예: 0.7 : 0.3)
                r = random_move_prob
                if len(top_moves) == 1:
                    uci = top_moves[0].uci()
                elif len(top_moves) == 2:
                    weights = [1.0 - r, r]
                    move = random.choices(top_moves, weights=weights, k=1)[0]
                    uci = move.uci()
                else:
                    weights = [1.0 - r, r * 0.7, r * 0.3]
                    move = random.choices(top_moves, weights=weights, k=1)[0]
                    uci = move.uci()
        if not uci:
            print("Stockfish analysis returned no usable move")
    except asyncio.TimeoutError:
        print(f"Stockfish timed out after {timeout_sec}s, using random move")
    except Exception as e:
        print(f"Stockfish analyse error: {e}")
    _chess_pending_task = None

    # 엔진이 수를 찾은 뒤, 인간처럼 약간의 랜덤 딜레이를 준다.
    if eff_think_max > 0 and eff_think_max >= eff_think_min:
        delay = random.uniform(eff_think_min, eff_think_max)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

    if not uci:
        uci = _pick_random_legal_move(board)
    if not uci or not _chess_game or _chess_game.get("status") != "active":
        return
    uci = _normalize_uci_promotion(board, uci)
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            uci = _pick_random_legal_move(board)
            if not uci:
                return
            move = chess.Move.from_uci(uci)
        now = time.time()
        turn_started = _chess_game.get("turn_started_at") or now
        elapsed = now - turn_started
        remaining = _chess_game.get(engine_time_key, CHESS_INITIAL_SECONDS) - elapsed
        remaining = max(0.0, remaining)
        if remaining <= 0:
            _chess_game["status"] = time_loss_status
            _chess_game[engine_time_key] = 0.0
            _chess_game["turn_started_at"] = now
            # Stockfish가 승부를 냈으므로 여기서 즉시 MMR 반영한다.
            _apply_chess_mmr_if_needed(_chess_game)
            await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
            return
        _chess_game[engine_time_key] = remaining + CHESS_INCREMENT_SECONDS
        cap = _get_captured_piece(board, move)
        if cap:
            _chess_game.setdefault(engine_captured_key, []).append(cap)
        board.push(move)
        _chess_game["turn_started_at"] = now
        status = "active"
        if board.is_checkmate():
            # board.push(move) 이후 board.turn은 체크메이트를 당한 쪽
            if board.turn == chess.WHITE:
                status = "checkmate_white"
            else:
                status = "checkmate_black"
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            status = "draw"
        _chess_game["status"] = status
        # Stockfish가 체크메이트/무승부로 게임을 끝냈다면 MMR을 반영한다.
        _apply_chess_mmr_if_needed(_chess_game)
        await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
    except (ValueError, chess.InvalidMoveError):
        uci_fb = _pick_random_legal_move(board)
        if uci_fb and _chess_game and _chess_game.get("status") == "active":
            try:
                move = chess.Move.from_uci(uci_fb)
                now = time.time()
                turn_started = _chess_game.get("turn_started_at") or now
                elapsed = now - turn_started
                remaining = _chess_game.get(engine_time_key, CHESS_INITIAL_SECONDS) - elapsed
                remaining = max(0.0, remaining)
                if remaining <= 0:
                    _chess_game["status"] = time_loss_status
                    _chess_game[engine_time_key] = 0.0
                    _chess_game["turn_started_at"] = now
                    # Stockfish가 승부를 냈으므로 여기서 즉시 MMR 반영한다.
                    _apply_chess_mmr_if_needed(_chess_game)
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                    return
                _chess_game[engine_time_key] = remaining + CHESS_INCREMENT_SECONDS
                board.push(move)
                _chess_game["turn_started_at"] = now
                status = "active"
                if board.is_checkmate():
                    if board.turn == chess.WHITE:
                        status = "checkmate_white"
                    else:
                        status = "checkmate_black"
                elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                    status = "draw"
                _chess_game["status"] = status
                # Stockfish가 체크메이트/무승부로 게임을 끝냈다면 MMR을 반영한다.
                _apply_chess_mmr_if_needed(_chess_game)
                await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
            except Exception:
                pass


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
    uci = _normalize_uci_promotion(board, uci)
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            uci = _pick_random_legal_move(board)
            if not uci:
                return
            move = chess.Move.from_uci(uci)
        now = time.time()
        turn_started = _chess_game.get("turn_started_at") or now
        elapsed = now - turn_started
        b = _chess_game.get("black_time_remaining", CHESS_INITIAL_SECONDS) - elapsed
        b = max(0.0, b)
        if b <= 0:
            _chess_game["status"] = "time_loss_black"
            _chess_game["black_time_remaining"] = 0.0
            _chess_game["turn_started_at"] = now
            await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
            return
        _chess_game["black_time_remaining"] = b + CHESS_INCREMENT_SECONDS
        cap = _get_captured_piece(board, move)
        if cap:
            _chess_game.setdefault("black_captured", []).append(cap)
        board.push(move)
        _chess_game["turn_started_at"] = now
        status = "active"
        if board.is_checkmate():
            status = "checkmate_white"
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            # can_claim_draw: 50수 규칙(캡처/폰 이동 없이 50수), 3회 반복 포함
            status = "draw"
        _chess_game["status"] = status
        await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
    except (ValueError, chess.InvalidMoveError):
        uci_fb = _pick_random_legal_move(board)
        if uci_fb and _chess_game and _chess_game.get("status") == "active":
            try:
                move = chess.Move.from_uci(uci_fb)
                now = time.time()
                turn_started = _chess_game.get("turn_started_at") or now
                elapsed = now - turn_started
                b = _chess_game.get("black_time_remaining", CHESS_INITIAL_SECONDS) - elapsed
                b = max(0.0, b)
                if b <= 0:
                    _chess_game["status"] = "time_loss_black"
                    _chess_game["black_time_remaining"] = 0.0
                    _chess_game["turn_started_at"] = now
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                    return
                _chess_game["black_time_remaining"] = b + CHESS_INCREMENT_SECONDS
                board.push(move)
                _chess_game["turn_started_at"] = now
                status = "active"
                if board.is_checkmate():
                    status = "checkmate_white"
                elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                    # can_claim_draw: 50수 규칙, 3회 반복 포함
                    status = "draw"
                _chess_game["status"] = status
                await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
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
    """참여자 목록 (user_id, username, mmr_rating, mmr_tier 포함, Serena는 user_id=null)."""
    out = []
    for _, u in _room_connections:
        uid = u.get("id")
        rating, _ = db.get_user_mmr(int(uid)) if uid else (None, 0)
        out.append({
            "user_id": uid,
            "username": u.get("username"),
            "name": u["name"],
            "avatar_url": u.get("avatar_url") or None,
            "mmr_rating": rating,
            "mmr_tier": _mmr_to_tier(rating),
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
    global _serena_invited, _serena_pending_task, _chess_game, _chess_pending_task, _gomoku_game, _gomoku_pending_task
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
            await websocket.send_text(json.dumps(_chess_state_payload(_chess_game), ensure_ascii=False))
        if _gomoku_game and _gomoku_game.get("status"):
            await websocket.send_text(json.dumps({
                "type": "gomoku_state",
                "board": [row[:] for row in _gomoku_game["board"]],
                "turn": _gomoku_game.get("turn", "black"),
                "status": _gomoku_game.get("status"),
                "black_player": _gomoku_game.get("black_player"),
                "white_player": _gomoku_game.get("white_player"),
                "mode": _gomoku_game.get("mode", "serena"),
                "last_move": _gomoku_game.get("last_move"),
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
                        if _chess_game and _chess_game.get("mode") == "stockfish":
                            asyncio.create_task(_close_stockfish_async())
                        _chess_game = None
                        _gomoku_game = None
                        if _gomoku_pending_task and not _gomoku_pending_task.done():
                            _gomoku_pending_task.cancel()
                            _gomoku_pending_task = None
                        await _room_broadcast({"type": "chess_state", "fen": None, "status": None}, exclude_ws=None)
                        await _room_broadcast({"type": "gomoku_state", "board": None, "status": None}, exclude_ws=None)
                        await _room_broadcast({"type": "system", "message": "Serena님이 퇴장했습니다."}, exclude_ws=None)
                        participants = _build_participants_list()
                        await _room_broadcast({"type": "participants", "list": participants}, exclude_ws=None)
                        await _room_broadcast({"type": "serena_status", "present": False}, exclude_ws=None)

                elif data.get("type") == "chess_start_practice":
                    # 연습 모드: 한 유저가 흑/백 모두 두는 모드 (시간 패배 없음)
                    if _chess_pending_task and not _chess_pending_task.done():
                        _chess_pending_task.cancel()
                        _chess_pending_task = None
                    board = chess.Board()
                    now = time.time()
                    white_user_id = None
                    black_user_id = None
                    for _ws, u in _room_connections:
                        if u.get("name") == nickname and white_user_id is None:
                            white_user_id = u.get("id")
                            black_user_id = u.get("id")
                    _chess_game = {
                        "board": board,
                        "white_player": nickname,
                        "black_player": nickname,
                        "white_user_id": white_user_id,
                        "black_user_id": black_user_id,
                        "mode": "practice",
                        "status": "active",
                        "paused": False,
                        "white_captured": [],
                        "black_captured": [],
                        "white_time_remaining": float(CHESS_INITIAL_SECONDS),
                        "black_time_remaining": float(CHESS_INITIAL_SECONDS),
                        "turn_started_at": now,
                    }
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)

                elif data.get("type") == "chess_start_pvp" and (data.get("opponent") or "").strip():
                    opp = (data["opponent"] or "").strip()[:32]
                    if opp == nickname:
                        continue
                    if not any(u.get("name") == opp for _, u in _room_connections):
                        continue
                    if _chess_pending_task and not _chess_pending_task.done():
                        _chess_pending_task.cancel()
                        _chess_pending_task = None
                    my_side = (data.get("my_side") or "").strip().lower()
                    challenger_wants_white = my_side in ("white", "w")
                    white_player = nickname if challenger_wants_white else opp
                    black_player = opp if challenger_wants_white else nickname
                    white_user_id = None
                    black_user_id = None
                    for _ws, u in _room_connections:
                        if u.get("name") == white_player and white_user_id is None:
                            white_user_id = u.get("id")
                        if u.get("name") == black_player and black_user_id is None:
                            black_user_id = u.get("id")
                    board = chess.Board()
                    now = time.time()
                    _chess_game = {
                        "board": board,
                        "white_player": white_player,
                        "black_player": black_player,
                        "white_user_id": white_user_id,
                        "black_user_id": black_user_id,
                        "mode": "pvp",
                        "status": "active",
                        "paused": False,
                        "white_captured": [],
                        "black_captured": [],
                        "white_time_remaining": float(CHESS_INITIAL_SECONDS),
                        "black_time_remaining": float(CHESS_INITIAL_SECONDS),
                        "turn_started_at": now,
                    }
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)

                elif data.get("type") == "chess_start_stockfish":
                    if not chess.engine:
                        await _room_broadcast({"type": "system", "message": "Stockfish를 사용할 수 없습니다. (python-chess 설치 확인)"}, exclude_ws=None)
                        continue
                    elo = int(data.get("elo") or 1200)
                    level_name, level_elo, skill_level, depth, blunder_rate, random_move_prob, use_clock = "Silver IV", 1200, 6, 5, 0.16, 0.25, False
                    for t in STOCKFISH_LEVELS:
                        if t[1] == elo:
                            level_name, level_elo, skill_level, depth, blunder_rate, random_move_prob, use_clock = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
                            break
                    think_min, think_max = _elo_to_think_range(level_elo)
                    stockfish_casual = bool(data.get("casual"))
                    try:
                        err_msg = await asyncio.wait_for(
                            _open_stockfish_async(skill_level, level_elo),
                            timeout=15.0,
                        )
                    except asyncio.TimeoutError:
                        await _room_broadcast({"type": "system", "message": "Stockfish 엔진 시작 시간 초과."}, exclude_ws=None)
                        continue
                    except Exception as e:
                        print(f"Stockfish start failed: {e}")
                        await _room_broadcast({"type": "system", "message": f"Stockfish 엔진을 시작할 수 없습니다. ({e})"}, exclude_ws=None)
                        continue
                    if err_msg:
                        msg = f"Stockfish 엔진을 시작할 수 없습니다. {err_msg} (설치: apt install stockfish 또는 STOCKFISH_PATH 환경변수 설정)"
                        await _room_broadcast({"type": "system", "message": msg}, exclude_ws=None)
                        continue
                    board = chess.Board()
                    now = time.time()
                    my_side = (data.get("my_side") or "white").strip().lower()
                    if my_side == "black":
                        white_player = "Stockfish"
                        black_player = nickname
                        stockfish_color = "white"
                    else:
                        white_player = nickname
                        black_player = "Stockfish"
                        stockfish_color = "black"
                    human_uid = None
                    for _ws, u in _room_connections:
                        if u.get("name") == nickname:
                            human_uid = u.get("id")
                            break
                    white_user_id = human_uid if white_player == nickname else None
                    black_user_id = human_uid if black_player == nickname else None
                    _chess_game = {
                        "board": board,
                        "white_player": white_player,
                        "black_player": black_player,
                        "white_user_id": white_user_id,
                        "black_user_id": black_user_id,
                        "mode": "stockfish",
                        "status": "active",
                        "paused": False,
                        "white_captured": [],
                        "black_captured": [],
                        "white_time_remaining": float(CHESS_INITIAL_SECONDS),
                        "black_time_remaining": float(CHESS_INITIAL_SECONDS),
                        "turn_started_at": now,
                        "stockfish_color": stockfish_color,
                        "stockfish_elo": level_elo,
                        "stockfish_skill_level": skill_level,
                        "stockfish_depth": depth,
                        "stockfish_blunder_rate": blunder_rate,
                        "stockfish_random_move_prob": random_move_prob,
                        "stockfish_use_clock": use_clock,
                        "stockfish_think_min": think_min,
                        "stockfish_think_max": think_max,
                        "stockfish_casual": stockfish_casual,
                    }
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                    # 내가 흑일 때는 Stockfish(백)가 선공이므로 바로 첫 수를 두도록 예약
                    if stockfish_color == "white":
                        _chess_pending_task = asyncio.create_task(_play_stockfish_chess_move())

                elif data.get("type") == "chess_undo" and _chess_game and _chess_game.get("status") == "active":
                    if _chess_game.get("paused"):
                        continue
                    um = _chess_game.get("mode")
                    if um == "stockfish" and _chess_game.get("stockfish_casual"):
                        if nickname not in (_chess_game.get("white_player"), _chess_game.get("black_player")):
                            continue
                        if nickname == "Stockfish":
                            continue
                        if _chess_pending_task and not _chess_pending_task.done():
                            _chess_pending_task.cancel()
                            try:
                                await _chess_pending_task
                            except asyncio.CancelledError:
                                pass
                            _chess_pending_task = None
                        board = _chess_game["board"]
                        if len(board.move_stack) < 1:
                            continue
                        last_move = board.peek()
                        cap = _get_captured_piece(board, last_move)
                        board.pop()
                        if cap:
                            if board.turn == chess.BLACK:
                                lst = _chess_game.get("black_captured") or []
                                if lst:
                                    _chess_game["black_captured"] = lst[:-1]
                            else:
                                lst = _chess_game.get("white_captured") or []
                                if lst:
                                    _chess_game["white_captured"] = lst[:-1]
                        _chess_game["turn_started_at"] = time.time()
                        _chess_game.pop("pending_undo", None)
                        await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                        if _chess_game.get("status") == "active":
                            engine_color = chess.WHITE if _chess_game.get("stockfish_color") == "white" else chess.BLACK
                            if board.turn == engine_color and (_chess_pending_task is None or _chess_pending_task.done()):
                                _chess_pending_task = asyncio.create_task(_play_stockfish_chess_move())
                        continue
                    if um != "pvp":
                        continue
                    board = _chess_game["board"]
                    if len(board.move_stack) < 1:
                        continue
                    # 무르기 요청자는 현재 턴이 아닌 쪽(직전에 둔 쪽). 수락/거절하는 쪽은 현재 턴인 쪽.
                    last_mover = _chess_game.get("white_player") if board.turn == chess.BLACK else _chess_game.get("black_player")
                    if nickname != last_mover:
                        continue
                    _chess_game["pending_undo"] = {"requested_by": nickname}
                    await _room_broadcast({"type": "chess_undo_request", "requested_by": nickname}, exclude_ws=None)

                elif data.get("type") == "chess_undo_accept" and _chess_game and _chess_game.get("status") == "active" and _chess_game.get("mode") == "pvp":
                    board = _chess_game["board"]
                    if len(board.move_stack) < 1:
                        _chess_game.pop("pending_undo", None)
                        continue
                    # 수락하는 쪽은 현재 턴인 쪽(직전에 둔 사람이 아님)
                    current_turn_player = _chess_game.get("white_player") if board.turn == chess.WHITE else _chess_game.get("black_player")
                    pending = _chess_game.get("pending_undo") or {}
                    if nickname != current_turn_player or pending.get("requested_by") is None:
                        continue
                    last_move = board.peek()
                    cap = _get_captured_piece(board, last_move)
                    board.pop()
                    turn_before = "black" if board.turn == chess.WHITE else "white"
                    if turn_before == "white":
                        _chess_game["white_time_remaining"] = _chess_game.get("white_time_remaining", CHESS_INITIAL_SECONDS) + CHESS_INCREMENT_SECONDS
                    else:
                        _chess_game["black_time_remaining"] = _chess_game.get("black_time_remaining", CHESS_INITIAL_SECONDS) + CHESS_INCREMENT_SECONDS
                    if cap:
                        if board.turn == chess.BLACK:
                            lst = _chess_game.get("black_captured") or []
                            if lst:
                                _chess_game["black_captured"] = lst[:-1]
                        else:
                            lst = _chess_game.get("white_captured") or []
                            if lst:
                                _chess_game["white_captured"] = lst[:-1]
                    _chess_game["turn_started_at"] = time.time()
                    _chess_game.pop("pending_undo", None)
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)

                elif data.get("type") == "chess_undo_practice" and _chess_game and _chess_game.get("status") == "active" and _chess_game.get("mode") == "practice":
                    board = _chess_game["board"]
                    if len(board.move_stack) < 1:
                        continue
                    last_move = board.peek()
                    cap = _get_captured_piece(board, last_move)
                    board.pop()
                    if cap:
                        if board.turn == chess.BLACK:
                            lst = _chess_game.get("black_captured") or []
                            if lst:
                                _chess_game["black_captured"] = lst[:-1]
                        else:
                            lst = _chess_game.get("white_captured") or []
                            if lst:
                                _chess_game["white_captured"] = lst[:-1]
                    _chess_game["status"] = "active"
                    _chess_game["turn_started_at"] = time.time()
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)

                elif data.get("type") == "chess_undo_reject" and _chess_game and _chess_game.get("status") == "active" and _chess_game.get("mode") == "pvp":
                    pending = _chess_game.get("pending_undo") or {}
                    # 거절하는 쪽은 현재 턴인 쪽
                    current_turn_player = _chess_game.get("white_player") if _chess_game["board"].turn == chess.WHITE else _chess_game.get("black_player")
                    if nickname == current_turn_player and pending.get("requested_by"):
                        req_by = pending["requested_by"]
                        _chess_game.pop("pending_undo", None)
                        await _room_broadcast({"type": "chess_undo_rejected", "requested_by": req_by}, exclude_ws=None)

                elif data.get("type") == "chess_pause" and _chess_game and _chess_game.get("status") == "active" and _chess_game.get("mode") in ("pvp", "stockfish"):
                    if nickname not in (_chess_game.get("white_player"), _chess_game.get("black_player")):
                        continue
                    now = time.time()
                    turn_started = _chess_game.get("turn_started_at") or now
                    elapsed = max(0.0, now - turn_started)
                    if _chess_game["board"].turn == chess.WHITE:
                        _chess_game["white_time_remaining"] = max(0.0, _chess_game.get("white_time_remaining", CHESS_INITIAL_SECONDS) - elapsed)
                    else:
                        _chess_game["black_time_remaining"] = max(0.0, _chess_game.get("black_time_remaining", CHESS_INITIAL_SECONDS) - elapsed)
                    _chess_game["turn_started_at"] = now
                    _chess_game["paused"] = True
                    # Stockfish 대국 중이면, 엔진 수 계산 태스크를 중단한다.
                    if _chess_pending_task and not _chess_pending_task.done():
                        _chess_pending_task.cancel()
                        try:
                            await _chess_pending_task
                        except asyncio.CancelledError:
                            pass
                        _chess_pending_task = None
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)

                elif data.get("type") == "chess_resume" and _chess_game and _chess_game.get("status") == "active" and _chess_game.get("mode") in ("pvp", "stockfish"):
                    if nickname not in (_chess_game.get("white_player"), _chess_game.get("black_player")):
                        continue
                    _chess_game["paused"] = False
                    _chess_game["turn_started_at"] = time.time()
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                    # Stockfish 대국에서 엔진 차례라면 재개 시 바로 다시 수를 두도록 예약
                    if _chess_game.get("mode") == "stockfish":
                        try:
                            board = _chess_game["board"]
                            engine_color = chess.WHITE if _chess_game.get("stockfish_color") == "white" else chess.BLACK
                            if board.turn == engine_color and (_chess_pending_task is None or _chess_pending_task.done()):
                                _chess_pending_task = asyncio.create_task(_play_stockfish_chess_move())
                        except Exception:
                            pass

                elif data.get("type") == "chess_legal_moves" and _chess_game:
                    from_sq = (data.get("from") or "").strip().lower()
                    if len(from_sq) == 2 and from_sq[0] in "abcdefgh" and from_sq[1] in "12345678":
                        try:
                            from_square = chess.parse_square(from_sq)
                            board = _chess_game["board"]
                            to_squares = [chess.SQUARE_NAMES[m.to_square] for m in board.legal_moves if m.from_square == from_square]
                            await websocket.send_json({"type": "chess_legal_moves", "from": from_sq, "to_squares": to_squares})
                        except (ValueError, KeyError):
                            pass

                elif data.get("type") == "chess_replay_ply":
                    if not _chess_game:
                        continue
                    st = _chess_game.get("status") or ""
                    if st not in CHESS_REPLAY_TERMINAL_STATUSES:
                        continue
                    board = _chess_game["board"]
                    mh = [m.uci() for m in board.move_stack]
                    try:
                        ply = int(data.get("ply", 0))
                    except (TypeError, ValueError):
                        ply = 0
                    ply = max(0, min(ply, len(mh)))
                    pos = _chess_position_at_ply(mh, ply)
                    try:
                        seq = int(data.get("seq", 0))
                    except (TypeError, ValueError):
                        seq = 0
                    await websocket.send_json({
                        "type": "chess_replay_view",
                        "seq": seq,
                        "ply": ply,
                        "ply_max": len(mh),
                        "fen": pos["fen"],
                        "white_captured": pos["white_captured"],
                        "black_captured": pos["black_captured"],
                        "last_move": pos["last_move"],
                        "turn": pos["turn"],
                        "in_check": pos["in_check"],
                    })

                elif data.get("type") == "chess_move" and (data.get("uci") or "").strip():
                    uci = (data["uci"] or "").strip().lower()[:10]
                    if not _chess_game or _chess_game.get("status") != "active":
                        continue
                    if _chess_game.get("paused"):
                        continue
                    board: chess.Board = _chess_game["board"]
                    uci = _normalize_uci_promotion(board, uci)
                    mode = _chess_game.get("mode", "serena")
                    white_p = _chess_game.get("white_player")
                    black_p = _chess_game.get("black_player")
                    if board.turn == chess.WHITE:
                        # 백 차례: Stockfish가 백이면 사람이 둘 수 없음
                        if mode == "stockfish" and white_p == "Stockfish":
                            continue
                        if nickname != white_p:
                            continue
                    else:
                        # 흑 차례: Stockfish가 흑이면 사람이 둘 수 없음
                        if mode == "stockfish" and black_p == "Stockfish":
                            continue
                        if nickname != black_p:
                            continue
                    try:
                        move = chess.Move.from_uci(uci)
                        if move not in board.legal_moves:
                            continue
                        now = time.time()
                        turn_started = _chess_game.get("turn_started_at") or now
                        elapsed = now - turn_started
                        if mode != "practice":
                            if board.turn == chess.WHITE:
                                w = _chess_game.get("white_time_remaining", CHESS_INITIAL_SECONDS) - elapsed
                                w = max(0.0, w)
                                if w <= 0:
                                    _chess_game["status"] = "time_loss_white"
                                    _chess_game["white_time_remaining"] = 0.0
                                    _chess_game["black_time_remaining"] = _chess_game.get("black_time_remaining", CHESS_INITIAL_SECONDS)
                                    _chess_game["turn_started_at"] = now
                                    _apply_chess_mmr_if_needed(_chess_game)
                                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                                    if mode == "stockfish":
                                        asyncio.create_task(_close_stockfish_async())
                                    continue
                                _chess_game["white_time_remaining"] = w + CHESS_INCREMENT_SECONDS
                            else:
                                b = _chess_game.get("black_time_remaining", CHESS_INITIAL_SECONDS) - elapsed
                                b = max(0.0, b)
                                if b <= 0:
                                    _chess_game["status"] = "time_loss_black"
                                    _chess_game["black_time_remaining"] = 0.0
                                    _chess_game["white_time_remaining"] = _chess_game.get("white_time_remaining", CHESS_INITIAL_SECONDS)
                                    _chess_game["turn_started_at"] = now
                                    _apply_chess_mmr_if_needed(_chess_game)
                                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                                    if mode == "stockfish":
                                        asyncio.create_task(_close_stockfish_async())
                                    continue
                                _chess_game["black_time_remaining"] = b + CHESS_INCREMENT_SECONDS
                        captured = _get_captured_piece(board, move)
                        if captured:
                            if board.turn == chess.WHITE:
                                _chess_game.setdefault("white_captured", []).append(captured)
                            else:
                                _chess_game.setdefault("black_captured", []).append(captured)
                        _chess_game.pop("pending_undo", None)
                        board.push(move)
                        _chess_game["turn_started_at"] = now
                        status = "active"
                        if board.is_checkmate():
                            # board.push(move) 이후 board.turn은 체크메이트를 당한 쪽
                            # 즉, 이제 둬야 할 차례인 쪽이 패배한 쪽
                            if board.turn == chess.WHITE:
                                status = "checkmate_white"   # 백이 체크메이트당함 → 흑 승리
                            else:
                                status = "checkmate_black"   # 흑이 체크메이트당함 → 백 승리
                        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                            # can_claim_draw: 50수 규칙(캡처/폰 이동 없이 50수), 3회 반복 포함
                            status = "draw"
                        _chess_game["status"] = status
                        _apply_chess_mmr_if_needed(_chess_game)
                        await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                        if mode == "stockfish" and status != "active":
                            asyncio.create_task(_close_stockfish_async())
                        if status == "active":
                            if mode == "stockfish":
                                # Stockfish는 흑/백 모두 가능하므로 stockfish_color 기준으로 차례 판단
                                engine_color = chess.WHITE if _chess_game.get("stockfish_color") == "white" else chess.BLACK
                                if board.turn == engine_color:
                                    _chess_pending_task = asyncio.create_task(_play_stockfish_chess_move())
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
                    _apply_chess_mmr_if_needed(_chess_game)
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                    if _chess_game.get("mode") == "stockfish":
                        asyncio.create_task(_close_stockfish_async())

                elif data.get("type") == "chess_end_practice" and _chess_game and _chess_game.get("mode") == "practice":
                    # 싱글플레이 종료: MMR 변동 없이 상태만 종료로 표시
                    if _chess_pending_task and not _chess_pending_task.done():
                        _chess_pending_task.cancel()
                        _chess_pending_task = None
                    _chess_game["status"] = "ended_practice"
                    _chess_game["turn_started_at"] = time.time()
                    await _room_broadcast(_chess_state_payload(_chess_game), exclude_ws=None)
                    _chess_game = None

                elif data.get("type") == "gomoku_start":
                    if not _serena_invited:
                        continue
                    if _gomoku_pending_task and not _gomoku_pending_task.done():
                        _gomoku_pending_task.cancel()
                        _gomoku_pending_task = None
                    _gomoku_game = {
                        "board": _gomoku_empty_board(),
                        "black_player": nickname,
                        "white_player": "Serena",
                        "mode": "serena",
                        "status": "active",
                        "turn": "black",
                        "last_move": None,
                    }
                    await _room_broadcast({
                        "type": "gomoku_state",
                        "board": [row[:] for row in _gomoku_game["board"]],
                        "black_player": _gomoku_game["black_player"],
                        "white_player": _gomoku_game["white_player"],
                        "mode": "serena",
                        "status": "active",
                        "turn": "black",
                        "last_move": None,
                    }, exclude_ws=None)

                elif data.get("type") == "gomoku_start_pvp" and (data.get("opponent") or "").strip():
                    opp = (data["opponent"] or "").strip()[:32]
                    if opp == nickname:
                        continue
                    if not any(u.get("name") == opp for _, u in _room_connections):
                        continue
                    if _gomoku_pending_task and not _gomoku_pending_task.done():
                        _gomoku_pending_task.cancel()
                        _gomoku_pending_task = None
                    _gomoku_game = {
                        "board": _gomoku_empty_board(),
                        "black_player": nickname,
                        "white_player": opp,
                        "mode": "pvp",
                        "status": "active",
                        "turn": "black",
                        "last_move": None,
                    }
                    await _room_broadcast({
                        "type": "gomoku_state",
                        "board": [row[:] for row in _gomoku_game["board"]],
                        "black_player": _gomoku_game["black_player"],
                        "white_player": _gomoku_game["white_player"],
                        "mode": "pvp",
                        "status": "active",
                        "turn": "black",
                        "last_move": None,
                    }, exclude_ws=None)

                elif data.get("type") == "gomoku_move":
                    row = data.get("row")
                    col = data.get("col")
                    if row is None or col is None or not _gomoku_game or _gomoku_game.get("status") != "active":
                        continue
                    r, c = int(row), int(col)
                    if not (0 <= r < GOMOKU_SIZE and 0 <= c < GOMOKU_SIZE):
                        continue
                    board = _gomoku_game["board"]
                    if board[r][c] != 0:
                        continue
                    turn = _gomoku_game.get("turn", "black")
                    if turn == "black":
                        if nickname != _gomoku_game.get("black_player"):
                            continue
                        board[r][c] = 1
                        if _gomoku_check_win(board, r, c, 1):
                            status = "win_black"
                        elif _gomoku_renju_forbidden_black(board, r, c):
                            board[r][c] = 0
                            continue  # 렌주 금수: 흑 쌍삼/쌍사/장목
                        else:
                            status = "active"
                            _gomoku_game["turn"] = "white"
                    else:
                        if nickname != _gomoku_game.get("white_player"):
                            continue
                        board[r][c] = 2
                        if _gomoku_check_win(board, r, c, 2):
                            status = "win_white"
                        else:
                            empty_count = sum(1 for row_ in board for cell in row_ if cell == 0)
                            status = "draw" if empty_count == 0 else "active"
                            if status == "active":
                                _gomoku_game["turn"] = "black"
                    if turn == "black" and status == "active":
                        empty_count = sum(1 for row_ in board for cell in row_ if cell == 0)
                        if empty_count == 0:
                            status = "draw"
                    _gomoku_game["status"] = status
                    _gomoku_game["last_move"] = [r, c]
                    await _room_broadcast({
                        "type": "gomoku_state",
                        "board": [row_[:] for row_ in board],
                        "turn": _gomoku_game.get("turn", "black"),
                        "status": status,
                        "black_player": _gomoku_game.get("black_player"),
                        "white_player": _gomoku_game.get("white_player"),
                        "mode": _gomoku_game.get("mode", "serena"),
                        "last_move": [r, c],
                    }, exclude_ws=None)
                    if status == "active" and _gomoku_game.get("turn") == "white" and _gomoku_game.get("mode") == "serena":
                        _gomoku_pending_task = asyncio.create_task(_play_gomoku_serena_move())

                elif data.get("type") == "gomoku_resign":
                    if not _gomoku_game or _gomoku_game.get("status") != "active":
                        continue
                    if _gomoku_pending_task and not _gomoku_pending_task.done():
                        _gomoku_pending_task.cancel()
                        _gomoku_pending_task = None
                    if nickname == _gomoku_game.get("black_player"):
                        status = "resign_black"
                    elif nickname == _gomoku_game.get("white_player"):
                        status = "resign_white"
                    else:
                        continue
                    _gomoku_game["status"] = status
                    await _room_broadcast({
                        "type": "gomoku_state",
                        "board": [row[:] for row in _gomoku_game["board"]],
                        "status": status,
                        "turn": _gomoku_game.get("turn"),
                        "black_player": _gomoku_game.get("black_player"),
                        "white_player": _gomoku_game.get("white_player"),
                        "mode": _gomoku_game.get("mode"),
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
        try:
            user_id = int(user_info.get("user_id") or 0)
        except (TypeError, ValueError):
            user_id = 0
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
            sid = status.get("sender_id")
            if key[0] == room_id and sid is not None and int(sid) == user_id and not status.get("read"):
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
                # 수신자(상대방)가 해당 1:1 채팅 화면을 활성화했을 때만 읽음 처리.
                # 작성자(발신자)가 들어왔을 때 보낸 viewed_room은 무시 → 배지는 상대가 읽었을 때만 사라짐.
                to_delete = []
                for key, status in _dm_message_read_status.items():
                    if key[0] != room_id:
                        continue
                    sid = status.get("sender_id")
                    if sid is not None and int(sid) == user_id:
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """브라우저 기본 favicon 요청 처리. 파일이 있으면 반환, 없으면 204로 404 방지."""
    path = STATIC_DIR / "favicon.ico"
    if path.exists():
        return FileResponse(path)
    return Response(status_code=204)


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
