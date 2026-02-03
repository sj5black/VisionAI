"""
영어 채팅 챗봇
- AI가 다양한 상황 중 하나를 골라 질문을 시작
- 사용자 답변에 대해 교정/피드백, 0-100 점수, 대화 이어가기
"""

import os
import re
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 다양한 대화 상황 (영어로 설명해 AI가 맥락을 이해하도록)
CONVERSATION_SITUATIONS = [
    "Job interview: You are the interviewer. Ask about the candidate's experience and goals.",
    "At a restaurant: You are a waiter. Take the customer's order and make small talk.",
    "Travel: You meet a fellow traveler at an airport. Ask about their trip and destination.",
    "Shopping: You are a store clerk. Help the customer find something and discuss preferences.",
    "First day at work: You are a friendly colleague. Welcome the new person and ask about their background.",
    "At a party: You are a host. Introduce yourself and ask what they do or how they know the host.",
    "Doctor's visit: You are the doctor. Ask about symptoms and lifestyle.",
    "Hotel check-in: You are the front desk staff. Confirm reservation and offer information.",
    "Coffee shop: You are a barista. Take their order and chat about the weather or their day.",
    "Booking a flight: You are an airline agent. Ask destination, dates, and preferences.",
]

# 응답 파싱용 구분자 (AI에게 이 형식으로 답하라고 지시)
CORRECTION_LABEL = "CORRECTION:"
SCORE_LABEL = "SCORE:"
REPLY_LABEL = "REPLY:"
KOREAN_LABEL = "KOREAN:"

SYSTEM_PROMPT = """You are a friendly English conversation partner and tutor. Your role:
1. You will be given a situation. Start the conversation with ONE short line in English—it can be a question, a greeting, a comment, or an observation. On the next line, add: KOREAN: (natural Korean translation of your line). Do not include CORRECTION/SCORE/REPLY in this first message.
2. When the user replies in English, you MUST respond with exactly this structure (use these exact labels, one per line):
   CORRECTION: (If their English has grammar mistakes or unnatural phrasing, correct it and briefly explain in Korean. If it's fine, say "Good job!" or similar in Korean.)
   SCORE: (A number from 0 to 100. 0-100 for how natural and grammatically correct their reply was. Be fair: minor mistakes 70-85, good 85-95, perfect 95-100.)
   REPLY: (Your next line in English only—continue the conversation naturally. Up to 4 sentences.)
   KOREAN: (Natural Korean translation of your REPLY only. One line.)

REPLY style—vary your responses naturally:
- Do NOT always end with a question. Mix it up.
- Sometimes use declarative statements only: share your thoughts, react to what they said, make small talk, or add information.
- Sometimes end with a question to invite them to reply.

Vocabulary & expression level:
- Use advanced vocabulary and expressions at TOEIC 900+ / IELTS 8.0+ level.
- Incorporate sophisticated words, idioms, phrasal verbs, and nuanced expressions (e.g., "in retrospect," "to a certain extent," "it goes without saying," "by and large," "on the flip side").
- Be specific and concrete rather than vague. Add detail where natural (e.g., describe feelings, situations, or context precisely).
- Aim for richness and variety in phrasing—avoid repetitive or overly simple structures.

Rules:
- Always write CORRECTION, SCORE, REPLY, and KOREAN in every response after the user's first reply.
- Keep the conversation in English except for the CORRECTION explanation (use Korean for that).
- Be encouraging. Score fairly based on grammar, word choice, and naturalness.
- Progress the conversation like a real chat: react, share, comment—not just ask questions."""


def get_client():
    """OpenAI 클라이언트 생성"""
    if not OPENAI_API_KEY:
        raise ValueError(".env에 OPENAI_API_KEY를 설정해 주세요.")
    return OpenAI(api_key=OPENAI_API_KEY)


def parse_ai_response(text: str) -> tuple[str | None, int | None, str, str | None]:
    """AI 응답에서 CORRECTION, SCORE, REPLY, KOREAN 추출"""
    correction = None
    score = None
    reply = None
    korean = None

    if CORRECTION_LABEL in text:
        try:
            start = text.index(CORRECTION_LABEL) + len(CORRECTION_LABEL)
            end = text.find(SCORE_LABEL, start)
            if end == -1:
                end = len(text)
            correction = text[start:end].strip()
        except ValueError:
            pass

    if SCORE_LABEL in text:
        match = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
        if match:
            score = min(100, max(0, int(match.group(1))))

    if REPLY_LABEL in text:
        try:
            start = text.index(REPLY_LABEL) + len(REPLY_LABEL)
            end = text.find(KOREAN_LABEL, start)
            if end == -1:
                reply = text[start:].strip()
            else:
                reply = text[start:end].strip()
        except ValueError:
            pass

    if KOREAN_LABEL in text:
        try:
            start = text.index(KOREAN_LABEL) + len(KOREAN_LABEL)
            korean = text[start:].strip()
        except ValueError:
            pass

    return correction, score, reply, korean


def run_chat():
    """챗봇 대화 루프"""
    client = get_client()
    situation = random.choice(CONVERSATION_SITUATIONS)

    # 시스템 메시지 + 상황 안내
    system_content = f"{SYSTEM_PROMPT}\n\nCurrent situation:\n{situation}"

    messages = [
        {"role": "system", "content": system_content},
    ]

    print("=" * 50)
    print("영어 채팅 챗봇")
    print("=" * 50)
    print(f"상황: {situation.split(':')[0].strip()}")
    print("영어로 대화해 보세요. 종료하려면 'quit' 또는 'exit' 입력\n")

    # AI가 먼저 질문/시작 멘트
    first = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_completion_tokens=300,
    )
    first_reply = first.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": first_reply})
    print(f"[AI] {first_reply}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("대화를 종료합니다.")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=500,
        )
        ai_text = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": ai_text})

        correction, score, reply, _ = parse_ai_response(ai_text)

        # 파싱된 내용이 있으면 구분해서 출력, 없으면 원문 전체 출력
        if correction is not None or score is not None or reply:
            if correction is not None:
                print(f"\n  [교정/피드백] {correction}")
            if score is not None:
                print(f"  [점수] {score}/100")
            if reply:
                print(f"\n[AI] {reply}\n")
            else:
                print(f"\n[AI] (파싱 실패 시 원문)\n{ai_text}\n")
        else:
            print(f"\n[AI] {ai_text}\n")


if __name__ == "__main__":
    run_chat()
