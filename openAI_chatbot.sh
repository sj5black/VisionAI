#!/bin/bash
# 영어 채팅 챗봇 웹 서비스 실행
# URL: http://175.197.131.234:8004

cd "$(dirname "$0")"

# .env 로드 (OPENAI_API_KEY 등)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

uvicorn chat_bot_web.main:app --host 0.0.0.0 --port 8004 --reload
