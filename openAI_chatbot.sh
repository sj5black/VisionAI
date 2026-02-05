#!/bin/bash
# 영어 채팅 챗봇 웹 서비스 실행
# URL: http://175.197.131.234:8004
# 실행 시 8004 포트 사용 중인 기존 프로세스를 종료한 뒤 재시작

cd "$(dirname "$0")"

# .env 로드 (OPENAI_API_KEY 등)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PORT=8004
OLD_PID=$(lsof -ti:"$PORT" 2>/dev/null)
if [ -n "$OLD_PID" ]; then
  echo "Stopping existing process on port $PORT (PID: $OLD_PID)..."
  kill $OLD_PID 2>/dev/null
  sleep 2
  if kill -0 $OLD_PID 2>/dev/null; then
    kill -9 $OLD_PID 2>/dev/null
    sleep 1
  fi
  echo "Stopped."
fi

LOG_FILE=".visionai_chatbot.log"
PID_FILE=".visionai_chatbot.pid"

echo "Starting chatbot server on port $PORT (nohup, background)..."
# 기존 로그 파일이 있으면 새로 쓴다 (재시작 시)
: > "$LOG_FILE"
nohup uvicorn chat_bot_web.main:app --host 0.0.0.0 --port $PORT --reload >> "$LOG_FILE" 2>&1 &
UVICORN_PID=$!
echo $UVICORN_PID > "$PID_FILE"
echo "Server started. PID: $UVICORN_PID"
echo "Following logs (Ctrl+C to stop following; server keeps running)..."
echo "To stop server: kill \$(cat $PID_FILE) or kill processes on port $PORT"
echo "---"
trap 'exit 0' INT
tail -f "$LOG_FILE"
