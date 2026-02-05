# OpenAI 모델 테스트 가이드

## ✅ 지원하는 모델

### **Chat Completions API** (기존 방식)
- `gpt-4o-mini`
- `gpt-4o`
- `chatgpt-4o-latest`
- 기타 gpt-4 계열

### **Responses API** (최신 방식) ✨
- `gpt-5-mini` ⭐ **추천**
- `gpt-5-nano`
- `gpt-5.1-chat-latest` ⭐ **현재 설정**
- `gpt-5.2-chat-latest`
- 모든 gpt-5.x 계열

---

## 🔧 모델 변경 방법

`.env` 파일 수정:

```bash
# 방법 1: gpt-5-mini 사용 (빠르고 저렴)
OPENAI_MODEL=gpt-5-mini

# 방법 2: gpt-5.1-chat-latest 사용 (현재 설정)
OPENAI_MODEL=gpt-5.1-chat-latest

# 방법 3: gpt-5.2-chat-latest 사용 (최신)
OPENAI_MODEL=gpt-5.2-chat-latest
```

변경 후 서버 재시작:
```bash
./openAI_chatbot.sh
```

---

## 🧪 테스트 방법

### 1. **웹 브라우저 테스트**
```
http://175.197.131.234:8004
```

### 2. **Serena 테스트 (멀티 채팅)**
- "사용자들과 채팅" 클릭
- 채팅방 입장
- "Serena 초대" 버튼 클릭
- 로그 확인:
  ```
  🔄 Using Responses API for model: gpt-5-mini
  ✅ Serena: Hey! Nice to meet you :)
  ```

### 3. **AI 영어 채팅 테스트**
- "AI와 채팅" 클릭
- 대화 시작
- AI 응답 확인

---

## 📊 API 자동 선택 로직

코드가 **자동으로 올바른 API**를 선택합니다:

| 모델명 포함 | 사용 API |
|------------|----------|
| `gpt-5-mini` | ✅ Responses API |
| `gpt-5-nano` | ✅ Responses API |
| `gpt-5.1` | ✅ Responses API |
| `gpt-5.2` | ✅ Responses API |
| 기타 | Chat Completions API |

---

## 🔍 로그 확인

```bash
# 서버 로그 실시간 확인
tail -f /tmp/server.log

# 또는 터미널에서 직접 실행
cd /home/teddy/VisionAI
./openAI_chatbot.sh
```

---

## ⚠️ 문제 해결

### 문제: "Empty response"
→ reasoning model (gpt-5-nano) 사용 시 발생
→ `.env`에서 `gpt-5-mini` 또는 `gpt-5.1-chat-latest`로 변경

### 문제: "Model not found"
→ 모델명 오타 확인
→ `python test_openai_model.py`로 사용 가능한 모델 확인

### 문제: API 키 에러
→ `.env` 파일에 `OPENAI_API_KEY` 확인
→ 서버 재시작

---

## 📝 코드 수정 내역

### 1. **Serena (_call_serena)**
- gpt-5.x 모델 감지
- Responses API 자동 사용
- `input` 파라미터 + `output_text` 응답

### 2. **AI 채팅 (api_start, api_chat)**
- 대화 시작 시 API 타입 저장
- 대화 계속 시 동일한 API 사용
- Responses API는 대화 히스토리를 텍스트로 변환

---

**✅ 모든 준비 완료!** `.env`에서 원하는 모델로 변경 후 테스트하세요.
