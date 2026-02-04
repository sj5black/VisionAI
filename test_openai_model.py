import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv()

# API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

print(f"API Key loaded: {api_key[:20]}..." if api_key else "❌ API Key not found")
print(f"Model configured: {model}")
print("\n사용 가능한 모델 목록:")
print("-" * 50)

client = OpenAI(api_key=api_key)
models = client.models.list()

# gpt-5 계열만 필터링
for m in models.data:
    if 'gpt-5' in m.id or 'gpt-4' in m.id:
        print(f"  ✓ {m.id}")
