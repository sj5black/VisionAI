# VisionAI 웹 배포 가이드

## 🌐 웹 서버 실행

### 빠른 시작

```bash
cd /home/teddy/VisionAI

# 서버 시작 (또는 재시작)
./restart_web.sh
```

서버가 포트 8003에서 실행됩니다.

### 접속 URL

- **로컬**: http://localhost:8003
- **네트워크**: http://175.197.131.234:8003

---

## 🎨 기능

### 모델 선택

웹 UI에서 3가지 모델 중 선택 가능:

1. **🆕 VisionAI Pipeline** (권장)
   - YOLOv8n 객체 탐지 (6.3 MB)
   - MobileNetV3 감정/자세 분석 (2.5 MB)
   - 행동 예측
   - **총 ~10 MB 경량 모델**
   - **실시간 처리 (25-30 FPS)**

2. **Faster R-CNN** (ResNet50)
   - 전통적 객체 탐지
   - 높은 정확도
   - ~150 MB

3. **RetinaNet** (ResNet50)
   - One-stage detector
   - 빠른 속도
   - ~150 MB

---

## 📊 VisionAI Pipeline 결과

### 출력 정보

| 항목 | 설명 | 예시 |
|------|------|------|
| **label** | 탐지된 객체 | dog, cat |
| **score** | 신뢰도 | 0.92 |
| **emotion** | 감정 | playful, relaxed, alert |
| **pose** | 자세 | running, sitting, lying |
| **state** | 통합 상태 | playing, resting |
| **predicted next** | 예측 행동 | resting, walking |

### 감정 클래스 (5개)

- `relaxed` - 편안함
- `alert` - 경계
- `fearful` - 두려움
- `aggressive` - 공격성
- `playful` - 장난기

### 자세 클래스 (5개)

- `sitting` - 앉기
- `standing` - 서기
- `lying` - 눕기
- `running` - 달리기
- `jumping` - 점프

---

## 🔧 서버 관리

### 서버 시작

```bash
./restart_web.sh
```

### 서버 중지

```bash
# PID 확인
cat .visionai_web.pid

# 중지
kill $(cat .visionai_web.pid)
```

### 로그 확인

```bash
# 실시간 로그
tail -f .visionai_web.log

# 전체 로그
cat .visionai_web.log
```

### 수동 실행 (디버깅용)

```bash
# 포그라운드 실행
uvicorn webapp.main:app --host 0.0.0.0 --port 8003

# 리로드 모드 (개발)
uvicorn webapp.main:app --host 0.0.0.0 --port 8003 --reload
```

---

## ⚙️ 환경 변수

### 디바이스 설정

```bash
# CUDA 사용
VISIONAI_DEVICE=cuda ./restart_web.sh

# CPU만 사용
VISIONAI_DEVICE=cpu ./restart_web.sh

# 특정 GPU
VISIONAI_DEVICE=cuda:0 ./restart_web.sh
```

### 기존 기능 비활성화

```bash
# Animal insights 비활성화 (OpenCLIP)
VISIONAI_ENABLE_ANIMAL_INSIGHTS=0 ./restart_web.sh
```

---

## 🚀 프로덕션 배포

### systemd 서비스 생성

```bash
sudo nano /etc/systemd/system/visionai.service
```

```ini
[Unit]
Description=VisionAI Web Server
After=network.target

[Service]
Type=simple
User=teddy
WorkingDirectory=/home/teddy/VisionAI
Environment="VISIONAI_DEVICE=cuda"
ExecStart=/usr/bin/uvicorn webapp.main:app --host 0.0.0.0 --port 8003
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable visionai
sudo systemctl start visionai

# 상태 확인
sudo systemctl status visionai

# 로그 확인
sudo journalctl -u visionai -f
```

### Nginx 리버스 프록시

```nginx
server {
    listen 80;
    server_name visionai.example.com;

    location / {
        proxy_pass http://127.0.0.1:8003;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 큰 파일 업로드 허용
        client_max_body_size 50M;
    }
}
```

---

## 📈 성능 모니터링

### 리소스 사용량 확인

```bash
# CPU/메모리
top -p $(cat .visionai_web.pid)

# GPU 사용량 (CUDA)
watch nvidia-smi

# 네트워크
netstat -an | grep 8003
```

### 벤치마크

```bash
# Apache Bench
ab -n 100 -c 10 -p test_image.jpg -T 'multipart/form-data' \
   http://localhost:8003/api/detect

# wrk
wrk -t4 -c100 -d30s http://localhost:8003/
```

---

## 🐛 문제 해결

### 포트 이미 사용 중

```bash
# 8003 포트 사용 프로세스 찾기
lsof -i :8003

# 프로세스 종료
kill -9 <PID>
```

### 모델 로드 실패

```bash
# 의존성 확인
pip list | grep -E "torch|ultralytics|pillow"

# 재설치
pip install -r pipeline_requirements.txt --force-reinstall
```

### CUDA Out of Memory

```bash
# CPU 모드로 전환
VISIONAI_DEVICE=cpu ./restart_web.sh
```

### 느린 첫 요청

첫 요청 시 모델이 로드되므로 시간이 걸립니다:
- YOLOv8: 자동 다운로드 (~9 MB)
- 첫 추론: 모델 로딩 시간 포함
- 이후 요청: 빠름 (캐시 사용)

---

## 📝 API 엔드포인트

### POST /api/detect

이미지 분석 API

**Request (multipart/form-data)**:
```
image: file (required)
threshold: float (0.0-1.0, default: 0.5)
max_detections: int (1-300, default: 100)
model: string (default: "visionai_pipeline")
  - "visionai_pipeline" (🆕)
  - "fasterrcnn_resnet50_fpn_v2"
  - "retinanet_resnet50_fpn_v2"
```

**Response (application/json)**:
```json
{
  "id": "uuid",
  "model": "visionai_pipeline",
  "threshold": 0.5,
  "object_types": ["dog", "cat"],
  "objects": [
    {
      "label": "dog",
      "score": 0.92,
      "box_xyxy": [100, 150, 400, 500],
      "pipeline_insights": {
        "emotion": "playful",
        "emotion_confidence": 0.85,
        "pose": "running",
        "pose_confidence": 0.91,
        "combined_state": "playing",
        "predicted_action": "resting",
        "prediction_confidence": 0.65
      }
    }
  ],
  "pipeline_enabled": true,
  "processing_time": 0.035,
  "original_image_url": "/files/{id}/original",
  "annotated_image_url": "/files/{id}/annotated"
}
```

### GET /files/{id}/original

원본 이미지 다운로드

### GET /files/{id}/annotated

탐지 결과가 그려진 이미지 다운로드

---

## 🔄 업데이트

### 코드 업데이트 후

```bash
cd /home/teddy/VisionAI

# Git pull (if using git)
git pull

# 의존성 업데이트
pip install -r pipeline_requirements.txt

# 서버 재시작
./restart_web.sh
```

### 모델 업데이트

```bash
# 학습된 모델 배포
cp trained_emotion.pth /home/teddy/VisionAI/models/

# 환경변수로 경로 지정
export EMOTION_MODEL_PATH=/home/teddy/VisionAI/models/trained_emotion.pth
./restart_web.sh
```

---

## 📊 사용 통계

로그 파일에서 통계 추출:

```bash
# 총 요청 수
grep "POST /api/detect" .visionai_web.log | wc -l

# 평균 처리 시간
grep "processing_time" .visionai_web.log | \
  grep -oP '"processing_time":\s*\K[0-9.]+' | \
  awk '{sum+=$1; n++} END {print sum/n}'

# 가장 많이 탐지된 객체
grep "object_types" .visionai_web.log | \
  grep -oP '"\w+"' | sort | uniq -c | sort -rn | head -10
```

---

## ✅ 체크리스트

배포 전:
- [ ] 의존성 설치 완료
- [ ] 테스트 실행 성공
- [ ] 포트 8003 사용 가능
- [ ] 방화벽 설정 확인

배포 후:
- [ ] 웹 UI 접속 확인
- [ ] 이미지 업로드 테스트
- [ ] VisionAI Pipeline 선택 가능
- [ ] 결과 표시 정상
- [ ] 로그 모니터링 설정

---

**최종 업데이트**: 2026-02-02  
**서버 주소**: http://175.197.131.234:8003
