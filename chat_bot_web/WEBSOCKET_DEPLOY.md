# 멀티 채팅방 WebSocket 연결

멀티 채팅방은 **WebSocket** (`/ws/room`)을 사용합니다.

## 직접 접속 (uvicorn만 사용)

`uvicorn chat_bot_web.main:app --host 0.0.0.0 --port 8004` 로 실행하고  
`http://175.197.131.234:8004` 로 접속하면 WebSocket도 같은 호스트/포트로 연결되므로 **별도 설정 없이** 동작합니다.

## nginx 등 리버스 프록시 뒤에 둘 때

사이트를 **80/443 포트의 nginx**로 서비스하고, 백엔드는 8004로 프록시하는 경우  
브라우저는 `http://도메인/` 으로 접속하지만 WebSocket은 **프록시가 업그레이드**해 줘야 합니다.

nginx 예시:

```nginx
location / {
    proxy_pass http://127.0.0.1:8004;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    # WebSocket 업그레이드
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
}

location /ws/ {
    proxy_pass http://127.0.0.1:8004;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
}
```

- `Upgrade`, `Connection "upgrade"` 가 있으면 WebSocket 요청이 백엔드(8004)로 전달됩니다.
- 프록시 뒤에서는 **같은 도메인/경로**로 접속해야 하므로, 클라이언트는 `wss://도메인/ws/room` 으로 자동 연결됩니다.

## 연결 실패 시 확인

1. **직접 접속**: `http://서버IP:8004` 로 접속했는지 확인 (프록시 없이 8004 포트).
2. **방화벽**: 8004 포트가 열려 있는지 확인.
3. **프록시 사용 시**: 위와 같이 nginx에 WebSocket 업그레이드 설정이 들어가 있는지 확인.
