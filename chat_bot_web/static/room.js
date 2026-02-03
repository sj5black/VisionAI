(function () {
  const nickSection = document.getElementById('nickSection');
  const roomSection = document.getElementById('roomSection');
  const nickInput = document.getElementById('nickInput');
  const enterBtn = document.getElementById('enterBtn');
  const nickError = document.getElementById('nickError');
  const participantCount = document.getElementById('participantCount');
  const participantList = document.getElementById('participantList');
  const roomMessages = document.getElementById('roomMessages');
  const roomInput = document.getElementById('roomInput');
  const roomSendBtn = document.getElementById('roomSendBtn');

  let ws = null;
  let myNickname = '';
  let contextMenuEl = null;

  function showNickError(msg) {
    nickError.textContent = msg || '';
    nickError.style.display = msg ? 'block' : 'none';
  }

  function scrollRoomBottom() {
    roomMessages.scrollTop = roomMessages.scrollHeight;
    requestAnimationFrame(function () {
      roomMessages.scrollTop = roomMessages.scrollHeight;
    });
  }

  function renderParticipants(list) {
    participantCount.textContent = list.length;
    participantList.textContent = (list || []).join(', ');
  }

  function ensureContextMenu() {
    if (contextMenuEl) return contextMenuEl;
    const menu = document.createElement('div');
    menu.className = 'roomContextMenu';
    menu.style.display = 'none';
    menu.innerHTML =
      '<button data-action="edit">메시지 수정</button>' +
      '<button data-action="delete" class="danger">메시지 삭제</button>' +
      '<button data-action="close">닫기</button>';
    document.body.appendChild(menu);
    contextMenuEl = menu;

    document.addEventListener('click', function () {
      hideContextMenu();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') hideContextMenu();
    });
    return menu;
  }

  function hideContextMenu() {
    if (!contextMenuEl) return;
    contextMenuEl.style.display = 'none';
    contextMenuEl.dataset.messageId = '';
  }

  function showContextMenu(x, y, messageId, currentText) {
    const menu = ensureContextMenu();
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';
    menu.style.display = 'block';
    menu.dataset.messageId = messageId || '';
    menu.dataset.currentText = currentText || '';
  }

  function setUnreadBadge(rowEl, unreadCount) {
    if (!rowEl) return;
    const badge = rowEl.querySelector('.unread-badge');
    if (unreadCount > 0) {
      if (badge) {
        badge.textContent = unreadCount;
      } else {
        const newBadge = document.createElement('span');
        newBadge.className = 'unread-badge';
        newBadge.textContent = unreadCount;
        // 내 메시지는 배지가 말풍선 "바깥(왼쪽)"에 위치하도록 앞에 추가
        rowEl.insertBefore(newBadge, rowEl.firstChild);
      }
    } else {
      if (badge) badge.remove();
    }
  }

  function appendRoomMessage(type, nickname, text, messageId, unreadCount) {
    if (type === 'system') {
      const sys = document.createElement('div');
      sys.className = 'roomMsg system';
      sys.textContent = text;
      roomMessages.appendChild(sys);
      scrollRoomBottom();
      return;
    }

    const row = document.createElement('div');
    row.className = 'roomMsgRow ' + type;
    if (messageId) row.dataset.messageId = messageId;

    const bubble = document.createElement('div');
    bubble.className = 'roomBubble ' + type;

    if (type === 'me') {
      bubble.textContent = text;
      // 배지는 말풍선 바깥(왼쪽)에
      if (unreadCount > 0) {
        const badge = document.createElement('span');
        badge.className = 'unread-badge';
        badge.textContent = unreadCount;
        row.appendChild(badge);
      }
      row.appendChild(bubble);

      // 우클릭: 수정/삭제 메뉴
      bubble.addEventListener('contextmenu', function (e) {
        e.preventDefault();
        if (!messageId) return;
        showContextMenu(e.clientX, e.clientY, messageId, bubble.textContent || '');
      });
    } else {
      const nickSpan = document.createElement('div');
      nickSpan.className = 'roomMsgNick';
      nickSpan.textContent = nickname;
      bubble.appendChild(nickSpan);
      const textSpan = document.createElement('span');
      textSpan.textContent = text;
      bubble.appendChild(textSpan);
      row.appendChild(bubble);
    }

    roomMessages.appendChild(row);
    scrollRoomBottom();
  }

  function connect() {
    var url = (window.location.origin.replace(/^http/, 'ws') + '/ws/room');
    ws = new WebSocket(url);

    ws.onopen = function () {
      ws.send(JSON.stringify({ type: 'join', nickname: myNickname }));
    };

    ws.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'error') {
          nickSection.style.display = 'block';
          roomSection.style.display = 'none';
          showNickError(data.message || '오류');
          ws.close();
          return;
        }
        if (data.type === 'participants') {
          renderParticipants(data.list || []);
          return;
        }
        if (data.type === 'system') {
          appendRoomMessage('system', null, data.message || '');
          return;
        }
        if (data.type === 'chat') {
          const isMe = data.nickname === myNickname;
          appendRoomMessage(
            isMe ? 'me' : 'other',
            data.nickname,
            data.text || '',
            data.message_id,
            isMe ? data.unread_count : 0
          );
          // 상대방 메시지는 자동으로 읽음 처리
          if (!isMe && data.message_id) {
            ws.send(JSON.stringify({ type: 'read', message_id: data.message_id }));
          }
          return;
        }
        if (data.type === 'read_update') {
          // 내가 보낸 메시지의 읽음 상태 업데이트
          const rowEl = roomMessages.querySelector('[data-message-id="' + data.message_id + '"]');
          setUnreadBadge(rowEl, data.unread_count || 0);
          return;
        }
        if (data.type === 'edit') {
          const rowEl = roomMessages.querySelector('[data-message-id="' + data.message_id + '"]');
          if (rowEl) {
            const bubble = rowEl.querySelector('.roomBubble');
            if (bubble) {
              // other는 닉네임 라인이 있으니 유지하고 텍스트만 교체
              if (rowEl.classList.contains('other')) {
                const nickEl = bubble.querySelector('.roomMsgNick');
                const textSpan = document.createElement('span');
                textSpan.textContent = data.text || '';
                bubble.textContent = '';
                if (nickEl) bubble.appendChild(nickEl);
                bubble.appendChild(textSpan);
              } else {
                bubble.textContent = data.text || '';
              }
            }
          }
          return;
        }
        if (data.type === 'delete') {
          const rowEl = roomMessages.querySelector('[data-message-id="' + data.message_id + '"]');
          if (rowEl) rowEl.remove();
          return;
        }
      } catch (e) {}
    };

    ws.onclose = function () {
      ws = null;
    };

    ws.onerror = function () {
      showNickError('연결 오류. 새로고침 후 다시 시도해 주세요.');
    };
  }

  enterBtn.addEventListener('click', function () {
    const nick = (nickInput.value || '').trim();
    if (nick.length < 2 || nick.length > 32) {
      showNickError('닉네임은 2~32자로 입력해 주세요.');
      return;
    }
    showNickError('');
    myNickname = nick;
    nickSection.style.display = 'none';
    roomSection.style.display = 'flex';
    roomSection.style.flexDirection = 'column';
    roomSection.style.flex = '1';
    roomSection.style.minHeight = '0';
    connect();
    roomInput.focus();
  });

  nickInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') enterBtn.click();
  });

  function sendMessage() {
    const text = (roomInput.value || '').trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'chat', text: text }));
    roomInput.value = '';
    roomInput.style.height = 'auto';
    roomInput.focus();
  }

  // textarea 자동 높이 조절
  roomInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
  });

  roomSendBtn.addEventListener('click', sendMessage);
  roomInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // 컨텍스트 메뉴 액션 처리
  document.addEventListener('click', function (e) {
    if (!contextMenuEl || contextMenuEl.style.display === 'none') return;
    const btn = e.target && e.target.closest ? e.target.closest('button[data-action]') : null;
    if (!btn || !contextMenuEl.contains(btn)) return;

    e.preventDefault();
    e.stopPropagation();

    const action = btn.dataset.action;
    const messageId = contextMenuEl.dataset.messageId;
    const currentText = contextMenuEl.dataset.currentText || '';
    hideContextMenu();

    if (!messageId || !ws || ws.readyState !== WebSocket.OPEN) return;

    if (action === 'edit') {
      const next = window.prompt('메시지 수정', currentText);
      if (next == null) return;
      const t = (next || '').trim();
      if (!t) return;
      ws.send(JSON.stringify({ type: 'edit', message_id: messageId, text: t }));
      return;
    }
    if (action === 'delete') {
      if (!window.confirm('이 메시지를 삭제할까요?')) return;
      ws.send(JSON.stringify({ type: 'delete', message_id: messageId }));
      return;
    }
  }, true);
})();
