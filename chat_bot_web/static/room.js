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
    participantList.innerHTML = '';
    (list || []).forEach(function (nick) {
      const li = document.createElement('li');
      li.textContent = nick;
      participantList.appendChild(li);
    });
  }

  function appendRoomMessage(type, nickname, text) {
    const div = document.createElement('div');
    div.className = 'roomMsg ' + type;
    if (type === 'system') {
      div.textContent = text;
    } else if (type === 'me') {
      div.textContent = text;
    } else {
      const nickSpan = document.createElement('div');
      nickSpan.className = 'roomMsgNick';
      nickSpan.textContent = nickname;
      div.appendChild(nickSpan);
      div.appendChild(document.createTextNode(text));
    }
    roomMessages.appendChild(div);
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
          appendRoomMessage(isMe ? 'me' : 'other', data.nickname, data.text || '');
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
    roomInput.focus();
  }

  roomSendBtn.addEventListener('click', sendMessage);
  roomInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  });
})();
