(function () {
  const newChatBtn = document.getElementById('newChatBtn');
  const chatMessages = document.getElementById('chatMessages');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const situationTitle = document.getElementById('situationTitle');
  const situationSelect = document.getElementById('situationSelect');
  const loading = document.getElementById('loading');
  const errorEl = document.getElementById('error');

  let conversationId = null;

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.style.display = 'block';
    setTimeout(() => {
      errorEl.style.display = 'none';
    }, 5000);
  }

  function setLoading(on) {
    loading.style.display = on ? 'block' : 'none';
    sendBtn.disabled = on;
  }

  /** AI 메시지를 문장 단위로 줄바꿈 */
  function formatAiMessage(text) {
    if (!text) return '';
    return text
      .replace(/([.!?])\s+/g, '$1\n')
      .replace(/\n{2,}/g, '\n\n')
      .trim();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
    requestAnimationFrame(function () {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  function appendMessage(role, content, extraClass) {
    const div = document.createElement('div');
    div.className = 'msg ' + (extraClass || role);
    if (role === 'ai' || extraClass === 'feedback') {
      div.textContent = formatAiMessage(content);
    } else {
      div.textContent = content;
    }
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function appendAiMessage(english, korean) {
    const wrap = document.createElement('div');
    wrap.className = 'msg-ai-wrap';
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg ai';
    msgDiv.textContent = formatAiMessage(english);
    wrap.appendChild(msgDiv);
    if (korean) {
      const btn = document.createElement('button');
      btn.className = 'korean-toggle-btn';
      btn.type = 'button';
      btn.textContent = '한국어로 보기';
      const krDiv = document.createElement('div');
      krDiv.className = 'korean-translation';
      krDiv.style.display = 'none';
      krDiv.textContent = formatAiMessage(korean);
      btn.addEventListener('click', function () {
        if (krDiv.style.display === 'none') {
          krDiv.style.display = 'block';
          btn.textContent = '한국어 숨기기';
        } else {
          krDiv.style.display = 'none';
          btn.textContent = '한국어로 보기';
        }
        scrollToBottom();
      });
      wrap.appendChild(btn);
      wrap.appendChild(krDiv);
    }
    chatMessages.appendChild(wrap);
    scrollToBottom();
  }

  function startNewChat() {
    setLoading(true);
    const idx = parseInt(situationSelect.value, 10);
    const body = idx >= 0 ? JSON.stringify({ situation_index: idx }) : '{}';
    fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body,
    })
      .then(res => {
        if (!res.ok) {
          return res.json().catch(() => ({ detail: res.statusText }))
            .then(j => Promise.reject(new Error(j.detail || res.statusText)));
        }
        return res.json();
      })
      .then(data => {
        conversationId = data.conversation_id;
        situationTitle.textContent = data.situation_display || data.situation_title;
        chatMessages.innerHTML = '';
        if (data.first_korean) {
          appendAiMessage(data.first_message, data.first_korean);
        } else {
          appendMessage('ai', data.first_message);
        }
        userInput.value = '';
        userInput.focus();
      })
      .catch(err => {
        showError(err.message || '시작 실패. OPENAI_API_KEY 확인해 주세요.');
      })
      .finally(() => setLoading(false));
  }

  function loadSituations() {
    fetch('/api/situations')
      .then(res => res.json())
      .then(list => {
        list.forEach(function (s) {
          const opt = document.createElement('option');
          opt.value = s.index;
          opt.textContent = s.title + ' (' + s.title_ko + ')';
          situationSelect.appendChild(opt);
        });
      })
      .catch(function () {});
  }

  function sendMessage() {
    const text = (userInput.value || '').trim();
    if (!text || !conversationId) return;

    appendMessage('user', text);
    userInput.value = '';
    userInput.style.height = 'auto';
    scrollToBottom();
    setLoading(true);

    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: conversationId, user_message: text }),
    })
      .then(res => {
        if (!res.ok) {
          return res.json().catch(() => ({ detail: res.statusText }))
            .then(j => Promise.reject(new Error(j.detail || res.statusText)));
        }
        return res.json();
      })
      .then(data => {
        if (data.score != null) {
          appendMessage('score', '점수: ' + data.score + '/100', 'score');
        }
        if (data.correction) {
          appendMessage('feedback', '교정/피드백: ' + data.correction, 'feedback');
        }
        if (data.korean_reply) {
          appendAiMessage(data.reply, data.korean_reply);
        } else {
          appendMessage('ai', data.reply);
        }
        scrollToBottom();
      })
      .catch(err => {
        showError(err.message || '전송 실패');
      })
      .finally(() => setLoading(false));

    userInput.focus();
  }

  function startNewConversation() {
    conversationId = null;
    startNewChat();
  }

  newChatBtn.addEventListener('click', startNewConversation);

  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        return;
      }
      e.preventDefault();
      sendMessage();
    }
  });
  userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
  });
  loadSituations();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startNewChat);
  } else {
    startNewChat();
  }
})();
