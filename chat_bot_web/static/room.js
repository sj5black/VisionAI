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
  const roomEmojiBtn = document.getElementById('roomEmojiBtn');
  const roomEmojiPanel = document.getElementById('roomEmojiPanel');
  const roomNickEditBtn = document.getElementById('roomNickEditBtn');
  const roomLeaveBtn = document.getElementById('roomLeaveBtn');
  const roomImageBtn = document.getElementById('roomImageBtn');
  const roomImageInput = document.getElementById('roomImageInput');

  var EMOJI_LIST = ['😀','😃','😄','😁','😅','😂','🤣','😊','😇','🙂','😉','😌','😍','🥰','😘','😗','😙','😚','😋','😛','😜','🤪','😝','🤑','🤗','🤭','🤫','🤔','🤐','🤨','😐','😑','😶','😏','😒','🙄','😬','🤥','😌','😔','😪','🤤','😴','😷','🤒','🤕','🤢','🤮','👍','👎','👌','✌️','🤞','🤟','🤘','🤙','👋','🤚','🖐️','✋','🖖','👏','🙌','👐','🤲','🙏','❤️','🧡','💛','💚','💙','💜','🖤','💔','❣️','💕','💞','💓','💗','💖','💘','💝','💟','✨','⭐','🌟','💫','🔥','💯'];

  let ws = null;
  let myNickname = '';
  let contextMenuEl = null;

  function escapeHtml(s) {
    var div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

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

  function isOnlyEmoji(text) {
    var t = (text || '').trim();
    if (!t.length || t.length > 48) return false;
    var rest = t;
    var listByLen = EMOJI_LIST.slice().sort(function (a, b) { return b.length - a.length; });
    while (rest.length) {
      var found = false;
      for (var i = 0; i < listByLen.length; i++) {
        var em = listByLen[i];
        if (rest.indexOf(em) === 0) {
          rest = rest.slice(em.length);
          found = true;
          break;
        }
      }
      if (!found) return false;
    }
    return true;
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
        rowEl.appendChild(newBadge);
      }
    } else {
      if (badge) badge.remove();
    }
  }

  function appendRoomMessage(type, nickname, text, messageId, unreadCount, imageUrl) {
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

    var isEmojiOnly = !imageUrl && isOnlyEmoji(text);

    if (isEmojiOnly) {
      if (type === 'me') {
        var emojiDiv = document.createElement('div');
        emojiDiv.className = 'roomMsgEmoji roomMsgEmoji--me';
        emojiDiv.textContent = text;
        row.appendChild(emojiDiv);
        if (unreadCount > 0) {
          var badge = document.createElement('span');
          badge.className = 'unread-badge';
          badge.textContent = unreadCount;
          row.appendChild(badge);
        }
        emojiDiv.addEventListener('contextmenu', function (e) {
          e.preventDefault();
          if (!messageId) return;
          showContextMenu(e.clientX, e.clientY, messageId, emojiDiv.textContent || '');
        });
      } else {
        var wrap = document.createElement('div');
        wrap.className = 'roomMsgOtherWrap';
        var nickLabel = document.createElement('div');
        nickLabel.className = 'roomMsgNickAbove';
        nickLabel.textContent = nickname;
        wrap.appendChild(nickLabel);
        var emojiDiv = document.createElement('div');
        emojiDiv.className = 'roomMsgEmoji roomMsgEmoji--other';
        emojiDiv.textContent = text;
        wrap.appendChild(emojiDiv);
        row.appendChild(wrap);
      }
    } else {
      var bubble = document.createElement('div');
      bubble.className = 'roomBubble ' + type;

      if (imageUrl) {
        var imgWrap = document.createElement('div');
        imgWrap.className = 'roomMsgImageWrap';
        var img = document.createElement('img');
        img.src = imageUrl;
        img.className = 'roomMsgImage';
        img.alt = '첨부 이미지';
        imgWrap.appendChild(img);
        bubble.appendChild(imgWrap);
      }
      if (text) {
        var textSpan = document.createElement('span');
        textSpan.className = 'roomMsgText';
        textSpan.textContent = text;
        bubble.appendChild(textSpan);
      }

      if (type === 'me') {
        row.appendChild(bubble);
        if (unreadCount > 0) {
          var badge = document.createElement('span');
          badge.className = 'unread-badge';
          badge.textContent = unreadCount;
          row.appendChild(badge);
        }
        bubble.addEventListener('contextmenu', function (e) {
          e.preventDefault();
          if (!messageId) return;
          var txtEl = bubble.querySelector('.roomMsgText');
          var txt = txtEl ? txtEl.textContent : bubble.textContent;
          showContextMenu(e.clientX, e.clientY, messageId, (txt || '').trim());
        });
      } else {
        var wrap = document.createElement('div');
        wrap.className = 'roomMsgOtherWrap';
        var nickLabel = document.createElement('div');
        nickLabel.className = 'roomMsgNickAbove';
        nickLabel.textContent = nickname;
        wrap.appendChild(nickLabel);
        wrap.appendChild(bubble);
        row.appendChild(wrap);
      }
    }

    row.dataset.rawText = text || '';
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
          if (roomNickEditBtn) roomNickEditBtn.style.display = 'none';
          if (roomLeaveBtn) roomLeaveBtn.style.display = 'none';
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
            isMe ? data.unread_count : 0,
            data.image_url || null
          );
          if (!isMe && data.message_id) {
            ws.send(JSON.stringify({ type: 'read', message_id: data.message_id }));
          }
          return;
        }
        if (data.type === 'link_preview') {
          var rowEl = roomMessages.querySelector('[data-message-id="' + data.message_id + '"]');
          if (rowEl && data.preview) {
            var card = document.createElement('a');
            card.href = data.preview.url;
            card.target = '_blank';
            card.rel = 'noopener noreferrer';
            card.className = 'roomLinkPreview';
            var html = '';
            if (data.preview.image) {
              html += '<img class="roomLinkPreviewImg" src="' + escapeHtml(data.preview.image) + '" alt="" />';
            }
            html += '<div class="roomLinkPreviewBody">';
            if (data.preview.title) html += '<div class="roomLinkPreviewTitle">' + escapeHtml(data.preview.title) + '</div>';
            if (data.preview.description) html += '<div class="roomLinkPreviewDesc">' + escapeHtml(data.preview.description) + '</div>';
            html += '</div>';
            card.innerHTML = html;
            var content = rowEl.querySelector('.roomBubble');
            if (content) content.appendChild(card);
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
            const newText = data.text || '';
            const bubble = rowEl.querySelector('.roomBubble');
            const emojiDiv = rowEl.querySelector('.roomMsgEmoji');
            if (bubble) {
              var txtEl = bubble.querySelector('.roomMsgText');
              var imgWrap = bubble.querySelector('.roomMsgImageWrap');
              if (isOnlyEmoji(newText) && !imgWrap) {
                var isMe = rowEl.classList.contains('me');
                var newEmojiDiv = document.createElement('div');
                newEmojiDiv.className = 'roomMsgEmoji roomMsgEmoji--' + (isMe ? 'me' : 'other');
                newEmojiDiv.textContent = newText;
                if (isMe) {
                  rowEl.replaceChild(newEmojiDiv, bubble);
                  newEmojiDiv.addEventListener('contextmenu', function (e) {
                    e.preventDefault();
                    if (!data.message_id) return;
                    showContextMenu(e.clientX, e.clientY, data.message_id, newEmojiDiv.textContent || '');
                  });
                } else {
                  bubble.parentNode.replaceChild(newEmojiDiv, bubble);
                }
              } else {
                if (txtEl) {
                  txtEl.textContent = newText;
                } else if (newText) {
                  var span = document.createElement('span');
                  span.className = 'roomMsgText';
                  span.textContent = newText;
                  bubble.appendChild(span);
                }
              }
            } else if (emojiDiv) {
              if (isOnlyEmoji(newText)) {
                emojiDiv.textContent = newText;
              } else {
                var isMe = rowEl.classList.contains('me');
                var newBubble = document.createElement('div');
                newBubble.className = 'roomBubble ' + (isMe ? 'me' : 'other');
                newBubble.textContent = newText;
                if (isMe) {
                  rowEl.replaceChild(newBubble, emojiDiv);
                  newBubble.addEventListener('contextmenu', function (e) {
                    e.preventDefault();
                    if (!data.message_id) return;
                    showContextMenu(e.clientX, e.clientY, data.message_id, newBubble.textContent || '');
                  });
                } else {
                  emojiDiv.parentNode.replaceChild(newBubble, emojiDiv);
                }
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

  function enterRoom(nick) {
    if (!nick || nick.length < 2 || nick.length > 32) return;
    showNickError('');
    myNickname = nick;
    nickInput.value = nick;
    nickSection.style.display = 'none';
    roomSection.style.display = 'flex';
    roomSection.style.flexDirection = 'column';
    roomSection.style.flex = '1';
    roomSection.style.minHeight = '0';
    if (roomNickEditBtn) roomNickEditBtn.style.display = 'flex';
    if (roomLeaveBtn) roomLeaveBtn.style.display = 'flex';
    connect();
    roomInput.focus();
  }

  function leaveRoom() {
    if (ws) {
      ws.close();
      ws = null;
    }
    nickSection.style.display = 'block';
    roomSection.style.display = 'none';
    if (roomNickEditBtn) roomNickEditBtn.style.display = 'none';
    if (roomLeaveBtn) roomLeaveBtn.style.display = 'none';
    showNickError('');
  }

  enterBtn.addEventListener('click', function () {
    const nick = (nickInput.value || '').trim();
    if (nick.length < 2 || nick.length > 32) {
      showNickError('닉네임은 2~32자로 입력해 주세요.');
      return;
    }
    enterRoom(nick);
  });

  nickInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') enterBtn.click();
  });

  // 저장된 닉네임이 있으면 해당 IP에서 자동 입장
  fetch('/api/room/saved-nickname')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      var nick = (data.nickname || '').trim();
      if (nick.length >= 2 && nick.length <= 32) {
        enterRoom(nick);
      }
    })
    .catch(function () {});

  if (roomNickEditBtn) {
    roomNickEditBtn.addEventListener('click', function (e) {
      e.preventDefault();
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      var newNick = window.prompt('새 닉네임 (2~32자)', myNickname);
      if (newNick == null) return;
      newNick = (newNick || '').trim();
      if (newNick.length < 2 || newNick.length > 32) {
        window.alert('닉네임은 2~32자로 입력해 주세요.');
        return;
      }
      myNickname = newNick;
      ws.send(JSON.stringify({ type: 'rename', nickname: newNick }));
    });
  }

  if (roomLeaveBtn) {
    roomLeaveBtn.addEventListener('click', function (e) {
      e.preventDefault();
      leaveRoom();
    });
  }

  window.addEventListener('beforeunload', function () {
    if (ws) ws.close();
  });

  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState !== 'visible') return;
    var inRoom = roomSection && roomSection.style.display !== 'none';
    var disconnected = !ws || ws.readyState !== WebSocket.OPEN;
    if (inRoom && disconnected && myNickname) {
      connect();
    }
  });

  function sendMessage(optText, optImageUrl) {
    var text = (optText != null ? optText : (roomInput && roomInput.value) || '').trim();
    var imageUrl = optImageUrl || null;
    if ((!text && !imageUrl) || !ws || ws.readyState !== WebSocket.OPEN) return;
    var payload = { type: 'chat', text: text };
    if (imageUrl) payload.image_url = imageUrl;
    ws.send(JSON.stringify(payload));
    if (roomInput) {
      roomInput.value = '';
      roomInput.style.height = 'auto';
    }
    roomInput.focus();
  }

  // textarea 자동 높이 조절
  roomInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
  });

  function isMobileRoom() {
    return window.matchMedia && window.matchMedia('(max-width: 768px)').matches;
  }

  function updateRoomInputPlaceholder() {
    if (!roomInput) return;
    roomInput.placeholder = isMobileRoom()
      ? '메시지 입력... (Enter: 줄바꿈)'
      : '메시지 입력... (Shift+Enter: 줄바꿈)';
  }
  updateRoomInputPlaceholder();
  if (window.matchMedia) {
    var mq = window.matchMedia('(max-width: 768px)');
    if (mq.addEventListener) mq.addEventListener('change', updateRoomInputPlaceholder);
    else if (mq.addListener) mq.addListener(updateRoomInputPlaceholder);
  }

  roomSendBtn.addEventListener('click', function () { sendMessage(); });

  if (roomImageBtn && roomImageInput) {
    roomImageBtn.addEventListener('click', function () { roomImageInput.click(); });
    roomImageInput.addEventListener('change', function () {
      var file = this.files && this.files[0];
      if (!file || !ws || ws.readyState !== WebSocket.OPEN) { this.value = ''; return; }
      var fd = new FormData();
      fd.append('file', file);
      fetch('/api/room/upload', { method: 'POST', body: fd })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          var url = data.url;
          if (url) {
            var text = (roomInput && roomInput.value || '').trim();
            sendMessage(text, url);
          }
        })
        .catch(function () {})
        .finally(function () { roomImageInput.value = ''; });
    });
  }
  roomInput.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter') return;
    if (isMobileRoom()) {
      return;
    }
    if (!e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function insertEmojiAtCursor(emoji) {
    var start = roomInput.selectionStart;
    var end = roomInput.selectionEnd;
    var text = roomInput.value;
    roomInput.value = text.slice(0, start) + emoji + text.slice(end);
    roomInput.selectionStart = roomInput.selectionEnd = start + emoji.length;
    roomInput.focus();
  }

  function initEmojiPanel() {
    if (roomEmojiPanel.querySelector('.roomEmojiPanelInner')) return;
    var header = document.createElement('div');
    header.className = 'roomEmojiPanelHeader';
    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'roomEmojiPanelClose';
    closeBtn.title = '닫기';
    closeBtn.innerHTML = '&times;';
    closeBtn.addEventListener('click', function () {
      roomEmojiPanel.style.display = 'none';
    });
    header.appendChild(closeBtn);
    roomEmojiPanel.appendChild(header);
    var inner = document.createElement('div');
    inner.className = 'roomEmojiPanelInner';
    EMOJI_LIST.forEach(function (emoji) {
      var span = document.createElement('span');
      span.textContent = emoji;
      span.setAttribute('role', 'button');
      span.tabIndex = 0;
      span.addEventListener('click', function () {
        insertEmojiAtCursor(emoji);
      });
      inner.appendChild(span);
    });
    roomEmojiPanel.appendChild(inner);
  }

  if (roomEmojiBtn && roomEmojiPanel) {
    roomEmojiBtn.addEventListener('click', function (e) {
      e.preventDefault();
      initEmojiPanel();
      var visible = roomEmojiPanel.style.display === 'flex' || roomEmojiPanel.style.display === 'grid' || roomEmojiPanel.style.display === 'block';
      roomEmojiPanel.style.display = visible ? 'none' : 'flex';
    });
    document.addEventListener('click', function (e) {
      if (roomEmojiPanel.style.display !== 'none' && !roomEmojiPanel.contains(e.target) && e.target !== roomEmojiBtn) {
        roomEmojiPanel.style.display = 'none';
      }
    });
  }

  function startInlineEdit(messageId, currentText) {
    const rowEl = roomMessages.querySelector('[data-message-id="' + messageId + '"]');
    if (!rowEl) return;
    const bubble = rowEl.querySelector('.roomBubble.me');
    if (!bubble) return;

    const textarea = document.createElement('textarea');
    textarea.className = 'roomBubble-edit';
    textarea.value = currentText;
    textarea.rows = 1;
    bubble.textContent = '';
    bubble.appendChild(textarea);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    function finishEdit(sendUpdate) {
      const newText = (textarea.value || '').trim();
      textarea.remove();
      bubble.textContent = sendUpdate && newText ? newText : currentText;
      if (sendUpdate && newText && newText !== currentText && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'edit', message_id: messageId, text: newText }));
      }
    }

    textarea.addEventListener('blur', function () {
      finishEdit(true);
    });
    textarea.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        finishEdit(true);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        finishEdit(false);
      }
    });
    textarea.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
  }

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
      startInlineEdit(messageId, currentText);
      return;
    }
    if (action === 'delete') {
      ws.send(JSON.stringify({ type: 'delete', message_id: messageId }));
      return;
    }
  }, true);
})();
