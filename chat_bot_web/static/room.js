(function () {
  const authSection = document.getElementById('authSection');
  const loadingSection = document.getElementById('loadingSection');
  const roomSection = document.getElementById('roomSection');
  const nickError = document.getElementById('authError');
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
  const serenaInviteBtn = document.getElementById('serenaInviteBtn');
  const chessGameBtn = document.getElementById('chessGameBtn');
  const chessPanel = document.getElementById('chessPanel');
  const visionaiFrame = document.getElementById('visionaiFrame');
  const chessBoard = document.getElementById('chessBoard');
  const chessStatus = document.getElementById('chessStatus');
  const chessStartBtn = document.getElementById('chessStartBtn');
  const chessUndoBtn = document.getElementById('chessUndoBtn');
  const chessPauseBtn = document.getElementById('chessPauseBtn');
  const chessResumeBtn = document.getElementById('chessResumeBtn');
  const chessResignBtn = document.getElementById('chessResignBtn');
  const chessPanelClose = document.getElementById('chessPanelClose');
  const chatPanelHideBtn = document.getElementById('chatPanelHideBtn');
  const chatPanelShowBtn = document.getElementById('chatPanelShowBtn');
  const chatFullscreenBtn = document.getElementById('chatFullscreenBtn');
  const roomLayout = document.getElementById('roomLayout');
  const withdrawBtn = document.getElementById('withdrawBtn');

  var EMOJI_LIST = ['😀','😃','😄','😁','😅','😂','🤣','😊','😇','🙂','😉','😌','😍','🥰','😘','😗','😙','😚','😋','😛','😜','🤪','😝','🤑','🤗','🤭','🤫','🤔','🤐','🤨','😐','😑','😶','😏','😒','🙄','😬','🤥','😌','😔','😪','🤤','😴','😷','🤒','🤕','🤢','🤮','👍','👎','👌','✌️','🤞','🤟','🤘','🤙','👋','🤚','🖐️','✋','🖖','👏','🙌','👐','🤲','🙏','❤️','🧡','💛','💚','💙','💜','🖤','💔','❣️','💕','💞','💓','💗','💖','💘','💝','💟','✨','⭐','🌟','💫','🔥','💯'];

  let ws = null;
  let wsDm = null;
  let myNickname = '';
  let myUser = null;
  let wsToken = null;
  let contextMenuEl = null;
  let serenaPresent = false;
  let currentDmRoomId = null;
  // DM 입력창 draft를 방(room_id)별로 분리 관리
  let dmDraftByRoomId = {};

  var chatTabs = [{ id: 'multi', label: '라운지', type: 'multi', closable: false }];
  var activeTabId = 'multi';

  var CHESS_INITIAL_SECONDS = 10 * 60;
  var chessState = { fen: null, turn: null, status: null, whitePlayer: null, blackPlayer: null, whiteRating: null, blackRating: null, mode: 'serena', lastMove: null, inCheck: false, whiteCaptured: [], blackCaptured: [], whiteTime: CHESS_INITIAL_SECONDS, blackTime: CHESS_INITIAL_SECONDS, turnStartedAt: null, paused: false };
  var chessSelected = null;
  var chessLegalTargets = null;
  var chessClockInterval = null;

  var gomokuState = { board: null, turn: 'black', status: null, blackPlayer: null, whitePlayer: null, mode: 'serena', lastMove: null };
  var GOMOKU_SIZE = 15;

  function createNickRow(nickname) {
    var row = document.createElement('div');
    row.className = 'roomMsgNickRow';
    if (nickname === 'Serena') {
      var avatar = document.createElement('img');
      avatar.src = '/serena.png';
      avatar.alt = 'Serena';
      avatar.className = 'roomMsgAvatar';
      avatar.addEventListener('click', function (e) { e.stopPropagation(); showSerenaPopup(); });
      avatar.addEventListener('error', function () { 
        console.error('Failed to load Serena avatar'); 
        this.style.background = '#6366f1';
      });
      row.appendChild(avatar);
    }
    var label = document.createElement('span');
    label.className = 'roomMsgNickAbove';
    label.textContent = nickname;
    row.appendChild(label);
    return row;
  }

  function escapeHtml(s) {
    var div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function linkify(text) {
    var urlRe = /(https?:\/\/[^\s<>"']+)/g;
    var parts = (text || '').split(urlRe);
    var out = '';
    for (var i = 0; i < parts.length; i++) {
      if (i % 2 === 1 && parts[i].match(/^https?:\/\//)) {
        out += '<a href="' + escapeHtml(parts[i]) + '" target="_blank" rel="noopener noreferrer" class="roomMsgLink">' + escapeHtml(parts[i]) + '</a>';
      } else {
        out += escapeHtml(parts[i]).replace(/\n/g, '<br>');
      }
    }
    return out || escapeHtml(text).replace(/\n/g, '<br>');
  }

  const dmPanel = document.getElementById('dmPanel');
  const multiPanel = document.getElementById('multiPanel');
  const dmChatHeader = document.getElementById('dmChatHeader');
  const dmCloseBtn = document.getElementById('dmCloseBtn');
  const dmMessages = document.getElementById('dmMessages');
  const dmInput = document.getElementById('dmInput');
  const dmSendBtn = document.getElementById('dmSendBtn');
  const dmEmojiBtn = document.getElementById('dmEmojiBtn');
  const dmEmojiPanel = document.getElementById('dmEmojiPanel');
  const dmImageBtn = document.getElementById('dmImageBtn');
  const dmImageInput = document.getElementById('dmImageInput');
  const notificationToggleBtn = document.getElementById('notificationToggleBtn');
  const notificationIconOn = document.getElementById('notificationIconOn');
  const notificationIconOff = document.getElementById('notificationIconOff');

  const imageLightbox = document.getElementById('imageLightbox');
  const imageLightboxImg = document.getElementById('imageLightboxImg');
  const imageLightboxContainer = imageLightbox ? imageLightbox.querySelector('.image-lightbox__container') : null;
  const imageLightboxBackdrop = imageLightbox ? imageLightbox.querySelector('.image-lightbox__backdrop') : null;
  const imageLightboxClose = imageLightbox ? imageLightbox.querySelector('.image-lightbox__close') : null;

  const chessPvpBtn = document.getElementById('chessPvpBtn');
  const chessStartPvpBtn = document.getElementById('chessStartPvpBtn');
  const chessPvpSelect = document.getElementById('chessPvpSelect');
  const chessPvpOpponent = document.getElementById('chessPvpOpponent');
  const chessPvpConfirm = document.getElementById('chessPvpConfirm');
  const chessCapturedLeft = document.getElementById('chessCapturedLeft');
  const chessCapturedRight = document.getElementById('chessCapturedRight');

  var participantListArr = [];
  var myChessMmr = null;

  const gomokuGameBtn = document.getElementById('gomokuGameBtn');
  const gomokuPvpBtn = document.getElementById('gomokuPvpBtn');
  const gomokuPanel = document.getElementById('gomokuPanel');
  const gomokuBoard = document.getElementById('gomokuBoard');
  const gomokuStatus = document.getElementById('gomokuStatus');
  const gomokuStartBtn = document.getElementById('gomokuStartBtn');
  const gomokuStartPvpBtn = document.getElementById('gomokuStartPvpBtn');
  const gomokuResignBtn = document.getElementById('gomokuResignBtn');
  const gomokuPanelClose = document.getElementById('gomokuPanelClose');
  const gomokuPvpSelect = document.getElementById('gomokuPvpSelect');
  const gomokuPvpOpponent = document.getElementById('gomokuPvpOpponent');
  const gomokuPvpConfirm = document.getElementById('gomokuPvpConfirm');

  function updateHeaderUserDisplay() {
    var un = document.getElementById('userName');
    if (!un) return;
    var name = myNickname || '';
    if (myChessMmr != null && myChessMmr !== '') {
      un.textContent = name + ' (' + Number(myChessMmr) + ')';
    } else {
      un.textContent = name;
    }
  }

  function updateSerenaBtn() {
    if (!serenaInviteBtn) return;
    serenaInviteBtn.textContent = serenaPresent ? 'Serena 강퇴' : 'Serena 초대';
    if (chessGameBtn) chessGameBtn.style.display = serenaPresent ? 'inline-block' : 'none';
    if (chessPvpBtn) chessPvpBtn.style.display = 'inline-block';
    if (gomokuGameBtn) gomokuGameBtn.style.display = serenaPresent ? 'inline-block' : 'none';
    if (gomokuPvpBtn) gomokuPvpBtn.style.display = 'inline-block';
  }

  /* 흑 기물과 동일한 글리프 사용, 백은 CSS로 흰색 적용 */
  var CHESS_PIECES = { K: '\u265A', Q: '\u265B', R: '\u265C', B: '\u265D', N: '\u265E', P: '\u265F', k: '\u265A', q: '\u265B', r: '\u265C', b: '\u265D', n: '\u265E', p: '\u265F' };

  function fenToBoard(fen) {
    if (!fen) return null;
    var parts = fen.split(' ');
    var rows = parts[0].split('/');
    var board = [];
    for (var r = 0; r < 8; r++) {
      var row = [];
      for (var i = 0; i < rows[r].length; i++) {
        var c = rows[r][i];
        if (/[1-8]/.test(c)) {
          for (var j = 0; j < parseInt(c, 10); j++) row.push(null);
        } else {
          row.push(c);
        }
      }
      board.push(row);
    }
    return board;
  }

  function countPieces(arr) {
    var o = {};
    arr.forEach(function (p) {
      o[p] = (o[p] || 0) + 1;
    });
    return o;
  }

  function renderCapturedPieces() {
    if (!chessCapturedLeft || !chessCapturedRight) return;
    var wc = chessState.whiteCaptured || [];
    var bc = chessState.blackCaptured || [];
    chessCapturedLeft.innerHTML = '';
    chessCapturedRight.innerHTML = '';
    var wCounts = countPieces(wc);
    ['q', 'r', 'n', 'b', 'p'].forEach(function (p) {
      if (!wCounts[p]) return;
      var wrap = document.createElement('span');
      wrap.className = 'chessCapturedPieceWrap';
      var icon = document.createElement('span');
      icon.className = 'chessCapturedPiece';
      icon.textContent = CHESS_PIECES[p] || p;
      wrap.appendChild(icon);
      var count = document.createElement('span');
      count.className = 'chessCapturedCount';
      count.textContent = 'x' + wCounts[p];
      wrap.appendChild(count);
      chessCapturedLeft.appendChild(wrap);
    });
    var bCounts = countPieces(bc);
    ['Q', 'R', 'N', 'B', 'P'].forEach(function (p) {
      if (!bCounts[p]) return;
      var wrap = document.createElement('span');
      wrap.className = 'chessCapturedPieceWrap';
      var icon = document.createElement('span');
      icon.className = 'chessCapturedPiece';
      icon.textContent = CHESS_PIECES[p] || p;
      wrap.appendChild(icon);
      var count = document.createElement('span');
      count.className = 'chessCapturedCount';
      count.textContent = 'x' + bCounts[p];
      wrap.appendChild(count);
      chessCapturedRight.appendChild(wrap);
    });
  }

  function renderChessBoard() {
    if (!chessBoard) return;
    var board = fenToBoard(chessState.fen);
    var lastMove = chessState.lastMove || '';
    var active = chessState.status === 'active';
    var isBlackView = chessState.blackPlayer === myNickname;
    chessBoard.innerHTML = '';
    chessBoard.classList.add('chessBoard--withLabels');
    for (var row = 0; row < 9; row++) {
      for (var col = 0; col < 9; col++) {
        var el = document.createElement('div');
        if (row === 8 && col === 0) {
          el.className = 'chessBoardLabel chessBoardLabel--empty';
          chessBoard.appendChild(el);
          continue;
        }
        if (row === 8) {
          el.className = 'chessBoardLabel chessBoardLabel--file';
          el.textContent = isBlackView ? String.fromCharCode(104 - (col - 1)) : String.fromCharCode(96 + col);
          chessBoard.appendChild(el);
          continue;
        }
        if (col === 0) {
          el.className = 'chessBoardLabel chessBoardLabel--rank';
          el.textContent = isBlackView ? String(row + 1) : String(8 - row);
          chessBoard.appendChild(el);
          continue;
        }
        var dr = row, dc = col - 1;
        var r = isBlackView ? 7 - dr : dr;
        var c = isBlackView ? 7 - dc : dc;
        var file = String.fromCharCode(97 + c);
        var rank = 8 - r;
        var sqId = file + rank;
        el.dataset.sq = sqId;
        el.className = 'chessSquare ' + ((r + c) % 2 === 0 ? 'light' : 'dark');
        if (lastMove && (sqId === lastMove.slice(0, 2) || sqId === lastMove.slice(2, 4))) el.classList.add('last-move');
        if (chessSelected === sqId) el.classList.add('selected');
        var piece = board && board[r] && board[r][c];
        var isMoveTarget = false, isCaptureTarget = false;
        if (chessSelected && chessLegalTargets && chessLegalTargets.from === chessSelected && chessLegalTargets.toSquares && chessLegalTargets.toSquares.indexOf(sqId) !== -1) {
          isMoveTarget = true;
          if (piece) {
            var isWhiteTurn = chessState.turn === 'white';
            if ((isWhiteTurn && /[kqrbnp]/.test(piece)) || (!isWhiteTurn && /[KQRBNP]/.test(piece))) isCaptureTarget = true;
          }
        }
        if (isMoveTarget) el.classList.add(isCaptureTarget ? 'chessSquare--capture' : 'chessSquare--move');
        if (piece) {
          var span = document.createElement('span');
          span.className = /[KQRBNP]/.test(piece) ? 'chessPiece chessPieceWhite' : 'chessPiece chessPieceBlack';
          span.textContent = CHESS_PIECES[piece] || piece;
          el.appendChild(span);
        }
        var canMove = active && !chessState.paused && ((chessState.turn === 'white' && chessState.whitePlayer === myNickname) || (chessState.turn === 'black' && chessState.blackPlayer === myNickname));
        if (canMove) {
          var isWhiteTurn = chessState.turn === 'white';
          var canSelect = (isWhiteTurn && piece && /[KQRBNP]/.test(piece)) || (!isWhiteTurn && piece && /[kqrbnp]/.test(piece));
          el.style.cursor = canSelect || chessSelected ? 'pointer' : 'default';
          el.addEventListener('click', function () { onChessSquareClick(this.dataset.sq); });
        }
        chessBoard.appendChild(el);
      }
    }
  }

  function onChessSquareClick(sq) {
    if (chessState.status !== 'active' || chessState.paused) return;
    var myTurnWhite = chessState.turn === 'white' && chessState.whitePlayer === myNickname;
    var myTurnBlack = chessState.turn === 'black' && chessState.blackPlayer === myNickname;
    if (!myTurnWhite && !myTurnBlack) return;
    if (chessSelected) {
      var uci = chessSelected + sq;
      if (uci.length === 4 && chessSelected !== sq) {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_move', uci: uci }));
        chessSelected = null;
        chessLegalTargets = null;
      } else if (sq === chessSelected) {
        chessSelected = null;
        chessLegalTargets = null;
      } else {
        var board = fenToBoard(chessState.fen);
        var r = 8 - parseInt(sq[1], 10);
        var c = sq.charCodeAt(0) - 97;
        var piece = board && board[r] && board[r][c];
        var isWhite = chessState.turn === 'white';
        if (piece && ((isWhite && /[KQRBNP]/.test(piece)) || (!isWhite && /[kqrbnp]/.test(piece)))) {
          chessSelected = sq;
          chessLegalTargets = null;
          if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_legal_moves', from: sq }));
        } else {
          chessSelected = null;
          chessLegalTargets = null;
        }
      }
    } else {
      var board = fenToBoard(chessState.fen);
      var r = 8 - parseInt(sq[1], 10);
      var c = sq.charCodeAt(0) - 97;
      var piece = board && board[r] && board[r][c];
      var isWhite = chessState.turn === 'white';
      if (piece && ((isWhite && /[KQRBNP]/.test(piece)) || (!isWhite && /[kqrbnp]/.test(piece)))) {
        chessSelected = sq;
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_legal_moves', from: sq }));
      }
    }
    renderChessBoard();
  }

  function formatNameWithRating(name, rating) {
    if (!name) return '';
    if (rating != null && rating !== '') return name + ' (' + Number(rating) + ')';
    return name;
  }

  function updateChessPanelTitle() {
    var titleEl = document.getElementById('chessPanelTitle');
    if (!titleEl) return;
    if (chessState.whitePlayer) {
      var wp = formatNameWithRating(chessState.whitePlayer, chessState.whiteRating);
      var bp = chessState.mode === 'serena' ? 'Serena' : formatNameWithRating(chessState.blackPlayer, chessState.blackRating);
      var title = (chessState.mode === 'serena') ? ('Serena vs ' + wp) : (wp + ' vs ' + bp);
      titleEl.textContent = '♔ ' + title;
    } else {
      titleEl.textContent = '♔ Serena vs (대기 중)';
    }
  }

  function updateChessButtons() {
    var hasGame = chessState.fen && chessState.status;
    var gameActive = chessState.status === 'active';
    var isChessPlayer = chessState.whitePlayer === myNickname || chessState.blackPlayer === myNickname;
    var showActions = !hasGame || isChessPlayer || !gameActive;
    var chessActionsTop = document.getElementById('chessActionsTop');
    if (chessActionsTop) chessActionsTop.style.display = showActions ? 'flex' : 'none';
    if (chessStartBtn) {
      chessStartBtn.style.display = serenaPresent ? 'inline-block' : 'none';
      var canStart = serenaPresent && (!chessState.fen || chessState.status !== 'active' || chessState.whitePlayer === myNickname);
      chessStartBtn.disabled = !canStart;
    }
    if (chessStartPvpBtn) {
      chessStartPvpBtn.style.display = (!chessState.fen || chessState.status !== 'active') ? 'inline-block' : 'none';
      chessStartPvpBtn.disabled = !!(chessState.fen && chessState.status === 'active' && chessState.whitePlayer !== myNickname);
    }
    var isPvpActive = chessState.fen && chessState.status === 'active' && chessState.mode === 'pvp';
    var isPlayer = isPvpActive && isChessPlayer;
    var canUndo = isPvpActive && chessState.lastMove && ((chessState.whitePlayer === myNickname && chessState.turn === 'white') || (chessState.blackPlayer === myNickname && chessState.turn === 'black'));
    if (chessUndoBtn) {
      chessUndoBtn.style.display = canUndo ? 'inline-block' : 'none';
      chessUndoBtn.disabled = !canUndo;
    }
    if (chessPauseBtn) chessPauseBtn.style.display = isPlayer && !chessState.paused ? 'inline-block' : 'none';
    if (chessResumeBtn) chessResumeBtn.style.display = isPlayer && chessState.paused ? 'inline-block' : 'none';
    if (chessResignBtn) chessResignBtn.style.display = isChessPlayer && gameActive ? 'inline-block' : 'none';
    if (chessPvpSelect) {
      chessPvpSelect.style.display = (chessPvpSelect.dataset.visible === '1') ? 'flex' : 'none';
    }
  }

  function formatChessTime(seconds) {
    var s = Math.max(0, Math.floor(Number(seconds) || 0));
    var m = Math.floor(s / 60);
    var sec = s % 60;
    return (m < 10 ? '0' : '') + m + ':' + (sec < 10 ? '0' : '') + sec;
  }

  function updateChessClocks() {
    var whiteEl = document.getElementById('chessClockWhite');
    var blackEl = document.getElementById('chessClockBlack');
    if (!whiteEl || !blackEl) return;
    var wt = chessState.whiteTime;
    var bt = chessState.blackTime;
    var started = chessState.turnStartedAt;
    var active = chessState.status === 'active';
    var paused = !!chessState.paused;
    if (active && !paused && started != null && typeof started === 'number') {
      var now = Date.now() / 1000;
      var elapsed = Math.max(0, now - started);
      if (chessState.turn === 'white') {
        wt = Math.max(0, (chessState.whiteTime || 0) - elapsed);
      } else {
        bt = Math.max(0, (chessState.blackTime || 0) - elapsed);
      }
    }
    var wtLabel = whiteEl.querySelector('.chessClockLabel');
    var wtTime = whiteEl.querySelector('.chessClockTime');
    var btLabel = blackEl.querySelector('.chessClockLabel');
    var btTime = blackEl.querySelector('.chessClockTime');
    if (wtLabel) wtLabel.textContent = chessState.whitePlayer ? formatNameWithRating(chessState.whitePlayer, chessState.whiteRating) : '흰색';
    if (btLabel) btLabel.textContent = chessState.blackPlayer ? formatNameWithRating(chessState.blackPlayer, chessState.blackRating) : '검은색';
    if (wtTime) wtTime.textContent = formatChessTime(wt);
    else whiteEl.textContent = formatChessTime(wt);
    if (btTime) btTime.textContent = formatChessTime(bt);
    else blackEl.textContent = formatChessTime(bt);
    var notPaused = !chessState.paused;
    whiteEl.classList.toggle('chessClock--active', active && notPaused && chessState.turn === 'white');
    blackEl.classList.toggle('chessClock--active', active && notPaused && chessState.turn === 'black');
  }

  function updateChessStatus() {
    if (!chessStatus) return;
    var s = chessState.status;
    var mode = chessState.mode || 'serena';
    if (!chessState.fen) { chessStatus.textContent = mode === 'serena' ? 'Serena를 초대한 뒤 새 게임을 시작하세요.' : '1:1 체스 도전을 하거나 대기하세요.'; return; }
    if (s === 'active') {
      if (chessState.paused) { chessStatus.textContent = '일시정지 중'; return; }
      var wp = formatNameWithRating(chessState.whitePlayer, chessState.whiteRating) || '흰색';
      var bp = formatNameWithRating(chessState.blackPlayer, chessState.blackRating) || '검은색';
      var base = chessState.turn === 'white' ? wp + ' (흰색) 차례' : bp + ' (검은색) 차례';
      chessStatus.textContent = chessState.inCheck ? base + ' – ⚠ 체크!' : base;
    } else if (s === 'checkmate_white') chessStatus.textContent = (formatNameWithRating(chessState.blackPlayer, chessState.blackRating) || '검은색') + ' 승리! (체크메이트)';
    else if (s === 'checkmate_black') chessStatus.textContent = (formatNameWithRating(chessState.whitePlayer, chessState.whiteRating) || '흰색') + ' 승리! (체크메이트)';
    else if (s === 'time_loss_white') chessStatus.textContent = (formatNameWithRating(chessState.blackPlayer, chessState.blackRating) || '검은색') + ' 승리! (흰색 시간 초과)';
    else if (s === 'time_loss_black') chessStatus.textContent = (formatNameWithRating(chessState.whitePlayer, chessState.whiteRating) || '흰색') + ' 승리! (검은색 시간 초과)';
    else if (s === 'resign_white') chessStatus.textContent = (formatNameWithRating(chessState.blackPlayer, chessState.blackRating) || '검은색') + ' 승리! (흰색 기권)';
    else if (s === 'resign_black') chessStatus.textContent = (formatNameWithRating(chessState.whitePlayer, chessState.whiteRating) || '흰색') + ' 승리! (검은색 기권)';
    else if (s === 'draw') chessStatus.textContent = '무승부';
    else chessStatus.textContent = '-';
  }

  function renderGomokuBoard() {
    if (!gomokuBoard) return;
    gomokuBoard.innerHTML = '';
    var board = gomokuState.board;
    if (!board || !board.length) {
      board = [];
      for (var i = 0; i < GOMOKU_SIZE; i++) {
        var row = [];
        for (var j = 0; j < GOMOKU_SIZE; j++) row.push(0);
        board.push(row);
      }
    }
    var active = gomokuState.status === 'active';
    var myTurnBlack = active && gomokuState.turn === 'black' && gomokuState.blackPlayer === myNickname;
    var myTurnWhite = active && gomokuState.turn === 'white' && gomokuState.whitePlayer === myNickname;
    var lastMove = gomokuState.lastMove;
    for (var r = 0; r < GOMOKU_SIZE; r++) {
      for (var c = 0; c < GOMOKU_SIZE; c++) {
        var cell = document.createElement('div');
        cell.className = 'gomokuCell';
        cell.dataset.row = r;
        cell.dataset.col = c;
        var val = board && board[r] && board[r][c];
        if (val === 1) { cell.classList.add('black'); cell.textContent = ''; }
        else if (val === 2) { cell.classList.add('white'); cell.textContent = ''; }
        else {
          cell.textContent = '';
          var canPlace = active && (myTurnBlack || myTurnWhite);
          if (canPlace) cell.classList.add('playable');
          cell.style.cursor = canPlace ? 'pointer' : 'default';
        }
        if (lastMove && lastMove[0] === r && lastMove[1] === c) cell.classList.add('last-move');
        cell.addEventListener('click', function (ev) {
          var row = parseInt(this.dataset.row, 10);
          var col = parseInt(this.dataset.col, 10);
          onGomokuCellClick(row, col);
        });
        gomokuBoard.appendChild(cell);
      }
    }
  }

  function onGomokuCellClick(row, col) {
    if (gomokuState.status !== 'active') return;
    var myTurnBlack = gomokuState.turn === 'black' && gomokuState.blackPlayer === myNickname;
    var myTurnWhite = gomokuState.turn === 'white' && gomokuState.whitePlayer === myNickname;
    if (!myTurnBlack && !myTurnWhite) return;
    var board = gomokuState.board;
    if (!board || !board[row] || board[row][col] !== 0) return;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'gomoku_move', row: row, col: col }));
  }

  function updateGomokuStatus() {
    if (!gomokuStatus) return;
    var s = gomokuState.status;
    if (!gomokuState.board) { gomokuStatus.textContent = 'Serena를 초대한 뒤 새 게임을 시작하세요.'; return; }
    if (s === 'active') {
      var bp = gomokuState.blackPlayer || '흑';
      var wp = gomokuState.whitePlayer || '백';
      gomokuStatus.textContent = gomokuState.turn === 'black' ? bp + ' (흑) 차례' : wp + ' (백) 차례';
    } else if (s === 'win_black') gomokuStatus.textContent = (gomokuState.blackPlayer || '흑') + ' 승리! (오목)';
    else if (s === 'win_white') gomokuStatus.textContent = (gomokuState.whitePlayer || '백') + ' 승리! (오목)';
    else if (s === 'resign_black') gomokuStatus.textContent = (gomokuState.whitePlayer || '백') + ' 승리! (흑 기권)';
    else if (s === 'resign_white') gomokuStatus.textContent = (gomokuState.blackPlayer || '흑') + ' 승리! (백 기권)';
    else if (s === 'draw') gomokuStatus.textContent = '무승부';
    else gomokuStatus.textContent = '-';
  }

  function updateGomokuPanelTitle() {
    var el = document.getElementById('gomokuPanelTitle');
    if (!el) return;
    if (gomokuState.blackPlayer) {
      var title = (gomokuState.mode === 'serena') ? (gomokuState.blackPlayer + ' vs Serena') : (gomokuState.blackPlayer + ' vs ' + gomokuState.whitePlayer);
      el.textContent = '● ' + title;
    } else el.textContent = '● 오목 (대기 중)';
  }

  function updateGomokuButtons() {
    if (gomokuStartBtn) {
      gomokuStartBtn.style.display = serenaPresent ? 'inline-block' : 'none';
      var canStart = serenaPresent && (!gomokuState.board || gomokuState.status !== 'active' || gomokuState.blackPlayer === myNickname);
      gomokuStartBtn.disabled = !canStart;
    }
    if (gomokuStartPvpBtn) {
      gomokuStartPvpBtn.style.display = (!gomokuState.board || gomokuState.status !== 'active') ? 'inline-block' : 'none';
      gomokuStartPvpBtn.disabled = !!(gomokuState.board && gomokuState.status === 'active' && gomokuState.blackPlayer !== myNickname);
    }
    if (gomokuPvpSelect) gomokuPvpSelect.style.display = (gomokuPvpSelect.dataset.visible === '1') ? 'flex' : 'none';
  }

  function showChessAlert(title, message) {
    var modal = document.getElementById('chessAlertModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'chessAlertModal';
      modal.className = 'chessAlertModal';
      modal.innerHTML = '<div class="chessAlertContent"><div class="chessAlertTitle"></div><div class="chessAlertMessage"></div><button type="button" class="chessAlertBtn">확인</button></div>';
      document.body.appendChild(modal);
      modal.addEventListener('click', function (e) {
        if (e.target === modal || e.target.classList.contains('chessAlertBtn')) modal.style.display = 'none';
      });
    }
    modal.querySelector('.chessAlertTitle').textContent = title || '';
    modal.querySelector('.chessAlertMessage').textContent = message || '';
    modal.style.display = 'flex';
  }

  function formatMmrDelta(delta) {
    if (delta == null || delta === '') return '0';
    var d = Number(delta);
    if (d > 0) return '+' + d;
    return String(d);
  }

  function showChessResultModal(resultTitle, resultMessage, whiteName, blackName, whiteDelta, blackDelta) {
    var modal = document.getElementById('chessResultModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'chessResultModal';
      modal.className = 'chessAlertModal chessResultModal';
      modal.innerHTML = '<div class="chessAlertContent"><div class="chessAlertTitle chessResultTitle"></div><div class="chessAlertMessage chessResultMessage"></div><div class="chessResultMmrs"></div><button type="button" class="chessAlertBtn chessResultCloseBtn">확인</button></div>';
      document.body.appendChild(modal);
      modal.querySelector('.chessResultCloseBtn').addEventListener('click', function () { modal.style.display = 'none'; });
      modal.addEventListener('click', function (e) {
        if (e.target === modal) modal.style.display = 'none';
      });
    }
    modal.querySelector('.chessResultTitle').textContent = resultTitle || '';
    modal.querySelector('.chessResultMessage').textContent = resultMessage || '';
    var mmrEl = modal.querySelector('.chessResultMmrs');
    mmrEl.innerHTML = '';
    if (whiteDelta != null || blackDelta != null) {
      var wStr = formatMmrDelta(whiteDelta);
      var bStr = formatMmrDelta(blackDelta);
      var wColor = whiteDelta > 0 ? 'chessMmrPlus' : (whiteDelta < 0 ? 'chessMmrMinus' : 'chessMmrZero');
      var bColor = blackDelta > 0 ? 'chessMmrPlus' : (blackDelta < 0 ? 'chessMmrMinus' : 'chessMmrZero');
      var line1 = document.createElement('div');
      line1.className = 'chessResultMmrLine';
      line1.innerHTML = (whiteName || '흰색') + ' <span class="' + wColor + '">' + escapeHtml(wStr) + '</span>';
      var line2 = document.createElement('div');
      line2.className = 'chessResultMmrLine';
      line2.innerHTML = (blackName || '검은색') + ' <span class="' + bColor + '">' + escapeHtml(bStr) + '</span>';
      mmrEl.appendChild(line1);
      mmrEl.appendChild(line2);
    }
    modal.style.display = 'flex';
  }

  function showChessUndoRequestModal(requestedBy) {
    var modal = document.getElementById('chessUndoRequestModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'chessUndoRequestModal';
      modal.className = 'chessAlertModal chessUndoRequestModal';
      modal.innerHTML = '<div class="chessAlertContent"><div class="chessAlertTitle">무르기 요청</div><div class="chessAlertMessage"></div><div class="chessUndoRequestActions"><button type="button" class="chessAlertBtn chessUndoAcceptBtn">수락</button><button type="button" class="chessAlertBtn chessUndoRejectBtn">거절</button></div></div>';
      document.body.appendChild(modal);
      modal.querySelector('.chessUndoAcceptBtn').addEventListener('click', function () {
        modal.style.display = 'none';
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_undo_accept' }));
      });
      modal.querySelector('.chessUndoRejectBtn').addEventListener('click', function () {
        modal.style.display = 'none';
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_undo_reject' }));
      });
      modal.addEventListener('click', function (e) {
        if (e.target === modal) modal.style.display = 'none';
      });
    }
    modal.querySelector('.chessAlertMessage').textContent = (requestedBy || '상대') + '님이 무르기를 요청했습니다. 수락하시겠습니까?';
    modal.style.display = 'flex';
  }

  function showSerenaPopup() {
    var modal = document.getElementById('serenaModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'serenaModal';
      modal.className = 'serenaModal';
      modal.innerHTML = '<div class="serenaModalContent"><img src="/serena.png" alt="Serena" /><button class="serenaModalClose">✕</button></div>';
      document.body.appendChild(modal);
      modal.addEventListener('click', function (e) {
        if (e.target === modal || e.target.className === 'serenaModalClose') {
          modal.style.display = 'none';
        }
      });
    }
    modal.style.display = 'flex';
  }

  function showNickError(msg) {
    if (nickError) {
      nickError.textContent = msg || '';
      nickError.style.display = msg ? 'block' : 'none';
    }
  }

  function scrollRoomBottom() {
    roomMessages.scrollTop = roomMessages.scrollHeight;
    requestAnimationFrame(function () {
      roomMessages.scrollTop = roomMessages.scrollHeight;
    });
  }

  function renderParticipants(list) {
    var arr = (list || []).filter(function (p) {
      var userId = p && p.user_id;
      var name = typeof p === 'string' ? p : (p && p.name);
      return !(myUser && userId === myUser.id) && name !== myNickname;
    });
    participantCount.textContent = arr.length;
    participantList.innerHTML = arr.map(function (p) {
      var name = typeof p === 'string' ? p : (p.name || p);
      var userId = p && p.user_id;
      var isMe = myUser && userId === myUser.id;
      var avatar = (p && p.avatar_url) || (name === 'Serena' ? '/serena.png' : '');
      if (name === 'Serena') {
        return '<span class="participantItem participantSerena"><img class="participantAvatar" src="/serena.png" alt="">Serena</span>';
      }
      var displayName = name;
      var rating = p && p.mmr_rating != null && p.mmr_rating !== '';
      if (rating) displayName = name + ' (' + Number(p.mmr_rating) + ')';
      var html = '<span class="participantItem' + (userId && !isMe ? ' participantWithDm" data-user-id="' + userId + '"' : '"') + '>';
      if (avatar) html += '<img class="participantAvatar" src="' + escapeHtml(avatar) + '" alt="">';
      if (userId && !isMe) {
        html += '<span class="participantName">' + escapeHtml(displayName) + '</span>';
        html += '<div class="participantDmPopover" style="display:none">';
        html += '<button type="button" class="dmFromParticipantBtn participantDmBtn" data-user-id="' + userId + '" title="1:1 대화">1:1 대화</button>';
        html += '</div>';
      } else {
        html += escapeHtml(name);
      }
      html += '</span>';
      return html;
    }).join(' ');
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
    contextMenuEl.dataset.isDm = '';
  }

  function showContextMenu(x, y, messageId, currentText, isDm) {
    const menu = ensureContextMenu();
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';
    menu.style.display = 'block';
    menu.dataset.messageId = messageId || '';
    menu.dataset.currentText = currentText || '';
    menu.dataset.isDm = isDm ? '1' : '';
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
    var badge = rowEl.querySelector('.unread-badge');
    var timeWrap = rowEl.querySelector('.roomMsgTimeWrap');
    if (unreadCount > 0) {
      if (badge) {
        badge.textContent = unreadCount;
      } else if (timeWrap) {
        var newBadge = document.createElement('span');
        newBadge.className = 'unread-badge';
        newBadge.textContent = unreadCount;
        var tsEl = timeWrap.querySelector('.roomMsgTimestamp');
        timeWrap.insertBefore(newBadge, tsEl || null);
      } else {
        var newBadge = document.createElement('span');
        newBadge.className = 'unread-badge';
        newBadge.textContent = unreadCount;
        rowEl.appendChild(newBadge);
      }
    } else {
      if (badge) badge.remove();
    }
  }

  // 알림 활성화 상태 (localStorage에 저장, 기본값: true)
  function getNotificationEnabled() {
    var enabled = localStorage.getItem('notificationEnabled');
    return enabled === null ? true : enabled === 'true';
  }

  function setNotificationEnabled(enabled) {
    localStorage.setItem('notificationEnabled', enabled ? 'true' : 'false');
    updateNotificationIcon();
  }

  function updateNotificationIcon() {
    if (!notificationToggleBtn || !notificationIconOn || !notificationIconOff) return;
    var enabled = getNotificationEnabled();
    if (enabled) {
      notificationIconOn.style.display = 'block';
      notificationIconOff.style.display = 'none';
      notificationToggleBtn.title = '알림 끄기';
    } else {
      notificationIconOn.style.display = 'none';
      notificationIconOff.style.display = 'block';
      notificationToggleBtn.title = '알림 켜기';
    }
  }

  function requestNotificationPermissionIfNeeded() {
    if (!('Notification' in window)) return Promise.resolve();
    if (Notification.permission === 'granted') return Promise.resolve();
    if (Notification.permission === 'denied') return Promise.resolve();
    // 알림이 필요할 때만 자동으로 권한 요청 (사용자 제스처 불필요)
    return Notification.requestPermission().then(function(permission) {
      console.log('Notification permission:', permission);
    }).catch(function(err) {
      console.error('Notification permission error:', err);
    });
  }

  function showChatNotification(title, body, tag) {
    if (!getNotificationEnabled()) return;
    if (!document.hidden) return;
    if (!('Notification' in window)) return;
    
    // 권한이 없으면 자동 요청
    if (Notification.permission === 'default') {
      requestNotificationPermissionIfNeeded().then(function() {
        if (Notification.permission === 'granted') {
          tryShowNotification(title, body, tag);
        }
      });
    } else if (Notification.permission === 'granted') {
      tryShowNotification(title, body, tag);
    }
  }

  function tryShowNotification(title, body, tag) {
    try {
      var opts = {
        body: body || '새 메시지',
        tag: tag || 'chat',
        requireInteraction: false,
        silent: false
      };
      var n = new Notification(title, opts);
      n.onclick = function () {
        window.focus();
        n.close();
      };
      setTimeout(function () { n.close(); }, 8000);
    } catch (e) {}
  }

  var imageLightboxState = { scale: 1, tx: 0, ty: 0, dragging: false, startX: 0, startY: 0, startTx: 0, startTy: 0 };

  function applyImageLightboxTransform() {
    if (!imageLightboxImg) return;
    var s = imageLightboxState;
    imageLightboxImg.style.transform = 'translate(' + s.tx + 'px,' + s.ty + 'px) scale(' + s.scale + ')';
  }

  function openImageLightbox(src) {
    if (!imageLightbox || !imageLightboxImg || !src) return;
    imageLightboxState = { scale: 1, tx: 0, ty: 0, dragging: false, startX: 0, startY: 0, startTx: 0, startTy: 0 };
    imageLightboxImg.src = src;
    applyImageLightboxTransform();
    imageLightbox.style.display = 'flex';
    imageLightbox.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  }

  function closeImageLightbox() {
    if (!imageLightbox) return;
    imageLightbox.style.display = 'none';
    imageLightbox.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    imageLightboxImg.src = '';
  }

  function setupImageLightbox() {
    if (!imageLightbox || !imageLightboxImg) return;
    var container = imageLightboxContainer;
    var s = imageLightboxState;

    function onWheel(e) {
      e.preventDefault();
      var delta = e.deltaY > 0 ? -0.15 : 0.15;
      s.scale = Math.max(0.5, Math.min(5, s.scale + delta));
      applyImageLightboxTransform();
    }

    function onMouseDown(e) {
      if (e.button !== 0) return;
      s.dragging = true;
      s.startX = e.clientX;
      s.startY = e.clientY;
      s.startTx = s.tx;
      s.startTy = s.ty;
    }

    function onMouseMove(e) {
      if (!s.dragging) return;
      s.tx = s.startTx + (e.clientX - s.startX);
      s.ty = s.startTy + (e.clientY - s.startY);
      applyImageLightboxTransform();
    }

    function onMouseUp() {
      s.dragging = false;
    }

    if (container) {
      container.addEventListener('wheel', onWheel, { passive: false });
    }
    imageLightboxImg.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);

    if (imageLightboxBackdrop) imageLightboxBackdrop.addEventListener('click', closeImageLightbox);
    if (imageLightboxClose) imageLightboxClose.addEventListener('click', closeImageLightbox);
    document.addEventListener('keydown', function onKey(e) {
      if (e.key === 'Escape' && imageLightbox && imageLightbox.style.display === 'flex') {
        closeImageLightbox();
      }
    });
  }

  function handleChatImageClick(e) {
    if (e.button !== 0) return;
    if (!e.target || typeof e.target.closest !== 'function') return;
    var wrap = e.target.closest('.roomMsgImageLargeWrap');
    if (!wrap) return;
    var img = wrap.querySelector('.roomMsgImageLarge') || wrap.querySelector('img');
    if (!img || !img.src || String(img.src).trim() === '') return;
    var container = e.target.closest('#roomMessages') || e.target.closest('#dmMessages');
    if (!container) return;
    e.preventDefault();
    e.stopPropagation();
    openImageLightbox(img.src);
  }

  document.addEventListener('click', function (e) {
    handleChatImageClick(e);
  }, true);

  if (imageLightbox && imageLightboxImg) {
    setupImageLightbox();
  }

  function addDmTab(roomId, otherUser) {
    var tabId = 'dm-' + roomId;
    var existing = chatTabs.find(function (t) { return t.id === tabId; });
    if (existing) {
      switchToTab(tabId);
      return;
    }
    chatTabs.push({ id: tabId, label: (otherUser && otherUser.name) || '1:1 대화', type: 'dm', roomId: roomId, otherUser: otherUser, closable: false });
    renderChatTabs();
    switchToTab(tabId);
  }

  function switchToTab(tabId) {
    // 탭 전환 직전: 기존 DM 방 draft 저장
    var prevTabId = activeTabId;
    var prevDmRoomId = currentDmRoomId;
    if (prevTabId && prevTabId.indexOf('dm-') === 0 && prevDmRoomId) {
      saveDmDraft(prevDmRoomId);
    }
    activeTabId = tabId;
    var tab = chatTabs.find(function (t) { return t.id === tabId; });
    if (!tab) return;
    renderChatTabs();
    if (tab.type === 'multi') {
      if (multiPanel) multiPanel.style.display = 'flex';
      if (dmPanel) dmPanel.style.display = 'none';
      if (wsDm) { wsDm.close(); wsDm = null; }
      currentDmRoomId = null;
      if (roomInput) roomInput.focus();
    } else {
      if (multiPanel) multiPanel.style.display = 'none';
      if (dmPanel) dmPanel.style.display = 'flex';
      openDmRoom(tab.roomId, tab.otherUser);
      if (dmInput) dmInput.focus();
    }
  }

  function removeTab(tabId) {
    if (tabId === 'multi') return;
    var idx = chatTabs.findIndex(function (t) { return t.id === tabId; });
    if (idx < 0) return;
    chatTabs.splice(idx, 1);
    if (activeTabId === tabId) {
      if (chatTabs.length > 0) switchToTab(chatTabs[Math.max(0, idx - 1)].id);
      else switchToTab('multi');
    } else {
      renderChatTabs();
    }
  }

  function renderChatTabs() {
    var el = document.getElementById('chatTabsList');
    if (!el) return;
    el.innerHTML = '';
    chatTabs.forEach(function (t) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chatTab' + (t.id === activeTabId ? ' active' : '') + (t.closable ? '' : ' multiOnly');
      btn.dataset.tabId = t.id;
      btn.innerHTML = '<span>' + escapeHtml(t.label) + '</span>';
      if (t.closable) {
        var closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'chatTabClose';
        closeBtn.innerHTML = '&times;';
        closeBtn.addEventListener('click', function (e) { e.stopPropagation(); removeTab(t.id); });
        btn.appendChild(closeBtn);
      }
      btn.addEventListener('click', function (e) { if (!e.target.classList.contains('chatTabClose')) switchToTab(t.id); });
      el.appendChild(btn);
    });
  }

  function startDmWithUser(otherUserId) {
    var id = parseInt(otherUserId, 10);
    if (isNaN(id) || id < 1) return;
    fetch('/api/dm/rooms/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ other_user_id: id })
    })
      .then(function (r) {
        return r.json().then(function (data) {
          if (!r.ok) {
            var d = data && data.detail;
            var msg = '요청 실패';
            if (typeof d === 'string') msg = d;
            else if (Array.isArray(d)) msg = d.map(function (x) { return x.msg || x; }).join(', ');
            else if (d && d.msg) msg = d.msg;
            alert(msg);
            return null;
          }
          return data;
        });
      })
      .then(function (data) {
        if (!data || !data.room) return;
        addDmTab(data.room.id, data.room.other_user);
      })
      .catch(function () {
        alert('1:1 대화방 생성 중 오류가 발생했습니다.');
      });
  }

  function saveDmDraft(roomId) {
    if (!dmInput || !roomId) return;
    dmDraftByRoomId[String(roomId)] = dmInput.value || '';
  }

  function restoreDmDraft(roomId) {
    if (!dmInput || !roomId) return;
    dmInput.value = dmDraftByRoomId[String(roomId)] || '';
    // textarea 높이 동기화
    dmInput.style.height = 'auto';
    dmInput.style.height = Math.min(dmInput.scrollHeight || 0, 120) + 'px';
  }

  function openDmRoom(roomId, otherUser) {
    // 방 이동 전 draft 저장
    if (currentDmRoomId && currentDmRoomId !== roomId) {
      saveDmDraft(currentDmRoomId);
    }
    currentDmRoomId = roomId;
    if (dmChatHeader) dmChatHeader.textContent = (otherUser && otherUser.name) || '1:1 대화';
    if (dmMessages) dmMessages.innerHTML = '';
    if (wsDm) {
      wsDm.close();
      wsDm = null;
    }
    // 방별 draft 복원 (다른 방의 입력이 보이지 않도록)
    restoreDmDraft(roomId);
    connectDm(roomId);
  }

  function sendDmViewedRoom() {
    if (wsDm && wsDm.readyState === WebSocket.OPEN && currentDmRoomId) {
      wsDm.send(JSON.stringify({ type: 'viewed_room' }));
    }
  }

  function connectDm(roomId) {
    var url = (window.location.origin.replace(/^http/, 'ws') + '/ws/dm');
    wsDm = new WebSocket(url);
    var socket = wsDm;
    socket.onopen = function () {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'join', ws_token: wsToken, room_id: roomId }));
        // 현재 활성 DM 방에서만 viewed_room 전송
        if (socket === wsDm && currentDmRoomId === roomId) {
          sendDmViewedRoom();
        }
      }
    };
    socket.onmessage = function (ev) {
      try {
        // 방 전환/재연결 시, 이전 소켓에서 오는 메시지는 무시 (다른 1:1 창에 노출 방지)
        if (socket !== wsDm || currentDmRoomId !== roomId) return;
        var d = JSON.parse(ev.data);
        if (d.type === 'error') { showNickError(d.message); return; }
        if (d.type === 'chat') {
          var isMe = d.is_me === true || d.nickname === myNickname;
          appendDmMessage(isMe, d.nickname, d.text, d.image_url, d.timestamp, d.message_id, d.unread_count || 0);
          if (!isMe && !d.is_history && document.hidden) {
            var dmPreview = (d.text && d.text.trim()) ? d.text.trim().slice(0, 50) + (d.text.length > 50 ? '...' : '') : '(사진)';
            showChatNotification('1:1 ' + (d.nickname || ''), dmPreview, 'dm-' + (d.message_id || ''));
          }
        }
        if (d.type === 'read_update') {
          // 내가 보낸 메시지의 읽음 상태 업데이트
          var rowEl = dmMessages ? dmMessages.querySelector('[data-message-id="' + d.message_id + '"]') : null;
          setUnreadBadge(rowEl, d.unread_count || 0);
        }
        if (d.type === 'edit') {
          var editRow = dmMessages ? dmMessages.querySelector('[data-message-id="' + d.message_id + '"]') : null;
          if (editRow) {
            var newText = d.text || '';
            editRow.dataset.rawText = newText;
            var bubble = editRow.querySelector('.roomBubble');
            var emojiDiv = editRow.querySelector('.roomMsgEmoji');
            var imgWrap = editRow.querySelector('.roomMsgImageLargeWrap');
            var cap = editRow.querySelector('.roomMsgImageCaption');
            if (isOnlyEmoji(newText) && !imgWrap) {
              if (bubble) {
                var newEmoji = document.createElement('div');
                newEmoji.className = editRow.classList.contains('me') ? 'roomMsgEmoji roomMsgEmoji--me' : 'roomMsgEmoji roomMsgEmoji--other';
                newEmoji.textContent = newText;
                bubble.parentNode.replaceChild(newEmoji, bubble);
                if (editRow.classList.contains('me') && d.message_id) {
                  newEmoji.addEventListener('contextmenu', function (ev) {
                    ev.preventDefault();
                    showContextMenu(ev.clientX, ev.clientY, d.message_id, newEmoji.textContent || '', true);
                  });
                }
              } else if (emojiDiv) {
                emojiDiv.textContent = newText;
              }
            } else if (imgWrap && cap) {
              cap.textContent = newText;
            } else if (imgWrap && newText) {
              var newCap = document.createElement('div');
              newCap.className = 'roomMsgImageCaption';
              newCap.textContent = newText;
              imgWrap.appendChild(newCap);
            } else if (bubble) {
              var txtEl = bubble.querySelector('.roomMsgText');
              if (txtEl) txtEl.innerHTML = linkify(newText);
              else bubble.innerHTML = '<span class="roomMsgText">' + linkify(newText) + '</span>';
            } else if (emojiDiv) {
              var newBubble = document.createElement('div');
              newBubble.className = 'roomBubble ' + (editRow.classList.contains('me') ? 'me' : 'other');
              newBubble.innerHTML = '<span class="roomMsgText">' + linkify(newText) + '</span>';
              emojiDiv.parentNode.replaceChild(newBubble, emojiDiv);
              if (editRow.classList.contains('me') && d.message_id) {
                newBubble.addEventListener('contextmenu', function (ev) {
                  ev.preventDefault();
                  var t = newBubble.querySelector('.roomMsgText');
                  showContextMenu(ev.clientX, ev.clientY, d.message_id, (t ? t.textContent : newBubble.textContent) || '', true);
                });
              }
            }
          }
        }
        if (d.type === 'delete') {
          var delRow = dmMessages ? dmMessages.querySelector('[data-message-id="' + d.message_id + '"]') : null;
          if (delRow) delRow.remove();
        }
      } catch (e) {}
    };
    socket.onclose = function () {
      // 이전 소켓 close가 새 소켓을 끊어버리지 않도록 보호
      if (wsDm === socket) wsDm = null;
    };
  }

  function appendDmMessage(isMe, nickname, text, imageUrl, timestamp, messageId, unreadCount) {
    var row = document.createElement('div');
    row.className = 'roomMsgRow ' + (isMe ? 'me' : 'other');
    if (messageId) row.dataset.messageId = messageId;
    row.dataset.rawText = text || '';
    var timeStr = timestamp || formatSeoulTime();
    var ts = document.createElement('span');
    ts.className = 'roomMsgTimestamp';
    ts.textContent = timeStr;
    var isEmojiOnly = !imageUrl && isOnlyEmoji(text);
    var hasImage = !!imageUrl;

    function makeTimeWrap() {
      var wrap = document.createElement('div');
      wrap.className = 'roomMsgTimeWrap';
      if (unreadCount > 0) {
        var badge = document.createElement('span');
        badge.className = 'unread-badge';
        badge.textContent = unreadCount;
        wrap.appendChild(badge);
      }
      wrap.appendChild(ts);
      return wrap;
    }

    if (isMe) {
      if (hasImage) {
        var imgW = document.createElement('div');
        imgW.className = 'roomMsgImageLargeWrap';
        var img = document.createElement('img');
        img.src = imageUrl;
        img.className = 'roomMsgImageLarge';
        img.alt = '첨부 이미지';
        imgW.appendChild(img);
        if (text) {
          var c = document.createElement('div');
          c.className = 'roomMsgImageCaption';
          c.textContent = text;
          imgW.appendChild(c);
        }
        row.appendChild(imgW);
        row.appendChild(makeTimeWrap());
        if (messageId) {
          imgW.addEventListener('contextmenu', function (e) {
            e.preventDefault();
            showContextMenu(e.clientX, e.clientY, messageId, text || '', true);
          });
        }
      } else if (isEmojiOnly) {
        var emojiDiv = document.createElement('div');
        emojiDiv.className = 'roomMsgEmoji roomMsgEmoji--me';
        emojiDiv.textContent = text;
        row.appendChild(emojiDiv);
        row.appendChild(makeTimeWrap());
        if (messageId) {
          emojiDiv.addEventListener('contextmenu', function (e) {
            e.preventDefault();
            showContextMenu(e.clientX, e.clientY, messageId, emojiDiv.textContent || '', true);
          });
        }
      } else {
        var b = document.createElement('div');
        b.className = 'roomBubble me';
        b.innerHTML = '<span class="roomMsgText">' + linkify(text || '') + '</span>';
        row.appendChild(b);
        row.appendChild(makeTimeWrap());
        if (messageId) {
          b.addEventListener('contextmenu', function (e) {
            e.preventDefault();
            var txtEl = b.querySelector('.roomMsgText');
            var txt = txtEl ? txtEl.textContent : b.textContent;
            showContextMenu(e.clientX, e.clientY, messageId, (txt || '').trim(), true);
          });
        }
      }
    } else {
      var wrap = document.createElement('div');
      wrap.className = 'roomMsgOtherWrap';
      var nr = document.createElement('div');
      nr.className = 'roomMsgNickRow';
      nr.innerHTML = '<span class="roomMsgNickAbove">' + escapeHtml(nickname) + '</span>';
      wrap.appendChild(nr);
      if (hasImage) {
        var imgW2 = document.createElement('div');
        imgW2.className = 'roomMsgImageLargeWrap';
        var img2 = document.createElement('img');
        img2.src = imageUrl;
        img2.className = 'roomMsgImageLarge';
        img2.alt = '첨부 이미지';
        imgW2.appendChild(img2);
        if (text) {
          var c2 = document.createElement('div');
          c2.className = 'roomMsgImageCaption';
          c2.textContent = text;
          imgW2.appendChild(c2);
        }
        wrap.appendChild(imgW2);
      } else if (isEmojiOnly) {
        var emojiDiv2 = document.createElement('div');
        emojiDiv2.className = 'roomMsgEmoji roomMsgEmoji--other';
        emojiDiv2.textContent = text;
        wrap.appendChild(emojiDiv2);
      } else {
        var b2 = document.createElement('div');
        b2.className = 'roomBubble other';
        b2.innerHTML = '<span class="roomMsgText">' + linkify(text || '') + '</span>';
        wrap.appendChild(b2);
      }
      wrap.appendChild(ts);
      row.appendChild(wrap);
    }
    if (dmMessages) {
      dmMessages.appendChild(row);
      dmMessages.scrollTop = dmMessages.scrollHeight;
    }
  }

  function formatSeoulTime() {
    var now = new Date();
    var options = { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit', hour12: false };
    return now.toLocaleTimeString('ko-KR', options);
  }

  function appendRoomMessage(type, nickname, text, messageId, unreadCount, imageUrl, timestamp) {
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

    var timeStr = timestamp || formatSeoulTime();
    var timeSpan = document.createElement('span');
    timeSpan.className = 'roomMsgTimestamp';
    timeSpan.textContent = timeStr;

    var isEmojiOnly = !imageUrl && isOnlyEmoji(text);
    var hasImage = !!imageUrl;

    if (hasImage) {
      var imgWrap = document.createElement('div');
      imgWrap.className = 'roomMsgImageLargeWrap';
      var img = document.createElement('img');
      img.src = imageUrl;
      img.className = 'roomMsgImageLarge';
      img.alt = '첨부 이미지';
      imgWrap.appendChild(img);
      if (text) {
        var cap = document.createElement('div');
        cap.className = 'roomMsgImageCaption';
        cap.textContent = text;
        imgWrap.appendChild(cap);
      }
      if (type === 'me') {
        row.appendChild(imgWrap);
        row.appendChild(timeSpan);
        if (unreadCount > 0) {
          var badge = document.createElement('span');
          badge.className = 'unread-badge';
          badge.textContent = unreadCount;
          row.appendChild(badge);
        }
        imgWrap.addEventListener('contextmenu', function (e) {
          e.preventDefault();
          if (!messageId) return;
          showContextMenu(e.clientX, e.clientY, messageId, text || '');
        });
      } else {
        var wrap = document.createElement('div');
        wrap.className = 'roomMsgOtherWrap';
        wrap.appendChild(createNickRow(nickname));
        wrap.appendChild(imgWrap);
        wrap.appendChild(timeSpan);
        row.appendChild(wrap);
      }
    } else if (isEmojiOnly) {
      if (type === 'me') {
        var emojiDiv = document.createElement('div');
        emojiDiv.className = 'roomMsgEmoji roomMsgEmoji--me';
        emojiDiv.textContent = text;
        row.appendChild(emojiDiv);
        row.appendChild(timeSpan);
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
        wrap.appendChild(createNickRow(nickname));
        var emojiDiv = document.createElement('div');
        emojiDiv.className = 'roomMsgEmoji roomMsgEmoji--other';
        emojiDiv.textContent = text;
        wrap.appendChild(emojiDiv);
        wrap.appendChild(timeSpan);
        row.appendChild(wrap);
      }
    } else {
      var bubble = document.createElement('div');
      bubble.className = 'roomBubble ' + type;

      if (text) {
        var textSpan = document.createElement('span');
        textSpan.className = 'roomMsgText';
        textSpan.innerHTML = linkify(text);
        bubble.appendChild(textSpan);
      }

      if (type === 'me') {
        row.appendChild(bubble);
        row.appendChild(timeSpan);
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
        wrap.appendChild(createNickRow(nickname));
        wrap.appendChild(bubble);
        wrap.appendChild(timeSpan);
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
      var socket = this;
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'join', ws_token: wsToken }));
      }
    };

    ws.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'error') {
          if (authSection) authSection.style.display = 'block';
          if (loadingSection) loadingSection.style.display = 'none';
          roomSection.style.display = 'none';
          if (roomNickEditBtn) roomNickEditBtn.style.display = 'none';
          if (roomLeaveBtn) roomLeaveBtn.style.display = 'none';
          if (withdrawBtn) withdrawBtn.style.display = 'none';
          if (notificationToggleBtn) notificationToggleBtn.style.display = 'none';
          showNickError(data.message || '오류');
          ws.close();
          return;
        }
        if (data.type === 'new_dm') {
          if (data.room_id && data.other_user) {
            addDmTab(data.room_id, data.other_user);
          }
          if (document.hidden) {
            var fromName = (data.other_user && data.other_user.name) ? data.other_user.name : '1:1 대화';
            showChatNotification('1:1 ' + fromName, data.preview || '새 메시지', 'dm-new-' + (data.room_id || ''));
          }
          return;
        }
        if (data.type === 'participants') {
          var list = data.list || [];
          var me = list.find(function (p) {
            var uid = p && p.user_id;
            var n = typeof p === 'string' ? p : (p && p.name);
            return (myUser && uid === myUser.id) || n === myNickname;
          });
          if (me && me.mmr_rating != null) myChessMmr = me.mmr_rating;
          participantListArr = list.filter(function (p) {
            var n = typeof p === 'string' ? p : (p && p.name);
            return n !== myNickname && n !== 'Serena' && (p && p.user_id);
          });
          renderParticipants(list);
          serenaPresent = list.some(function (p) { return (p && p.name) === 'Serena'; });
          updateSerenaBtn();
          updateHeaderUserDisplay();
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
            data.image_url || null,
            data.timestamp || null
          );
          if (!isMe && data.message_id) {
            ws.send(JSON.stringify({ type: 'read', message_id: data.message_id }));
          }
          if (!isMe && document.hidden) {
            var preview = (data.text && data.text.trim()) ? data.text.trim().slice(0, 50) + (data.text.length > 50 ? '...' : '') : (data.image_url ? '(사진)' : '(메시지)');
            showChatNotification('라운지', data.nickname + ': ' + preview, 'room-' + (data.message_id || ''));
          }
          return;
        }
        if (data.type === 'link_preview') {
          var rowEl = roomMessages.querySelector('[data-message-id="' + data.message_id + '"]');
          if (rowEl && data.preview) {
            var wrap = document.createElement('div');
            wrap.className = 'roomLinkPreviewWrap';
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
            wrap.appendChild(card);
            rowEl.appendChild(wrap);
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
                  txtEl.innerHTML = linkify(newText);
                } else if (newText) {
                  var span = document.createElement('span');
                  span.className = 'roomMsgText';
                  span.innerHTML = linkify(newText);
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
                var span = document.createElement('span');
                span.className = 'roomMsgText';
                span.innerHTML = linkify(newText);
                newBubble.appendChild(span);
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
        if (data.type === 'serena_status') {
          serenaPresent = !!data.present;
          updateSerenaBtn();
          if (!data.present) {
            chessState = { fen: null, turn: null, status: null, whitePlayer: null, blackPlayer: null, whiteRating: null, blackRating: null, mode: 'serena', lastMove: null, inCheck: false, whiteCaptured: [], blackCaptured: [], whiteTime: CHESS_INITIAL_SECONDS, blackTime: CHESS_INITIAL_SECONDS, turnStartedAt: null, paused: false };
            chessSelected = null;
            if (chessClockInterval) { clearInterval(chessClockInterval); chessClockInterval = null; }
            renderChessBoard();
            updateChessClocks();
            updateChessStatus();
            updateChessButtons();
          }
          return;
        }
        if (data.type === 'chess_state') {
          var prevStatus = chessState.status;
          var prevInCheck = chessState.inCheck;
          chessState.fen = data.fen || null;
          chessState.turn = data.turn || null;
          chessState.status = data.status || null;
          chessState.whitePlayer = data.white_player || null;
          chessState.blackPlayer = data.black_player || null;
          chessState.whiteRating = data.white_rating != null ? Number(data.white_rating) : null;
          chessState.blackRating = data.black_rating != null ? Number(data.black_rating) : null;
          chessState.mode = data.mode || 'serena';
          chessState.lastMove = data.last_move || null;
          chessState.inCheck = !!data.in_check;
          chessState.whiteCaptured = data.white_captured || [];
          chessState.blackCaptured = data.black_captured || [];
          chessState.whiteTime = data.white_time != null ? Number(data.white_time) : CHESS_INITIAL_SECONDS;
          chessState.blackTime = data.black_time != null ? Number(data.black_time) : CHESS_INITIAL_SECONDS;
          chessState.turnStartedAt = data.turn_started_at != null ? Number(data.turn_started_at) : null;
          chessState.paused = !!data.paused;
          chessSelected = null;
          chessLegalTargets = null;
          renderChessBoard();
          renderCapturedPieces();
          updateChessClocks();
          updateChessStatus();
          updateChessButtons();
          updateChessPanelTitle();
          if (data.white_player === myNickname || data.black_player === myNickname) {
            var wasActive = prevStatus === 'active';
            var justBecameCheck = wasActive && data.in_check && data.status === 'active' && !prevInCheck;
            if (justBecameCheck) {
              var inCheckTurn = data.turn || '';
              var inCheckName = inCheckTurn === 'white' ? (data.white_player || '흰색') : (data.black_player || '검은색');
              var inCheckSide = inCheckTurn === 'white' ? '흰색' : '검은색';
              showChessAlert('⚠ 체크!', inCheckName + ' (' + inCheckSide + ')이(가) 체크 상태입니다. 왕을 피하세요!');
            }
          }
          var terminalStatuses = ['checkmate_white', 'checkmate_black', 'time_loss_white', 'time_loss_black', 'resign_white', 'resign_black', 'draw'];
          if (terminalStatuses.indexOf(data.status) !== -1 && (data.white_player === myNickname || data.black_player === myNickname)) {
            var resultTitle = '게임 종료';
            var resultMessage = '';
            var wp = data.white_player || '흰색';
            var bp = data.black_player || '검은색';
            if (data.status === 'checkmate_white') { resultMessage = bp + ' 승리 (체크메이트)'; }
            else if (data.status === 'checkmate_black') { resultMessage = wp + ' 승리 (체크메이트)'; }
            else if (data.status === 'time_loss_white') { resultMessage = bp + ' 승리 (흰색 시간 초과)'; }
            else if (data.status === 'time_loss_black') { resultMessage = wp + ' 승리 (검은색 시간 초과)'; }
            else if (data.status === 'resign_white') { resultMessage = bp + ' 승리 (흰색 기권)'; }
            else if (data.status === 'resign_black') { resultMessage = wp + ' 승리 (검은색 기권)'; }
            else if (data.status === 'draw') { resultMessage = '무승부'; }
            var wDelta = data.white_mmr_delta != null ? Number(data.white_mmr_delta) : null;
            var bDelta = data.black_mmr_delta != null ? Number(data.black_mmr_delta) : null;
            showChessResultModal(resultTitle, resultMessage, wp, bp, wDelta, bDelta);
          }
          if (chessState.status === 'active') {
            if (!chessClockInterval) chessClockInterval = setInterval(function () { updateChessClocks(); }, 1000);
          } else {
            if (chessClockInterval) { clearInterval(chessClockInterval); chessClockInterval = null; }
          }
          if (data.status === 'active' && (data.white_player === myNickname || data.black_player === myNickname)) {
            if (chessPanel) { chessPanel.style.display = 'flex'; if (visionaiFrame) visionaiFrame.classList.add('room-iframe--hidden'); if (gomokuPanel) gomokuPanel.style.display = 'none'; updateChessClocks(); if (!chessClockInterval) chessClockInterval = setInterval(function () { updateChessClocks(); }, 1000); }
          }
          return;
        }
        if (data.type === 'chess_legal_moves') {
          chessLegalTargets = (data.from != null && Array.isArray(data.to_squares))
            ? { from: data.from, toSquares: data.to_squares }
            : null;
          renderChessBoard();
          return;
        }
        if (data.type === 'chess_undo_request') {
          var amLastMover = (chessState.whitePlayer === myNickname && chessState.turn === 'black') || (chessState.blackPlayer === myNickname && chessState.turn === 'white');
          if (amLastMover) showChessUndoRequestModal(data.requested_by || '상대');
          return;
        }
        if (data.type === 'chess_undo_rejected') {
          if (data.requested_by === myNickname) showChessAlert('무르기 거절', '무르기가 거절되었습니다.');
          return;
        }
        if (data.type === 'gomoku_state') {
          gomokuState.board = data.board || null;
          gomokuState.turn = data.turn || 'black';
          gomokuState.status = data.status || null;
          gomokuState.blackPlayer = data.black_player || null;
          gomokuState.whitePlayer = data.white_player || null;
          gomokuState.mode = data.mode || 'serena';
          gomokuState.lastMove = data.last_move || null;
          renderGomokuBoard();
          updateGomokuStatus();
          updateGomokuButtons();
          updateGomokuPanelTitle();
          if (data.status === 'active' && (data.black_player === myNickname || data.white_player === myNickname)) {
            if (gomokuPanel) { gomokuPanel.style.display = 'flex'; if (visionaiFrame) visionaiFrame.classList.add('room-iframe--hidden'); if (chessPanel) chessPanel.style.display = 'none'; }
          }
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

  function getDmRoomIdFromUrl() {
    var m = (window.location.search || '').match(/[?&]dm=(\d+)/);
    return m ? parseInt(m[1], 10) : null;
  }

  function enterRoom(user) {
    if (!user || !user.name) return;
    showNickError('');
    myUser = user;
    myNickname = user.name;
    if (loadingSection) loadingSection.style.display = 'none';
    if (authSection) authSection.style.display = 'none';
    roomSection.style.display = 'flex';
    roomSection.style.flexDirection = 'column';
    roomSection.style.flex = '1';
    roomSection.style.minHeight = '0';
    var headerUser = document.getElementById('headerUser');
    if (headerUser) {
      headerUser.style.display = 'flex';
      updateHeaderUserDisplay();
    }
    if (roomNickEditBtn) roomNickEditBtn.style.display = 'flex';
    if (roomLeaveBtn) roomLeaveBtn.style.display = 'flex';
    if (withdrawBtn) withdrawBtn.style.display = 'inline-block';
    if (chatPanelHideBtn) chatPanelHideBtn.style.display = 'flex';
    if (chatFullscreenBtn) chatFullscreenBtn.style.display = 'flex';
    if (notificationToggleBtn) notificationToggleBtn.style.display = 'flex';
    updateNotificationIcon();
    var chatTabsBar = document.getElementById('chatTabsBar');
    if (chatTabsBar) chatTabsBar.style.display = 'block';
    chatTabs = [{ id: 'multi', label: '라운지', type: 'multi', closable: false }];
    activeTabId = 'multi';
    updateSerenaBtn();
    var dmRoomId = getDmRoomIdFromUrl();
    connect();
    fetch('/api/dm/rooms', { credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var rooms = data.rooms || [];
        var existingIds = {};
        chatTabs.forEach(function (t) { existingIds[t.id] = true; });
        rooms.forEach(function (r) {
          var tabId = 'dm-' + r.id;
          if (!existingIds[tabId]) {
            existingIds[tabId] = true;
            chatTabs.push({
              id: tabId,
              label: (r.other_user && r.other_user.name) || '1:1 대화',
              type: 'dm',
              roomId: r.id,
              otherUser: r.other_user,
              closable: false
            });
          }
        });
        renderChatTabs();
      })
      .catch(function () { renderChatTabs(); });
    if (dmRoomId) {
      fetch('/api/dm/rooms/' + dmRoomId, { credentials: 'same-origin' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.room) {
            addDmTab(data.room.id, data.room.other_user);
            if (dmInput) dmInput.focus();
          } else {
            showNickError('대화방을 찾을 수 없습니다.');
            if (roomInput) roomInput.focus();
          }
        })
        .catch(function () {
          showNickError('대화방을 불러올 수 없습니다.');
          if (roomInput) roomInput.focus();
        });
    } else {
      if (multiPanel) multiPanel.style.display = 'flex';
      if (dmPanel) dmPanel.style.display = 'none';
      if (roomInput) roomInput.focus();
    }
  }

  function leaveRoom() {
    if (ws) { ws.close(); ws = null; }
    if (wsDm) { wsDm.close(); wsDm = null; }
    if (loadingSection) loadingSection.style.display = 'block';
    roomSection.style.display = 'none';
    if (roomNickEditBtn) roomNickEditBtn.style.display = 'none';
    if (roomLeaveBtn) roomLeaveBtn.style.display = 'none';
    if (withdrawBtn) withdrawBtn.style.display = 'none';
    if (chatPanelHideBtn) chatPanelHideBtn.style.display = 'none';
    if (chatFullscreenBtn) chatFullscreenBtn.style.display = 'none';
    if (roomLayout) {
      roomLayout.classList.remove('chat-panel-hidden');
      roomLayout.classList.remove('chat-fullscreen');
    }
    if (notificationToggleBtn) notificationToggleBtn.style.display = 'none';
    if (document.getElementById('chatTabsBar')) document.getElementById('chatTabsBar').style.display = 'none';
    serenaPresent = false;
    updateSerenaBtn();
    showNickError('');
  }

  // 로그인 상태 확인 후 입장
  fetch('/api/auth/me', { credentials: 'same-origin' })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (!data.logged_in) {
        if (loadingSection) loadingSection.style.display = 'none';
        if (authSection) authSection.style.display = 'block';
        return;
      }
      return fetch('/api/auth/ws-token').then(function (r) { return r.json(); })
        .then(function (tok) {
          wsToken = tok.token;
          enterRoom(data.user);
        });
    })
    .catch(function () {
      if (loadingSection) loadingSection.style.display = 'none';
      if (authSection) authSection.style.display = 'block';
    });

  function doLogin() {
    var username = (document.getElementById('loginId').value || '').trim();
    var password = document.getElementById('loginPassword').value || '';
    showNickError('');
    if (!username) { showNickError('아이디를 입력하세요.'); return; }
    if (!password) { showNickError('비밀번호를 입력하세요.'); return; }
    fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username, password: password })
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || '로그인 실패'); });
        return r.json();
      })
      .then(function (data) {
        showNickError('');
        return fetch('/api/auth/ws-token').then(function (r) { return r.json(); })
          .then(function (tok) {
            wsToken = tok.token;
            enterRoom(data.user);
          });
      })
      .catch(function (err) { showNickError(err.message || '로그인에 실패했습니다.'); });
  }

  function doSignup() {
    var username = (document.getElementById('signupId').value || '').trim();
    var password = document.getElementById('signupPassword').value || '';
    var passwordConfirm = document.getElementById('signupPasswordConfirm').value || '';
    var name = (document.getElementById('signupName').value || '').trim();
    showNickError('');
    if (!username) { showNickError('아이디를 입력하세요.'); return; }
    if (username.length < 2) { showNickError('아이디는 2자 이상이어야 합니다.'); return; }
    if (!password) { showNickError('비밀번호를 입력하세요.'); return; }
    if (password.length < 4) { showNickError('비밀번호는 4자 이상이어야 합니다.'); return; }
    if (password !== passwordConfirm) { showNickError('비밀번호가 일치하지 않습니다.'); return; }
    if (!name) { showNickError('닉네임을 입력하세요.'); return; }
    fetch('/api/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username, name: name, password: password, password_confirm: passwordConfirm })
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || '가입 실패'); });
        return r.json();
      })
      .then(function (data) {
        showNickError('');
        return fetch('/api/auth/ws-token').then(function (r) { return r.json(); })
          .then(function (tok) {
            wsToken = tok.token;
            enterRoom(data.user);
          });
      })
      .catch(function (err) { showNickError(err.message || '회원가입에 실패했습니다.'); });
  }

  if (document.getElementById('authTabLogin')) {
    document.getElementById('authTabLogin').addEventListener('click', function () {
      document.getElementById('authTabLogin').classList.add('active');
      document.getElementById('authTabSignup').classList.remove('active');
      document.getElementById('loginForm').style.display = 'flex';
      document.getElementById('signupForm').style.display = 'none';
      showNickError('');
    });
  }
  if (document.getElementById('authTabSignup')) {
    document.getElementById('authTabSignup').addEventListener('click', function () {
      document.getElementById('authTabSignup').classList.add('active');
      document.getElementById('authTabLogin').classList.remove('active');
      document.getElementById('signupForm').style.display = 'flex';
      document.getElementById('loginForm').style.display = 'none';
      showNickError('');
    });
  }
  if (document.getElementById('loginBtn')) document.getElementById('loginBtn').addEventListener('click', doLogin);
  if (document.getElementById('signupBtn')) document.getElementById('signupBtn').addEventListener('click', doSignup);

  // 참여자 목록: 이름 클릭 시 1:1 대화 버튼 표시, 버튼 클릭 시 탭에 추가
  if (participantList) {
    participantList.addEventListener('click', function (e) {
      var btn = e.target.closest('.dmFromParticipantBtn');
      if (btn && btn.dataset.userId) {
        e.stopPropagation();
        startDmWithUser(parseInt(btn.dataset.userId, 10));
        hideParticipantDmBtns();
        return;
      }
      var nameEl = e.target.closest('.participantName');
      if (nameEl) {
        e.stopPropagation();
        var item = nameEl.closest('.participantWithDm');
        if (item) {
          var popover = item.querySelector('.participantDmPopover');
          var wasVisible = popover && popover.style.display === 'block';
          hideParticipantDmBtns();
          if (popover && !wasVisible) popover.style.display = 'block';
        }
        return;
      }
    });
  }

  function hideParticipantDmBtns() {
    var popovers = participantList && participantList.querySelectorAll('.participantDmPopover');
    if (popovers) for (var i = 0; i < popovers.length; i++) popovers[i].style.display = 'none';
  }

  document.addEventListener('click', function () { hideParticipantDmBtns(); });

  // 알림 켜기/끄기 버튼 (켤 때 사용자 제스처로 권한 요청 → 허용 시 비활성 탭에서도 우측/하단 팝업 알림)
  if (notificationToggleBtn) {
    notificationToggleBtn.addEventListener('click', function () {
      var enabled = getNotificationEnabled();
      if (enabled) {
        setNotificationEnabled(false);
        return;
      }
      if (!('Notification' in window)) {
        setNotificationEnabled(false);
        return;
      }
      if (Notification.permission === 'denied') {
        alert('이 사이트의 알림이 브라우저에서 차단되어 있습니다.\n주소창 왼쪽 자물쇠(또는 설정) → 사이트 설정 → 알림을 허용해 주세요.');
        return;
      }
      if (Notification.permission === 'default') {
        Notification.requestPermission().then(function (perm) {
          if (perm === 'granted') {
            setNotificationEnabled(true);
          } else {
            setNotificationEnabled(false);
          }
          updateNotificationIcon();
        }).catch(function () {
          setNotificationEnabled(false);
          updateNotificationIcon();
        });
        return;
      }
      setNotificationEnabled(true);
    });
  }

  // DM 대화방 나가기 버튼
  if (dmCloseBtn) {
    dmCloseBtn.addEventListener('click', function () {
      if (activeTabId && activeTabId !== 'multi') {
        removeTab(activeTabId);
        switchToTab('multi');
      }
    });
  }

  if (document.getElementById('dmBackLink')) {
    document.getElementById('dmBackLink').addEventListener('click', function () {
      switchToTab('multi');
    });
  }

  function sendDmMessage(text, imageUrl) {
    if (!wsDm || wsDm.readyState !== WebSocket.OPEN || !currentDmRoomId) return;
    var txt = (text || '').trim();
    if (!txt && !imageUrl) return;
    wsDm.send(JSON.stringify({ type: 'chat', text: txt || '', image_url: imageUrl || '' }));
    if (dmInput) {
      dmInput.value = '';
      dmInput.style.height = 'auto';
    }
    // 전송 후 현재 방 draft 초기화
    if (currentDmRoomId) dmDraftByRoomId[String(currentDmRoomId)] = '';
  }

  if (dmSendBtn && dmInput) {
    dmSendBtn.addEventListener('click', function () {
      var txt = (dmInput.value || '').trim();
      sendDmMessage(txt);
    });
  }

  if (dmInput) {
    dmInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        if (!e.shiftKey) {
          e.preventDefault();
          sendDmMessage((dmInput.value || '').trim());
        }
      }
    });
    // DM textarea 자동 높이 조절 + 방별 draft 저장
    dmInput.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 120) + 'px';
      if (currentDmRoomId) dmDraftByRoomId[String(currentDmRoomId)] = this.value || '';
    });
  }

  if (dmImageBtn && dmImageInput) {
    dmImageBtn.addEventListener('click', function () { dmImageInput.click(); });
    dmImageInput.addEventListener('change', function () {
      var file = this.files && this.files[0];
      if (!file || !wsDm || wsDm.readyState !== WebSocket.OPEN || !currentDmRoomId) { this.value = ''; return; }
      var fd = new FormData();
      fd.append('file', file, file.name || 'image.jpg');
      fetch('/api/room/upload', { method: 'POST', body: fd })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || '업로드 실패'); });
          return r.json();
        })
        .then(function (data) {
          if (data.url) sendDmMessage((dmInput && dmInput.value || '').trim(), data.url);
        })
        .catch(function (err) { showNickError(err.message || '사진 업로드에 실패했습니다.'); })
        .finally(function () { dmImageInput.value = ''; });
    });
  }

  function insertEmojiIntoTextarea(ta, emoji) {
    if (!ta) return;
    var start = ta.selectionStart;
    var end = ta.selectionEnd;
    var text = ta.value;
    ta.value = text.slice(0, start) + emoji + text.slice(end);
    ta.selectionStart = ta.selectionEnd = start + emoji.length;
    ta.focus();
    // 프로그램적으로 값 변경 시 input 이벤트가 발생하지 않으므로 강제 트리거
    try { ta.dispatchEvent(new Event('input', { bubbles: true })); } catch (e) {}
  }

  function initDmEmojiPanel() {
    if (!dmEmojiPanel || dmEmojiPanel.querySelector('.roomEmojiPanelInner')) return;
    var header = document.createElement('div');
    header.className = 'roomEmojiPanelHeader';
    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'roomEmojiPanelClose';
    closeBtn.title = '닫기';
    closeBtn.innerHTML = '&times;';
    closeBtn.addEventListener('click', function () { dmEmojiPanel.style.display = 'none'; });
    header.appendChild(closeBtn);
    dmEmojiPanel.appendChild(header);
    var inner = document.createElement('div');
    inner.className = 'roomEmojiPanelInner';
    EMOJI_LIST.forEach(function (emoji) {
      var span = document.createElement('span');
      span.textContent = emoji;
      span.setAttribute('role', 'button');
      span.tabIndex = 0;
      span.addEventListener('click', function () { insertEmojiIntoTextarea(dmInput, emoji); });
      inner.appendChild(span);
    });
    dmEmojiPanel.appendChild(inner);
  }

  if (dmEmojiBtn && dmEmojiPanel) {
    dmEmojiBtn.addEventListener('click', function (e) {
      e.preventDefault();
      initDmEmojiPanel();
      var visible = dmEmojiPanel.style.display === 'flex' || dmEmojiPanel.style.display === 'grid' || dmEmojiPanel.style.display === 'block';
      dmEmojiPanel.style.display = visible ? 'none' : 'flex';
    });
    document.addEventListener('click', function (e) {
      if (dmEmojiPanel && dmEmojiPanel.style.display !== 'none' && !dmEmojiPanel.contains(e.target) && !(dmEmojiBtn && dmEmojiBtn.contains(e.target))) {
        dmEmojiPanel.style.display = 'none';
      }
    });
  }

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
      window.location.href = '/api/auth/logout';
    });
    if (withdrawBtn) {
      withdrawBtn.addEventListener('click', function (e) {
        e.preventDefault();
        if (!window.confirm('정말 회원탈퇴하시겠습니까? 탈퇴 시 계정과 모든 1:1 대화 내역이 삭제되며 복구할 수 없습니다.')) return;
        var pwd = window.prompt('비밀번호를 입력해 주세요.');
        if (pwd == null) return;
        fetch(window.location.origin + '/api/auth/withdraw', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: pwd || '' })
        }).then(function (r) {
          if (r.redirected) {
            window.location.href = r.url;
            return;
          }
          if (!r.ok) {
            return r.json().then(function (d) {
              showNickError(d.detail || '탈퇴에 실패했습니다.');
            });
          }
          window.location.href = '/';
        }).catch(function () {
          showNickError('탈퇴 요청에 실패했습니다.');
        });
      });
    }
    if (chatPanelHideBtn && roomLayout) {
      chatPanelHideBtn.addEventListener('click', function () {
        roomLayout.classList.add('chat-panel-hidden');
      });
    }
    if (chatPanelShowBtn && roomLayout) {
      chatPanelShowBtn.addEventListener('click', function () {
        roomLayout.classList.remove('chat-panel-hidden');
      });
    }
    if (chatFullscreenBtn && roomLayout) {
      chatFullscreenBtn.addEventListener('click', function () {
        var isFull = roomLayout.classList.toggle('chat-fullscreen');
        var expandIcon = chatFullscreenBtn.querySelector('.icon-expand');
        var shrinkIcon = chatFullscreenBtn.querySelector('.icon-shrink');
        if (expandIcon && shrinkIcon) {
          expandIcon.style.display = isFull ? 'none' : 'block';
          shrinkIcon.style.display = isFull ? 'block' : 'none';
        }
        chatFullscreenBtn.title = isFull ? '전체화면 해제' : '채팅 전체화면';
        chatFullscreenBtn.setAttribute('aria-label', isFull ? '전체화면 해제' : '채팅 전체화면');
      });
    }
  }

  if (serenaInviteBtn) {
    serenaInviteBtn.addEventListener('click', function () {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      serenaInviteBtn.disabled = true;
      var action = serenaPresent ? 'kick_serena' : 'invite_serena';
      ws.send(JSON.stringify({ type: action }));
      setTimeout(function () { serenaInviteBtn.disabled = false; }, 5000);
    });
  }

  function setChessPanelInVisionArea(visible) {
    if (!chessPanel) return;
    if (visible) {
      chessPanel.style.display = 'flex';
      if (visionaiFrame) visionaiFrame.classList.add('room-iframe--hidden');
      if (gomokuPanel) gomokuPanel.style.display = 'none';
      updateChessClocks();
      if (chessState.status === 'active' && !chessClockInterval) {
        chessClockInterval = setInterval(function () { updateChessClocks(); }, 1000);
      }
    } else {
      chessPanel.style.display = 'none';
      if (visionaiFrame) visionaiFrame.classList.remove('room-iframe--hidden');
    }
  }

  function setGomokuPanelInVisionArea(visible) {
    if (!gomokuPanel) return;
    if (visible) {
      gomokuPanel.style.display = 'flex';
      if (visionaiFrame) visionaiFrame.classList.add('room-iframe--hidden');
      if (chessPanel) chessPanel.style.display = 'none';
    } else {
      gomokuPanel.style.display = 'none';
      if (visionaiFrame) visionaiFrame.classList.remove('room-iframe--hidden');
    }
  }

  if (chessGameBtn) {
    chessGameBtn.addEventListener('click', function () {
      var show = chessPanel.style.display === 'none' || !chessPanel.style.display;
      setChessPanelInVisionArea(show);
      if (show) {
        renderChessBoard();
        renderCapturedPieces();
        updateChessClocks();
        updateChessStatus();
        updateChessButtons();
        updateChessPanelTitle();
      }
    });
  }
  if (chessPvpBtn) {
    chessPvpBtn.addEventListener('click', function () {
      var show = chessPanel.style.display === 'none' || !chessPanel.style.display;
      setChessPanelInVisionArea(show);
      if (show) {
        renderChessBoard();
        renderCapturedPieces();
        updateChessClocks();
        updateChessStatus();
        updateChessButtons();
        updateChessPanelTitle();
      }
    });
  }
  if (chessPanelClose) {
    chessPanelClose.addEventListener('click', function () { setChessPanelInVisionArea(false); });
  }
  if (chessStartBtn) {
    chessStartBtn.addEventListener('click', function () {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_start' }));
    });
  }
  if (chessStartPvpBtn) {
    chessStartPvpBtn.addEventListener('click', function () {
      var showSelect = true;
      if (chessPvpSelect) {
        if (chessPvpSelect.dataset.visible === '1') { showSelect = false; }
        chessPvpSelect.dataset.visible = showSelect ? '1' : '0';
        chessPvpSelect.style.display = showSelect ? 'flex' : 'none';
        if (chessPvpOpponent && showSelect) {
          chessPvpOpponent.innerHTML = participantListArr.map(function (p) {
            var n = p && p.name;
            return n ? '<option value="' + escapeHtml(n) + '">' + escapeHtml(n) + '</option>' : '';
          }).join('') || '<option value="">참여자 없음</option>';
        }
      }
      if (showSelect && chessPanel) {
        setChessPanelInVisionArea(true);
      }
    });
  }
  if (chessPvpConfirm && chessPvpOpponent) {
    chessPvpConfirm.addEventListener('click', function () {
      var opp = chessPvpOpponent.value;
      var mySideEl = document.getElementById('chessPvpMySide');
      var mySide = mySideEl ? (mySideEl.value || 'black').toLowerCase() : 'black';
      if (opp && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'chess_start_pvp', opponent: opp, my_side: mySide }));
        if (chessPvpSelect) {
          chessPvpSelect.dataset.visible = '0';
          chessPvpSelect.style.display = 'none';
        }
      }
    });
  }
  if (chessResignBtn) {
    chessResignBtn.addEventListener('click', function () {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_resign' }));
    });
  }
  if (chessUndoBtn) {
    chessUndoBtn.addEventListener('click', function () {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_undo' }));
    });
  }
  if (chessPauseBtn) {
    chessPauseBtn.addEventListener('click', function () {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_pause' }));
    });
  }
  if (chessResumeBtn) {
    chessResumeBtn.addEventListener('click', function () {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'chess_resume' }));
    });
  }

  if (gomokuGameBtn) {
    gomokuGameBtn.addEventListener('click', function () {
      var show = gomokuPanel.style.display === 'none' || !gomokuPanel.style.display;
      setGomokuPanelInVisionArea(show);
      if (show) {
        renderGomokuBoard();
        updateGomokuStatus();
        updateGomokuButtons();
        updateGomokuPanelTitle();
      }
    });
  }
  if (gomokuPvpBtn) {
    gomokuPvpBtn.addEventListener('click', function () {
      var show = gomokuPanel.style.display === 'none' || !gomokuPanel.style.display;
      setGomokuPanelInVisionArea(show);
      if (show) {
        renderGomokuBoard();
        updateGomokuStatus();
        updateGomokuButtons();
        updateGomokuPanelTitle();
      }
    });
  }
  if (gomokuPanelClose) {
    gomokuPanelClose.addEventListener('click', function () { setGomokuPanelInVisionArea(false); });
  }
  if (gomokuStartBtn) {
    gomokuStartBtn.addEventListener('click', function () {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'gomoku_start' }));
    });
  }
  if (gomokuStartPvpBtn) {
    gomokuStartPvpBtn.addEventListener('click', function () {
      if (gomokuPvpSelect) {
        gomokuPvpSelect.dataset.visible = gomokuPvpSelect.dataset.visible === '1' ? '0' : '1';
        gomokuPvpSelect.style.display = gomokuPvpSelect.dataset.visible === '1' ? 'flex' : 'none';
        if (gomokuPvpOpponent && gomokuPvpSelect.dataset.visible === '1') {
          gomokuPvpOpponent.innerHTML = participantListArr.map(function (p) {
            var n = p && p.name;
            return n ? '<option value="' + escapeHtml(n) + '">' + escapeHtml(n) + '</option>' : '';
          }).join('') || '<option value="">참여자 없음</option>';
        }
      }
    });
  }
  if (gomokuPvpConfirm && gomokuPvpOpponent) {
    gomokuPvpConfirm.addEventListener('click', function () {
      var opp = gomokuPvpOpponent.value;
      if (opp && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'gomoku_start_pvp', opponent: opp }));
        if (gomokuPvpSelect) {
          gomokuPvpSelect.dataset.visible = '0';
          gomokuPvpSelect.style.display = 'none';
        }
      }
    });
  }
  if (gomokuResignBtn) {
    gomokuResignBtn.addEventListener('click', function () {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'gomoku_resign' }));
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
    if (inRoom && activeTabId && activeTabId.indexOf('dm-') === 0 && currentDmRoomId) {
      sendDmViewedRoom();
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
      fd.append('file', file, file.name || 'image.jpg');
      fetch('/api/room/upload', { method: 'POST', body: fd })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || '업로드 실패'); });
          return r.json();
        })
        .then(function (data) {
          var url = data.url;
          if (url) {
            var text = (roomInput && roomInput.value || '').trim();
            sendMessage(text, url);
          }
        })
        .catch(function (err) { showNickError(err.message || '사진 업로드에 실패했습니다.'); })
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
      if (roomEmojiPanel.style.display !== 'none' && !roomEmojiPanel.contains(e.target) && !(roomEmojiBtn && roomEmojiBtn.contains(e.target))) {
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
      var disp = sendUpdate && newText ? newText : currentText;
      var span = document.createElement('span');
      span.className = 'roomMsgText';
      span.innerHTML = linkify(disp);
      bubble.textContent = '';
      bubble.appendChild(span);
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

  function startInlineEditDm(messageId, currentText) {
    if (!dmMessages || !wsDm || wsDm.readyState !== WebSocket.OPEN) return;
    var rowEl = dmMessages.querySelector('[data-message-id="' + messageId + '"]');
    if (!rowEl) return;
    var bubble = rowEl.querySelector('.roomBubble.me');
    var emojiDiv = rowEl.querySelector('.roomMsgEmoji--me');
    var imgWrap = rowEl.querySelector('.roomMsgImageLargeWrap');
    var cap = imgWrap ? imgWrap.querySelector('.roomMsgImageCaption') : null;
    var contentEl = bubble || emojiDiv || cap;
    if (!contentEl && imgWrap) {
      contentEl = document.createElement('div');
      contentEl.className = 'roomMsgImageCaption';
      imgWrap.appendChild(contentEl);
    }
    if (!contentEl) return;

    var textarea = document.createElement('textarea');
    textarea.className = 'roomBubble-edit';
    textarea.value = currentText;
    textarea.rows = 1;
    var origContent = contentEl.textContent || '';
    contentEl.textContent = '';
    contentEl.appendChild(textarea);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    function finishEdit(sendUpdate) {
      var newText = (textarea.value || '').trim();
      textarea.remove();
      if (bubble) {
        contentEl.innerHTML = '<span class="roomMsgText">' + linkify(sendUpdate && newText ? newText : origContent) + '</span>';
      } else if (emojiDiv) {
        contentEl.textContent = sendUpdate && newText ? newText : origContent;
      } else {
        contentEl.textContent = sendUpdate && newText ? newText : origContent;
      }
      rowEl.dataset.rawText = sendUpdate && newText ? newText : origContent;
      if (sendUpdate && newText && newText !== origContent && wsDm && wsDm.readyState === WebSocket.OPEN) {
        wsDm.send(JSON.stringify({ type: 'edit', message_id: messageId, text: newText }));
      }
    }

    textarea.addEventListener('blur', function () { finishEdit(true); });
    textarea.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' && !ev.shiftKey) {
        ev.preventDefault();
        finishEdit(true);
      } else if (ev.key === 'Escape') {
        ev.preventDefault();
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
    const isDm = contextMenuEl.dataset.isDm === '1';
    hideContextMenu();

    if (isDm) {
      if (!messageId || !wsDm || wsDm.readyState !== WebSocket.OPEN) return;
      if (action === 'edit') {
        startInlineEditDm(messageId, currentText);
        return;
      }
      if (action === 'delete') {
        wsDm.send(JSON.stringify({ type: 'delete', message_id: messageId }));
        return;
      }
      return;
    }

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

  function loadVisionAiBackend() {
    var iframe = document.getElementById('visionaiFrame');
    var labelEl = document.getElementById('visionaiBackendLabel');
    if (!iframe || !iframe.src || !labelEl) return;
    try {
      var origin = new URL(iframe.src).origin;
      fetch(origin + '/api/emotion-backend', { credentials: 'omit' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data && data.emotion_backend) {
            labelEl.textContent = data.emotion_backend;
          }
        })
        .catch(function () { labelEl.textContent = '—'; });
    } catch (e) { labelEl.textContent = '—'; }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      setTimeout(loadVisionAiBackend, 800);
    });
  } else {
    setTimeout(loadVisionAiBackend, 800);
  }
})();
