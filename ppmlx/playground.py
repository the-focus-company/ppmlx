"""Built-in web chat playground for ppmlx.

Serves a self-contained HTML/CSS/JS chat UI at ``/ui`` with no external
dependencies.  The UI talks to the existing ``/v1/chat/completions``
endpoint for streaming responses and ``/v1/models`` for the model picker.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["playground"])

_PLAYGROUND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ppmlx Playground</title>
<style>
/* ── Reset & Variables ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #ffffff; --bg-secondary: #f7f7f8; --bg-tertiary: #ececf1;
  --text: #1a1a2e; --text-secondary: #6b6b80;
  --border: #d9d9e3; --border-light: #e5e5ea;
  --accent: #6c63ff; --accent-hover: #5a52d5; --accent-light: #eeedff;
  --user-bg: #6c63ff; --user-text: #ffffff;
  --assistant-bg: #f7f7f8; --assistant-text: #1a1a2e;
  --error: #ef4444; --success: #22c55e;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --radius: 12px; --radius-sm: 8px;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  --sidebar-width: 260px;
  --max-chat-width: 780px;
}

[data-theme="dark"] {
  --bg: #1a1a2e; --bg-secondary: #232340; --bg-tertiary: #2d2d4a;
  --text: #e4e4f0; --text-secondary: #9898b0;
  --border: #3a3a5c; --border-light: #2d2d4a;
  --accent: #8b83ff; --accent-hover: #9d96ff; --accent-light: #2d2b4a;
  --user-bg: #6c63ff; --user-text: #ffffff;
  --assistant-bg: #232340; --assistant-text: #e4e4f0;
  --error: #f87171; --success: #4ade80;
  --shadow: 0 1px 3px rgba(0,0,0,0.3);
}

body {
  font-family: var(--font); background: var(--bg); color: var(--text);
  height: 100vh; display: flex; overflow: hidden;
  transition: background 0.2s, color 0.2s;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
.sidebar {
  width: var(--sidebar-width); background: var(--bg-secondary);
  border-right: 1px solid var(--border); display: flex; flex-direction: column;
  flex-shrink: 0; transition: transform 0.25s ease;
}

.sidebar-header {
  padding: 16px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}

.sidebar-header h1 {
  font-size: 18px; font-weight: 700;
  background: linear-gradient(135deg, var(--accent), #ff6b6b);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}

.sidebar-actions { padding: 12px; display: flex; flex-direction: column; gap: 6px; }

.sidebar-btn {
  display: flex; align-items: center; gap: 8px; padding: 10px 12px;
  border: none; border-radius: var(--radius-sm); cursor: pointer;
  font-size: 14px; font-family: var(--font);
  background: transparent; color: var(--text); text-align: left;
  transition: background 0.15s;
}
.sidebar-btn:hover { background: var(--bg-tertiary); }
.sidebar-btn.primary { background: var(--accent); color: #fff; font-weight: 600; }
.sidebar-btn.primary:hover { background: var(--accent-hover); }

/* Conversation list */
.conv-list { flex: 1; overflow-y: auto; padding: 4px 12px; }
.conv-item {
  padding: 10px 12px; border-radius: var(--radius-sm); cursor: pointer;
  font-size: 13px; color: var(--text-secondary); margin-bottom: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  transition: background 0.15s;
}
.conv-item:hover { background: var(--bg-tertiary); }
.conv-item.active { background: var(--accent-light); color: var(--accent); font-weight: 600; }

.sidebar-footer {
  padding: 12px; border-top: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}

/* ── Main content ──────────────────────────────────────────────────── */
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

.topbar {
  padding: 12px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px; flex-shrink: 0;
}

.model-select {
  padding: 8px 12px; border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg); color: var(--text); font-size: 14px;
  font-family: var(--font); cursor: pointer; min-width: 200px;
  appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%236b6b80' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 10px center;
  padding-right: 30px;
}
.model-select:focus { outline: 2px solid var(--accent); border-color: transparent; }

.topbar-spacer { flex: 1; }

.icon-btn {
  width: 36px; height: 36px; border: none; border-radius: var(--radius-sm);
  background: transparent; color: var(--text-secondary); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; transition: background 0.15s, color 0.15s;
}
.icon-btn:hover { background: var(--bg-tertiary); color: var(--text); }

/* ── Chat area ─────────────────────────────────────────────────────── */
.chat-area {
  flex: 1; overflow-y: auto; padding: 24px 20px;
  scroll-behavior: smooth;
}

.chat-inner {
  max-width: var(--max-chat-width); margin: 0 auto;
  display: flex; flex-direction: column; gap: 16px;
}

/* Welcome screen */
.welcome {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; height: 100%; text-align: center;
  padding: 40px 20px;
}
.welcome h2 {
  font-size: 28px; font-weight: 700; margin-bottom: 8px;
  background: linear-gradient(135deg, var(--accent), #ff6b6b);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.welcome p { color: var(--text-secondary); font-size: 15px; max-width: 420px; }

/* Message bubbles */
.message { display: flex; gap: 12px; max-width: 100%; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }

.message-avatar {
  width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; font-weight: 700;
}
.message.user .message-avatar { background: var(--user-bg); color: var(--user-text); }
.message.assistant .message-avatar { background: var(--bg-tertiary); color: var(--accent); }

.message-content {
  padding: 12px 16px; border-radius: var(--radius);
  max-width: 85%; line-height: 1.6; font-size: 15px;
  word-wrap: break-word; overflow-wrap: break-word;
}
.message.user .message-content {
  background: var(--user-bg); color: var(--user-text);
  border-bottom-right-radius: 4px;
}
.message.assistant .message-content {
  background: var(--assistant-bg); color: var(--assistant-text);
  border-bottom-left-radius: 4px;
}

/* Markdown in assistant messages */
.message-content pre {
  background: var(--bg-tertiary); border-radius: var(--radius-sm);
  padding: 12px 16px; overflow-x: auto; margin: 8px 0;
  font-size: 13px; line-height: 1.5;
}
.message-content code {
  font-family: var(--mono); font-size: 0.9em;
  background: var(--bg-tertiary); padding: 2px 6px;
  border-radius: 4px;
}
.message-content pre code { background: none; padding: 0; font-size: inherit; }
.message-content p { margin: 4px 0; }
.message-content ul, .message-content ol { margin: 4px 0 4px 20px; }
.message-content li { margin: 2px 0; }
.message-content blockquote {
  border-left: 3px solid var(--accent); padding-left: 12px;
  margin: 8px 0; color: var(--text-secondary);
}
.message-content h1, .message-content h2, .message-content h3,
.message-content h4, .message-content h5, .message-content h6 {
  margin: 12px 0 4px; font-weight: 600;
}
.message-content table { border-collapse: collapse; margin: 8px 0; width: 100%; }
.message-content th, .message-content td {
  border: 1px solid var(--border); padding: 6px 10px; text-align: left;
}
.message-content th { background: var(--bg-tertiary); font-weight: 600; }

/* Streaming cursor */
.streaming-cursor::after {
  content: ''; display: inline-block; width: 8px; height: 16px;
  background: var(--accent); border-radius: 2px; margin-left: 2px;
  animation: blink 0.8s step-end infinite; vertical-align: text-bottom;
}
@keyframes blink { 50% { opacity: 0; } }

/* Error message */
.message-error {
  background: color-mix(in srgb, var(--error) 10%, transparent);
  color: var(--error); padding: 12px 16px;
  border-radius: var(--radius); border: 1px solid var(--error);
  font-size: 14px; max-width: var(--max-chat-width); margin: 0 auto;
}

/* ── Input area ────────────────────────────────────────────────────── */
.input-area {
  padding: 16px 20px 24px; border-top: 1px solid var(--border);
  flex-shrink: 0;
}

.input-inner {
  max-width: var(--max-chat-width); margin: 0 auto;
  display: flex; gap: 10px; align-items: flex-end;
}

.input-wrapper {
  flex: 1; position: relative;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--bg); transition: border-color 0.15s;
}
.input-wrapper:focus-within { border-color: var(--accent); }

#chat-input {
  width: 100%; border: none; outline: none; resize: none;
  padding: 12px 16px; font-size: 15px; font-family: var(--font);
  background: transparent; color: var(--text);
  max-height: 200px; line-height: 1.5; min-height: 48px;
}
#chat-input::placeholder { color: var(--text-secondary); }

.send-btn {
  width: 48px; height: 48px; border: none; border-radius: var(--radius);
  background: var(--accent); color: #fff; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; flex-shrink: 0; transition: background 0.15s, opacity 0.15s;
}
.send-btn:hover { background: var(--accent-hover); }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.stop-btn {
  background: var(--error);
}
.stop-btn:hover { background: color-mix(in srgb, var(--error) 80%, black); }

.input-hint {
  text-align: center; margin-top: 8px; font-size: 12px;
  color: var(--text-secondary);
}

/* ── Responsive ────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .sidebar { position: fixed; left: 0; top: 0; bottom: 0; z-index: 100;
    transform: translateX(-100%); }
  .sidebar.open { transform: translateX(0); }
  .sidebar-overlay { display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.4); z-index: 99; }
  .sidebar.open + .sidebar-overlay { display: block; }
  .menu-btn { display: flex !important; }
}
@media (min-width: 769px) {
  .menu-btn { display: none !important; }
}
</style>
</head>
<body>

<!-- Sidebar -->
<aside class="sidebar" id="sidebar">
  <div class="sidebar-header">
    <h1>ppmlx</h1>
  </div>
  <div class="sidebar-actions">
    <button class="sidebar-btn primary" onclick="newConversation()" title="New chat">
      + New Chat
    </button>
  </div>
  <div class="conv-list" id="conv-list"></div>
  <div class="sidebar-footer">
    <span style="font-size:12px;color:var(--text-secondary)">ppmlx Playground</span>
    <button class="icon-btn" onclick="toggleTheme()" title="Toggle theme" id="theme-btn">
      &#9790;
    </button>
  </div>
</aside>
<div class="sidebar-overlay" onclick="closeSidebar()"></div>

<!-- Main -->
<div class="main">
  <div class="topbar">
    <button class="icon-btn menu-btn" onclick="toggleSidebar()" title="Menu">&#9776;</button>
    <select class="model-select" id="model-select" title="Select model">
      <option value="">Loading models...</option>
    </select>
    <div class="topbar-spacer"></div>
    <button class="icon-btn" onclick="exportConversation()" title="Export chat">&#8681;</button>
    <button class="icon-btn" onclick="clearConversation()" title="Clear chat">&#128465;</button>
  </div>

  <div class="chat-area" id="chat-area">
    <div class="chat-inner" id="chat-inner">
      <div class="welcome" id="welcome">
        <h2>ppmlx Playground</h2>
        <p>Chat with your local LLMs running on Apple Silicon. Select a model above and start typing.</p>
      </div>
    </div>
  </div>

  <div class="input-area">
    <div class="input-inner">
      <div class="input-wrapper">
        <textarea id="chat-input" placeholder="Type a message..." rows="1"
                  onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      </div>
      <button class="send-btn" id="send-btn" onclick="sendMessage()" title="Send">
        &#9654;
      </button>
    </div>
    <div class="input-hint">Enter to send &middot; Shift+Enter for newline</div>
  </div>
</div>

<script>
/* ── State ─────────────────────────────────────────────────────────── */
let conversations = JSON.parse(localStorage.getItem('ppmlx_convs') || '[]');
let activeConvId = null;
let isStreaming = false;
let abortController = null;

/* ── Theme ─────────────────────────────────────────────────────────── */
function initTheme() {
  const saved = localStorage.getItem('ppmlx_theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', theme);
  updateThemeBtn(theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('ppmlx_theme', next);
  updateThemeBtn(next);
}

function updateThemeBtn(theme) {
  const btn = document.getElementById('theme-btn');
  btn.innerHTML = theme === 'dark' ? '&#9788;' : '&#9790;';
}

/* ── Models ────────────────────────────────────────────────────────── */
async function loadModels() {
  const select = document.getElementById('model-select');
  try {
    const resp = await fetch('/v1/models');
    const data = await resp.json();
    const models = data.data || data.models || [];
    select.innerHTML = '';
    if (models.length === 0) {
      select.innerHTML = '<option value="">No models available</option>';
      return;
    }
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.id;
      select.appendChild(opt);
    });
    // Restore last used model
    const last = localStorage.getItem('ppmlx_model');
    if (last && models.some(m => m.id === last)) {
      select.value = last;
    }
  } catch (e) {
    select.innerHTML = '<option value="">Failed to load models</option>';
  }
}

document.getElementById('model-select').addEventListener('change', function() {
  localStorage.setItem('ppmlx_model', this.value);
});

/* ── Conversations ─────────────────────────────────────────────────── */
const MAX_CONVERSATIONS = 50;

function saveConversations() {
  if (conversations.length > MAX_CONVERSATIONS) {
    conversations = conversations.slice(0, MAX_CONVERSATIONS);
  }
  localStorage.setItem('ppmlx_convs', JSON.stringify(conversations));
}

function newConversation() {
  const conv = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    title: 'New Chat',
    messages: [],
    created: Date.now(),
  };
  conversations.unshift(conv);
  activeConvId = conv.id;
  saveConversations();
  renderConvList();
  renderMessages();
  closeSidebar();
  document.getElementById('chat-input').focus();
}

function switchConversation(id) {
  activeConvId = id;
  renderConvList();
  renderMessages();
  closeSidebar();
}

function getActiveConv() {
  return conversations.find(c => c.id === activeConvId) || null;
}

function renderConvList() {
  const list = document.getElementById('conv-list');
  list.innerHTML = '';
  conversations.forEach(conv => {
    const el = document.createElement('div');
    el.className = 'conv-item' + (conv.id === activeConvId ? ' active' : '');
    el.textContent = conv.title;
    el.onclick = () => switchConversation(conv.id);
    list.appendChild(el);
  });
}

function clearConversation() {
  const conv = getActiveConv();
  if (!conv) return;
  if (conv.messages.length > 0 && !confirm('Clear this conversation?')) return;
  conv.messages = [];
  conv.title = 'New Chat';
  saveConversations();
  renderConvList();
  renderMessages();
}

function exportConversation() {
  const conv = getActiveConv();
  if (!conv || conv.messages.length === 0) return;
  const text = conv.messages.map(m =>
    `## ${m.role === 'user' ? 'You' : 'Assistant'}\n\n${m.content}\n`
  ).join('\n---\n\n');
  const blob = new Blob([text], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (conv.title || 'chat').replace(/[^a-z0-9]/gi, '_') + '.md';
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ── Markdown Rendering ────────────────────────────────────────────── */
function renderMarkdown(text) {
  // Escape HTML first
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Code blocks (fenced)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`
  );

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers
  html = html.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
  html = html.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
  html = html.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');

  // Bold & italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Blockquotes
  html = html.replace(/^&gt;\s+(.+)$/gm, '<blockquote>$1</blockquote>');

  // Unordered lists
  html = html.replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

  // Ordered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Horizontal rules
  html = html.replace(/^---+$/gm, '<hr>');

  // Paragraphs: convert double newlines
  html = html.replace(/\n\n/g, '</p><p>');
  html = '<p>' + html + '</p>';
  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '');
  // Don't wrap block elements in <p>
  html = html.replace(/<p>\s*(<(?:pre|h[1-6]|ul|ol|blockquote|hr|table)[^>]*>)/g, '$1');
  html = html.replace(/(<\/(?:pre|h[1-6]|ul|ol|blockquote|hr|table)>)\s*<\/p>/g, '$1');

  return html;
}

/* ── Render Messages ───────────────────────────────────────────────── */
function renderMessages() {
  const conv = getActiveConv();
  const inner = document.getElementById('chat-inner');
  const welcome = document.getElementById('welcome');

  if (!conv || conv.messages.length === 0) {
    inner.innerHTML = '';
    inner.appendChild(welcome);
    welcome.style.display = 'flex';
    return;
  }

  welcome.style.display = 'none';
  inner.innerHTML = '';

  conv.messages.forEach((msg, idx) => {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;
    div.id = `msg-${idx}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = msg.role === 'user' ? 'U' : 'A';

    const content = document.createElement('div');
    content.className = 'message-content';

    if (msg.role === 'user') {
      content.textContent = msg.content;
    } else {
      content.innerHTML = renderMarkdown(msg.content);
    }

    if (msg.role === 'user') {
      div.appendChild(content);
      div.appendChild(avatar);
    } else {
      div.appendChild(avatar);
      div.appendChild(content);
    }

    inner.appendChild(div);
  });

  scrollToBottom();
}

function scrollToBottom() {
  const area = document.getElementById('chat-area');
  area.scrollTop = area.scrollHeight;
}

/* ── Send Message ──────────────────────────────────────────────────── */
async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || isStreaming) return;

  const model = document.getElementById('model-select').value;
  if (!model) { alert('Please select a model first.'); return; }

  // Create conversation if needed
  if (!activeConvId) newConversation();
  const conv = getActiveConv();

  // Add user message
  conv.messages.push({ role: 'user', content: text });

  // Auto-title from first message
  if (conv.messages.length === 1) {
    conv.title = text.slice(0, 50) + (text.length > 50 ? '...' : '');
    renderConvList();
  }

  input.value = '';
  autoResize(input);

  // Add placeholder assistant message
  conv.messages.push({ role: 'assistant', content: '' });
  const msgIdx = conv.messages.length - 1;
  renderMessages();

  // Show streaming cursor on last message
  const lastContent = document.querySelector(`#msg-${msgIdx} .message-content`);
  if (lastContent) lastContent.classList.add('streaming-cursor');

  // Switch send button to stop button
  isStreaming = true;
  updateSendButton();

  abortController = new AbortController();

  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model,
        messages: conv.messages.slice(0, -1).map(m => ({
          role: m.role, content: m.content
        })),
        stream: true,
      }),
      signal: abortController.signal,
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || errData.error?.message || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;

        try {
          const parsed = JSON.parse(data);
          if (parsed.error) {
            throw new Error(parsed.error.message || 'Stream error');
          }
          const delta = parsed.choices?.[0]?.delta;
          if (delta?.content) {
            conv.messages[msgIdx].content += delta.content;
            // Update the rendered content in place
            if (lastContent) {
              lastContent.innerHTML = renderMarkdown(conv.messages[msgIdx].content);
            }
            scrollToBottom();
          }
        } catch (e) {
          if (e.message && e.message !== 'Stream error') {
            // JSON parse error on incomplete chunk; ignore
          } else {
            throw e;
          }
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      // User cancelled
      if (conv.messages[msgIdx].content === '') {
        conv.messages.pop(); // Remove empty assistant message
      }
    } else {
      conv.messages[msgIdx].content = '';
      conv.messages.pop(); // Remove empty assistant message
      // Show error
      const inner = document.getElementById('chat-inner');
      const errDiv = document.createElement('div');
      errDiv.className = 'message-error';
      errDiv.textContent = 'Error: ' + e.message;
      inner.appendChild(errDiv);
      scrollToBottom();
    }
  }

  // Remove streaming cursor
  if (lastContent) lastContent.classList.remove('streaming-cursor');

  isStreaming = false;
  abortController = null;
  updateSendButton();
  saveConversations();
  document.getElementById('chat-input').focus();
}

function stopStreaming() {
  if (abortController) {
    abortController.abort();
  }
}

function updateSendButton() {
  const btn = document.getElementById('send-btn');
  if (isStreaming) {
    btn.innerHTML = '&#9632;'; // Stop square
    btn.className = 'send-btn stop-btn';
    btn.onclick = stopStreaming;
    btn.disabled = false;
    btn.title = 'Stop generation';
  } else {
    btn.innerHTML = '&#9654;'; // Play triangle
    btn.className = 'send-btn';
    btn.onclick = sendMessage;
    btn.disabled = false;
    btn.title = 'Send';
  }
}

/* ── Input handling ────────────────────────────────────────────────── */
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (isStreaming) return;
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

/* ── Sidebar (mobile) ──────────────────────────────────────────────── */
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
}

/* ── Init ──────────────────────────────────────────────────────────── */
initTheme();
loadModels();

if (conversations.length > 0) {
  activeConvId = conversations[0].id;
} else {
  newConversation();
}
renderConvList();
renderMessages();
document.getElementById('chat-input').focus();
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse)
async def playground_ui() -> HTMLResponse:
    """Serve the built-in chat playground."""
    return HTMLResponse(content=_PLAYGROUND_HTML)
