#!/usr/bin/env python3
"""Persistent browser chat UI and reverse proxy for a loopback-only vLLM server.

The gateway owns conversation state. vLLM remains stateless: each request gets a
bounded context assembled from the active session, its rolling summary, and the
user's cross-session memory.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


UPSTREAM_HOST = os.environ.get("VLLM_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("VLLM_UPSTREAM_PORT", "8001"))
LISTEN_HOST = os.environ.get("VLLM_WEB_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("VLLM_WEB_PORT", "8000"))
DB_PATH = os.environ.get("VLLM_CHAT_DB", os.path.expanduser("~/vllm-chat.sqlite3"))

# These are deliberately conservative approximations for a 32K-context model.
# vLLM applies the real tokenizer; keeping the gateway below this budget leaves
# room for the chat template and an 8K completion.
RECENT_CONTEXT_CHARS = 32_000
SESSION_SUMMARY_CHARS = 9_000
USER_MEMORY_CHARS = 6_000
COMPACTION_TRIGGER_CHARS = 52_000
KEEP_RECENT_MESSAGES = 10
MAX_USER_MESSAGE_CHARS = 24_000
MAX_OUTPUT_TOKENS = 8_192

PAGE = rb"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Qwen</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#10131a;color:#edf2f7;font:15px system-ui,sans-serif}.app{height:100vh;display:grid;grid-template-columns:270px minmax(0,1fr)}aside{padding:16px;background:#141925;border-right:1px solid #2d3545;overflow:auto}.brand{font-size:18px;font-weight:750;margin-bottom:16px}.new{width:100%;padding:10px;border:0;border-radius:8px;background:#38bdf8;color:#082f49;font-weight:750;cursor:pointer}.sessions{margin-top:14px}.session-row{display:flex;align-items:center;gap:3px}.session{min-width:0;flex:1;display:block;text-align:left;padding:10px;border:0;border-radius:8px;background:transparent;color:#cbd5e1;cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.session:hover,.session.active{background:#273247;color:white}.session small{display:block;color:#9aa7b8;margin-top:3px}.delete{padding:7px 9px;background:transparent;color:#9aa7b8}.delete:hover{background:#43252b;color:#fca5a5}main{min-width:0;min-height:0;display:flex;flex-direction:column}.top{padding:16px 24px;border-bottom:1px solid #2d3545}.top h1{font-size:18px;margin:0}.note{color:#9aa7b8;font-size:13px;margin:6px 0 0}.chat{min-height:0;flex:1;overflow:auto;padding:12px max(18px,calc((100vw - 980px)/2));white-space:pre-wrap;overflow-wrap:anywhere}.msg{padding:14px 0;border-bottom:1px solid #2d3545}.role{font-weight:700;color:#7dd3fc;margin-bottom:6px}.assistant .role{color:#86efac}.thinking summary{color:#fbbf24;font-weight:700;cursor:pointer}.thinking div{padding-top:8px;color:#cbd5e1}.composer{padding:14px max(18px,calc((100vw - 980px)/2));border-top:1px solid #2d3545;background:#10131a}.composer form{display:flex;gap:10px}.controls{display:flex;align-items:center;gap:8px;margin-top:10px}.controls label{color:#9aa7b8;font-size:13px}.controls select{border:1px solid #3b465a;border-radius:7px;background:#171c26;color:inherit;padding:6px;font:inherit}textarea{flex:1;min-height:68px;resize:vertical;padding:12px;border-radius:8px;border:1px solid #3b465a;background:#171c26;color:inherit;font:inherit}button{padding:0 18px;border:0;border-radius:8px;background:#38bdf8;color:#082f49;font-weight:700;cursor:pointer}button:disabled{opacity:.5}@media(max-width:700px){.app{grid-template-columns:1fr}aside{display:none}.chat,.composer{padding-left:14px;padding-right:14px}}
</style></head><body><div class="app"><aside><div class="brand">Local Qwen</div><button id="new" class="new">+ New chat</button><div id="sessions" class="sessions"></div></aside><main><header class="top"><h1 id="model-name">Local Qwen</h1><p class="note">Chats are saved locally. Public endpoint: do not submit secrets or sensitive code.</p></header><section id="chat" class="chat"></section><div class="composer"><form id="form"><textarea id="prompt" placeholder="Ask Qwen..." required></textarea><button id="send">Send</button></form><div class="controls"><label for="mode">Response mode</label><select id="mode"><option value="auto" selected>Auto</option><option value="think">Think</option><option value="fast">Fast answer</option></select><button id="retry" type="button">Retry last prompt</button><span id="mode-note" class="note">Auto selects reasoning for multi-step work.</span></div></div></main></div>
<script>
const USER_ID='0',chat=document.querySelector('#chat'),sessionsEl=document.querySelector('#sessions'),form=document.querySelector('#form'),prompt=document.querySelector('#prompt'),send=document.querySelector('#send'),retry=document.querySelector('#retry'),mode=document.querySelector('#mode'),modeNote=document.querySelector('#mode-note'),modelName=document.querySelector('#model-name');
let activeModel='',currentSession=null,sessionPoll=null;
const THINK_PATTERN=/```|\b(debug|diagnos|traceback|error|bug|implement|code|program|algorithm|complexity|math|calculate|deriv|proof|reason|analy[sz]|compare|trade[- ]?off|design|architect|plan|strateg|optim[isz]|step[- ]by[- ]step|why)\b|[=<>]{1,3}|\b(if|while|for)\s*\(/i;
async function api(path,options={}){const r=await fetch(path,options);if(!r.ok)throw new Error((await r.text())||r.statusText);return r;}
async function loadModel(){try{const r=await api('/v1/models');const data=await r.json();activeModel=data.data?.[0]?.id||'';if(activeModel)modelName.textContent=activeModel;}catch(_){modelName.textContent='Local Qwen (model unavailable)';}return activeModel;}
let followStream=true;
chat.addEventListener('scroll',()=>{followStream=chat.scrollHeight-chat.scrollTop-chat.clientHeight<80;});
function scrollLatest(force=false){if(force||followStream)requestAnimationFrame(()=>{chat.scrollTop=chat.scrollHeight;});}
function add(role,text){const el=document.createElement('div');el.className='msg '+role;el.innerHTML='<div class="role"></div><div></div>';el.firstChild.textContent=role;el.lastChild.textContent=text;chat.append(el);scrollLatest();return el.lastChild;}
function addThinking(text='',open=true){const el=document.createElement('details');el.className='msg thinking';el.open=open;el.innerHTML='<summary>Thinking</summary><div></div>';el.lastChild.textContent=text;chat.append(el);scrollLatest();return {el,content:el.lastChild};}
function clearChat(){chat.replaceChildren();}
function renderMessage(m){if(m.role==='assistant'&&m.reasoning)addThinking(m.reasoning,m.status==='streaming');const text=m.status==='streaming'&&!m.content?'(Generating...)':m.content;add(m.role,text);}
function visibleReply(content){return (content||'').replace(/<think>[\s\S]*?<\/think>\s*/g,'').trim()||'(No visible response)';}
function selectedMode(text){if(mode.value==='think')return 'think';if(mode.value==='fast')return 'fast';return THINK_PATTERN.test(text)||text.length>500?'think':'fast';}
function updateModeNote(){modeNote.textContent=mode.value==='auto'?'Auto selects reasoning for multi-step work.':mode.value==='think'?'Reasoning is enabled for every request.':'Reasoning is disabled for faster replies.';}
mode.addEventListener('change',updateModeNote);
async function refreshSessions(){const r=await api('/api/sessions?user_id='+encodeURIComponent(USER_ID));const data=await r.json();sessionsEl.replaceChildren();for(const s of data.sessions){const row=document.createElement('div');row.className='session-row';const b=document.createElement('button');b.className='session'+(s.id===currentSession?' active':'');b.innerHTML='<span></span><small></small>';b.firstChild.textContent=s.title;b.lastChild.textContent=new Date(s.updated_at*1000).toLocaleString();b.onclick=()=>loadSession(s.id);const del=document.createElement('button');del.className='delete';del.title='Delete chat';del.innerHTML='&times;';del.onclick=async e=>{e.stopPropagation();if(confirm('Delete this chat permanently?'))await deleteSession(s.id);};row.append(b,del);sessionsEl.append(row);}if(!currentSession&&data.sessions[0])await loadSession(data.sessions[0].id);}
async function newSession(){const r=await api('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:USER_ID})});const s=await r.json();currentSession=s.id;await refreshSessions();await loadSession(s.id);}
async function loadSession(id){if(sessionPoll)clearTimeout(sessionPoll);const r=await api('/api/sessions/'+encodeURIComponent(id)+'?user_id='+encodeURIComponent(USER_ID));const s=await r.json();currentSession=s.id;followStream=true;clearChat();for(const m of s.messages)renderMessage(m);scrollLatest(true);await refreshSessions();if(s.messages.some(m=>m.status==='streaming'))sessionPoll=setTimeout(()=>{if(currentSession===id)loadSession(id).catch(showError);},700);}
async function deleteSession(id){await api('/api/sessions/'+encodeURIComponent(id)+'?user_id='+encodeURIComponent(USER_ID),{method:'DELETE'});if(currentSession===id){currentSession=null;clearChat();}await refreshSessions();if(!currentSession)await newSession();}
document.querySelector('#new').onclick=()=>newSession().catch(showError);
function showError(err){add('assistant','Request failed: '+err.message);}
function consumeSSE(buffer,handle){const events=buffer.split('\n\n');const tail=events.pop();for(const event of events){const data=event.split('\n').filter(line=>line.startsWith('data:')).map(line=>line.slice(5).trim()).join('');if(data&&data!=='[DONE]')handle(JSON.parse(data));}return tail;}
function removeLastAssistantDisplay(){while(chat.lastElementChild&&(chat.lastElementChild.classList.contains('assistant')||chat.lastElementChild.classList.contains('thinking')))chat.lastElementChild.remove();}
async function runChat(payload,retryMode=false){try{if(!currentSession)await newSession();if(!activeModel)await loadModel();if(!activeModel)throw new Error('vLLM is unavailable');if(!retryMode){add('user',payload.content);prompt.value='';}else removeLastAssistantDisplay();send.disabled=true;retry.disabled=true;const answer=add('assistant','');let thinking=null,raw='',buffer='',finish='';const r=await api('/api/sessions/'+encodeURIComponent(currentSession)+'/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...payload,user_id:USER_ID,model:activeModel})});if(r.headers.get('X-Context-Compressed')==='1')add('assistant','[Earlier messages were compacted into this chat\'s summary.]');const reader=r.body.getReader(),decoder=new TextDecoder();for(;;){const next=await reader.read();if(next.done)break;buffer+=decoder.decode(next.value,{stream:true});buffer=consumeSSE(buffer,chunk=>{const choice=chunk.choices?.[0];if(!choice)return;finish=choice.finish_reason||finish;const reasoning=choice.delta?.reasoning_content||choice.delta?.reasoning||'';if(reasoning){thinking??=addThinking();thinking.content.textContent+=reasoning;scrollLatest();}raw+=choice.delta?.content||'';answer.textContent=visibleReply(raw);scrollLatest();});}let reply=visibleReply(raw);if(finish==='length')reply+='\n\n[Response stopped at the context/output limit.]';answer.textContent=reply;if(thinking)thinking.el.open=false;await refreshSessions();}catch(err){if(retryMode)loadSession(currentSession).catch(showError);showError(err)}finally{send.disabled=false;retry.disabled=false;prompt.focus();}}
form.addEventListener('submit',async e=>{e.preventDefault();const text=prompt.value.trim();if(text)await runChat({content:text,mode:selectedMode(text)});});
retry.onclick=()=>runChat({retry:true,mode:mode.value==='think'?'think':mode.value==='fast'?'fast':'auto'},true);
loadModel();refreshSessions().catch(showError);
</script></body></html>"""

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}
THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
AUTO_THINK = re.compile(r"```|\b(debug|diagnos|traceback|error|bug|implement|code|program|algorithm|complexity|math|calculate|deriv|proof|reason|analy[sz]|compare|trade[- ]?off|design|architect|plan|strateg|optim[isz]|step[- ]by[- ]step|why)\b|[=<>]{1,3}|\b(if|while|for)\s*\(", re.IGNORECASE)
SESSION_LOCKS: dict[str, threading.Lock] = {}
SESSION_LOCKS_GUARD = threading.Lock()


def now() -> int:
    return int(time.time())


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def session_lock(session_id: str) -> threading.Lock:
    """Serialize turns in one chat while allowing different chats to run together."""
    with SESSION_LOCKS_GUARD:
        return SESSION_LOCKS.setdefault(session_id, threading.Lock())


def init_db() -> None:
    with db() as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                compressed_until_id INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                reasoning TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'complete' CHECK(status IN ('streaming', 'complete', 'error')),
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_session_id ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT PRIMARY KEY,
                memory TEXT NOT NULL DEFAULT '',
                last_message_id INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS machine_sessions (
                user_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id)
            );
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(messages)")}
        if "reasoning" not in columns:
            connection.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT NOT NULL DEFAULT ''")
        if "status" not in columns:
            connection.execute("ALTER TABLE messages ADD COLUMN status TEXT NOT NULL DEFAULT 'complete'")


def create_session(user_id: str, title: str = "New chat") -> dict:
    session = {"id": uuid.uuid4().hex, "user_id": user_id, "title": title, "created_at": now(), "updated_at": now()}
    with db() as connection:
        connection.execute(
            "INSERT INTO sessions(id,user_id,title,created_at,updated_at) VALUES(:id,:user_id,:title,:created_at,:updated_at)", session
        )
    return session


def session_for_user(session_id: str, user_id: str) -> sqlite3.Row | None:
    with db() as connection:
        return connection.execute("SELECT * FROM sessions WHERE id=? AND user_id=?", (session_id, user_id)).fetchone()


def list_sessions(user_id: str) -> list[dict]:
    with db() as connection:
        rows = connection.execute("SELECT id,title,created_at,updated_at FROM sessions WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall()
    return [dict(row) for row in rows]


def get_messages(session_id: str, after_id: int = 0) -> list[sqlite3.Row]:
    with db() as connection:
        return connection.execute("SELECT id,role,content,reasoning,status,created_at FROM messages WHERE session_id=? AND id>? ORDER BY id", (session_id, after_id)).fetchall()


def insert_message(session_id: str, role: str, content: str, reasoning: str = "", status: str = "complete") -> int:
    timestamp = now()
    with db() as connection:
        cursor = connection.execute("INSERT INTO messages(session_id,role,content,reasoning,status,created_at) VALUES(?,?,?,?,?,?)", (session_id, role, content, reasoning, status, timestamp))
        connection.execute("UPDATE sessions SET updated_at=? WHERE id=?", (timestamp, session_id))
        return int(cursor.lastrowid)


def update_message(message_id: int, content: str, reasoning: str, status: str) -> None:
    with db() as connection:
        connection.execute("UPDATE messages SET content=?,reasoning=?,status=? WHERE id=?", (content, reasoning, status, message_id))


def delete_session(session_id: str, user_id: str) -> bool:
    with db() as connection:
        exists = connection.execute("SELECT 1 FROM sessions WHERE id=? AND user_id=?", (session_id, user_id)).fetchone()
        if not exists:
            return False
        connection.execute("DELETE FROM machine_sessions WHERE session_id=?", (session_id,))
        connection.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return True


def retry_turn(session_id: str) -> tuple[int, str] | None:
    """Remove the latest completed assistant response and return its user turn."""
    rows = get_messages(session_id)
    if not rows or rows[-1]["role"] != "assistant" or rows[-1]["status"] == "streaming":
        return None
    assistant = rows[-1]
    user = next((row for row in reversed(rows[:-1]) if row["role"] == "user"), None)
    if not user:
        return None
    with db() as connection:
        connection.execute("DELETE FROM messages WHERE id=?", (assistant["id"],))
        connection.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), session_id))
    return int(user["id"]), str(user["content"])


def choose_mode(content: str, requested: str) -> str:
    if requested == "think":
        return "think"
    if requested == "auto" and (len(content) > 500 or AUTO_THINK.search(content)):
        return "think"
    return "fast"


def title_session_if_needed(session_id: str, user_text: str) -> None:
    with db() as connection:
        row = connection.execute("SELECT title FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row and row["title"] == "New chat":
            title = " ".join(user_text.split())[:72] or "New chat"
            connection.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))


def get_user_memory(user_id: str) -> str:
    with db() as connection:
        row = connection.execute("SELECT memory FROM user_memory WHERE user_id=?", (user_id,)).fetchone()
    return row["memory"] if row else ""


def vllm_json(method: str, path: str, payload: dict | None = None, timeout: int = 900) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    connection = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        response = connection.getresponse()
        raw = response.read()
        if response.status >= 300:
            raise RuntimeError(f"vLLM returned {response.status}: {raw.decode(errors='replace')[:500]}")
        return json.loads(raw)
    finally:
        connection.close()


def default_model() -> str:
    models = vllm_json("GET", "/v1/models")
    data = models.get("data", [])
    if not data:
        raise RuntimeError("vLLM did not report a model")
    return str(data[0]["id"])


def complete(model: str, messages: list[dict], max_tokens: int) -> str:
    response = vllm_json("POST", "/v1/chat/completions", {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens, "stream": False})
    message = response["choices"][0]["message"]
    return str(message.get("content") or message.get("reasoning_content") or "")


def maybe_compact_session(session_id: str, model: str) -> bool:
    """Summarize older messages while retaining all originals for the UI."""
    with db() as connection:
        session = connection.execute("SELECT summary,compressed_until_id FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not session:
        return False
    pending = get_messages(session_id, int(session["compressed_until_id"]))
    if len(pending) <= KEEP_RECENT_MESSAGES or sum(len(row["content"]) for row in pending) < COMPACTION_TRIGGER_CHARS:
        return False
    source = pending[:-KEEP_RECENT_MESSAGES]
    transcript = "\n\n".join(f"{row['role'].upper()}: {row['content']}" for row in source)
    prompt = [
        {"role": "system", "content": "Maintain a compact, factual conversation summary. Preserve decisions, requirements, unresolved work, and user preferences. Do not invent facts. This is internal context, not a response to the user."},
        {"role": "user", "content": f"Existing summary:\n{session['summary'] or '(none)'}\n\nMessages to fold in:\n{transcript}\n\nWrite an updated summary under 1,500 words. /no_think"},
    ]
    summary = complete(model, prompt, 1_500).strip()[:SESSION_SUMMARY_CHARS]
    if not summary:
        return False
    with db() as connection:
        connection.execute("UPDATE sessions SET summary=?,compressed_until_id=?,updated_at=? WHERE id=?", (summary, source[-1]["id"], now(), session_id))
    return True


def maybe_update_user_memory(user_id: str, session_id: str, model: str) -> None:
    """Periodically update durable cross-session memory after a completed turn."""
    with db() as connection:
        row = connection.execute("SELECT memory,last_message_id FROM user_memory WHERE user_id=?", (user_id,)).fetchone()
        latest = connection.execute("SELECT MAX(id) AS id FROM messages WHERE session_id=?", (session_id,)).fetchone()["id"] or 0
    previous_id = int(row["last_message_id"]) if row else 0
    if latest - previous_id < 8:
        return
    recent = get_messages(session_id, max(previous_id, latest - 16))
    transcript = "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent)
    prompt = [
        {"role": "system", "content": "Maintain durable user memory across chats. Include only stable facts, preferences, projects, and explicitly requested follow-ups stated by the user. Exclude secrets, transient questions, model claims, and guesses. Keep it concise."},
        {"role": "user", "content": f"Existing memory:\n{row['memory'] if row else '(none)'}\n\nRecent conversation:\n{transcript}\n\nReturn the revised memory as bullets, under 800 words. /no_think"},
    ]
    memory = complete(model, prompt, 800).strip()[:USER_MEMORY_CHARS]
    if not memory:
        return
    with db() as connection:
        connection.execute(
            "INSERT INTO user_memory(user_id,memory,last_message_id,updated_at) VALUES(?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET memory=excluded.memory,last_message_id=excluded.last_message_id,updated_at=excluded.updated_at",
            (user_id, memory, latest, now()),
        )


def build_context(session_id: str, user_id: str, current_message_id: int, mode: str) -> list[dict]:
    with db() as connection:
        session = connection.execute("SELECT summary,compressed_until_id FROM sessions WHERE id=?", (session_id,)).fetchone()
    summary = (session["summary"] if session else "")[-SESSION_SUMMARY_CHARS:]
    memory = get_user_memory(user_id)[-USER_MEMORY_CHARS:]
    rows = get_messages(session_id, int(session["compressed_until_id"]) if session else 0)
    selected: list[sqlite3.Row] = []
    used = 0
    for row in reversed(rows):
        size = len(row["content"])
        if selected and used + size > RECENT_CONTEXT_CHARS:
            break
        selected.append(row)
        used += size
    selected.reverse()
    system = "You are a helpful assistant. Use the supplied conversation summary and user memory only as context; do not mention them unless asked."
    if memory:
        system += f"\n\nDurable user memory:\n{memory}"
    if summary:
        system += f"\n\nSummary of earlier messages in this chat:\n{summary}"
    messages: list[dict] = [{"role": "system", "content": system}]
    for row in selected:
        content = row["content"]
        if row["id"] == current_message_id:
            content += "\n\n/think" if mode == "think" else "\n\n/no_think"
        messages.append({"role": row["role"], "content": content})
    return messages


def visible_content(raw: str) -> str:
    return THINK_BLOCK.sub("", raw).strip() or "(No visible response)"


def machine_session(user_id: str) -> str:
    with db() as connection:
        row = connection.execute("SELECT session_id FROM machine_sessions WHERE user_id=?", (user_id,)).fetchone()
    if row and session_for_user(row["session_id"], user_id):
        return str(row["session_id"])
    session = create_session(user_id, "Machine context")
    with db() as connection:
        connection.execute("INSERT INTO machine_sessions(user_id,session_id) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET session_id=excluded.session_id", (user_id, session["id"]))
    return session["id"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if not size:
            return {}
        return json.loads(self.rfile.read(size))

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if parsed.path == "/api/sessions":
            user_id = parse_qs(parsed.query).get("user_id", ["0"])[0]
            self.send_json(200, {"sessions": list_sessions(user_id)})
            return
        if parsed.path.startswith("/api/sessions/"):
            session_id = parsed.path.removeprefix("/api/sessions/").split("/", 1)[0]
            user_id = parse_qs(parsed.query).get("user_id", ["0"])[0]
            session = session_for_user(session_id, user_id)
            if not session:
                self.send_json(404, {"error": "Session not found"})
                return
            self.send_json(200, {"id": session["id"], "title": session["title"], "messages": [dict(row) for row in get_messages(session_id)]})
            return
        self.proxy()

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/api/sessions":
                data = self.read_json()
                session = create_session(str(data.get("user_id", "0")))
                self.send_json(201, session)
                return
            if parsed.path == "/api/machine-chat":
                data = self.read_json()
                user_id = str(data.get("user_id", "0"))
                self.stream_chat(machine_session(user_id), user_id, data)
                return
            if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/chat"):
                session_id = parsed.path.removeprefix("/api/sessions/").removesuffix("/chat").rstrip("/")
                data = self.read_json()
                user_id = str(data.get("user_id", "0"))
                if not session_for_user(session_id, user_id):
                    self.send_json(404, {"error": "Session not found"})
                    return
                self.stream_chat(session_id, user_id, data)
                return
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        self.proxy()

    def do_DELETE(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path.startswith("/api/sessions/"):
            session_id = parsed.path.removeprefix("/api/sessions/").split("/", 1)[0]
            user_id = parse_qs(parsed.query).get("user_id", ["0"])[0]
            with session_lock(session_id):
                deleted = delete_session(session_id, user_id)
            if deleted:
                self.send_json(200, {"deleted": session_id})
            else:
                self.send_json(404, {"error": "Session not found"})
            return
        self.send_json(404, {"error": "Not found"})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def stream_chat(self, session_id: str, user_id: str, payload: dict) -> None:
        retry = bool(payload.get("retry"))
        content = str(payload.get("content", "")).strip()
        if not content and not retry:
            self.send_json(400, {"error": "content is required"})
            return
        if content and len(content) > MAX_USER_MESSAGE_CHARS:
            self.send_json(413, {"error": f"Message is too large; limit is {MAX_USER_MESSAGE_CHARS} characters."})
            return
        requested_mode = str(payload.get("mode", "auto"))
        # Each session is a sequential conversation. Separate sessions still
        # stream concurrently, but two turns in the same session are ordered.
        with session_lock(session_id):
            model = str(payload.get("model") or default_model())
            if retry:
                previous_turn = retry_turn(session_id)
                if not previous_turn:
                    self.send_json(409, {"error": "Only a completed latest response can be retried."})
                    return
                user_message_id, content = previous_turn
            else:
                user_message_id = insert_message(session_id, "user", content)
                title_session_if_needed(session_id, content)
            mode = choose_mode(content, requested_mode)
            compacted = maybe_compact_session(session_id, model)
            messages = build_context(session_id, user_id, user_message_id, mode)
            assistant_message_id = insert_message(session_id, "assistant", "", status="streaming")
            body = json.dumps({"model": model, "messages": messages, "temperature": 0.7, "max_tokens": MAX_OUTPUT_TOKENS, "stream": True}).encode()
            connection = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=900)
            raw_answer = ""
            raw_reasoning = ""
            buffer = ""
            last_save = 0.0
            try:
                connection.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                if response.status >= 300:
                    error = response.read().decode(errors="replace")
                    update_message(assistant_message_id, f"[Generation failed: {error[:500]}]", "", "error")
                    self.send_json(response.status, {"error": error})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Context-Compressed", "1" if compacted else "0")
                self.end_headers()
                while data := response.read(8192):
                    self.wfile.write(data)
                    self.wfile.flush()
                    buffer += data.decode("utf-8", errors="replace")
                    events = buffer.split("\n\n")
                    buffer = events.pop()
                    for event in events:
                        lines = [line[5:].strip() for line in event.splitlines() if line.startswith("data:")]
                        joined = "".join(lines)
                        if not joined or joined == "[DONE]":
                            continue
                        try:
                            choice = json.loads(joined).get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            raw_answer += delta.get("content", "") or ""
                            raw_reasoning += delta.get("reasoning_content", "") or delta.get("reasoning", "") or ""
                        except (ValueError, IndexError, TypeError):
                            pass
                    if time.monotonic() - last_save >= 0.5:
                        progress = visible_content(raw_answer) if raw_answer else ("(Thinking...)" if raw_reasoning else "(Generating...)")
                        update_message(assistant_message_id, progress, raw_reasoning, "streaming")
                        last_save = time.monotonic()
                answer = visible_content(raw_answer)
                update_message(assistant_message_id, answer, raw_reasoning, "complete")
                threading.Thread(target=self.maintain_context, args=(user_id, session_id, model), daemon=True).start()
            except OSError as exc:
                update_message(assistant_message_id, f"[Generation interrupted: {exc}]", raw_reasoning, "error")
                if not self.wfile.closed:
                    self.send_json(502, {"error": f"vLLM upstream unavailable: {exc}"})
            finally:
                connection.close()
                self.close_connection = True

    @staticmethod
    def maintain_context(user_id: str, session_id: str, model: str) -> None:
        try:
            maybe_compact_session(session_id, model)
            maybe_update_user_memory(user_id, session_id, model)
        except Exception as exc:  # Context maintenance must never break a reply.
            print(f"Context maintenance failed: {exc}", flush=True)

    def proxy(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size) if size else None
        headers = {key: value for key, value in self.headers.items() if key.lower() not in HOP_BY_HOP | {"host", "content-length"}}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        connection = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=900)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while data := response.read(8192):
                self.wfile.write(data)
                self.wfile.flush()
            self.close_connection = True
        except OSError as exc:
            message = f"vLLM upstream unavailable: {exc}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
        finally:
            connection.close()


if __name__ == "__main__":
    init_db()
    print(f"Serving persistent UI on http://{LISTEN_HOST}:{LISTEN_PORT}; proxying to {UPSTREAM_HOST}:{UPSTREAM_PORT}; database={DB_PATH}", flush=True)
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
