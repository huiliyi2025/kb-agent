import { SSEParser } from "./sse.mjs";

const $ = (id) => document.getElementById(id);
let token = sessionStorage.getItem("kb_token") || "";
let sessionId = sessionStorage.getItem("kb_session") || null;
let busy = false;
let documents = [];
const selected = new Set();

function controls() {
  for (const id of ["question", "send-btn", "agent-btn", "upload-btn", "new-session-btn", "sessions", "logout-btn"])
    $(id).disabled = busy || !token;
  for (const id of ["login-btn", "register-btn"]) $(id).disabled = busy;
  for (const input of $("docs").querySelectorAll("input, button")) input.disabled = busy;
  $("doc-box").classList.toggle("hidden", !token);
  $("logout-btn").classList.toggle("hidden", !token);
  $("username").disabled = Boolean(token);
  $("password").closest(".row").classList.toggle("hidden", Boolean(token));
  $("login-btn").closest(".row").classList.toggle("hidden", Boolean(token));
  $("register-btn").classList.toggle("hidden", Boolean(token));
}

function setSession(value) {
  sessionId = value || null;
  if (sessionId) sessionStorage.setItem("kb_session", sessionId);
  else sessionStorage.removeItem("kb_session");
  $("sessions").value = sessionId || "";
}

function logout() {
  token = "";
  sessionStorage.removeItem("kb_token");
  setSession(null);
  documents = [];
  selected.clear();
  $("docs").replaceChildren();
  $("messages").replaceChildren();
  $("who").textContent = "未登录";
  controls();
}

async function api(path, options = {}) {
  const headers = { ...options.headers };
  if (token) headers.Authorization = "Bearer " + token;
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail)) detail = data.detail.map(x => x.msg).join("；");
    } catch { /* 保留 HTTP 状态 */ }
    if (response.status === 401 && !path.startsWith("/api/auth/")) logout();
    throw new Error(detail);
  }
  return response;
}

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "user" : "bot");
  for (const [className, value] of [["who", role === "user" ? "你" : "kb-agent"], ["bubble", text], ["citations", ""], ["meta-line", ""]]) {
    const child = document.createElement("div");
    child.className = className;
    child.textContent = value;
    el.append(child);
  }
  $("messages").append(el);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function renderCitations(el, citations) {
  const box = el.querySelector(".citations");
  box.replaceChildren();
  for (const cite of citations || []) {
    const card = document.createElement("div");
    card.className = "cite";
    card.textContent = `[${cite.index}] ${[cite.filename, cite.heading, cite.page ? `第 ${cite.page} 页` : null].filter(Boolean).join(" / ")}\n${cite.snippet}`;
    box.append(card);
  }
}

async function loadHealth() {
  try {
    const data = await (await api("/api/health")).json();
    $("health").textContent = `${data.llm} · ${data.embedder}`;
  } catch { $("health").textContent = "服务不可用"; }
}

async function loadDocs() {
  const old = new Set(documents.map(d => d.id));
  documents = await (await api("/api/documents")).json();
  for (const doc of documents) if (!old.has(doc.id) && doc.status === "ready") selected.add(doc.id);
  for (const id of selected) if (!documents.some(d => d.id === id)) selected.delete(id);
  $("docs").replaceChildren();
  for (const doc of documents) {
    const li = document.createElement("li");
    const row = document.createElement("div");
    row.className = "doc-actions";
    const label = document.createElement("label");
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = selected.has(doc.id);
    check.onchange = () => check.checked ? selected.add(doc.id) : selected.delete(doc.id);
    label.append(check, document.createTextNode(doc.filename));
    const remove = document.createElement("button");
    remove.textContent = "删除";
    remove.onclick = async () => {
      if (!confirm(`删除「${doc.filename}」及其索引？`)) return;
      try {
        await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
        await loadDocs();
      } catch (error) { $("upload-err").textContent = error.message; }
    };
    row.append(label, remove);
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${doc.chunk_count} 片段 · ${doc.status}${doc.error ? " · " + doc.error : ""}`;
    li.append(row, meta);
    $("docs").append(li);
  }
  if (!documents.length) $("docs").textContent = "还没有文档，先上传一份。";
  controls();
}

async function loadSessions() {
  const sessions = await (await api("/api/sessions")).json();
  $("sessions").replaceChildren(new Option("新会话", ""));
  for (const session of sessions) $("sessions").add(new Option(session.title, session.id));
  $("sessions").value = sessionId || "";
}

async function restoreSession(id) {
  if (!id) { setSession(null); $("messages").replaceChildren(); return; }
  const messages = await (await api(`/api/sessions/${encodeURIComponent(id)}/messages`)).json();
  setSession(id);
  $("messages").replaceChildren();
  for (const message of messages) {
    const el = addMessage(message.role, message.content);
    if (message.citations) renderCitations(el, JSON.parse(message.citations));
  }
}

async function authenticate(mode) {
  $("auth-err").textContent = "";
  try {
    const data = await (await api(`/api/auth/${mode}`, { method: "POST", body: JSON.stringify({
      username: $("username").value, password: $("password").value,
    }) })).json();
    logout();
    token = data.access_token;
    sessionStorage.setItem("kb_token", token);
    $("who").textContent = data.username;
    $("password").value = "";
    controls();
    await Promise.all([loadDocs(), loadSessions()]);
  } catch (error) { $("auth-err").textContent = error.message; }
}

$("login-btn").onclick = () => authenticate("login");
$("register-btn").onclick = () => authenticate("register");
$("logout-btn").onclick = logout;
$("new-session-btn").onclick = () => restoreSession(null);
$("sessions").onchange = async () => {
  try { await restoreSession($("sessions").value); }
  catch (error) { $("auth-err").textContent = error.message; setSession(null); }
};
$("upload-btn").onclick = async () => {
  const file = $("file").files[0];
  if (!file) { $("upload-err").textContent = "请先选择文件"; return; }
  const form = new FormData();
  form.append("file", file);
  busy = true;
  controls();
  $("upload-err").textContent = "正在解析并建立索引…";
  try {
    const doc = await (await api("/api/documents", { method: "POST", body: form })).json();
    $("upload-err").textContent = doc.status !== "ready" ? `当前状态：${doc.status}，稍后重新上传可查询状态` : doc.reused ? "已复用相同内容的索引" : "索引完成";
  } catch (error) { $("upload-err").textContent = error.message; }
  finally {
    try { await loadDocs(); } catch { /* 失效令牌由 api 处理 */ }
    busy = false;
    controls();
  }
};

async function ask(agent = false) {
  const question = $("question").value.trim();
  if (!question || busy || !token) return;
  busy = true;
  controls();
  $("question").value = "";
  addMessage("user", question);
  const bot = addMessage("assistant", "");
  const bubble = bot.querySelector(".bubble");
  const meta = bot.querySelector(".meta-line");
  meta.textContent = "检索中…";
  try {
    if (agent) {
      const data = await (await api("/api/agent", { method: "POST", body: JSON.stringify({ question }) })).json();
      bubble.textContent = data.answer;
      renderCitations(bot, data.citations);
      meta.textContent = `${data.steps.map(s => s.action).join(" → ")} · ${data.elapsed_ms} ms · Agent 结果不写入会话历史`;
    } else {
      const response = await api("/api/chat/stream", { method: "POST", body: JSON.stringify({
        question, session_id: sessionId, doc_ids: [...selected],
      }) });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let finished = false;
      const parser = new SSEParser((event, data) => {
        if (event === "meta") {
          if (data.session_id) setSession(data.session_id);
          meta.textContent = `检索 ${data.retrieved.length} 条 · 可回答性分数 ${data.confidence} · 生成中，引用待校验`;
        } else if (event === "token") {
          bubble.textContent += data.text;
          $("messages").scrollTop = $("messages").scrollHeight;
        } else if (event === "done") {
          finished = true;
          setSession(data.session_id);
          bubble.textContent = data.answer;
          renderCitations(bot, data.citations);
          meta.textContent = `${data.refused ? "资料不足或引用校验未通过" : "引用编号校验完成，请核对原文"} · ${data.latency_ms} ms · ${data.model}`;
        } else if (event === "error") throw new Error(data.message);
      });
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) { parser.feed(decoder.decode()); break; }
          parser.feed(decoder.decode(value, { stream: true }));
        }
        if (!finished) throw new Error("连接提前中断，回答未完成，请重试");
      } finally { await reader.cancel(); reader.releaseLock(); }
      await loadSessions();
    }
  } catch (error) {
    bubble.textContent = "回答未完成，请重新提交问题。";
    meta.textContent = error.message;
    meta.classList.add("err");
    $("question").value = question;
  } finally { busy = false; controls(); }
}

$("send-btn").onclick = () => ask(false);
$("agent-btn").onclick = () => ask(true);
$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); ask(false); }
});

async function boot() {
  controls();
  await loadHealth();
  if (!token) return;
  try {
    const user = await (await api("/api/me")).json();
    $("who").textContent = user.username;
    $("username").value = user.username;
    await Promise.all([loadDocs(), loadSessions()]);
    if (sessionId) await restoreSession(sessionId);
  } catch (error) { $("auth-err").textContent = error.message; setSession(null); }
}
boot();
