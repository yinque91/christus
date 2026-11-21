const $ = (s) => document.querySelector(s);
const state = {
  userId: localStorage.getItem("christus_user") || "local-user",
  sessionId: localStorage.getItem("christus_session") || null,
  busy: false,
};
localStorage.setItem("christus_user", state.userId);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}
function fmtTime(t) {
  if (!t) return "";
  return new Date(t * 1000).toLocaleString("zh-CN", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"});
}
async function api(path, opts={}) {
  const r = await fetch(path, {headers:{"Content-Type":"application/json"}, ...opts});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function setStatus(ok, text) {
  $("#dot").classList.toggle("ok", !!ok);
  $("#statusText").textContent = text;
}

function showView(name) {
  document.querySelectorAll(".nav button").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + name));
  const titles = {chat:["交流","和同一个持续主体对话"], state:["主体状态","真实运行数据，不是装饰性指标"],
    history:["自身历史","互动、经历与来源"], learning:["学习","知识摄取与纠正记录"],
    web:["网络信息","她从哪里学到的"], settings:["系统","运行、数据与安全"]};
  $("#viewTitle").textContent = titles[name][0];
  $("#viewSub").textContent = titles[name][1];
  if (name === "state") loadState();
  if (name === "history") loadHistory();
  if (name === "learning") loadLearning();
  if (name === "web") loadKnowledge();
  if (name === "settings") loadSettings();
}

function renderMessage(m) {
  const row = document.createElement("div");
  row.className = "msg " + (m.role === "user" ? "user" : "assistant");
  const src = m.source === "user" ? "human" : m.source;
  const cites = (m.citations || []).map(c => `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title || c.url)}</a>`).join(" · ");
  row.innerHTML = `<div class="avatar">${m.role === "user" ? "你" : "C"}</div>
    <div><div class="bubble">${esc(m.text)}</div>
    <div class="meta"><span>${esc(src || "local")}</span>${cites ? `<span>${cites}</span>` : ""}</div></div>`;
  $("#messages").appendChild(row);
  $("#welcome").style.display = "none";
  $("#chat").scrollTop = $("#chat").scrollHeight;
}

async function loadSessions() {
  try {
    const d = await api(`/api/sessions?user_id=${encodeURIComponent(state.userId)}`);
    const box = $("#sessionList"); box.innerHTML = "";
    for (const s of d.sessions || []) {
      const b = document.createElement("button");
      b.className = "session" + (s.id === state.sessionId ? " active" : "");
      b.textContent = s.title || "新的对话";
      b.title = fmtTime(s.updated_at);
      b.onclick = () => openSession(s.id);
      box.appendChild(b);
    }
  } catch (e) { console.warn(e); }
}

async function openSession(id) {
  state.sessionId = id;
  localStorage.setItem("christus_session", id);
  $("#messages").innerHTML = "";
  $("#welcome").style.display = "none";
  try {
    const d = await api(`/api/history?session_id=${encodeURIComponent(id)}`);
    for (const m of d.messages || []) renderMessage(m);
    if (!d.messages || !d.messages.length) $("#welcome").style.display = "block";
    await loadSessions();
  } catch (e) { console.warn(e); }
}

async function loadState() {
  try {
    const s = await api("/api/state");
    setStatus(true, "主体运行中");
    $("#subjectPill").textContent = "subject " + (s.subject_id || "—").slice(0, 10);
    $("#stStep").textContent = s.step ?? "—";
    $("#stH").textContent = Number(s.h_norm || 0).toFixed(3);
    $("#stSubject").innerHTML = `<span>action</span><b>${esc(s.action)}</b><span>turns</span><b>${s.turns}</b><span>reward</span><b>${Number(s.reward).toFixed(3)}</b><span>checkpoint</span><b>${s.last_checkpoint ? "saved" : "—"}</b>`;
    $("#stCore").innerHTML = `<span>W/P/Q</span><b>${s.core_weights_unchanged ? "unchanged" : "changed"}</b><span>language</span><b>${esc(s.language_source)}</b>`;
    const aff = $("#stAffect"); aff.innerHTML = "";
    for (const [k,v] of Object.entries(s.affect || {})) {
      const n = Math.max(0, Math.min(1, Number(v)));
      aff.insertAdjacentHTML("beforeend", `<div class="barRow"><span>${esc(k)}</span><div class="bar"><i style="width:${n*100}%"></i></div><span>${n.toFixed(2)}</span></div>`);
    }
    const mem = s.memory || {};
    $("#stMemory").innerHTML = Object.entries(mem).map(([k,v]) => `<span>${esc(k)}</span><b>${esc(v)}</b>`).join("");
    const con = s.concept || {};
    $("#stConcept").innerHTML = Object.entries(con).map(([k,v]) => `<span>${esc(k)}</span><b>${esc(v)}</b>`).join("");
    const sm = s.self_model || {};
    $("#stSelf").innerHTML = sm.available
      ? `<span>mode</span><b>${esc(sm.mode)}</b><span>source probe</span><b>${sm.source_probe}</b><span>temporal</span><b>${sm.temporal}</b><span>integration</span><b>${esc(sm.note)}</b>`
      : `<span>status</span><b>not available</b>`;
    $("#langSource").textContent = "language: " + (s.language_source || "—");
    $("#turnHint").textContent = `${s.turns || 0} turns · step ${s.step || 0}`;
    $("#pauseBtn").textContent = s.paused ? "恢复学习" : "暂停学习";
  } catch (e) { setStatus(false, "连接失败"); console.warn(e); }
}

async function loadHistory() {
  const body = $("#historyBody"); body.innerHTML = "";
  try {
    const d = await api(`/api/sessions?user_id=${encodeURIComponent(state.userId)}`);
    for (const s of d.sessions || []) {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = `<b>${esc(s.title)}</b><small>${fmtTime(s.updated_at)} · ${esc(s.id)}</small>`;
      div.onclick = () => { showView("chat"); openSession(s.id); };
      body.appendChild(div);
    }
    if (!body.children.length) body.innerHTML = '<div class="item muted">还没有历史。</div>';
  } catch (e) { body.innerHTML = `<div class="item">${esc(e.message)}</div>`; }
}

async function loadLearning() {
  const body = $("#learningBody"); body.innerHTML = "";
  try {
    const d = await api("/api/learning");
    for (const ev of d.events || []) {
      const div = document.createElement("div"); div.className = "item";
      const c = ev.content || {};
      const text = c.text || c.query || c.reply || JSON.stringify(c).slice(0, 240);
      div.innerHTML = `<span class="tag">${esc(ev.kind)}</span><b>${esc(text).slice(0, 220)}</b><small>${fmtTime(ev.created_at)}</small>`;
      body.appendChild(div);
    }
    if (!body.children.length) body.innerHTML = '<div class="item muted">还没有学习事件。</div>';
  } catch (e) { body.innerHTML = `<div class="item">${esc(e.message)}</div>`; }
}

async function loadKnowledge() {
  const body = $("#knowledgeBody"); body.innerHTML = "";
  try {
    const d = await api("/api/knowledge");
    for (const k of d.items || []) {
      const div = document.createElement("div"); div.className = "item";
      div.innerHTML = `<b>${esc(k.title || k.query)}</b><small>${esc(k.license)} · ${esc(k.trust)} · ${fmtTime(k.retrieved_at)}</small><p class="prose">${esc((k.summary || "").slice(0, 260))}</p>${k.url ? `<a href="${esc(k.url)}" target="_blank" rel="noopener" style="color:#68c8ba;font-size:11px">${esc(k.url)}</a>` : ""}`;
      body.appendChild(div);
    }
    if (!body.children.length) body.innerHTML = '<div class="item muted">还没有网络知识记录。</div>';
  } catch (e) { body.innerHTML = `<div class="item">${esc(e.message)}</div>`; }
}

async function loadSettings() {
  try {
    const d = await api("/api/settings");
    $("#settingsBody").innerHTML = Object.entries(d).map(([k,v]) => `<span>${esc(k)}</span><b>${esc(v)}</b>`).join("");
  } catch (e) { $("#settingsBody").textContent = e.message; }
}

async function sendMessage() {
  if (state.busy) return;
  const input = $("#input"); const text = input.value.trim();
  if (!text) return;
  state.busy = true; input.value = ""; input.style.height = "auto";
  $("#welcome").style.display = "none";
  renderMessage({role:"user", text, source:"human"});
  const assistant = document.createElement("div");
  assistant.className = "msg assistant";
  assistant.innerHTML = `<div class="avatar">C</div><div><div class="bubble"></div><div class="meta"><span>streaming</span></div></div>`;
  $("#messages").appendChild(assistant);
  const bubble = assistant.querySelector(".bubble");
  let sid = state.sessionId;
  try {
    const r = await fetch("/api/chat/stream", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({text, session_id: sid, user_id: state.userId})});
    if (!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream:true});
      const parts = buf.split("\n\n"); buf = parts.pop();
      for (const p of parts) {
        const line = p.replace(/^data:\s*/, "").trim(); if (!line) continue;
        const d = JSON.parse(line);
        if (d.delta) { bubble.textContent += d.delta; $("#chat").scrollTop = $("#chat").scrollHeight; }
        if (d.error) { bubble.textContent += "\n[error] " + d.error; }
        if (d.done) {
          sid = d.session_id; state.sessionId = sid; localStorage.setItem("christus_session", sid);
          const meta = assistant.querySelector(".meta");
          const cites = (d.citations || []).map(c => `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title || c.url)}</a>`).join(" · ");
          meta.innerHTML = `<span>${esc(d.source)}</span>${cites ? `<span>${cites}</span>` : ""}`;
        }
      }
    }
    await loadSessions(); await loadState();
  } catch (e) {
    bubble.textContent += "\n请求失败：" + e.message;
  } finally {
    state.busy = false; input.focus();
  }
}

document.querySelectorAll(".nav button").forEach(b => b.onclick = () => showView(b.dataset.view));
document.querySelectorAll("[data-suggest]").forEach(b => b.onclick = () => { $("#input").value = b.dataset.suggest; sendMessage(); });
$("#newChat").onclick = () => {
  state.sessionId = null; localStorage.removeItem("christus_session");
  $("#messages").innerHTML = ""; $("#welcome").style.display = "block"; showView("chat"); loadSessions();
};
$("#send").onclick = sendMessage;
$("#input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
$("#input").addEventListener("input", e => { e.target.style.height = "auto"; e.target.style.height = Math.min(140, e.target.scrollHeight) + "px"; });
$("#checkpointBtn").onclick = async () => { try { const d = await api("/api/checkpoint", {method:"POST", body:"{}"}); alert("检查点已保存：\n" + JSON.stringify(d.checkpoint)); loadState(); } catch (e) { alert(e.message); } };
$("#backupBtn").onclick = async () => { try { const d = await api("/api/backup", {method:"POST", body:"{}"}); alert("备份完成：\n" + d.backup); } catch (e) { alert(e.message); } };
$("#refreshBtn").onclick = loadState;
$("#pauseBtn").onclick = async () => { const cur = $("#pauseBtn").textContent.includes("暂停"); try { await api("/api/pause", {method:"POST", body: JSON.stringify({paused: cur})}); loadState(); } catch (e) { alert(e.message); } };

(async function boot() {
  await loadState(); await loadSessions();
  if (state.sessionId) await openSession(state.sessionId);
})();
