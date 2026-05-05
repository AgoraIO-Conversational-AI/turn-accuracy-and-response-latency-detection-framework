const state = {
  ws: null,
  turns: [],
  results: {},  // turn index -> result
  currentTurn: null,
  running: false,
  batchRunning: false,
};

// --- WebSocket ---

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  state.ws = new WebSocket(`${proto}//${location.host}/ws`);

  state.ws.onopen = () => {
    document.getElementById("connection-status").className = "status-dot connected";
    document.getElementById("connection-status").title = "Connected";
  };

  state.ws.onclose = () => {
    document.getElementById("connection-status").className = "status-dot disconnected";
    document.getElementById("connection-status").title = "Disconnected";
    setTimeout(connect, 2000);
  };

  state.ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMessage(msg);
  };
}

function send(msg) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(msg));
  }
}

// --- Message Handling ---

function handleMessage(msg) {
  switch (msg.type) {
    case "init":
      renderDevices(msg.devices);
      break;

    case "source_changed":
      renderSourceSelect(msg.sources);
      state.results = {};
      renderSummary({});
      loadTurns();
      break;

    case "run_start":
      state.running = true;
      state.batchRunning = true;
      state.results = {};
      updateControls();
      break;

    case "turn_start":
      state.currentTurn = msg.turn;
      state.running = true;
      setTurnStatus(msg.turn, "playing");
      renderCurrentTurn(msg);
      highlightRow(msg.turn);
      updateControls();
      break;

    case "waiting_response":
      setTurnStatus(msg.turn, "listening");
      updateCurrentTurnMeta("Waiting for agent response...");
      break;

    case "response_detected":
      setTurnTtfa(msg.turn, msg.ttfa_ms);
      updateCurrentTurnMeta(`TTFA: ${msg.ttfa_ms.toFixed(0)}ms — recording response...`);
      break;

    case "turn_done":
      state.results[msg.turn] = msg;
      state.currentTurn = null;
      // only clear running if not in a batch run
      if (!state.batchRunning) {
        state.running = false;
        updateControls();
      }
      setTurnStatus(msg.turn, msg.status);
      if (msg.ttfa_ms != null) {
        setTurnTtfa(msg.turn, msg.ttfa_ms);
      }
      if (msg.summary) {
        renderSummary(msg.summary);
      }
      clearHighlight();
      break;

    case "run_complete":
      state.running = false;
      state.batchRunning = false;
      state.currentTurn = null;
      updateControls();
      renderSummary(msg);
      renderCurrentTurnIdle();
      break;

    case "reset":
      state.results = {};
      state.running = false;
      state.batchRunning = false;
      state.currentTurn = null;
      updateControls();
      renderSummary({});
      renderCurrentTurnIdle();
      resetTableStatuses();
      break;

    case "stopped":
      state.running = false;
      state.batchRunning = false;
      state.currentTurn = null;
      updateControls();
      clearHighlight();
      break;

    case "results":
      renderSummary(msg);
      break;
  }
}

// --- Load Sources & Turns ---

async function loadSources() {
  const resp = await fetch("/api/sources");
  const data = await resp.json();
  renderSourceSelect(data.sources);
}

function renderSourceSelect(sources) {
  const sel = document.getElementById("source-select");
  sel.innerHTML = "";
  for (const src of sources) {
    const opt = document.createElement("option");
    opt.value = src.key;
    opt.textContent = src.label;
    if (src.active) opt.selected = true;
    sel.appendChild(opt);
  }
}

function changeSource() {
  const sourceKey = document.getElementById("source-select").value;
  send({ action: "set_source", source: sourceKey });
}

async function loadTurns() {
  const speaker = document.getElementById("speaker-select").value;
  const url = speaker ? `/api/turns?speaker=${speaker}` : "/api/turns";
  const resp = await fetch(url);
  const data = await resp.json();
  state.turns = data.turns;
  renderTurnTable();
}

// --- UI Rendering ---

function renderDevices(active) {
  const el = document.getElementById("device-list");
  const labels = {
    blackhole_2ch: "BlackHole 2ch (to browser)",
    speakers: "Speakers (monitoring)",
    blackhole_16ch: "BlackHole 16ch (agent capture)",
  };
  let html = "";
  for (const [key, idx] of Object.entries(active)) {
    const label = labels[key] || key;
    const status = idx != null ? `Device #${idx}` : "Not found";
    const color = idx != null ? "#4caf50" : "#f44336";
    html += `<div><span style="color:${color}">\u25cf</span> ${label}: ${status}</div>`;
  }
  el.innerHTML = html;
}

function renderTurnTable() {
  const tbody = document.getElementById("turn-table-body");
  let html = "";

  for (const turn of state.turns) {
    const hesCount = turn.hesitations ? turn.hesitations.length : 0;
    const maxHes = turn.max_hesitation_ms || 0;
    const hesText = hesCount > 0 ? `${hesCount} (max ${maxHes}ms)` : "—";

    html += `
      <tr id="turn-row-${turn.turn}" data-turn="${turn.turn}">
        <td>${turn.turn}</td>
        <td>S${turn.speaker}</td>
        <td class="text-cell" title="${escapeHtml(turn.text)}">${escapeHtml(turn.text)}</td>
        <td>${(turn.duration_ms / 1000).toFixed(1)}s</td>
        <td>${hesText}</td>
        <td class="ttfa-cell" id="ttfa-${turn.turn}">—</td>
        <td id="status-${turn.turn}"><span class="status-badge status-pending">pending</span></td>
        <td><button class="btn-play-single" onclick="playSingle(${turn.turn})">Play</button></td>
      </tr>
    `;
  }

  tbody.innerHTML = html;
}

function setTurnStatus(turnIdx, status) {
  const el = document.getElementById(`status-${turnIdx}`);
  if (el) {
    const label = status.replace("_", " ");
    el.innerHTML = `<span class="status-badge status-${status}">${label}</span>`;
  }
  const row = document.getElementById(`turn-row-${turnIdx}`);
  if (row) {
    row.classList.toggle("barge-in", status === "barge_in");
  }
}

function setTurnTtfa(turnIdx, ttfa) {
  const el = document.getElementById(`ttfa-${turnIdx}`);
  if (!el) return;

  const ms = Math.round(ttfa);
  let cls;
  if (ms < 0) {
    cls = "ttfa-negative";  // agent interrupted before turn ended
  } else if (ms > 1500) {
    cls = "ttfa-red";
  } else if (ms > 500) {
    cls = "ttfa-yellow";
  } else {
    cls = "ttfa-green";
  }

  el.innerHTML = `<span class="${cls}">${ms}ms</span>`;
}

function highlightRow(turnIdx) {
  clearHighlight();
  const row = document.getElementById(`turn-row-${turnIdx}`);
  if (row) {
    row.classList.add("active");
    // only scroll if the row is not already visible
    const rect = row.getBoundingClientRect();
    const inView = rect.top >= 0 && rect.bottom <= window.innerHeight;
    if (!inView) {
      row.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }
}

function clearHighlight() {
  document.querySelectorAll("tr.active").forEach(r => r.classList.remove("active"));
}

function resetTableStatuses() {
  for (const turn of state.turns) {
    setTurnStatus(turn.turn, "pending");
    const ttfaEl = document.getElementById(`ttfa-${turn.turn}`);
    if (ttfaEl) ttfaEl.textContent = "—";
  }
}

function renderCurrentTurn(msg) {}
function updateCurrentTurnMeta(text) {}
function renderCurrentTurnIdle() {}

function renderSummary(data) {
  document.getElementById("stat-total").textContent = data.total_turns ?? "—";
  document.getElementById("stat-completed").textContent = data.completed ?? "—";
  document.getElementById("stat-bargein").textContent = data.barge_in_count ?? "—";
  document.getElementById("stat-noresp").textContent = data.no_response_count ?? "—";
  document.getElementById("stat-avg").textContent =
    data.ttfa_avg_ms != null ? `${Math.round(data.ttfa_avg_ms)}ms` : "—";
  document.getElementById("stat-median").textContent =
    data.ttfa_median_ms != null ? `${Math.round(data.ttfa_median_ms)}ms` : "—";
  document.getElementById("stat-p95").textContent =
    data.ttfa_p95_ms != null ? `${Math.round(data.ttfa_p95_ms)}ms` : "—";
}

function updateControls() {
  document.getElementById("btn-run-all").disabled = state.running;
  document.getElementById("btn-stop").disabled = !state.running;
  updatePlayPauseIcon();
}

function updatePlayPauseIcon() {
  const btn = document.getElementById("btn-play-pause");
  const playIcon = document.getElementById("icon-play");
  const pauseIcon = document.getElementById("icon-pause");
  if (!btn) return;
  if (state.running) {
    playIcon.classList.add("hidden");
    pauseIcon.classList.remove("hidden");
    btn.classList.add("active");
  } else {
    playIcon.classList.remove("hidden");
    pauseIcon.classList.add("hidden");
    btn.classList.remove("active");
  }
}

function togglePlayPause() {
  if (state.running) {
    stopRun();
  } else {
    runAll();
  }
}

// --- Actions ---

function runAll() {
  const speaker = document.getElementById("speaker-select").value;
  send({
    action: "run_all",
    speaker: speaker !== "" ? parseInt(speaker) : null,
  });
}

function stopRun() {
  send({ action: "stop" });
}

function resetRun() {
  send({ action: "reset" });
}

function playSingle(turnIdx) {
  send({ action: "run_single", turn: turnIdx });
}

// --- Util ---

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// --- Init ---

document.getElementById("btn-run-all").addEventListener("click", runAll);
document.getElementById("btn-stop").addEventListener("click", stopRun);
document.getElementById("btn-reset").addEventListener("click", resetRun);
document.getElementById("speaker-select").addEventListener("change", loadTurns);
document.getElementById("source-select").addEventListener("change", changeSource);

connect();
loadSources();
loadTurns();
