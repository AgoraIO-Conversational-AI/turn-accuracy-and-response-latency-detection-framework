const state = {
  ws: null,
  turns: [],
  results: {},  // turn index -> result
  currentTurn: null,
  running: false,
  batchRunning: false,
};

// Base path the app is mounted under (e.g. "/benchmark/" behind nginx, "/"
// for a bare uvicorn). Derived from the current document URL so the same
// build works both at the root and behind a proxy.
const BASE = location.pathname.endsWith("/")
  ? location.pathname
  : location.pathname + "/";

// --- WebSocket ---

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  state.ws = new WebSocket(`${proto}//${location.host}${BASE}ws`);

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
      // Server-side device map ignored; the browser drives audio now.
      break;

    case "browser_result":
      // No-op: we already updated the table locally; the broadcast just
      // tells other open tabs to refresh if they care.
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
      // Browser-mode already painted the cell locally in phaseHandler.
      if (msg.source !== "browser") setTurnTtfa(msg.turn, msg.ttfa_ms);
      updateCurrentTurnMeta(`TTFA: ${msg.ttfa_ms.toFixed(0)}ms — recording response...`);
      break;

    case "turn_done":
      // Browser-mode submits POST /api/results/submit which the server
      // echoes back as turn_done. We already painted the row locally in
      // runBrowserTurnFlow — only treat this WS event as authoritative
      // for the summary stats, and skip the TTFA cell write for
      // barge-in rows (which should remain blank).
      state.results[msg.turn] = msg;
      state.currentTurn = null;
      if (!state.batchRunning) {
        state.running = false;
        updateControls();
      }
      if (msg.source !== "browser") {
        // Server-driven path (legacy) still owns the row paint.
        setTurnStatus(msg.turn, msg.status);
        if (!msg.barge_in && msg.ttfa_ms != null) {
          setTurnTtfa(msg.turn, msg.ttfa_ms);
        }
        if (!msg.barge_in && msg.ttfa2_ms != null) {
          setTurnTtfa2(msg.turn, msg.ttfa2_ms);
        }
      }
      if (msg.summary) {
        renderSummary(msg.summary);
      }
      updateSummaryRow();
      clearHighlight();
      break;

    case "run_complete":
      state.running = false;
      state.batchRunning = false;
      state.currentTurn = null;
      updateControls();
      renderSummary(msg);
      renderCurrentTurnIdle();
      updateSummaryRow();
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
      updateSummaryRow();
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
  const resp = await fetch(`${BASE}api/sources`);
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
  const url = speaker ? `${BASE}api/turns?speaker=${speaker}` : `${BASE}api/turns`;
  const resp = await fetch(url);
  const data = await resp.json();
  state.turns = data.turns;
  renderTurnTable();
}

// --- UI Rendering ---

// Browser-side device picker \u2014 populated from
// navigator.mediaDevices.enumerateDevices(). Three slots:
//   - output  \u2192 primary playback target (BlackHole 2ch in Mode A,
//               real speakers in Mode B).
//   - monitor \u2192 optional second output, useful in Mode A so the
//               operator can hear what the agent is hearing.
//   - input   \u2192 capture device (BlackHole 16ch in Mode A,
//               built-in mic in Mode B).
const DEVICE_SLOTS = [
  {
    key: "output",
    label: "Output 1 (to agent under test)",
    hint: "Where turn audio plays. Mode A: BlackHole 2ch. Mode B: real speakers.",
    direction: "output",
  },
  {
    key: "monitor",
    label: "Output 2 (to hear locally)",
    hint: "Optional parallel output so you can hear what's playing. Leave blank to skip.",
    direction: "output",
    optional: true,
  },
  {
    key: "input",
    label: "Input (audio from agent under test)",
    hint: "Where the page listens for the agent. Mode A: BlackHole 16ch. Mode B: your built-in mic.",
    direction: "input",
  },
];

const DEVICES_STORAGE_KEY = "benchmark.deviceIds.v2";

// Storage shape: { output: {id, pinned}, monitor: {...}, input: {...} }
// pinned=true means the user explicitly picked this slot (or cleared it),
// so auto-detect should NOT overwrite on the next load. pinned=false (or
// the slot missing entirely) means we've never been told otherwise and
// are free to auto-detect.
function loadDeviceSelections() {
  try {
    const raw = JSON.parse(localStorage.getItem(DEVICES_STORAGE_KEY) || "{}");
    const out = {};
    for (const k of ["output", "monitor", "input"]) {
      const v = raw[k];
      if (typeof v === "string") out[k] = { id: v, pinned: true };
      else if (v && typeof v === "object") out[k] = { id: v.id || "", pinned: !!v.pinned };
      else out[k] = { id: "", pinned: false };
    }
    return out;
  } catch {
    return { output: { id: "", pinned: false }, monitor: { id: "", pinned: false }, input: { id: "", pinned: false } };
  }
}

function persistDeviceSelections() {
  localStorage.setItem(
    DEVICES_STORAGE_KEY,
    JSON.stringify(state.browserDevices || {}),
  );
}

// Auto-detect sensible defaults for any slot whose user has not pinned
// a choice. Mode A pattern: BlackHole 2ch → output, BlackHole 16ch →
// input, system default → monitor.
function autoFillDevices() {
  const cat = state.deviceCatalog;
  if (!cat) return;
  const dev = state.browserDevices || {};
  const findOutput = (rx) =>
    cat.outputs.find((d) => d.label && rx.test(d.label))?.deviceId;
  const findInput = (rx) =>
    cat.inputs.find((d) => d.label && rx.test(d.label))?.deviceId;
  const defaultOutput = () =>
    cat.outputs.find((d) => d.deviceId === "default")?.deviceId ||
    cat.outputs[0]?.deviceId;
  const defaultInput = () =>
    cat.inputs.find((d) => d.deviceId === "default")?.deviceId ||
    cat.inputs[0]?.deviceId;

  let changed = false;
  if (!dev.output.pinned && !dev.output.id) {
    const id = findOutput(/blackhole\s*2ch/i);
    if (id) { dev.output.id = id; changed = true; }
  }
  if (!dev.input.pinned && !dev.input.id) {
    const id = findInput(/blackhole\s*16ch/i) || defaultInput();
    if (id) { dev.input.id = id; changed = true; }
  }
  if (!dev.monitor.pinned && !dev.monitor.id) {
    // Prefer a non-BlackHole output — that's almost certainly what the
    // operator wants to monitor through. Fall back to system default.
    const real = cat.outputs.find(
      (d) => d.label && !/blackhole/i.test(d.label) && d.deviceId !== "default",
    );
    const id = real?.deviceId || defaultOutput();
    if (id) { dev.monitor.id = id; changed = true; }
  }
  state.browserDevices = dev;
  if (changed) persistDeviceSelections();
}

function renderDevices() {
  const el = document.getElementById("device-list");
  const cat = state.deviceCatalog;
  if (!cat) {
    el.innerHTML =
      '<div class="device-note">Click <em>Grant mic access</em> to populate device labels.' +
      ' <button class="btn" id="btn-grant" onclick="grantMicAccess()">Grant mic access</button>' +
      "</div>";
    return;
  }
  const browserState = state.browserDevices || {};
  const supportsSink = typeof state.canSelectOutput === "boolean" ? state.canSelectOutput : true;

  let html = '<div class="device-grid">';
  if (!supportsSink) {
    html +=
      '<div class="device-note device-warn">\u26a0 Your browser does not support choosing the output device (HTMLMediaElement.setSinkId). ' +
      "Playback will go to the system default. Use Chrome / Edge / Opera for output-device selection.</div>";
  }
  for (const slot of DEVICE_SLOTS) {
    const isOutput = slot.direction === "output";
    const opts = isOutput ? cat.outputs : cat.inputs;
    const entry = browserState[slot.key] || { id: "", pinned: false };
    const current = entry.id;
    const found = !!current;
    const dotColor = found ? "#4caf50" : slot.optional ? "#999" : "#f44336";
    const autoTag = found && !entry.pinned
      ? '<span class="device-auto-tag" title="Auto-picked. Change the dropdown to pin a manual choice.">auto</span>'
      : "";

    html += `
      <div class="device-row">
        <div class="device-label">
          <span class="device-dot" style="color:${dotColor}">\u25cf</span>
          <span><strong>${slot.label}</strong></span>
          ${autoTag}
        </div>
        <select class="device-select" data-slot="${slot.key}" onchange="setBrowserDevice('${slot.key}', this.value)">
          <option value="">\u2014 Not set \u2014</option>`;
    for (const d of opts) {
      const selected = d.deviceId === current ? " selected" : "";
      const label =
        escapeHtml(d.label || `(${isOutput ? "output" : "input"} ${d.deviceId.slice(0, 6)})`);
      html += `<option value="${d.deviceId}"${selected}>${label}</option>`;
    }
    html += `
        </select>
        <div class="device-hint">${slot.hint}</div>`;
    if (slot.key === "input") {
      // Live RMS meter so the operator can see immediately whether
      // anything is reaching the chosen input. Threshold marker = the
      // browser VAD trigger.
      html += `
        <div class="meter-wrap" aria-hidden="true">
          <div class="meter-bar"><div class="meter-fill" id="meter-fill"></div>
            <div class="meter-threshold" title="VAD speech threshold"></div>
          </div>
          <div class="meter-readout"><span id="meter-rms">\u2014</span> <span class="meter-label" id="meter-state"></span></div>
        </div>`;
    }
    html += `</div>`;
  }
  html += "</div>";
  el.innerHTML = html;
  // (Re)start the live input meter for whatever the input slot points at.
  syncInputMeter();
}

// --- Live input meter ---

const METER_MAX_RMS = 0.2; // 0.2 RMS \u2248 moderately loud speech
let _meterHandle = null;
let _meterDeviceId = null;

async function syncInputMeter() {
  const dev = state.browserDevices || {};
  const targetId = dev.input?.id || "";
  if (!targetId) {
    if (_meterHandle) { _meterHandle.stop(); _meterHandle = null; _meterDeviceId = null; }
    paintMeter(null);
    return;
  }
  if (_meterHandle && _meterDeviceId === targetId) return;
  if (_meterHandle) { _meterHandle.stop(); _meterHandle = null; }
  _meterDeviceId = targetId;
  try {
    _meterHandle = await window.benchHarness.startMeter(targetId, paintMeter);
  } catch (e) {
    console.warn("startMeter failed:", e);
    paintMeter({ rms: null, error: String(e?.message || e) });
  }
}

function paintMeter(level) {
  const fill = document.getElementById("meter-fill");
  const readout = document.getElementById("meter-rms");
  const stateEl = document.getElementById("meter-state");
  if (!fill || !readout) return;
  if (!level) {
    fill.style.width = "0%";
    readout.textContent = "\u2014";
    if (stateEl) stateEl.textContent = "";
    return;
  }
  if (level.error) {
    fill.style.width = "0%";
    readout.textContent = "error";
    if (stateEl) stateEl.textContent = level.error;
    return;
  }
  const pct = Math.min(100, (level.rms / METER_MAX_RMS) * 100);
  const threshold = window.benchHarness?.VAD_RMS_THRESHOLD ?? 0.003;
  fill.style.width = `${pct}%`;
  fill.classList.toggle("over-threshold", level.rms >= threshold);
  readout.textContent = `RMS ${level.rms.toFixed(4)}`;
  if (stateEl) {
    if (level.rms >= threshold) stateEl.textContent = "(speech)";
    else if (level.rms >= 0.001) stateEl.textContent = "(quiet)";
    else stateEl.textContent = "(silent)";
  }
}

async function grantMicAccess() {
  const ok = await window.benchHarness.primePermission();
  if (!ok) return;
  await refreshDeviceCatalog();
}

async function refreshDeviceCatalog() {
  state.canSelectOutput = window.benchHarness.canSelectOutputDevice();
  try {
    state.deviceCatalog = await window.benchHarness.listDevices();
  } catch (e) {
    console.warn("listDevices failed:", e);
    return;
  }
  // Drop stale IDs that no longer exist (devices unplugged etc), but
  // keep pinned=true so when the device comes back it doesn't get
  // overridden by auto-detect.
  const valid = new Set([
    ...state.deviceCatalog.outputs.map((d) => d.deviceId),
    ...state.deviceCatalog.inputs.map((d) => d.deviceId),
  ]);
  let changed = false;
  for (const k of Object.keys(state.browserDevices || {})) {
    const e = state.browserDevices[k];
    if (e && e.id && !valid.has(e.id)) {
      e.id = "";
      changed = true;
    }
  }
  // Fill any unpinned slot with a sensible auto-pick now that we have labels.
  autoFillDevices();
  if (changed) persistDeviceSelections();
  renderDevices();
}

function setBrowserDevice(slot, deviceId) {
  state.browserDevices = state.browserDevices || {};
  state.browserDevices[slot] = { id: deviceId || "", pinned: true };
  persistDeviceSelections();
  renderDevices();
}
window.grantMicAccess = grantMicAccess;
window.setBrowserDevice = setBrowserDevice;

function renderTurnTable() {
  const tbody = document.getElementById("turn-table-body");
  let html = "";

  for (const turn of state.turns) {
    const maxHes = turn.max_hesitation_ms || 0;
    const silenceText = maxHes > 0 ? `${maxHes}ms` : "—";

    // type column: category badge only
    const cat = turn.category || "";
    let typeHtml = "";
    if (cat === "normal") {
      typeHtml = `<span class="category-badge cat-normal">normal</span>`;
    } else if (cat === "hesitation") {
      typeHtml = `<span class="category-badge cat-hesitation">hesitation</span>`;
    } else if (cat === "hesitation2") {
      typeHtml = `<span class="category-badge cat-hesitation2">hesitation2</span>`;
    } else if (cat === "pause") {
      typeHtml = `<span class="category-badge cat-pause">pause</span>`;
    } else if (cat === "ambiguous") {
      typeHtml = `<span class="category-badge cat-ambiguous">ambiguous</span>`;
    }

    html += `
      <tr id="turn-row-${turn.turn}" data-turn="${turn.turn}">
        <td><a class="turn-num" title="Preview this turn's audio locally (no test run, no capture, no measurement)" onclick="previewTurn(${turn.turn});return false;" href="#">${turn.turn}</a></td>
        <td>S${turn.speaker}</td>
        <td class="type-cell">${typeHtml}</td>
        <td class="text-cell" title="${escapeHtml(turn.text)}">${escapeHtml(turn.text)}</td>
        <td>${(turn.duration_ms / 1000).toFixed(1)}s</td>
        <td>${silenceText}</td>
        <td class="ttfa-cell" id="ttfa-${turn.turn}">—</td>
        <td class="ttfa-cell" id="ttfa2-${turn.turn}">—</td>
        <td id="status-${turn.turn}"><span class="status-badge status-pending">pending</span></td>
        <td><button class="btn-play-single" title="Plays turn 0 (opener / instructions) first to prime the agent, then resumes from this turn through the end" onclick="playSingle(${turn.turn})">Start</button></td>
      </tr>
    `;
  }

  // summary footer row — single inline strip across all columns. Stats
  // are populated by renderSummary(); the layout sits where "Average"
  // used to sit so screen recordings end on the headline numbers.
  html += `
    <tr id="turn-row-summary" class="summary-row">
      <td colspan="10" class="summary-strip">
        <span>Turns <b id="stat-total">—</b></span>
        <span>Completed <b id="stat-completed">—</b></span>
        <span>Barge-ins <b id="stat-bargein">—</b></span>
        <span>No Response <b id="stat-noresp">—</b></span>
        <span>Avg TTFA <b id="stat-avg">—</b></span>
        <span>Avg TTFA2 <b id="stat-avg2">—</b></span>
        <span>P95 TTFA <b id="stat-p95">—</b></span>
        <span>P95 TTFA2 <b id="stat-p952">—</b></span>
      </td>
    </tr>
  `;

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
  paintTtfaCell(el, ttfa, false);
}

function setTurnTtfa2(turnIdx, ttfa) {
  const el = document.getElementById(`ttfa2-${turnIdx}`);
  paintTtfaCell(el, ttfa, true);
}

function paintTtfaCell(el, ttfa, allowNegative) {
  if (!el) return;

  // Canonical TTFA keeps negative/barge rows blank. TTFA2 may show a
  // negative value when the new timing disagrees, so the discrepancy is
  // visible without changing the run status.
  if (ttfa == null || (!allowNegative && ttfa < 0)) {
    el.textContent = "—";
    return;
  }

  const ms = Math.round(ttfa);
  let cls;
  if (ms < 0 || ms > 1500) {
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
    const ttfa2El = document.getElementById(`ttfa2-${turn.turn}`);
    if (ttfa2El) ttfa2El.textContent = "—";
  }
}

function summarizeMetric(field) {
  const values = Object.values(state.results)
    .filter(r => !r.barge_in && r[field] != null && r[field] >= 0)
    .map(r => r[field])
    .sort((a, b) => a - b);
  if (values.length === 0) return { avg: null, p95: null, count: 0 };
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const idx = (values.length - 1) * 0.95;
  const low = Math.floor(idx);
  const high = Math.min(low + 1, values.length - 1);
  const frac = idx - low;
  const p95 = values[low] + frac * (values[high] - values[low]);
  return { avg, p95, count: values.length };
}

function setSummaryMetric(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value != null ? `${Math.round(value)}ms` : "—";
}

function updateSummaryRow() {
  // Client-side fallback so metrics tick live between server summary
  // events. Server-side renderSummary() supersedes this when its
  // summary message arrives for TTFA. TTFA2 is browser-only, so it is
  // summarized here from the submitted result objects.
  const ttfa = summarizeMetric("ttfa_ms");
  const ttfa2 = summarizeMetric("ttfa2_ms");
  if (ttfa.count > 0) {
    setSummaryMetric("stat-avg", ttfa.avg);
    setSummaryMetric("stat-p95", ttfa.p95);
  }
  setSummaryMetric("stat-avg2", ttfa2.avg);
  setSummaryMetric("stat-p952", ttfa2.p95);
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
  document.getElementById("stat-p95").textContent =
    data.ttfa_p95_ms != null ? `${Math.round(data.ttfa_p95_ms)}ms` : "—";
  document.getElementById("stat-avg2").textContent = "—";
  document.getElementById("stat-p952").textContent = "—";
  updateSummaryRow();
}

function updateControls() {
  document.getElementById("btn-run-all").disabled = state.running;
  document.getElementById("btn-stop").disabled = !state.running;
}

// --- Actions ---

function isBrowserMode() {
  // Toggle has been removed — browser mode is the only mode now. The
  // server-mode branches in stopRun/resetRun/etc are kept for the
  // (rare) case the harness is run standalone without the new UI.
  return true;
}

function getCurrentSourceKey() {
  return document.getElementById("source-select").value;
}

// --- bench logger ---
// Every event is tagged with the current runId so that overlapping
// callbacks from a stopped/orphaned run are obvious in the console.
function benchLog(...args) {
  const tag = state.runId != null ? `bench#${state.runId}` : "bench";
  console.log(`[${tag}]`, ...args);
}

function newRunId() {
  state.runId = (state.runId || 0) + 1;
  return state.runId;
}

// Abort the current run cleanly. The signal is plumbed through
// runBrowserTurnFlow → runBrowserTurn so the poll loop bails out
// instead of dragging on while the next run starts.
function abortCurrentRun(reason) {
  if (state.abortCtl) {
    benchLog("abort:", reason);
    state.abortCtl.abort(reason);
    state.abortCtl = null;
  }
}

async function runAll() {
  if (!isBrowserMode()) {
    const speaker = document.getElementById("speaker-select").value;
    send({
      action: "run_all",
      speaker: speaker !== "" ? parseInt(speaker) : null,
    });
    return;
  }
  if (!ensureDevicesConfigured()) return;
  // Cancel any in-flight run before starting a new one.
  abortCurrentRun("new run_all");
  const runId = newRunId();
  const abortCtl = new AbortController();
  state.abortCtl = abortCtl;
  state.batchRunning = true;
  state.running = true;
  state.stopRequested = false;
  state.results = {};
  resetTableStatuses();
  renderSummary({});
  updateSummaryRow();
  updateControls();
  const turns = state.turns.slice();
  benchLog("run_all start —", turns.length, "turns");
  // Persistent capture: one AudioContext + getUserMedia + ScriptProcessor
  // for the whole run instead of recreating per turn.
  let runCap = null;
  try {
    const devs = deviceIdsForHarness();
    runCap = await window.benchHarness.startCapture(devs.input);
    benchLog(`persistent capture opened (sr=${runCap.actualSampleRate})`);
  } catch (e) {
    console.error("[bench] failed to open persistent capture:", e);
    benchLog("falling back to per-turn capture");
  }
  // Prefetch all turn WAVs into the browser HTTP cache so canplaythrough
  // doesn't have to wait for Chrome to fetch + decode each WAV
  // individually. ~600KB each, fired in parallel; the browser handles
  // the actual concurrency. Doesn't decode here — just primes the
  // cache so mkAudio's src= request is a cache hit.
  prefetchAllWavs(turns).catch((e) => console.warn("[bench] prefetch:", e));
  try {
    for (const turn of turns) {
      if (abortCtl.signal.aborted) {
        benchLog("run_all aborted before turn", turn.turn);
        break;
      }
      if (runCap && window.benchHarness.waitForSilence) {
        const settled = await window.benchHarness.waitForSilence(
          runCap, 400, 5000, abortCtl.signal,
        );
        if (!settled) benchLog(`turn ${turn.turn} pre-arm silence gate timed out`);
      }
      if (abortCtl.signal.aborted) break;
      await runBrowserTurnFlow(turn, abortCtl.signal, runId, runCap);
      if (abortCtl.signal.aborted) break;
    }
  } finally {
    if (runCap) {
      try { runCap.stop(); } catch {}
      benchLog("persistent capture closed");
    }
  }
  // If this run wasn't superseded by a newer one, clear shared state.
  if (state.abortCtl === abortCtl) state.abortCtl = null;
  state.batchRunning = false;
  state.running = false;
  state.currentTurn = null;
  updateControls();
  benchLog("run_all done");
}

function stopRun() {
  if (isBrowserMode()) {
    benchLog("stopRun pressed");
    abortCurrentRun("stop");
    state.stopRequested = true;
    state.running = false;
    state.batchRunning = false;
    state.currentTurn = null;
    updateControls();
    clearHighlight();
    return;
  }
  send({ action: "stop" });
}

function resetRun() {
  if (isBrowserMode()) {
    benchLog("resetRun pressed");
    abortCurrentRun("reset");
    state.results = {};
    state.running = false;
    state.batchRunning = false;
    state.currentTurn = null;
    updateControls();
    renderSummary({});
    resetTableStatuses();
    updateSummaryRow();
    return;
  }
  send({ action: "reset" });
}

// Prime the browser HTTP cache with all turn WAVs in parallel. The
// `<audio>.src=` fetch later in mkAudio will then hit the cache and
// canplaythrough should fire fast instead of stalling 3-11 s. We use
// the same URL shape the per-turn code uses so cache keys match.
async function prefetchAllWavs(turns) {
  if (!turns?.length) return;
  const source = getCurrentSourceKey();
  if (!source) return;
  const t0 = performance.now();
  const fetches = turns.map((turn) => {
    const url = `${BASE}api/wav/${source}/${turn.speaker}/${turn.turn}`;
    // cache: 'force-cache' so the browser actually stores it rather
    // than discarding immediately after the response is consumed.
    return fetch(url, { cache: "force-cache" })
      .then((r) => r.arrayBuffer())  // drain so the body completes
      .then(() => null)
      .catch((e) => {
        console.warn(`[bench] prefetch failed for turn ${turn.turn}:`, e);
        return null;
      });
  });
  // Don't await — let prefetch proceed in background while turn 0
  // runs. We just kick off the requests synchronously.
  Promise.all(fetches).then(() => {
    const dt = (performance.now() - t0).toFixed(0);
    benchLog(`prefetched ${turns.length} WAVs in ${dt}ms (background)`);
  });
}

// Standalone preview — click the turn number in the leftmost column to
// play just that turn's WAV through the operator's local speakers
// (the "Monitor" output). Routes through Monitor, NOT the primary
// output, because the primary is BlackHole 2ch → agent under test —
// the operator can't hear that locally. Falls back to the browser's
// default output if no Monitor device is configured. No capture, no
// test sequence, no measurement; clicking another number stops the
// prior preview.
let _previewAudio = null;
async function previewTurn(turnIdx) {
  const source = getCurrentSourceKey();
  if (!source) {
    benchLog("preview: no source selected");
    return;
  }
  const turn = (state.turns || []).find((t) => t.turn === turnIdx);
  if (!turn) {
    benchLog(`preview: turn ${turnIdx} not in state.turns (have ${(state.turns||[]).length})`);
    return;
  }

  if (_previewAudio) {
    try { _previewAudio.pause(); } catch {}
    try { _previewAudio.src = ""; _previewAudio.load(); } catch {}
    _previewAudio = null;
  }

  const url = `${BASE}api/wav/${source}/${turn.speaker}/${turn.turn}`;
  const el = new Audio(url);
  el.preload = "auto";

  // Prefer the Monitor output (operator's local speakers). Falls back
  // to the default output if not configured / not yet selected.
  const monitorId = state.browserDevices?.monitor?.id || "";
  const sinkLabel = monitorId
    ? state.browserDevices?.monitor?.label || monitorId
    : "(default)";

  _previewAudio = el;
  const clear = () => { if (_previewAudio === el) _previewAudio = null; };
  el.addEventListener("ended", clear);
  el.addEventListener("error", (e) => {
    benchLog(`preview turn ${turnIdx} error:`, el.error && el.error.message);
    clear();
  });

  // setSinkId before play so the first sample lands on the right device.
  // Errors are logged but non-fatal — we still try to play through the
  // default sink so the operator gets *some* sound.
  if (monitorId && el.setSinkId) {
    try { await el.setSinkId(monitorId); }
    catch (e) { console.warn(`preview setSinkId(${monitorId}) failed:`, e); }
  }

  benchLog(`preview turn ${turnIdx} → ${sinkLabel} (${url})`);
  try {
    await el.play();
  } catch (e) {
    benchLog(`preview turn ${turnIdx} play() failed: ${e.message}`);
    clear();
  }
}
window.previewTurn = previewTurn;

// "Start" per turn = prime the agent with turn 0 (the opener that
// explains the test rules to the agent), then resume from this turn
// to the end of the corpus. Equivalent to Run All but skipping turns
// 1..(N-1). Turn 0 itself is played in both cases so the agent
// always hears the instructions.
async function playSingle(turnIdx) {
  if (!isBrowserMode()) {
    send({ action: "run_single", turn: turnIdx });
    return;
  }
  if (!ensureDevicesConfigured()) return;
  const startTurn = state.turns.find((t) => t.turn === turnIdx);
  if (!startTurn) return;

  // Build the sequence: turn 0 (if not already turnIdx) then turnIdx..end.
  const turn0 = state.turns.find((t) => t.turn === 0);
  const allFromStart = state.turns
    .filter((t) => t.turn >= turnIdx)
    .sort((a, b) => a.turn - b.turn);
  const sequence =
    turnIdx === 0 || !turn0
      ? allFromStart
      : [turn0, ...allFromStart];

  abortCurrentRun("new playSingle");
  const runId = newRunId();
  const abortCtl = new AbortController();
  state.abortCtl = abortCtl;
  state.batchRunning = true;
  state.running = true;
  state.stopRequested = false;
  state.results = {};
  resetTableStatuses();
  renderSummary({});
  updateSummaryRow();
  updateControls();
  benchLog(
    `Start from turn ${turnIdx} — sequence: ${sequence.map((t) => t.turn).join(", ")}`,
  );

  let runCap = null;
  try {
    const devs = deviceIdsForHarness();
    runCap = await window.benchHarness.startCapture(devs.input);
    benchLog(`persistent capture opened (sr=${runCap.actualSampleRate})`);
  } catch (e) {
    console.error("[bench] failed to open persistent capture:", e);
    benchLog("falling back to per-turn capture");
  }
  prefetchAllWavs(sequence).catch((e) => console.warn("[bench] prefetch:", e));
  try {
    for (const turn of sequence) {
      if (abortCtl.signal.aborted) {
        benchLog("aborted before turn", turn.turn);
        break;
      }
      if (runCap && window.benchHarness.waitForSilence) {
        const settled = await window.benchHarness.waitForSilence(
          runCap, 400, 5000, abortCtl.signal,
        );
        if (!settled) benchLog(`turn ${turn.turn} pre-arm silence gate timed out`);
      }
      if (abortCtl.signal.aborted) break;
      await runBrowserTurnFlow(turn, abortCtl.signal, runId, runCap);
      if (abortCtl.signal.aborted) break;
    }
  } finally {
    if (runCap) {
      try { runCap.stop(); } catch {}
      benchLog("persistent capture closed");
    }
  }
  if (state.abortCtl === abortCtl) state.abortCtl = null;
  state.batchRunning = false;
  state.running = false;
  state.currentTurn = null;
  updateControls();
  benchLog("Start sequence done");
}

function ensureDevicesConfigured() {
  const dev = state.browserDevices || {};
  if (!dev.output?.id || !dev.input?.id) {
    alert(
      'Please set the "Playback" output and "Capture" input under Audio Devices first.',
    );
    return false;
  }
  return true;
}

// Flatten the persisted {id, pinned} structure into plain device-IDs
// for the harness module.
function deviceIdsForHarness() {
  const d = state.browserDevices || {};
  return {
    output: d.output?.id || "",
    monitor: d.monitor?.id || "",
    input: d.input?.id || "",
  };
}

// Wraps a single turn in the local UI lifecycle + the browser harness
// runner + result submission to the server. signal/runId let us identify
// and abort orphan runs after a Stop or Reset.
async function runBrowserTurnFlow(turn, signal, runId, cap) {
  state.currentTurn = turn.turn;
  setTurnStatus(turn.turn, "playing");
  highlightRow(turn.turn);
  benchLog(`turn ${turn.turn} start (speaker ${turn.speaker})`);
  const phaseHandler = (p, partial) => {
    if (signal && signal.aborted) return;
    benchLog(`turn ${turn.turn} phase=${p}`, partial?.ttfa_ms != null ? `ttfa=${partial.ttfa_ms.toFixed(0)}ms` : "");
    if (p === "playing") {
      setTurnStatus(turn.turn, "playing");
    } else if (p === "response_detected") {
      setTurnStatus(turn.turn, "listening");
      if (partial && partial.ttfa_ms != null) {
        setTurnTtfa(turn.turn, partial.ttfa_ms);
      }
    } else if (p === "barge_in") {
      setTurnStatus(turn.turn, "barge_in");
    }
  };
  let result;
  try {
    result = await window.benchHarness.runBrowserTurn(
      turn,
      getCurrentSourceKey(),
      deviceIdsForHarness(),
      BASE,
      phaseHandler,
      signal,
      cap ? { cap } : undefined,
    );
  } catch (e) {
    if (e && (e.name === "AbortError" || signal?.aborted)) {
      benchLog(`turn ${turn.turn} aborted`);
      return;
    }
    console.error(`[bench] turn ${turn.turn} failed:`, e);
    setTurnStatus(turn.turn, "skipped");
    return;
  }
  // Don't write into a fresher run's state if Stop/Reset fired during
  // our await chain.
  if (signal && signal.aborted) {
    benchLog(`turn ${turn.turn} finished after abort — discarding result`);
    return;
  }
  if (runId != null && runId !== state.runId) {
    benchLog(`turn ${turn.turn} finished but runId mismatch (was ${runId}, now ${state.runId}) — discarding`);
    return;
  }
  state.results[turn.turn] = result;
  setTurnStatus(turn.turn, result.status);
  if (!result.barge_in && result.ttfa_ms != null) {
    setTurnTtfa(turn.turn, result.ttfa_ms);
  }
  if (!result.barge_in && result.ttfa2_ms != null) {
    setTurnTtfa2(turn.turn, result.ttfa2_ms);
  }
  benchLog(
    `turn ${turn.turn} done — status=${result.status} ` +
    `ttfa=${result.ttfa_ms?.toFixed?.(0) ?? "—"} ` +
    `ttfa2=${result.ttfa2_ms?.toFixed?.(0) ?? "—"} ` +
    `barge=${result.barge_in}`,
  );
  // POST to server so it goes into the canonical results store.
  await window.benchHarness.submitResult(
    { ...result, speaker: turn.speaker },
    BASE,
  );
  updateSummaryRow();
  clearHighlight();
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

// --- Setup modal ---
const setupModal = document.getElementById("setup-modal");
document.getElementById("btn-setup").addEventListener("click", () => {
  setupModal.classList.remove("hidden");
  // Lazy-augment any <pre><code> blocks with a Copy button. Cheap to
  // re-run; we tag once so it isn't redone.
  setupModal.querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    const code = pre.querySelector("code");
    if (!code) return;
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        const prev = btn.textContent;
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = prev;
          btn.classList.remove("copied");
        }, 1200);
      } catch (e) {
        console.warn("clipboard write failed:", e);
        btn.textContent = "Failed";
        setTimeout(() => (btn.textContent = "Copy"), 1200);
      }
    });
    pre.appendChild(btn);
  });
});
setupModal.querySelectorAll("[data-close-modal]").forEach((el) => {
  el.addEventListener("click", () => setupModal.classList.add("hidden"));
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setupModal.classList.add("hidden");
});

// --- Browser-mode bootstrap ---
state.browserDevices = loadDeviceSelections();
state.canSelectOutput = true; // updated once the harness module loads
// React to hot-plug events.
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    if (state.deviceCatalog) refreshDeviceCatalog();
  });
}

// Dynamically import the harness module and expose it for runners +
// device-picker code. Falls back gracefully if the browser is too old.
(async () => {
  try {
    const mod = await import(`${BASE}static/browser_harness.js`);
    window.benchHarness = mod;
    state.canSelectOutput = mod.canSelectOutputDevice();
    // If permission was granted in an earlier session, we can list
    // immediately. enumerateDevices() succeeds but labels are empty
    // until permission is primed once.
    const list = await mod.listDevices();
    const haveLabels = [...list.inputs, ...list.outputs].some((d) => d.label);
    if (haveLabels) {
      state.deviceCatalog = list;
      autoFillDevices();
      renderDevices();
    } else {
      renderDevices(); // shows the Grant button
    }
  } catch (e) {
    console.error("benchHarness import failed:", e);
  }
})();

connect();
loadSources();
loadTurns();
