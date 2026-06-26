// browser_harness.js
// In-browser playback + capture + VAD + TTFA / barge-in detection.
// Mirrors src/harness/{audio_engine,vad_engine,turn_manager}.py so the
// numbers match when the harness is hosted remotely but driven from a
// user's Mac (BlackHole installed there).

// --- Constants (mirror Python harness) ---
export const VAD_RATE = 16000;
export const VAD_FRAME = 512;          // 32 ms @ 16 kHz
export const VAD_RMS_THRESHOLD = 0.003;
export const VAD_MIN_SILENCE_MS = 300;
export const RESPONSE_SILENCE_TIMEOUT_S = 1.5;
export const BARGE_IN_SILENCE_TIMEOUT_S = 3.0;
export const MAX_WAIT_FOR_RESPONSE_S = 8.0;
export const OUTPUT_RMS_THRESHOLD = 0.005;
export const OUTPUT_RMS_WINDOW_MS = 20;
export const PLAYING_FALLBACK_MS = 250;

// --- Device enumeration ---

let _permissionPrimed = false;

// enumerateDevices() returns labelled entries only after the user has
// granted mic permission at least once in this session. We can prime
// that with a momentary getUserMedia + immediate stop.
export async function primePermission() {
  if (_permissionPrimed) return true;
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach((t) => t.stop());
    _permissionPrimed = true;
    return true;
  } catch (e) {
    console.warn("mic permission denied:", e);
    return false;
  }
}

export async function listDevices() {
  const all = await navigator.mediaDevices.enumerateDevices();
  return {
    inputs: all.filter((d) => d.kind === "audioinput"),
    outputs: all.filter((d) => d.kind === "audiooutput"),
  };
}

export function canSelectOutputDevice() {
  // setSinkId is Chrome / Edge / Opera only at this point.
  const proto = HTMLMediaElement && HTMLMediaElement.prototype;
  return !!(proto && typeof proto.setSinkId === "function");
}

// --- VAD ---

class BrowserVad {
  constructor() {
    this.minSilenceSamples = Math.round((VAD_RATE * VAD_MIN_SILENCE_MS) / 1000);
    this.reset();
  }
  reset() {
    this.active = false;
    this.silence = 0;
    this.totalSamples = 0;
    this.buffer = new Float32Array(0);
  }
  // Returns events: {type, timeS}
  processChunk(chunk) {
    const merged = new Float32Array(this.buffer.length + chunk.length);
    merged.set(this.buffer, 0);
    merged.set(chunk, this.buffer.length);
    this.buffer = merged;
    const events = [];

    while (this.buffer.length >= VAD_FRAME) {
      const frame = this.buffer.subarray(0, VAD_FRAME);
      this.buffer = this.buffer.subarray(VAD_FRAME);

      let sumSq = 0;
      for (let i = 0; i < VAD_FRAME; i++) sumSq += frame[i] * frame[i];
      const rms = Math.sqrt(sumSq / VAD_FRAME);

      if (rms >= VAD_RMS_THRESHOLD) {
        this.silence = 0;
        if (!this.active) {
          this.active = true;
          events.push({
            type: "speech_start",
            timeS: this.totalSamples / VAD_RATE,
          });
        }
      } else if (this.active) {
        this.silence += VAD_FRAME;
        if (this.silence >= this.minSilenceSamples) {
          this.active = false;
          events.push({
            type: "speech_end",
            timeS: this.totalSamples / VAD_RATE,
          });
          this.silence = 0;
        }
      }
      this.totalSamples += VAD_FRAME;
    }
    return events;
  }
}

// --- Capture ---

// Wraps getUserMedia + an AudioContext running at 16 kHz so the
// ScriptProcessor frames are exactly VAD_FRAME = 32 ms. Returns an
// async iterator-ish handle with start time, current VAD state, and
// stop().
async function startCapture(inputDeviceId) {
  // 16 kHz AudioContext — Chromium honors the sampleRate hint exactly,
  // so the ScriptProcessor frames are VAD_FRAME = 32 ms (matches Python
  // VAD). Safari / Firefox silently override the sampleRate hint, which
  // would skew the VAD's per-frame timing; that's the main reason we
  // require Chrome / Edge in the setup docs.
  const ctx = new AudioContext({ sampleRate: VAD_RATE });
  const actualSr = ctx.sampleRate;
  if (actualSr !== VAD_RATE) {
    console.warn(
      `[harness] AudioContext sampleRate hint rejected: asked for ${VAD_RATE}, got ${actualSr}. Wall-clock TTFA will be used.`,
    );
  }
  const constraints = {
    audio: {
      deviceId: inputDeviceId ? { exact: inputDeviceId } : undefined,
      // Critical: leave the agent's TTS untouched. Echo cancellation
      // and noise suppression would chew through the agent's voice
      // (which the OS sees as "speaker output coming back into the
      // mic") and TTFA would never trigger.
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(VAD_FRAME, 1, 1);

  // VAD processes ALL frames for the lifetime of the capture, even
  // between turns. armForTurn() only resets the per-turn tracking
  // (firstSpeechWall, callbackCount, etc.) — `vad.active` keeps
  // reflecting real-time input state so callers can ask "is the
  // input currently quiet?" while we're transitioning between turns.
  // (Disarming the VAD would freeze isSpeechActive at whatever value
  // it had at disarm time, defeating the inter-turn silence gate.)
  const vad = new BrowserVad();
  let armWall = performance.now();
  let speechStartedAt = null;     // audio-time (s) of first speech_start since arm
  let firstSpeechWall = null;     // performance.now() at first speech_start frame
  let lastSpeechWall = null;      // performance.now() at most recent speech_active frame
  let callbackCount = 0;          // since arm — diagnoses batching
  // Peak RMS observed since arm — answers "was audio arriving at all?"
  // independently of whether it crossed the VAD threshold. Distinguishes
  // true silence (peak ≈ 0) from below-threshold audio (peak < VAD_RMS_THRESHOLD)
  // from clean speech (peak above VAD_RMS_THRESHOLD) when the turn
  // ends in no_response.
  let peakRmsThisTurn = 0;

  proc.onaudioprocess = (ev) => {
    const data = ev.inputBuffer.getChannelData(0);
    // Single pass RMS for the whole buffer — cheap, and gives us a
    // per-callback aggregate without re-walking the samples.
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) sumSq += data[i] * data[i];
    const rms = Math.sqrt(sumSq / data.length);
    if (rms > peakRmsThisTurn) peakRmsThisTurn = rms;

    const events = vad.processChunk(data);
    callbackCount++;
    for (const e of events) {
      if (e.type === "speech_start" && speechStartedAt === null) {
        // First speech since the most recent armForTurn(). Audio-time
        // measured from arm point so the per-turn math stays correct.
        speechStartedAt = e.timeS - (armAudioTime);
        firstSpeechWall = performance.now();
      }
    }
    if (vad.active) lastSpeechWall = performance.now();
  };

  // Audio-time offset captured at each arm so per-turn `speechStartedAt`
  // is relative to the arm point, not the absolute start of capture.
  let armAudioTime = 0;
  const getAudioTimeSeconds = () => vad.totalSamples / VAD_RATE;

  const armForTurn = () => {
    // VAD keeps running between turns (so isSpeechActive stays
    // truthful for the inter-turn silence gate), but we explicitly
    // mark "not currently mid-utterance" here so the next loud frame
    // counts as a fresh speech_start for THIS turn. Without this, if
    // the agent's response from turn N-1 was still ringing during the
    // silence gate, vad.active could stay continuously true across
    // the turn boundary and we'd never see a new speech_start
    // (manifests as a spurious no_response on the next turn).
    vad.active = false;
    vad.silence = 0;
    armWall = performance.now();
    armAudioTime = getAudioTimeSeconds();
    speechStartedAt = null;
    firstSpeechWall = null;
    lastSpeechWall = null;
    callbackCount = 0;
    peakRmsThisTurn = 0;
  };
  // disarm is a no-op now (VAD keeps running). Kept on the API for
  // backwards-compatibility with the runBrowserTurn caller.
  const disarm = () => {};

  // Arm immediately so single-turn callers (playSingle) keep working
  // without an explicit arm step.
  armForTurn();

  // Silent gain so ScriptProcessor stays alive (it only fires when in
  // the audio graph upstream of destination) without piping the mic
  // back through the speakers — that would feed the agent's voice
  // straight to the room in Mode B and cause feedback.
  const muteSink = ctx.createGain();
  muteSink.gain.value = 0;
  src.connect(proc);
  proc.connect(muteSink);
  muteSink.connect(ctx.destination);

  return {
    actualSampleRate: actualSr,
    // Per-turn arm/disarm. The poll loop calls armForTurn() before each
    // turn so state is fresh; the captured audio graph survives across
    // turns to avoid getUserMedia / AudioContext setup cost.
    armForTurn,
    disarm,
    // startWall is the arm time of the CURRENT turn (was: the time
    // capture was opened). Existing callers read this as the zero
    // point for the turn, which is exactly what arm time provides.
    get startWall() { return armWall; },
    get speechStartedAtMs() {
      return speechStartedAt !== null ? speechStartedAt * 1000 : null;
    },
    get firstSpeechWallMs() {
      return firstSpeechWall;
    },
    get lastSpeechWallMs() {
      return lastSpeechWall;
    },
    get isSpeechActive() {
      return vad.active;
    },
    // Loudest RMS the input has shown since the most recent
    // armForTurn(). Used by the no_response diagnostic to tell true
    // silence from below-threshold audio.
    get peakRms() {
      return peakRmsThisTurn;
    },
    get callbackCount() {
      return callbackCount;
    },
    stop() {
      try { proc.disconnect(); } catch {}
      try { src.disconnect(); } catch {}
      try { stream.getTracks().forEach((t) => t.stop()); } catch {}
      try { ctx.close(); } catch {}
    },
  };
}

// --- Playback ---

function formatError(e) {
  return e?.message || String(e);
}

async function decodeWavTiming(wavUrl) {
  const resp = await fetch(wavUrl, { cache: "force-cache" });
  if (!resp.ok) {
    throw new Error(`fetch ${resp.status}`);
  }
  const bytes = await resp.arrayBuffer();
  const DecodeCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = DecodeCtx
    ? new DecodeCtx(1, 1, 48000)
    : new AudioContext();
  let audioBuffer;
  try {
    audioBuffer = await ctx.decodeAudioData(bytes.slice(0));
  } finally {
    if (typeof ctx.close === "function") {
      try { ctx.close(); } catch {}
    }
  }

  const sampleRate = audioBuffer.sampleRate;
  const durationMs = audioBuffer.duration * 1000;
  const windowSamples = Math.max(1, Math.round(sampleRate * OUTPUT_RMS_WINDOW_MS / 1000));
  const channels = [];
  for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
    channels.push(audioBuffer.getChannelData(c));
  }

  let firstActiveWindow = null;
  let lastActiveWindow = null;
  const totalWindows = Math.ceil(audioBuffer.length / windowSamples);
  for (let w = 0; w < totalWindows; w++) {
    const start = w * windowSamples;
    const end = Math.min(audioBuffer.length, start + windowSamples);
    let sumSq = 0;
    let n = 0;
    for (let i = start; i < end; i++) {
      let sample = 0;
      for (let c = 0; c < channels.length; c++) sample += channels[c][i];
      sample /= channels.length || 1;
      sumSq += sample * sample;
      n++;
    }
    const rms = n > 0 ? Math.sqrt(sumSq / n) : 0;
    if (rms >= OUTPUT_RMS_THRESHOLD) {
      if (firstActiveWindow === null) firstActiveWindow = w;
      lastActiveWindow = w;
    }
  }

  const firstOutputSpeechMs =
    firstActiveWindow !== null ? firstActiveWindow * OUTPUT_RMS_WINDOW_MS : null;
  const lastOutputSpeechMs =
    lastActiveWindow !== null
      ? Math.min(durationMs, (lastActiveWindow + 1) * OUTPUT_RMS_WINDOW_MS)
      : null;

  return {
    decoded_duration_ms: durationMs,
    first_output_speech_ms: firstOutputSpeechMs,
    last_output_speech_ms: lastOutputSpeechMs,
    output_tail_ms: lastOutputSpeechMs !== null ? Math.max(0, durationMs - lastOutputSpeechMs) : null,
    output_rms_threshold: OUTPUT_RMS_THRESHOLD,
    output_rms_window_ms: OUTPUT_RMS_WINDOW_MS,
    output_sample_rate: sampleRate,
  };
}

// Plays a wav URL through one or two output devices via <audio> +
// setSinkId. Returns a controller {startWall, playbackStart, ended, stop()}.
// stop() halts playback and resolves `ended` immediately so a poll
// loop awaiting it can move on.
async function startPlayback(wavUrl, primaryOutputId, monitorOutputId, armBeforePlay, log) {
  const t_call = performance.now();
  const _log = log || (() => {});

  // mkAudio: create the Audio element, point at WAV. Cheap, sync-ish.
  const mkAudio = async (sinkId, kind) => {
    const t_mk = performance.now();
    const el = new Audio();
    el.preload = "auto";
    el.src = wavUrl;
    el.crossOrigin = "anonymous";
    const dt_create = performance.now() - t_mk;
    let dt_sink = 0;
    if (sinkId && el.setSinkId) {
      const t_sink = performance.now();
      try { await el.setSinkId(sinkId); }
      catch (e) { console.warn(`setSinkId(${kind}) failed:`, e); }
      dt_sink = performance.now() - t_sink;
    }
    _log(`  ${kind}: create=${dt_create.toFixed(0)}ms setSinkId=${dt_sink.toFixed(0)}ms`);
    return el;
  };

  const t_primary = performance.now();
  const primary = await mkAudio(primaryOutputId, "primary");
  const t_after_primary = performance.now();
  _log(`  primary ready t+${(t_after_primary - t_call).toFixed(0)}ms`);

  const monitor =
    monitorOutputId && monitorOutputId !== primaryOutputId
      ? await mkAudio(monitorOutputId, "monitor")
      : null;
  const t_after_monitor = performance.now();
  if (monitor) _log(`  monitor ready t+${(t_after_monitor - t_call).toFixed(0)}ms`);

  // canplaythrough wait — usually the variable cost on first turn or
  // big WAVs. We treat 'error' the same as ready so a broken decode
  // doesn't hang us forever.
  const t_pre_cpt = performance.now();
  const cptResult = await new Promise((resolve) => {
    if (primary.readyState >= 4) return resolve("already_ready");
    primary.addEventListener("canplaythrough", () => resolve("canplaythrough"), { once: true });
    primary.addEventListener("error", () => resolve("error"), { once: true });
  });
  const dt_cpt = performance.now() - t_pre_cpt;
  _log(`  ${cptResult} after ${dt_cpt.toFixed(0)}ms (total setup ${(performance.now() - t_call).toFixed(0)}ms)`);

  let endedResolve;
  const ended = new Promise((resolve) => { endedResolve = resolve; });
  let playbackStartResolve;
  let playbackStartSettled = false;
  let playbackStartTimer = null;
  const playbackStart = new Promise((resolve) => { playbackStartResolve = resolve; });
  const settlePlaybackStart = (wall, source) => {
    if (playbackStartSettled) return;
    playbackStartSettled = true;
    if (playbackStartTimer !== null) clearTimeout(playbackStartTimer);
    playbackStartResolve({ wall, source });
  };
  primary.addEventListener("ended", () => endedResolve(), { once: true });
  // The Audio element's own 'playing' event fires when output actually
  // begins. Surfacing it lets us see browser-side play scheduling vs.
  // when we called .play().
  primary.addEventListener("playing", () => {
    const wall = performance.now();
    _log(`  Audio element 'playing' event fired (browser confirms playback) t+${(wall - t_call).toFixed(0)}ms`);
    settlePlaybackStart(wall, "playing");
  }, { once: true });
  // Arm immediately before play() so startWall and the VAD's
  // "first speech since arm" clock align to the same instant.
  if (typeof armBeforePlay === "function") {
    try { armBeforePlay(); } catch (e) { console.warn("armBeforePlay threw:", e); }
  }
  const startWall = performance.now();
  playbackStartTimer = setTimeout(
    () => settlePlaybackStart(startWall, "play_call_fallback"),
    PLAYING_FALLBACK_MS,
  );
  _log(`  → primary.play() called at t+${(startWall - t_call).toFixed(0)}ms`);
  // Catch play() rejections so the caller doesn't hang on `await
  // playPromise` if the browser refuses (autoplay block, user-gesture
  // expiry, MediaError after canplaythrough). On primary failure we
  // resolve `ended` immediately so the run can move on; the upstream
  // result will record no-response / timeout instead of stalling the
  // queue indefinitely.
  primary.play().catch((e) => {
    console.warn("primary.play() rejected:", e);
    _log(`  ✗ primary.play() rejected: ${e?.message || e}`);
    settlePlaybackStart(startWall, "play_rejected");
    endedResolve();
  });
  if (monitor) {
    monitor.play().catch((e) => {
      // Monitor failure isn't fatal — operator just won't hear the
      // playback locally. Log and carry on.
      console.warn("monitor.play() rejected:", e);
      _log(`  ✗ monitor.play() rejected (non-fatal): ${e?.message || e}`);
    });
  }
  // Cleanup releases the underlying MediaElement audio node + decoder
  // so we don't leak one (or two) per turn. Without this, Run All over
  // 25 turns leaves ~50 elements holding sample data + sink references,
  // and Chrome starts to delay AudioContext / ScriptProcessor scheduling
  // — observed as TTFA inflation that grows over the run.
  const release = (el) => {
    try { el.pause(); } catch {}
    try { el.src = ""; } catch {}
    try { el.removeAttribute("src"); } catch {}
    try { el.load(); } catch {}
  };
  const stop = () => {
    release(primary);
    if (monitor) release(monitor);
    settlePlaybackStart(startWall, "stopped_before_playing");
    endedResolve();
  };
  // Auto-release once playback finishes (the natural path through
  // poll loop -> playPromise resolves). stop() handles the abort path.
  ended.then(() => {
    release(primary);
    if (monitor) release(monitor);
  });
  return { startWall, playbackStart, ended, stop };
}

// --- Run a single turn ---

// turn: object from /api/turns. devices: {output, monitor, input}.
// signal (optional): AbortSignal — when aborted, capture is torn down,
// playback stopped, and the function throws an AbortError so the
// caller can identify orphan completions and discard them.
// Returns the result payload that POST /api/results/submit expects.
export async function runBrowserTurn(turn, sourceKey, devices, baseUrl, onPhase, signal, opts) {
  // Every log line shows time since *runBrowserTurn was entered* so the
  // operator can spot delays between steps. Also keeps the turn id in
  // every line for grep-ability.
  const t_turn_start = performance.now();
  const log = (...a) => {
    const ts = `+${(performance.now() - t_turn_start).toFixed(0)}ms`;
    console.log(`[harness#turn${turn.turn} ${ts}]`, ...a);
  };
  const throwIfAborted = () => {
    if (signal && signal.aborted) {
      const err = new Error("aborted");
      err.name = "AbortError";
      throw err;
    }
  };
  const phase = (p, partial) => { if (onPhase) onPhase(p, partial); };
  const wavUrl = `${baseUrl}api/wav/${sourceKey}/${turn.speaker}/${turn.turn}`;
  const expectedEndMs = turn.duration_ms;
  const hesitations = turn.hesitations || [];
  const hesWindows = hesitations.map((h) => {
    const gapStart = h.at_ms - (turn.start_ms || 0);
    return [gapStart, gapStart + h.duration_ms];
  });
  const outputTimingPromise = decodeWavTiming(wavUrl).catch((e) => ({
    error: formatError(e),
  }));

  const emit = (p, partial) => phase && phase(p, partial);
  throwIfAborted();

  // Persistent capture: if the caller passes one in (Run All), reuse
  // it. We deliberately DON'T armForTurn() here — that would zero
  // the speech-start clock while we're still doing mkAudio /
  // setSinkId / canplaythrough, and any leftover audio from the
  // previous turn would be counted as a barge-in once arm + play
  // are reconciled. We arm right before primary.play() below.
  let cap = opts?.cap;
  const ownCap = !cap;
  if (cap) {
    log("reusing persistent capture (arm deferred to play)");
  } else {
    emit("capture_starting");
    log("capture_starting input=", devices.input?.slice?.(0, 8));
    cap = await startCapture(devices.input);
  }
  if (signal && signal.aborted) {
    if (ownCap) cap.stop();
    throwIfAborted();
  }

  emit("preparing");
  log(`requesting playback: wav=${wavUrl} expectedEnd=${expectedEndMs}ms`);
  log(`setup begins (mkAudio → setSinkId → canplaythrough → play)`);
  let playbackEndWall = null;
  let playbackError = null;
  let playController = null;
  try {
    playController = await startPlayback(
      wavUrl, devices.output, devices.monitor,
      // armBeforePlay: invoked synchronously immediately before
      // primary.play() so the VAD's "first speech since arm" clock
      // is t=0 at the moment audio actually begins. Anything the
      // mic heard during the setSinkId / canplaythrough wait is
      // leftover from prior turns and is correctly ignored.
      () => cap.armForTurn(),
      log,
    );
    // play() has been called and the audio element will start very
    // soon (browser-side scheduling — usually a few ms). This is the
    // "actual playing" emission point, distinct from the upstream
    // 'preparing' phase that fired when we requested playback.
    emit("playing");
    log(`PLAYING — primary.play() returned; expect agent response soon`);
  } catch (e) {
    playbackError = e;
    playbackEndWall = performance.now();
    log(`startPlayback threw: ${e?.message || e}`);
  }
  const playPromise = playController
    ? playController.ended.then(() => { playbackEndWall = performance.now(); })
    : Promise.resolve();

  // Stop pipeline + abort listener: if the caller aborts, stop the
  // audio elements + bail the poll loop ASAP.
  const onAbort = () => {
    log("abort received — stopping playback");
    try { playController?.stop(); } catch {}
  };
  if (signal) signal.addEventListener("abort", onAbort, { once: true });

  const result = {
    turn: turn.turn,
    ttfa_ms: null,
    ttfa2_ms: null,
    barge_in: false,
    barge_in_at_ms: null,
    response_duration_ms: null,
    status: "playing",
  };
  let firstSpeechAudioMs = null;

  const startWall = cap.startWall;
  while (true) {
    if (signal && signal.aborted) {
      log("poll loop sees abort — exiting");
      break;
    }
    await new Promise((r) => setTimeout(r, 10));
    const nowWall = performance.now();
    const elapsedS = (nowWall - startWall) / 1000;
    const pastExpectedEnd = elapsedS * 1000 > expectedEndMs;

    if (firstSpeechAudioMs === null && cap.speechStartedAtMs !== null) {
      firstSpeechAudioMs = cap.speechStartedAtMs;
      // Zero point for TTFA is *playback start*, not capture start.
      // Capture opens before mkAudio()/setSinkId()/canplaythrough are
      // ready, and that capture→play gap is non-zero (sometimes
      // multi-second when the page accumulates leaked Audio elements).
      // `playController.startWall` is the actual moment primary.play()
      // was called.
      const playStartWall = playController?.startWall ?? cap.startWall;
      const captureToPlayMs = playStartWall - cap.startWall;
      // Audio-time TTFA, offset by the capture→play gap.
      const audioFromPlayMs = firstSpeechAudioMs - captureToPlayMs;
      const ttfaAudio = audioFromPlayMs - expectedEndMs;
      // Wall-clock TTFA — independent of audio-time math.
      const wallFromPlayMs = cap.firstSpeechWallMs != null
        ? (cap.firstSpeechWallMs - playStartWall)
        : null;
      const ttfaWall = wallFromPlayMs != null
        ? (wallFromPlayMs - expectedEndMs)
        : null;
      const rateMismatch = cap.actualSampleRate !== VAD_RATE;
      // Use wall-clock when the rate hint was rejected, otherwise the
      // audio-time number (which has slightly less bias).
      const ttfa = rateMismatch && ttfaWall != null ? ttfaWall : ttfaAudio;
      log(
        `speech_start: ` +
        `audio-time-from-capture=${firstSpeechAudioMs.toFixed(0)}ms ` +
        `wall-from-capture=${cap.firstSpeechWallMs != null ? (cap.firstSpeechWallMs - cap.startWall).toFixed(0) : "?"}ms ` +
        `capture→play=${captureToPlayMs.toFixed(0)}ms ` +
        `expectedEnd=${expectedEndMs}ms ` +
        `ttfa(audio)=${ttfaAudio.toFixed(0)}ms ` +
        `ttfa(wall)=${ttfaWall?.toFixed?.(0) ?? "?"}ms ` +
        `sr=${cap.actualSampleRate} cb=${cap.callbackCount} ` +
        `→ using ${rateMismatch ? "WALL" : "AUDIO"}`,
      );
      if (ttfa < 0) {
        result.barge_in = true;
        result.barge_in_at_ms = firstSpeechAudioMs;
      } else {
        for (const [gs, ge] of hesWindows) {
          if (firstSpeechAudioMs >= gs && firstSpeechAudioMs <= ge) {
            result.barge_in = true;
            result.barge_in_at_ms = firstSpeechAudioMs;
            break;
          }
        }
      }
      result.ttfa_ms = ttfa;
      result.ttfa_audio_ms = ttfaAudio;
      result.ttfa_wall_ms = ttfaWall;
      result.capture_to_play_ms = captureToPlayMs;
      result.sample_rate = cap.actualSampleRate;
      if (result.barge_in) result.status = "barge_in";
      // Fire the phase event with a snapshot of the result so the UI
      // can flip the row's status badge + TTFA cell the instant we
      // know — without waiting for the rest of the turn to complete.
      emit(result.barge_in ? "barge_in" : "response_detected", { ...result });
    }

    // no-response timeout
    if (pastExpectedEnd && firstSpeechAudioMs === null) {
      const sinceEnd = elapsedS - expectedEndMs / 1000;
      if (sinceEnd > MAX_WAIT_FOR_RESPONSE_S) {
        // Three-way classification of why no audio crossed the VAD
        // threshold:
        //   peakRms ≈ 0           → true silence, input mic dead / agent disconnected
        //   peakRms < threshold   → audio arrived but below the browser VAD threshold — TTS too quiet, gain too low, routing OK
        //   peakRms ≥ threshold + no speech_start → real logic bug; investigate
        const peak = cap.peakRms ?? 0;
        const threshold = VAD_RMS_THRESHOLD;
        let interpretation;
        if (peak < 0.0005) {
          interpretation = "TRUE SILENCE (no audio at all — agent dead/disconnected or mic routing broken)";
        } else if (peak < threshold) {
          interpretation = `AUDIO PRESENT BUT BELOW VAD THRESHOLD (peak ${peak.toFixed(4)} < ${threshold}). Boost agent TTS volume or lower threshold.`;
        } else {
          interpretation = `audio crossed threshold (peak ${peak.toFixed(4)}) but no speech_start — possible logic bug, investigate.`;
        }
        const sawAnyActivity = cap.lastSpeechWallMs != null;
        const lastActivityAgo = sawAnyActivity
          ? `${(performance.now() - cap.lastSpeechWallMs).toFixed(0)}ms ago`
          : "never";
        log(
          `no_response: waited ${MAX_WAIT_FOR_RESPONSE_S}s past expectedEnd. ` +
          `peakRms=${peak.toFixed(4)} ` +
          `vad.isSpeechActive=${cap.isSpeechActive} ` +
          `lastSpeechWall=${lastActivityAgo} ` +
          `cb=${cap.callbackCount} ` +
          `→ ${interpretation}`,
        );
        result.status = "no_response";
        break;
      }
    }

    // done condition: spoke, now silent past threshold
    if (firstSpeechAudioMs !== null && cap.lastSpeechWallMs !== null) {
      if (!cap.isSpeechActive && pastExpectedEnd) {
        const silenceDurS = (nowWall - cap.lastSpeechWallMs) / 1000;
        const timeoutS = result.barge_in
          ? BARGE_IN_SILENCE_TIMEOUT_S
          : RESPONSE_SILENCE_TIMEOUT_S;
        if (result.barge_in) {
          const sinceExpEnd = elapsedS - expectedEndMs / 1000;
          if (sinceExpEnd < timeoutS) continue;
        }
        if (silenceDurS >= timeoutS) {
          const speakStartedWall =
            startWall + (firstSpeechAudioMs / 1000) * 1000;
          const responseDurMs =
            nowWall - silenceDurS * 1000 - speakStartedWall;
          result.response_duration_ms = Math.max(0, responseDurMs);
          if (!result.barge_in) result.status = "done";
          else result.status = "barge_in";
          break;
        }
      }
    }
  }

  try { await playPromise; } catch {}
  let outputTiming = null;
  let playbackStartInfo = null;
  if (!signal?.aborted) {
    try {
      outputTiming = await outputTimingPromise;
    } catch (e) {
      outputTiming = { error: formatError(e) };
    }
    try {
      playbackStartInfo = playController?.playbackStart
        ? await playController.playbackStart
        : { wall: playController?.startWall ?? cap.startWall, source: "missing_playback_controller" };
    } catch (e) {
      playbackStartInfo = {
        wall: playController?.startWall ?? cap.startWall,
        source: "playback_start_error",
        error: formatError(e),
      };
    }

    if (outputTiming?.error) {
      result.ttfa2_ms = null;
      result.playback_start_source = "decode_failed";
      result.ttfa2_error = outputTiming.error;
    } else if (playbackStartInfo && outputTiming) {
      const playStartWall = playbackStartInfo.wall;
      const inputSpeechStartWall = cap.firstSpeechWallMs;
      const indexSpeechEndMs =
        turn.speech_end_ms != null ? Number(turn.speech_end_ms) : null;
      const hasIndexSpeechEnd = Number.isFinite(indexSpeechEndMs);
      const promptEndOffsetMs = hasIndexSpeechEnd
        ? indexSpeechEndMs
        : outputTiming.last_output_speech_ms;
      const promptEndMethod = hasIndexSpeechEnd
        ? "index_speech_end_ms"
        : "decoded_wav_rms_fallback";
      const promptEndWall =
        promptEndOffsetMs !== null && promptEndOffsetMs !== undefined
          ? playStartWall + promptEndOffsetMs
          : null;

      Object.assign(result, {
        decoded_duration_ms: outputTiming.decoded_duration_ms,
        first_output_speech_ms: outputTiming.first_output_speech_ms,
        last_output_speech_ms: outputTiming.last_output_speech_ms,
        output_tail_ms: outputTiming.output_tail_ms,
        output_rms_threshold: outputTiming.output_rms_threshold,
        output_rms_window_ms: outputTiming.output_rms_window_ms,
        output_sample_rate: outputTiming.output_sample_rate,
        playback_start_source: playbackStartInfo.source,
        media_ended_minus_playing_ms: playbackEndWall != null ? playbackEndWall - playStartWall : null,
        first_input_from_play_ms: inputSpeechStartWall != null ? inputSpeechStartWall - playStartWall : null,
        ttfa_index_ms: firstSpeechAudioMs != null ? firstSpeechAudioMs - expectedEndMs : null,
        prompt_end_ms: promptEndOffsetMs,
        prompt_tail_ms: promptEndOffsetMs != null ? Math.max(0, expectedEndMs - promptEndOffsetMs) : null,
        prompt_end_wall_method: promptEndMethod,
      });

      if (inputSpeechStartWall != null && promptEndWall != null) {
        result.ttfa2_ms = inputSpeechStartWall - promptEndWall;
        result.ttfa2_hard_barge_in = inputSpeechStartWall < promptEndWall;
        result.ttfa2_gap_barge_in = hesWindows.some(([gapStartMs, gapEndMs]) => {
          const gapStartWall = playStartWall + gapStartMs;
          const gapEndWall = playStartWall + gapEndMs;
          return inputSpeechStartWall >= gapStartWall && inputSpeechStartWall <= gapEndWall;
        });
      } else if (promptEndWall == null) {
        result.ttfa2_error = "no_output_speech_detected";
      }
    }

    console.info("[bench timing]", {
      turn: turn.turn,
      turn_duration_ms: expectedEndMs,
      decoded_duration_ms: result.decoded_duration_ms ?? null,
      ended_minus_playing_ms: result.media_ended_minus_playing_ms ?? null,
      measured_output_active_ms: result.last_output_speech_ms ?? null,
      prompt_end_ms: result.prompt_end_ms ?? null,
      prompt_tail_ms: result.prompt_tail_ms ?? null,
      output_tail_ms: result.output_tail_ms ?? null,
      ttfa_ms: result.ttfa_ms,
      ttfa_index_ms: result.ttfa_index_ms ?? null,
      ttfa_wall_ms: result.ttfa_wall_ms ?? null,
      ttfa2_ms: result.ttfa2_ms,
      first_input_from_play_ms: result.first_input_from_play_ms ?? null,
      playback_start_source: result.playback_start_source ?? null,
      prompt_end_wall_method: result.prompt_end_wall_method ?? null,
      ttfa2_hard_barge_in: result.ttfa2_hard_barge_in ?? false,
      ttfa2_gap_barge_in: result.ttfa2_gap_barge_in ?? false,
      ttfa2_error: result.ttfa2_error ?? null,
      vad_frame_ms: (VAD_FRAME / VAD_RATE) * 1000,
    });
    console.info(
      `[bench timing flat] turn=${turn.turn} ` +
      `ttfa=${result.ttfa_ms?.toFixed?.(1) ?? "null"} ` +
      `ttfa2=${result.ttfa2_ms?.toFixed?.(1) ?? "null"} ` +
      `ttfa_wall=${result.ttfa_wall_ms?.toFixed?.(1) ?? "null"} ` +
      `ttfa_index=${result.ttfa_index_ms?.toFixed?.(1) ?? "null"} ` +
      `input_from_play=${result.first_input_from_play_ms?.toFixed?.(1) ?? "null"} ` +
      `turn_duration=${expectedEndMs?.toFixed?.(1) ?? expectedEndMs} ` +
      `index_speech_end=${turn.speech_end_ms != null ? Number(turn.speech_end_ms).toFixed(1) : "null"} ` +
      `prompt_end=${result.prompt_end_ms?.toFixed?.(1) ?? "null"} ` +
      `prompt_tail=${result.prompt_tail_ms?.toFixed?.(1) ?? "null"} ` +
      `decoded_duration=${result.decoded_duration_ms?.toFixed?.(1) ?? "null"} ` +
      `last_output_speech=${result.last_output_speech_ms?.toFixed?.(1) ?? "null"} ` +
      `output_tail=${result.output_tail_ms?.toFixed?.(1) ?? "null"} ` +
      `ended_minus_playing=${result.media_ended_minus_playing_ms?.toFixed?.(1) ?? "null"} ` +
      `playback_start_source=${result.playback_start_source ?? "null"} ` +
      `prompt_end_method=${result.prompt_end_wall_method ?? "null"} ` +
      `ttfa2_error=${result.ttfa2_error ?? "null"}`,
    );

    if (turn.speech_end_ms != null && outputTiming?.last_output_speech_ms != null) {
      const drift = Math.abs(outputTiming.last_output_speech_ms - turn.speech_end_ms);
      if (drift > 30) {
        console.warn(
          `[bench drift] turn ${turn.turn}: decoded last_output_speech ` +
          `${outputTiming.last_output_speech_ms.toFixed(1)}ms vs index ` +
          `${Number(turn.speech_end_ms).toFixed(1)}ms (drift ${drift.toFixed(1)}ms)`,
        );
      }
    }
  }
  if (ownCap) cap.stop();
  else cap.disarm();  // persistent: VAD keeps running but ignores frames between turns
  if (signal) signal.removeEventListener("abort", onAbort);
  // If we exited because of an abort, raise so the caller can discard
  // any in-flight state instead of writing a half-baked result.
  throwIfAborted();
  log(`done status=${result.status} ttfa=${result.ttfa_ms?.toFixed?.(0) ?? "—"} barge=${result.barge_in}`);
  emit("done");
  if (playbackError) console.warn("playback error:", playbackError);
  return result;
}

// --- Persistent capture helpers exposed to the orchestrator ---

// Open a capture the orchestrator (app.js Run All) can hand to each
// runBrowserTurn call so we don't tear down + recreate the audio graph
// between turns.
export { startCapture };

// Wait until the persistent capture has been silent (VAD inactive) for
// `minSilenceMs` continuous ms. Used as a gate between turns so a late
// response from a previous turn doesn't bleed into the next turn's
// detection window. Bails if signal is aborted or after timeoutMs.
export async function waitForSilence(cap, minSilenceMs = 500, timeoutMs = 5000, signal) {
  const start = performance.now();
  let quietSince = cap.isSpeechActive ? null : performance.now();
  while (true) {
    if (signal?.aborted) return false;
    if (performance.now() - start > timeoutMs) return false;
    if (cap.isSpeechActive) {
      quietSince = null;
    } else if (quietSince === null) {
      quietSince = performance.now();
    } else if (performance.now() - quietSince >= minSilenceMs) {
      return true;
    }
    await new Promise((r) => setTimeout(r, 20));
  }
}

// --- Live input meter ---

// Continuous RMS monitor for an input device. AnalyserNode is cheap
// and doesn't need to be in the audio graph downstream of destination,
// so nothing leaks to the speakers. onLevel({rms}) is called every
// animation frame (~16 ms at 60Hz). The same device can be opened
// concurrently by runBrowserTurn() — macOS multiplexes inputs fine.
export async function startMeter(inputDeviceId, onLevel) {
  const ctx = new AudioContext();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: inputDeviceId ? { exact: inputDeviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (e) {
    try { ctx.close(); } catch {}
    throw e;
  }
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0;
  src.connect(analyser);

  const buf = new Float32Array(analyser.fftSize);
  let raf = null;
  let peak = 0;
  let peakAt = 0;

  const tick = () => {
    analyser.getFloatTimeDomainData(buf);
    let sumSq = 0;
    let frameMax = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = buf[i];
      sumSq += v * v;
      const a = Math.abs(v);
      if (a > frameMax) frameMax = a;
    }
    const rms = Math.sqrt(sumSq / buf.length);
    const now = performance.now();
    // 1 s peak hold so the user can spot brief transients.
    if (frameMax > peak || now - peakAt > 1000) {
      peak = frameMax;
      peakAt = now;
    }
    onLevel({ rms, peak });
    raf = requestAnimationFrame(tick);
  };
  tick();

  return {
    stop() {
      if (raf !== null) cancelAnimationFrame(raf);
      try { src.disconnect(); } catch {}
      try { stream.getTracks().forEach((t) => t.stop()); } catch {}
      try { ctx.close(); } catch {}
    },
  };
}

// --- Result submission ---

export async function submitResult(result, baseUrl) {
  const resp = await fetch(`${baseUrl}api/results/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(result),
  });
  if (!resp.ok) {
    console.warn("submitResult failed:", resp.status, await resp.text());
  }
  return resp.ok;
}
