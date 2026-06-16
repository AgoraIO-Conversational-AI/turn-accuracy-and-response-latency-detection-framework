// browser_harness.js
// In-browser playback + capture + VAD + TTFA / barge-in detection.
// Mirrors src/harness/{audio_engine,vad_engine,turn_manager}.py so the
// numbers match when the harness is hosted remotely but driven from a
// user's Mac (BlackHole installed there).

// --- Constants (mirror Python harness) ---
export const VAD_RATE = 16000;
export const VAD_FRAME = 512;          // 32 ms @ 16 kHz
export const VAD_RMS_THRESHOLD = 0.01;
export const VAD_MIN_SILENCE_MS = 300;
export const RESPONSE_SILENCE_TIMEOUT_S = 1.5;
export const BARGE_IN_SILENCE_TIMEOUT_S = 3.0;
export const MAX_WAIT_FOR_RESPONSE_S = 10.0;

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
  // ScriptProcessor is deprecated but still ships everywhere and gives
  // us synchronous frames; an AudioWorklet would require a separate
  // file. Buffer size must be a power of 2; 512 = exactly one VAD frame
  // at 16 kHz so no buffering glitches.
  const proc = ctx.createScriptProcessor(VAD_FRAME, 1, 1);

  const vad = new BrowserVad();
  let speechStartedAt = null; // audio-time (s) of first speech_start
  let lastSpeechWall = null;  // performance.now() at most recent speech_active frame
  const startWall = performance.now();

  proc.onaudioprocess = (ev) => {
    const data = ev.inputBuffer.getChannelData(0);
    const events = vad.processChunk(data);
    for (const e of events) {
      if (e.type === "speech_start" && speechStartedAt === null) {
        speechStartedAt = e.timeS;
      }
    }
    if (vad.active) lastSpeechWall = performance.now();
  };

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
    startWall,
    get speechStartedAtMs() {
      return speechStartedAt !== null ? speechStartedAt * 1000 : null;
    },
    get lastSpeechWallMs() {
      return lastSpeechWall;
    },
    get isSpeechActive() {
      return vad.active;
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

// Plays a wav URL through one or two output devices via <audio> +
// setSinkId. Returns a controller {startWall, ended (Promise), stop()}.
// stop() halts playback and resolves `ended` immediately so a poll
// loop awaiting it can move on.
async function startPlayback(wavUrl, primaryOutputId, monitorOutputId) {
  const mkAudio = async (sinkId) => {
    const el = new Audio();
    el.preload = "auto";
    el.src = wavUrl;
    el.crossOrigin = "anonymous";
    if (sinkId && el.setSinkId) {
      try { await el.setSinkId(sinkId); }
      catch (e) { console.warn("setSinkId failed:", e); }
    }
    return el;
  };

  const primary = await mkAudio(primaryOutputId);
  const monitor =
    monitorOutputId && monitorOutputId !== primaryOutputId
      ? await mkAudio(monitorOutputId)
      : null;

  await new Promise((resolve) => {
    if (primary.readyState >= 4) return resolve();
    primary.addEventListener("canplaythrough", resolve, { once: true });
    primary.addEventListener("error", resolve, { once: true });
  });

  let endedResolve;
  const ended = new Promise((resolve) => { endedResolve = resolve; });
  primary.addEventListener("ended", () => endedResolve(), { once: true });
  const startWall = performance.now();
  primary.play();
  if (monitor) monitor.play();
  const stop = () => {
    try { primary.pause(); } catch {}
    try { if (monitor) monitor.pause(); } catch {}
    endedResolve();
  };
  return { startWall, ended, stop };
}

// --- Run a single turn ---

// turn: object from /api/turns. devices: {output, monitor, input}.
// signal (optional): AbortSignal — when aborted, capture is torn down,
// playback stopped, and the function throws an AbortError so the
// caller can identify orphan completions and discard them.
// Returns the result payload that POST /api/results/submit expects.
export async function runBrowserTurn(turn, sourceKey, devices, baseUrl, onPhase, signal) {
  const log = (...a) => console.log(`[harness#turn${turn.turn}]`, ...a);
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

  const emit = (p, partial) => phase && phase(p, partial);
  throwIfAborted();
  emit("capture_starting");
  log("capture_starting input=", devices.input?.slice?.(0, 8));
  const cap = await startCapture(devices.input);
  if (signal && signal.aborted) { cap.stop(); throwIfAborted(); }

  emit("playing");
  log(`playing wav=${wavUrl} expectedEnd=${expectedEndMs}ms`);
  let playbackEndWall = null;
  let playbackError = null;
  let playController = null;
  try {
    playController = await startPlayback(wavUrl, devices.output, devices.monitor);
  } catch (e) {
    playbackError = e;
    playbackEndWall = performance.now();
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
    barge_in: false,
    barge_in_at_ms: null,
    response_duration_ms: null,
    status: "playing",
  };
  let firstSpeechAudioMs = null;
  let ttfaAnnounced = false;

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
      const ttfa = firstSpeechAudioMs - expectedEndMs;
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
      if (result.barge_in) result.status = "barge_in";
      // Fire the phase event with a snapshot of the result so the UI
      // can flip the row's status badge + TTFA cell the instant we
      // know — without waiting for the rest of the turn to complete.
      emit(result.barge_in ? "barge_in" : "response_detected", { ...result });
      ttfaAnnounced = true;
    }

    // no-response timeout
    if (pastExpectedEnd && firstSpeechAudioMs === null) {
      const sinceEnd = elapsedS - expectedEndMs / 1000;
      if (sinceEnd > MAX_WAIT_FOR_RESPONSE_S) {
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
  cap.stop();
  if (signal) signal.removeEventListener("abort", onAbort);
  // If we exited because of an abort, raise so the caller can discard
  // any in-flight state instead of writing a half-baked result.
  throwIfAborted();
  log(`done status=${result.status} ttfa=${result.ttfa_ms?.toFixed?.(0) ?? "—"} barge=${result.barge_in}`);
  emit("done");
  if (playbackError) console.warn("playback error:", playbackError);
  return result;
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
