"""FastAPI server with WebSocket for real-time turn playback control."""

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio_engine import AudioEngine, list_devices
from .turn_manager import TurnManager, TurnState
from .vad_engine import VadEngine

app = FastAPI(title="Turn Playback Harness")

STATIC_DIR = Path(__file__).parent / "static"

# global instances — initialized on startup
audio_engine: AudioEngine | None = None
vad_engine: VadEngine | None = None
turn_manager: TurnManager | None = None

# connected WebSocket clients
ws_clients: set[WebSocket] = set()

# async queue for events from turn_manager (thread-safe bridge)
event_queue: asyncio.Queue | None = None


async def broadcast(msg: dict):
    """Send JSON message to all connected WebSocket clients."""
    global ws_clients
    text = json.dumps(msg)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


def on_turn_event(event_type: str, data: dict):
    """Callback from TurnManager (called from worker threads).

    Puts events into the async queue for the event loop to pick up.
    """
    if event_queue is not None:
        try:
            event_queue.put_nowait({"type": event_type, **data})
        except asyncio.QueueFull:
            pass


async def event_dispatcher():
    """Background task that forwards events from the queue to WebSocket clients."""
    while True:
        msg = await event_queue.get()
        await broadcast(msg)


@app.on_event("startup")
async def startup():
    global audio_engine, vad_engine, turn_manager, event_queue

    event_queue = asyncio.Queue(maxsize=1000)
    audio_engine = AudioEngine()
    vad_engine = VadEngine()
    turn_manager = TurnManager(
        audio_engine=audio_engine,
        vad_engine=vad_engine,
        on_event=on_turn_event,
    )

    asyncio.create_task(event_dispatcher())


# --- REST endpoints ---

@app.get("/api/devices")
async def get_devices():
    devices = list_devices()
    active = audio_engine.devices if audio_engine else {}
    return {"devices": devices, "active": active}


@app.get("/api/turns")
async def get_turns(speaker: int | None = None):
    turns = turn_manager.get_turns(speaker) if turn_manager else []
    return {"turns": turns}


@app.get("/api/sources")
async def get_sources():
    return {"sources": turn_manager.get_sources() if turn_manager else []}


@app.post("/api/sources/{source_key}")
async def set_source(source_key: str):
    if turn_manager is None:
        return {"error": "not initialized"}
    try:
        turn_manager.set_source(source_key)
        return {"sources": turn_manager.get_sources()}
    except ValueError as e:
        return {"error": str(e)}


@app.get("/api/results")
async def get_results():
    if turn_manager is None:
        return {"results": []}
    return turn_manager.get_summary()


@app.post("/api/devices/configure")
async def configure_devices(config: dict):
    """Update audio device assignments."""
    if audio_engine is None:
        return {"error": "not initialized"}
    if "blackhole_2ch" in config:
        audio_engine.bh2_idx = config["blackhole_2ch"]
    if "speakers" in config:
        audio_engine.spk_idx = config["speakers"]
    if "blackhole_16ch" in config:
        audio_engine.bh16_idx = config["blackhole_16ch"]
    return {"active": audio_engine.devices}


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)

    # send initial state
    await ws.send_text(json.dumps({
        "type": "init",
        "devices": audio_engine.devices if audio_engine else {},
        "state": turn_manager.run.state.value if turn_manager else "idle",
    }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "run_all":
                if turn_manager.run.running or turn_manager.run.state != TurnState.IDLE:
                    await ws.send_text(json.dumps({"type": "error", "error": "run already in progress"}))
                else:
                    speaker = msg.get("speaker")
                    asyncio.create_task(turn_manager.run_all(speaker=speaker))

            elif action == "run_single":
                if turn_manager.run.running or turn_manager.run.state != TurnState.IDLE:
                    await ws.send_text(json.dumps({"type": "error", "error": "run already in progress"}))
                else:
                    turn_idx = msg.get("turn", 0)
                    asyncio.create_task(turn_manager.run_single_turn(turn_idx))

            elif action == "stop":
                turn_manager.stop()
                await broadcast({"type": "stopped"})

            elif action == "reset":
                turn_manager.reset()
                await ws.send_text(json.dumps({"type": "reset"}))

            elif action == "set_source":
                source_key = msg.get("source")
                if source_key:
                    try:
                        turn_manager.set_source(source_key)
                        await broadcast({
                            "type": "source_changed",
                            "sources": turn_manager.get_sources(),
                        })
                    except ValueError as e:
                        await ws.send_text(json.dumps({"type": "error", "error": str(e)}))

            elif action == "get_results":
                summary = turn_manager.get_summary()
                await ws.send_text(json.dumps({"type": "results", **summary}))

    except WebSocketDisconnect:
        ws_clients.discard(ws)


# --- Static files ---

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    uvicorn.run(
        "src.harness.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
