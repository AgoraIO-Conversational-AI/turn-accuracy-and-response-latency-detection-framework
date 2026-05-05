# L1 — Security

## API Keys

- Stored in `.env` at project root (gitignored)
- Never committed — `.env.example` provides the template
- Keys are loaded lazily by `load_env()` only when running diarization comparison
- The harness server does not require any API keys

## Network Exposure

- FastAPI server binds to `0.0.0.0:8000` — accessible on local network
- No authentication on the WebSocket or REST endpoints
- Intended for local development/testing only
- Do not expose to the internet without adding auth

## Audio Data

- Fixture files contain real conversation audio (non-sensitive test content)
- Generated output WAVs are gitignored
- No audio data is transmitted to external services by the harness
- Diarization comparison sends audio to Deepgram/Soniox/Speechmatics APIs

## Dependencies

- No pinned hashes in requirements.txt — consider adding for production
- sounddevice links to system PortAudio library
- scipy used for audio resampling (only if non-48kHz WAVs encountered)
