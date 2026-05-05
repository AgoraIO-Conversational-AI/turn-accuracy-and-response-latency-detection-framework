# L2 — Diarization Providers

## Provider Architecture

Each provider follows the same pattern:
1. Upload audio file to provider API
2. Wait for batch processing to complete (polling)
3. Parse response into normalized word list: `[{word, start, end, speaker, confidence}]`

## Deepgram

- **API**: Single synchronous POST to `/v1/listen` with audio body
- **Model**: nova-3
- **Response format**: Words array with speaker labels (0-indexed)
- **Processing**: 2-3s for 90s audio
- **Notes**: Fastest, best punctuation, correct name recognition

```python
url = "https://api.deepgram.com/v1/listen?model=nova-3&diarize=true&utterances=true"
# Auth: Token header
# Body: raw audio bytes
# Response: results.channels[0].alternatives[0].words[]
```

## Soniox

- **API**: Three-step: upload file → create transcription → poll → get transcript
- **Model**: stt-async-v4
- **Response format**: Sub-word tokens that need merging
- **Processing**: 8-9s for 90s audio
- **Notes**: Speaker labels are 1-indexed (needs `-1` normalization)

Token merging logic: new word starts when token has leading space or speaker changes.

```python
# Upload: POST /v1/files (multipart)
# Create: POST /v1/transcriptions (json config)
# Poll: GET /v1/transcriptions/{id} until status=completed
# Get: GET /v1/transcriptions/{id}/transcript
```

## Speechmatics

- **API**: Two-step: submit job → poll → get transcript
- **Model**: Default batch model
- **Response format**: Word-level results with speaker labels (S1, S2...)
- **Processing**: 5-6s for 90s audio
- **Notes**: No punctuation in batch mode, speaker labels like "S1" need parsing

```python
# Submit: POST /v2/jobs/ (multipart: data_file + config json)
# Poll: GET /v2/jobs/{id} until job.status=done
# Get: GET /v2/jobs/{id}/transcript?format=json-v2
```

## Adding a New Provider

1. Add function `diarize_newprovider(audio_path, api_key)` returning `(words, raw_response)`
2. Words must be normalized: `[{word: str, start: float_seconds, end: float_seconds, speaker: int_0indexed, confidence: float}]`
3. Add CLI argument `--newprovider-key`
4. Add env var to `.env.example`
5. Add dispatch in `main()` function

## Comparison Algorithm

`compare_providers()` works by:
1. Dividing the audio timeline into 0.5s bins
2. For each bin, determining which speaker each provider assigns
3. Normalizing speaker labels across providers (co-occurrence mapping)
4. Counting agreement/disagreement bins

The label normalization handles cases where Provider A calls someone "Speaker 0" while Provider B calls the same person "Speaker 1".
