# DeepFilterNet Speech Enhancement API

A FastAPI-based REST API for real-time speech enhancement using [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet). Removes background noise and improves speech clarity using deep learning frequency-domain filtering.

## Features

- **Async job tracking** — Submit audio, get a `job_id` instantly, poll for progress, download when ready
- **Per-file attenuation** — Set different noise reduction levels for each file in a batch
- **Single file enhancement** — Upload one audio file, track and download result
- **Batch enhancement** — Upload multiple files with individual settings, get a ZIP or download files individually
- **WebSocket streaming** — Stream audio chunks in real-time for live enhancement
- **Any audio format** — Supports WAV, MP3, M4A/AAC, OGG, OPUS, FLAC, WebM, WMA, and more (auto-converts via ffmpeg)
- **Browser UI** — Upload page with per-file controls and live job polling at `/`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Browser-friendly upload page |
| `GET` | `/health` | Health check (model info, active jobs) |
| `POST` | `/enhance` | Submit single file → returns `job_id` |
| `POST` | `/enhance/batch` | Submit multiple files with per-file attenuation → returns `job_id` |
| `GET` | `/status/{job_id}` | Check job progress (per-file status) |
| `GET` | `/download/{job_id}` | Download all results (WAV for single, ZIP for batch) |
| `GET` | `/download/{job_id}/{filename}` | Download a specific file from a batch (no need to wait for all) |
| `GET` | `/jobs` | List all jobs |
| `WS` | `/enhance/stream` | Real-time streaming enhancement via WebSocket |

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install API dependencies
pip install -r requirements.txt

# Install ffmpeg for multi-format support
pip install imageio-ffmpeg
```

> **Note:** `deepfilternet` requires Rust to compile its native extension. Install Rust via [rustup.rs](https://rustup.rs/) if not already available.

### 2. Run the server

```bash
python app.py
# Server starts on http://0.0.0.0:8000
```

### 3. Interactive docs

- **Browser UI**: `http://localhost:8000` — upload page with per-file attenuation and job polling
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Usage Examples

### Single File Enhancement

```bash
# Submit
curl -X POST http://localhost:8000/enhance \
  -F "file=@noisy_audio.wav" \
  -F "atten_lim_db=20"

# Response: {"job_id": "abc123", "status": "queued", "message": "..."}

# Poll status
curl http://localhost:8000/status/abc123

# Response: {"job_id": "abc123", "status": "completed", "progress": "1/1", ...}

# Download result
curl http://localhost:8000/download/abc123 -o enhanced.wav
```

### Batch Enhancement with Per-File Attenuation

Each file gets its own noise reduction level:

```bash
# Submit with per-file settings
curl -X POST http://localhost:8000/enhance/batch \
  -F "files=@meeting.wav" \
  -F "files=@call.m4a" \
  -F "files=@interview.mp3" \
  -F 'settings_json={"meeting.wav": 10, "call.m4a": 25, "interview.mp3": null}'

# Response:
# {
#   "job_id": "def456",
#   "total_files": 3,
#   "files": {
#     "meeting.wav":   {"atten_lim_db": 10,   "status": "queued"},
#     "call.m4a":      {"atten_lim_db": 25,   "status": "queued"},
#     "interview.mp3": {"atten_lim_db": null,  "status": "queued"}
#   }
# }
```

- `10` = light noise reduction (10 dB limit)
- `25` = heavy noise reduction (25 dB limit)
- `null` = full enhancement (no limit)

### Polling Job Status

```bash
curl http://localhost:8000/status/def456

# Response:
# {
#   "job_id": "def456",
#   "status": "processing",
#   "progress": "2/3",
#   "files": {
#     "meeting.wav":   {"status": "completed"},
#     "call.m4a":      {"status": "processing"},
#     "interview.mp3": {"status": "queued"}
#   },
#   "download_url": null
# }
```

### Downloading Results

```bash
# Download all as ZIP (when job is completed)
curl http://localhost:8000/download/def456 -o batch.zip

# Download a specific file (as soon as it's done, no need to wait for others)
curl http://localhost:8000/download/def456/meeting.wav -o meeting_enhanced.wav
```

### List All Jobs

```bash
curl http://localhost:8000/jobs

# Response:
# [
#   {"job_id": "def456", "status": "completed", "job_type": "batch", "total_files": 3, "completed": 3},
#   {"job_id": "abc123", "status": "completed", "job_type": "single", "total_files": 1, "completed": 1}
# ]
```

### Python Client Example

```python
import requests, json, time

BASE = "http://localhost:8000"

# Submit batch with per-file attenuation
files = [
    ("files", ("meeting.wav", open("meeting.wav", "rb"))),
    ("files", ("call.m4a", open("call.m4a", "rb"))),
]
settings = {"meeting.wav": 15, "call.m4a": 25}

r = requests.post(
    f"{BASE}/enhance/batch",
    files=files,
    data={"settings_json": json.dumps(settings)},
)
job_id = r.json()["job_id"]
print(f"Submitted: {job_id}")

# Poll until done
while True:
    status = requests.get(f"{BASE}/status/{job_id}").json()
    print(f"Progress: {status['progress']} — {status['status']}")
    if status["status"] == "completed":
        break
    time.sleep(2)

# Download ZIP
r = requests.get(f"{BASE}/download/{job_id}")
with open("enhanced_batch.zip", "wb") as f:
    f.write(r.content)
print("Downloaded!")
```

## WebSocket Streaming Protocol

For real-time audio enhancement (e.g., live microphone input):

```
1. Connect to ws://localhost:8000/enhance/stream
2. Send JSON config:  {"sample_rate": 16000, "channels": 1, "atten_lim_db": null}
3. Send binary frames of raw PCM int16 audio chunks
4. Receive enhanced PCM int16 audio chunks back
5. Send text "END" to signal completion
```

The server auto-resamples if your sample rate differs from the model's 48kHz.

### Python streaming example

```python
import asyncio, json, websockets, soundfile as sf, numpy as np

async def stream_enhance(input_file, output_file):
    data, sr = sf.read(input_file, dtype="int16")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    channels = data.shape[1]
    raw = data.tobytes()
    chunk_bytes = int(sr * 0.2) * channels * 2  # 200ms chunks

    enhanced_chunks = []
    async with websockets.connect("ws://localhost:8000/enhance/stream") as ws:
        await ws.send(json.dumps({"sample_rate": sr, "channels": channels}))
        config = json.loads(await ws.recv())

        offset = 0
        while offset < len(raw):
            await ws.send(raw[offset : offset + chunk_bytes])
            enhanced_chunks.append(await ws.recv())
            offset += chunk_bytes

        await ws.send("END")
        json.loads(await ws.recv())

    out_raw = b"".join(enhanced_chunks)
    out_data = np.frombuffer(out_raw, dtype=np.int16).reshape(-1, channels)
    sf.write(output_file, out_data, sr, subtype="PCM_16")

asyncio.run(stream_enhance("noisy.wav", "enhanced.wav"))
```

## Expose via Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

This gives you a public `https://*.trycloudflare.com` URL instantly.

## Supported Audio Formats

| Format | Extensions |
|--------|-----------|
| WAV | `.wav` |
| MP3 | `.mp3` |
| M4A/AAC | `.m4a`, `.aac` |
| OGG/Vorbis | `.ogg` |
| Opus | `.opus` |
| FLAC | `.flac` |
| WebM | `.webm` |
| WMA | `.wma` |
| Any ffmpeg-supported format | ... |

Non-WAV formats are automatically converted using ffmpeg before processing.

## Tech Stack

- [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) — Deep learning speech enhancement model
- [FastAPI](https://fastapi.tiangolo.com/) — Async web framework
- [PyTorch](https://pytorch.org/) — Model inference
- [ffmpeg](https://ffmpeg.org/) (via imageio-ffmpeg) — Multi-format audio conversion

## License

The DeepFilterNet model is licensed under MIT/Apache-2.0. See [DeepFilterNet LICENSE](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE).
