# DeepFilterNet Speech Enhancement API

A FastAPI-based REST API for real-time speech enhancement using [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet). Removes background noise and improves speech clarity using deep learning frequency-domain filtering.

## Features

- **Single file enhancement** — Upload one audio file, get back enhanced WAV
- **Batch enhancement** — Upload multiple files, get back a ZIP of enhanced WAVs
- **WebSocket streaming** — Stream audio chunks in real-time for live enhancement
- **Any audio format** — Supports WAV, MP3, M4A/AAC, OGG, OPUS, FLAC, WebM, WMA, and more (auto-converts via ffmpeg)
- **Adjustable noise reduction** — Optional `atten_lim_db` parameter to control enhancement intensity

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/` | Browser-friendly upload page |
| `POST` | `/enhance` | Enhance a single audio file |
| `POST` | `/enhance/batch` | Enhance multiple files (returns ZIP) |
| `WS` | `/enhance/stream` | Real-time streaming enhancement |

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

### 3. Test

```bash
# Health check
curl http://localhost:8000/health

# Enhance a single file
curl -X POST http://localhost:8000/enhance \
  -F "file=@noisy_audio.wav" \
  -o enhanced.wav

# Enhance an M4A file with attenuation limit
curl -X POST http://localhost:8000/enhance \
  -F "file=@recording.m4a" \
  -F "atten_lim_db=20" \
  -o enhanced.wav

# Batch enhance multiple files
curl -X POST http://localhost:8000/enhance/batch \
  -F "files=@file1.wav" \
  -F "files=@file2.m4a" \
  -F "files=@file3.mp3" \
  -o enhanced_batch.zip
```

### 4. Interactive docs

Open `http://localhost:8000/docs` for Swagger UI or `http://localhost:8000` for a simple browser upload page.

## WebSocket Streaming Protocol

```
1. Connect to ws://localhost:8000/enhance/stream
2. Send JSON config:  {"sample_rate": 16000, "channels": 1, "atten_lim_db": null}
3. Send binary frames of raw PCM int16 audio chunks
4. Receive enhanced PCM int16 audio chunks back
5. Send text "END" to signal completion
```

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

## Tech Stack

- [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) — Deep learning speech enhancement model
- [FastAPI](https://fastapi.tiangolo.com/) — Async web framework
- [PyTorch](https://pytorch.org/) — Model inference
- [ffmpeg](https://ffmpeg.org/) (via imageio-ffmpeg) — Multi-format audio conversion

## License

The DeepFilterNet model is licensed under MIT/Apache-2.0. See [DeepFilterNet LICENSE](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE).
