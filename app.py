"""
DeepFilterNet Speech Enhancement API
=====================================
Provides REST endpoints for real-time noise reduction using DeepFilterNet3.

Endpoints:
  POST /enhance          - Enhance a single audio file
  POST /enhance/batch    - Enhance multiple audio files
  WS   /enhance/stream   - Streaming enhancement via WebSocket
  GET  /health           - Health check
"""

import io
import os
import asyncio
import subprocess
import tempfile
import uuid
import zipfile
from typing import Optional

import numpy as np
import torch
import torchaudio
import imageio_ffmpeg
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
SUPPORTED_NATIVE = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


# ── Model Initialization ────────────────────────────────────────────────────
from df.enhance import init_df, enhance
from df.io import load_audio, resample

print("Loading DeepFilterNet3 model...")
model, df_state, _ = init_df()
MODEL_SR = df_state.sr()  # 48000
print(f"Model ready  –  sample rate: {MODEL_SR} Hz")

app = FastAPI(
    title="DeepFilterNet Speech Enhancement API",
    version="1.0.0",
    description="Real-time speech enhancement using DeepFilterNet3",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _enhance_tensor(audio: torch.Tensor, atten_lim_db: Optional[float] = None) -> torch.Tensor:
    """Run enhancement on a [C, T] tensor already at MODEL_SR."""
    return enhance(model, df_state, audio, atten_lim_db=atten_lim_db)


def _convert_to_wav(input_path: str) -> str:
    """Convert any audio format to WAV using ffmpeg. Returns path to WAV file."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in SUPPORTED_NATIVE:
        return input_path
    wav_path = input_path + ".converted.wav"
    result = subprocess.run(
        [FFMPEG_BIN, "-y", "-i", input_path, "-ar", str(MODEL_SR), "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()[-500:]}")
    return wav_path


def _load_and_enhance(audio_bytes: bytes, filename: str, atten_lim_db: Optional[float] = None) -> bytes:
    """Load raw upload bytes (any format), enhance, return WAV bytes."""
    uid = uuid.uuid4().hex
    tmp_in = os.path.join(tempfile.gettempdir(), f"dfin_{uid}_{filename}")
    wav_path = None
    try:
        with open(tmp_in, "wb") as f:
            f.write(audio_bytes)

        # Convert to WAV if needed (m4a, mp3, aac, opus, webm, etc.)
        wav_path = _convert_to_wav(tmp_in)

        audio, _ = load_audio(wav_path, sr=MODEL_SR, verbose=False)
        enhanced = _enhance_tensor(audio, atten_lim_db=atten_lim_db)

        buf = io.BytesIO()
        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)
        # Save as 16-bit PCM WAV
        enhanced_int16 = (enhanced * (1 << 15)).to(torch.int16)
        torchaudio.save(buf, enhanced_int16, MODEL_SR, format="wav")
        buf.seek(0)
        return buf.read()
    finally:
        if os.path.exists(tmp_in):
            os.unlink(tmp_in)
        if wav_path and wav_path != tmp_in and os.path.exists(wav_path):
            os.unlink(wav_path)


# ── Routes ───────────────────────────────────────────────────────────────────

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def home():
    """Browser-friendly test page for all endpoints."""
    return """<!DOCTYPE html>
<html><head><title>DeepFilterNet API</title>
<style>
body{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#f5f5f5}
h1{color:#333}h2{color:#555;border-bottom:1px solid #ddd;padding-bottom:8px}
.card{background:#fff;padding:20px;border-radius:8px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)}
input[type=file]{margin:10px 0}button{background:#2563eb;color:#fff;border:none;padding:10px 24px;border-radius:6px;cursor:pointer;font-size:15px}
button:hover{background:#1d4ed8}
#status{margin-top:12px;padding:10px;border-radius:4px;display:none}
.ok{background:#d1fae5;color:#065f46;display:block!important}.err{background:#fee2e2;color:#991b1b;display:block!important}
</style></head><body>
<h1>DeepFilterNet Speech Enhancement API</h1>
<div class="card"><h2>Single File</h2>
<form id="singleForm"><input type="file" name="file" accept="audio/*" required>
<br><label>Attenuation limit (dB, optional): <input type="number" name="atten_lim_db" placeholder="e.g. 20" step="1"></label>
<br><br><button type="submit">Enhance</button></form>
<div id="singleStatus" id="status"></div></div>

<div class="card"><h2>Batch Files</h2>
<form id="batchForm"><input type="file" name="files" accept="audio/*" multiple required>
<p style="color:#666;font-size:13px">Hold Ctrl/Cmd to select multiple files</p>
<label>Attenuation limit (dB, optional): <input type="number" name="atten_lim_db" placeholder="e.g. 20" step="1"></label>
<br><br><button type="submit">Enhance Batch</button></form>
<div id="batchStatus" id="status"></div></div>

<div class="card"><h2>API Docs</h2>
<p><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p></div>

<script>
async function submitForm(formId, url, statusId, fileField) {
  const form = document.getElementById(formId);
  const status = document.getElementById(statusId);
  const fd = new FormData(form);
  status.className='';status.style.display='block';status.textContent='Processing...';
  try {
    const r = await fetch(url, {method:'POST', body:fd});
    if(!r.ok) throw new Error(await r.text());
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = url.includes('batch') ? 'enhanced_batch.zip' : 'enhanced_audio.wav';
    a.click();
    status.className='ok';status.textContent='Done! Download started.';
  } catch(e) {status.className='err';status.textContent='Error: '+e.message;}
}
document.getElementById('singleForm').onsubmit=e=>{e.preventDefault();submitForm('singleForm','/enhance','singleStatus','file')};
document.getElementById('batchForm').onsubmit=e=>{e.preventDefault();submitForm('batchForm','/enhance/batch','batchStatus','files')};
</script></body></html>"""


@app.get("/health")
async def health():
    return {"status": "ok", "model": "DeepFilterNet3", "sample_rate": MODEL_SR}


@app.post("/enhance")
async def enhance_single(
    file: UploadFile = File(...),
    atten_lim_db: Optional[float] = Form(None, description="Noise attenuation limit in dB (e.g. 20). None = full enhancement"),
):
    """Enhance a single audio file. Returns enhanced WAV."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty file")

    loop = asyncio.get_event_loop()
    enhanced_wav = await loop.run_in_executor(
        None, _load_and_enhance, audio_bytes, file.filename or "input.wav", atten_lim_db
    )

    return StreamingResponse(
        io.BytesIO(enhanced_wav),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="enhanced_{file.filename}"'},
    )


@app.post("/enhance/batch")
async def enhance_batch(
    files: list[UploadFile] = File(..., description="Select multiple audio files (hold Ctrl/Cmd to select more than one)"),
    atten_lim_db: Optional[float] = Form(None, description="Noise attenuation limit in dB (e.g. 20). None = full enhancement"),
):
    """
    Enhance multiple audio files at once. Returns a ZIP containing all enhanced WAVs.

    **How to upload multiple files:**
    - **Swagger UI**: Click "Add string item" or hold Ctrl/Cmd while selecting files
    - **cURL**: Use multiple `-F "files=@file.wav"` flags
    - **Python requests**: Pass a list of tuples: `files=[("files", open("a.wav","rb")), ("files", open("b.wav","rb"))]`

    Supports: wav, mp3, m4a, aac, ogg, opus, flac, webm, wma, and more.
    """
    if not files:
        raise HTTPException(400, "No files provided")

    loop = asyncio.get_event_loop()

    async def process_one(f: UploadFile):
        data = await f.read()
        return f.filename, await loop.run_in_executor(
            None, _load_and_enhance, data, f.filename or "input.wav", atten_lim_db
        )

    results = await asyncio.gather(*(process_one(f) for f in files))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, wav_bytes in results:
            zf.writestr(f"enhanced_{fname}", wav_bytes)
    zip_buf.seek(0)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="enhanced_batch.zip"'},
    )


# ── WebSocket Streaming ─────────────────────────────────────────────────────

@app.websocket("/enhance/stream")
async def enhance_stream(ws: WebSocket):
    """
    Streaming speech enhancement over WebSocket.

    Protocol:
      1. Client sends a JSON config message:
         {"sample_rate": 16000, "channels": 1, "atten_lim_db": null}
      2. Client sends binary frames of raw PCM int16 audio chunks.
      3. Server replies with binary frames of enhanced PCM int16 audio at 48 kHz.
      4. Client sends text "END" to signal end of stream.

    The model requires 48 kHz input. If the client's sample_rate differs,
    the server resamples automatically in both directions.
    """
    await ws.accept()
    try:
        # Step 1: receive config
        config = await ws.receive_json()
        client_sr = int(config.get("sample_rate", MODEL_SR))
        channels = int(config.get("channels", 1))
        atten_lim_db = config.get("atten_lim_db", None)

        await ws.send_json({"status": "ready", "model_sr": MODEL_SR})

        # Step 2: process audio chunks
        while True:
            msg = await ws.receive()

            if msg.get("text") == "END":
                await ws.send_json({"status": "done"})
                break

            raw = msg.get("bytes")
            if raw is None:
                continue

            # Decode PCM int16 → float tensor
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / (1 << 15)
            audio = torch.from_numpy(pcm).reshape(channels, -1)

            # Resample to model SR if needed
            if client_sr != MODEL_SR:
                audio = resample(audio, client_sr, MODEL_SR)

            # Enhance
            loop = asyncio.get_event_loop()
            enhanced = await loop.run_in_executor(None, _enhance_tensor, audio, atten_lim_db)

            # Resample back to client SR if needed
            if client_sr != MODEL_SR:
                enhanced = resample(enhanced, MODEL_SR, client_sr)

            # Convert back to int16 bytes
            out_pcm = (enhanced * (1 << 15)).to(torch.int16).numpy().tobytes()
            await ws.send_bytes(out_pcm)

    except WebSocketDisconnect:
        pass


# ── Fix Swagger UI for batch file upload ─────────────────────────────────────

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title, version=app.version, description=app.description, routes=app.routes,
    )
    # Fix the batch endpoint schema so Swagger UI shows file upload buttons (not "Add string item")
    for name, comp in schema.get("components", {}).get("schemas", {}).items():
        if "enhance_batch" in name:
            props = comp.get("properties", {})
            if "files" in props:
                props["files"] = {
                    "type": "array",
                    "items": {"type": "string", "format": "binary"},
                    "title": "Audio Files",
                    "description": "Select multiple audio files (use Add Item to add more)",
                }
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
