"""
DeepFilterNet Speech Enhancement API
=====================================
Provides REST endpoints for real-time noise reduction using DeepFilterNet3.

Endpoints:
  POST /enhance              - Submit single file for enhancement (returns job_id)
  POST /enhance/batch        - Submit multiple files with per-file attenuation (returns job_id)
  GET  /status/{job_id}      - Check job progress
  GET  /download/{job_id}    - Download enhanced result when ready
  GET  /download/{job_id}/{filename} - Download a specific file from batch job
  WS   /enhance/stream       - Real-time streaming enhancement via WebSocket
  GET  /health               - Health check
  GET  /                     - Browser UI
"""

import io
import json
import os
import asyncio
import subprocess
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
import soundfile as sf_io
import imageio_ffmpeg
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
SUPPORTED_NATIVE = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

# ── Model Initialization ────────────────────────────────────────────────────
from df.enhance import init_df, enhance
from df.io import load_audio, resample

print("Loading DeepFilterNet3 model...")
model, df_state, _ = init_df()
MODEL_SR = df_state.sr()  # 48000
print(f"Model ready  –  sample rate: {MODEL_SR} Hz")

# ── CUDA Warmup (eliminates cold-start latency on first request) ────────────
print("Warming up CUDA kernels...")
_warmup_audio = torch.randn(1, MODEL_SR * 2)  # 2 seconds of dummy audio
_ = enhance(model, df_state, _warmup_audio)
del _warmup_audio
torch.cuda.synchronize() if torch.cuda.is_available() else None
print("Warmup complete — ready for low-latency inference")

app = FastAPI(
    title="DeepFilterNet Speech Enhancement API",
    version="2.0.0",
    description="Real-time speech enhancement using DeepFilterNet3 with async job tracking",
)

executor = ThreadPoolExecutor(max_workers=1)  # model is not thread-safe, serialize inference
_processing_queue: asyncio.Queue = None  # initialized on startup


async def _job_worker():
    """Background worker that processes files from the queue one at a time."""
    global _processing_queue
    _processing_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    while True:
        job_id, filename, audio_bytes, atten_lim_db = await _processing_queue.get()
        try:
            job = jobs.get(job_id)
            if not job or filename not in job.files:
                continue
            fr = job.files[filename]
            fr.status = JobStatus.PROCESSING
            _update_job_status(job)
            fr.result_bytes = await loop.run_in_executor(
                executor, _load_and_enhance, audio_bytes, filename, atten_lim_db
            )
            fr.status = JobStatus.COMPLETED
        except Exception as e:
            fr.status = JobStatus.FAILED
            fr.error = str(e)
        finally:
            _update_job_status(job)
            _processing_queue.task_done()


@app.on_event("startup")
async def startup():
    asyncio.create_task(_job_worker())


# ── Job Store ────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileResult:
    filename: str
    atten_lim_db: Optional[float] = None
    status: JobStatus = JobStatus.QUEUED
    result_bytes: Optional[bytes] = None
    error: Optional[str] = None


@dataclass
class Job:
    job_id: str
    created_at: float
    job_type: str  # "single" or "batch"
    status: JobStatus = JobStatus.QUEUED
    files: dict[str, FileResult] = field(default_factory=dict)  # filename -> FileResult
    error: Optional[str] = None


# In-memory job store
jobs: dict[str, Job] = {}


# ── Audio Processing Helpers ─────────────────────────────────────────────────

def _enhance_tensor(audio: torch.Tensor, atten_lim_db: Optional[float] = None) -> torch.Tensor:
    return enhance(model, df_state, audio, atten_lim_db=atten_lim_db)


def _convert_to_wav(input_path: str) -> str:
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
    uid = uuid.uuid4().hex
    tmp_in = os.path.join(tempfile.gettempdir(), f"dfin_{uid}_{filename}")
    wav_path = None
    try:
        with open(tmp_in, "wb") as f:
            f.write(audio_bytes)
        wav_path = _convert_to_wav(tmp_in)
        audio, _ = load_audio(wav_path, sr=MODEL_SR, verbose=False)
        enhanced = _enhance_tensor(audio, atten_lim_db=atten_lim_db)
        buf = io.BytesIO()
        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)
        enhanced_int16 = (enhanced * (1 << 15)).to(torch.int16).numpy().T
        sf_io.write(buf, enhanced_int16, MODEL_SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
    finally:
        if os.path.exists(tmp_in):
            os.unlink(tmp_in)
        if wav_path and wav_path != tmp_in and os.path.exists(wav_path):
            os.unlink(wav_path)



def _update_job_status(job: Job):
    """Derive overall job status from individual file statuses."""
    statuses = [fr.status for fr in job.files.values()]
    if all(s == JobStatus.COMPLETED for s in statuses):
        job.status = JobStatus.COMPLETED
    elif any(s == JobStatus.FAILED for s in statuses) and not any(s in (JobStatus.QUEUED, JobStatus.PROCESSING) for s in statuses):
        job.status = JobStatus.COMPLETED  # partially completed
    elif any(s == JobStatus.PROCESSING for s in statuses):
        job.status = JobStatus.PROCESSING
    else:
        job.status = JobStatus.PROCESSING


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    """Browser-friendly test page."""
    return """<!DOCTYPE html>
<html><head><title>DeepFilterNet API v2</title>
<style>
body{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#f5f5f5}
h1{color:#333}h2{color:#555;border-bottom:1px solid #ddd;padding-bottom:8px}
.card{background:#fff;padding:20px;border-radius:8px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)}
input[type=file],input[type=number]{margin:8px 0}
button{background:#2563eb;color:#fff;border:none;padding:10px 24px;border-radius:6px;cursor:pointer;font-size:15px;margin:4px}
button:hover{background:#1d4ed8}
button.dl{background:#059669}button.dl:hover{background:#047857}
.status-box{margin-top:12px;padding:12px;border-radius:4px;background:#f0f0f0;font-family:monospace;font-size:13px;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;margin:10px 0}td,th{padding:6px 10px;border:1px solid #ddd;text-align:left;font-size:13px}
th{background:#f9f9f9}.ok{color:#059669;font-weight:bold}.err{color:#dc2626;font-weight:bold}.prog{color:#d97706;font-weight:bold}
</style></head><body>
<h1>DeepFilterNet Speech Enhancement API <small style="color:#999">v2</small></h1>

<div class="card"><h2>Single File</h2>
<input type="file" id="singleFile" accept="audio/*">
<br><label>Attenuation limit (dB): <input type="number" id="singleAtten" placeholder="optional" step="1"></label>
<br><button onclick="submitSingle()">Submit</button>
<div id="singleOut" class="status-box" style="display:none"></div></div>

<div class="card"><h2>Batch Files (per-file attenuation)</h2>
<input type="file" id="batchFiles" accept="audio/*" multiple>
<p style="color:#666;font-size:13px">After selecting files, set attenuation per file below:</p>
<div id="batchConfig"></div>
<button onclick="submitBatch()">Submit Batch</button>
<div id="batchOut" class="status-box" style="display:none"></div></div>

<div class="card"><h2>Check Job Status</h2>
<input type="text" id="jobIdInput" placeholder="Enter job ID" style="width:320px;padding:6px">
<button onclick="checkStatus()">Check</button>
<div id="statusOut" class="status-box" style="display:none"></div></div>

<div class="card"><h2>API Docs</h2>
<p><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p></div>

<script>
const B=window.location.origin;
document.getElementById('batchFiles').onchange=function(){
  const c=document.getElementById('batchConfig');c.innerHTML='';
  Array.from(this.files).forEach((f,i)=>{
    c.innerHTML+=`<div style="margin:4px 0"><b>${f.name}</b> — atten_lim_db: <input type="number" id="atten_${i}" placeholder="null" step="1" style="width:80px"></div>`;
  });
};
async function submitSingle(){
  const out=document.getElementById('singleOut');out.style.display='block';out.textContent='Uploading...';
  const fd=new FormData();fd.append('file',document.getElementById('singleFile').files[0]);
  const a=document.getElementById('singleAtten').value;if(a)fd.append('atten_lim_db',a);
  const r=await fetch('/enhance',{method:'POST',body:fd});const j=await r.json();
  out.textContent=JSON.stringify(j,null,2);pollJob(j.job_id,'singleOut');
}
async function submitBatch(){
  const out=document.getElementById('batchOut');out.style.display='block';out.textContent='Uploading...';
  const fd=new FormData();const files=document.getElementById('batchFiles').files;
  const settings={};
  Array.from(files).forEach((f,i)=>{
    fd.append('files',f);
    const v=document.getElementById('atten_'+i).value;
    settings[f.name]=v?parseFloat(v):null;
  });
  fd.append('settings_json',JSON.stringify(settings));
  const r=await fetch('/enhance/batch',{method:'POST',body:fd});const j=await r.json();
  out.textContent=JSON.stringify(j,null,2);pollJob(j.job_id,'batchOut');
}
async function pollJob(id,outId){
  const out=document.getElementById(outId);
  const iv=setInterval(async()=>{
    const r=await fetch('/status/'+id);const j=await r.json();
    let txt=JSON.stringify(j,null,2);
    if(j.status==='completed'){
      clearInterval(iv);
      txt+='\\n\\n--- Download ready ---';
      txt+='\\nGET '+B+'/download/'+id;
    }
    out.textContent=txt;
  },300);
}
async function checkStatus(){
  const id=document.getElementById('jobIdInput').value.trim();if(!id)return;
  const out=document.getElementById('statusOut');out.style.display='block';
  const r=await fetch('/status/'+id);out.textContent=JSON.stringify(await r.json(),null,2);
}
</script></body></html>"""


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "DeepFilterNet3",
        "sample_rate": MODEL_SR,
        "active_jobs": sum(1 for j in jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)),
    }


# ── Submit Endpoints (return job_id immediately) ─────────────────────────────

@app.post("/enhance")
async def enhance_single(
    file: UploadFile = File(...),
    atten_lim_db: Optional[float] = Form(None, description="Noise attenuation limit in dB (e.g. 20). None = full enhancement"),
):
    """Submit a single audio file for enhancement. Returns a job_id to track progress."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty file")

    filename = file.filename or "input.wav"
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id=job_id, created_at=time.time(), job_type="single")
    job.files[filename] = FileResult(filename=filename, atten_lim_db=atten_lim_db)
    jobs[job_id] = job

    await _processing_queue.put((job_id, filename, audio_bytes, atten_lim_db))

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Processing '{filename}'. Check GET /status/{job_id} for progress.",
    }


@app.post("/enhance/batch")
async def enhance_batch(
    files: list[UploadFile] = File(..., description="Select multiple audio files"),
    settings_json: Optional[str] = Form(
        None,
        description='JSON object mapping filename to atten_lim_db. Example: {"file1.wav": 15, "file2.m4a": null, "file3.mp3": 25}',
    ),
):
    """
    Submit multiple audio files for enhancement with **per-file attenuation settings**.

    Returns a `job_id` immediately. Each file is processed independently — track progress
    via `GET /status/{job_id}` and download results via `GET /download/{job_id}`.

    **Per-file attenuation:** Pass `settings_json` as a JSON string mapping each filename
    to its `atten_lim_db` value. Files not in the map use full enhancement (null).

    **cURL example:**
    ```
    curl -X POST /enhance/batch \\
      -F "files=@meeting.wav" \\
      -F "files=@call.m4a" \\
      -F 'settings_json={"meeting.wav": 15, "call.m4a": 25}'
    ```
    """
    if not files:
        raise HTTPException(400, "No files provided")

    # Parse per-file settings
    settings = {}
    if settings_json:
        try:
            settings = json.loads(settings_json)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid settings_json — must be valid JSON")

    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id=job_id, created_at=time.time(), job_type="batch")

    # Read all files and queue them
    file_data = []
    for f in files:
        audio_bytes = await f.read()
        filename = f.filename or f"input_{uuid.uuid4().hex[:6]}.wav"
        atten = settings.get(filename, None)
        job.files[filename] = FileResult(filename=filename, atten_lim_db=atten)
        file_data.append((filename, audio_bytes, atten))

    jobs[job_id] = job

    for filename, audio_bytes, atten in file_data:
        await _processing_queue.put((job_id, filename, audio_bytes, atten))

    return {
        "job_id": job_id,
        "status": "queued",
        "total_files": len(job.files),
        "files": {
            fname: {"atten_lim_db": fr.atten_lim_db, "status": fr.status.value}
            for fname, fr in job.files.items()
        },
        "message": f"Processing {len(job.files)} files. Check GET /status/{job_id} for progress.",
    }


# ── Status & Download ───────────────────────────────────────────────────────

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check the progress of a job. Shows per-file status for batch jobs."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    job = jobs[job_id]
    completed = sum(1 for fr in job.files.values() if fr.status == JobStatus.COMPLETED)
    failed = sum(1 for fr in job.files.values() if fr.status == JobStatus.FAILED)
    total = len(job.files)

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "job_type": job.job_type,
        "progress": f"{completed + failed}/{total}",
        "completed": completed,
        "failed": failed,
        "total": total,
        "files": {
            fname: {
                "status": fr.status.value,
                "atten_lim_db": fr.atten_lim_db,
                "error": fr.error,
            }
            for fname, fr in job.files.items()
        },
        "download_url": f"/download/{job.job_id}" if job.status == JobStatus.COMPLETED else None,
    }


@app.get("/download/{job_id}")
async def download_job(job_id: str):
    """
    Download enhanced results.
    - Single file job: returns the enhanced WAV directly.
    - Batch job: returns a ZIP containing all successfully enhanced WAVs.
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        pending = sum(1 for fr in job.files.values() if fr.status in (JobStatus.QUEUED, JobStatus.PROCESSING))
        raise HTTPException(202, f"Job still processing — {pending} file(s) remaining")

    completed_files = {fn: fr for fn, fr in job.files.items() if fr.status == JobStatus.COMPLETED and fr.result_bytes}

    if not completed_files:
        raise HTTPException(500, "No files were successfully enhanced")

    # Single file — return WAV directly
    if job.job_type == "single" and len(completed_files) == 1:
        fname, fr = next(iter(completed_files.items()))
        return StreamingResponse(
            io.BytesIO(fr.result_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="enhanced_{fname}"'},
        )

    # Batch — return ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, fr in completed_files.items():
            out_name = os.path.splitext(fname)[0] + "_enhanced.wav"
            zf.writestr(out_name, fr.result_bytes)
    zip_buf.seek(0)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="enhanced_batch_{job_id}.zip"'},
    )


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a specific enhanced file from a batch job (no need to wait for all files)."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    job = jobs[job_id]
    if filename not in job.files:
        raise HTTPException(404, f"File '{filename}' not found in job '{job_id}'")

    fr = job.files[filename]
    if fr.status == JobStatus.FAILED:
        raise HTTPException(500, f"File failed: {fr.error}")
    if fr.status != JobStatus.COMPLETED or not fr.result_bytes:
        raise HTTPException(202, f"File '{filename}' still processing")

    out_name = os.path.splitext(filename)[0] + "_enhanced.wav"
    return StreamingResponse(
        io.BytesIO(fr.result_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.get("/jobs")
async def list_jobs():
    """List all jobs (most recent first)."""
    return [
        {
            "job_id": j.job_id,
            "status": j.status.value,
            "job_type": j.job_type,
            "total_files": len(j.files),
            "completed": sum(1 for fr in j.files.values() if fr.status == JobStatus.COMPLETED),
            "created_at": j.created_at,
        }
        for j in sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)
    ]


# ── WebSocket Streaming ─────────────────────────────────────────────────────

@app.websocket("/enhance/stream")
async def enhance_stream(ws: WebSocket):
    """
    Streaming speech enhancement over WebSocket.

    Protocol:
      1. Client sends JSON config: {"sample_rate": 16000, "channels": 1, "atten_lim_db": null}
      2. Client sends binary frames of raw PCM int16 audio chunks.
      3. Client sends text "END" to signal end of stream.
      4. Server buffers all audio, enhances in one pass (fastest), then sends back
         the full enhanced audio as a single binary frame + done JSON.
    """
    await ws.accept()
    try:
        config = await ws.receive_json()
        client_sr = int(config.get("sample_rate", MODEL_SR))
        channels = int(config.get("channels", 1))
        atten_lim_db = config.get("atten_lim_db", None)

        await ws.send_json({"status": "ready", "model_sr": MODEL_SR})

        # Buffer all incoming audio chunks
        audio_chunks = []
        while True:
            msg = await ws.receive()
            if msg.get("text") == "END":
                break
            raw = msg.get("bytes")
            if raw is None:
                continue
            audio_chunks.append(raw)

        # Combine all chunks into one tensor
        all_raw = b"".join(audio_chunks)
        pcm = np.frombuffer(all_raw, dtype=np.int16).astype(np.float32) / (1 << 15)
        audio = torch.from_numpy(pcm).reshape(channels, -1)
        if client_sr != MODEL_SR:
            audio = resample(audio, client_sr, MODEL_SR)

        # Enhance entire audio in one pass (much faster than per-chunk)
        loop = asyncio.get_event_loop()
        enhanced = await loop.run_in_executor(None, _enhance_tensor, audio, atten_lim_db)

        if client_sr != MODEL_SR:
            enhanced = resample(enhanced, MODEL_SR, client_sr)

        out_pcm = (enhanced * (1 << 15)).to(torch.int16).numpy().tobytes()
        await ws.send_bytes(out_pcm)
        await ws.send_json({"status": "done", "samples": len(out_pcm) // 2})

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, ws_max_size=50 * 1024 * 1024)
