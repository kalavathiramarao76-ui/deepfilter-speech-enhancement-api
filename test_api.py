"""
Quick test script for the DeepFilterNet API.
Start the server first:  python app.py
Then run:  python test_api.py <path_to_noisy_audio.wav>
"""

import sys
import json
import asyncio
import requests
import websockets
import numpy as np

BASE = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    print("Health:", r.json())


def test_single(filepath: str):
    print(f"\n--- Single enhance: {filepath} ---")
    with open(filepath, "rb") as f:
        r = requests.post(f"{BASE}/enhance", files={"file": (filepath, f, "audio/wav")})
    out = filepath.replace(".wav", "_enhanced.wav")
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"Saved → {out}  ({len(r.content)} bytes)")


def test_batch(filepaths: list[str]):
    print(f"\n--- Batch enhance: {filepaths} ---")
    files = [("files", (fp, open(fp, "rb"), "audio/wav")) for fp in filepaths]
    r = requests.post(f"{BASE}/enhance/batch", files=files)
    out = "enhanced_batch.zip"
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"Saved → {out}  ({len(r.content)} bytes)")


async def test_stream(filepath: str, chunk_ms: int = 100):
    """Send audio in chunks over WebSocket and collect enhanced output."""
    import wave

    print(f"\n--- Stream enhance: {filepath} ---")

    with wave.open(filepath, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    chunk_samples = int(sr * chunk_ms / 1000) * channels
    chunk_bytes = chunk_samples * 2  # int16

    enhanced_chunks = []
    async with websockets.connect("ws://localhost:8000/enhance/stream") as ws:
        # Send config
        await ws.send(json.dumps({"sample_rate": sr, "channels": channels}))
        config_resp = json.loads(await ws.recv())
        print("Config response:", config_resp)

        # Send chunks
        offset = 0
        while offset < len(raw):
            chunk = raw[offset : offset + chunk_bytes]
            await ws.send(chunk)
            enhanced = await ws.recv()
            enhanced_chunks.append(enhanced)
            offset += chunk_bytes

        # Signal end
        await ws.send("END")
        done_resp = json.loads(await ws.recv())
        print("Done response:", done_resp)

    # Save output
    out_raw = b"".join(enhanced_chunks)
    out = filepath.replace(".wav", "_stream_enhanced.wav")
    import wave as w
    with w.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_raw)
    print(f"Saved → {out}  ({len(out_raw)} bytes)")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None

    test_health()

    if filepath:
        test_single(filepath)
        test_batch([filepath])
        asyncio.run(test_stream(filepath))
    else:
        print("\nPass an audio file path to test enhance endpoints.")
        print("Usage: python test_api.py noisy_audio.wav")
