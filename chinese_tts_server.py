#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import traceback
import cgi
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import soundfile as sf
import numpy as np


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "chinese-tts-web"
OUTPUT_DIR = ROOT / "runs" / "chinese-tts-live"
UPLOAD_DIR = ROOT / "runs" / "chinese-tts-uploads"
MODEL_PATH = os.environ.get("VOXCPM_MODEL", "pretrained_models/VoxCPM2")
HOST = os.environ.get("VOXCPM_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOXCPM_PORT", "8792"))
MAX_TEXT_CHARS = int(os.environ.get("VOXCPM_MAX_TEXT_CHARS", "420"))
MAX_UPLOAD_BYTES = int(os.environ.get("VOXCPM_MAX_UPLOAD_BYTES", str(60 * 1024 * 1024)))

model = None


@dataclass
class Chunk:
    index: int
    text: str


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def load_model():
    global model
    if model is None:
        from voxcpm import VoxCPM

        started = time.perf_counter()
        model = VoxCPM.from_pretrained(
            MODEL_PATH,
            load_denoiser=False,
            optimize=False,
            device="auto",
        )
        print(f"[tts] model loaded in {time.perf_counter() - started:.2f}s", flush=True)
    return model


def clean_name(text: str) -> str:
    compact = re.sub(r"\s+", "-", text.strip())
    compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "", compact)
    return compact[:28] or "tts"


def split_text(text: str, max_chars: int) -> list[Chunk]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    chunks: list[str] = []
    parts = re.split(r"([\u3002\uff01\uff1f!?\uff1b;\uff0c,\u3001])", text)
    units: list[str] = []
    for idx in range(0, len(parts), 2):
        body = parts[idx]
        punct = parts[idx + 1] if idx + 1 < len(parts) else ""
        unit = (body + punct).strip()
        if unit:
            units.append(unit)

    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(unit), max_chars):
                chunks.append(unit[start : start + max_chars])
            continue
        if current and len(current) + len(unit) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current += unit
    if current:
        chunks.append(current)

    return [Chunk(index=i + 1, text=chunk) for i, chunk in enumerate(chunks)]


def concatenate_wavs(wavs: list[np.ndarray], sample_rate: int, gap_seconds: float = 0.18) -> np.ndarray:
    if not wavs:
        return np.array([], dtype=np.float32)
    gap = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for idx, wav in enumerate(wavs):
        pieces.append(np.asarray(wav, dtype=np.float32))
        if idx < len(wavs) - 1:
            pieces.append(gap)
    return np.concatenate(pieces)


def form_value(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    if name not in form:
        return default
    item = form[name]
    if isinstance(item, list):
        item = item[0]
    value = item.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def bool_value(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def save_uploaded_sample(form: cgi.FieldStorage, stamp: str) -> Path | None:
    if "voice_sample" not in form:
        return None
    item = form["voice_sample"]
    if isinstance(item, list):
        item = item[0]
    if not getattr(item, "filename", ""):
        return None

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    original_name = Path(str(item.filename)).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        suffix = ".wav"
    out_path = UPLOAD_DIR / f"{stamp}-sample-{clean_name(Path(original_name).stem)}{suffix}"

    written = 0
    with out_path.open("wb") as out:
        while True:
            block = item.file.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                out_path.unlink(missing_ok=True)
                raise ValueError(f"voice_sample_too_large>{MAX_UPLOAD_BYTES}")
            out.write(block)
    return out_path


def read_tts_request(handler: SimpleHTTPRequestHandler, stamp: str) -> tuple[dict, Path | None]:
    content_type = handler.headers.get("Content-Type", "")
    if content_type.startswith("multipart/form-data"):
        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
            },
        )
        payload = {
            "text": form_value(form, "text"),
            "voice": form_value(form, "voice"),
            "prompt_text": form_value(form, "prompt_text"),
            "cfg_value": form_value(form, "cfg_value", "2.0"),
            "inference_timesteps": form_value(form, "inference_timesteps", "10"),
            "stable_voice": bool_value(form_value(form, "stable_voice", "true"), True),
        }
        return payload, save_uploaded_sample(form, stamp)

    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8")), None


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean = unquote(parsed.path)
        if clean.startswith("/outputs/"):
            return str(OUTPUT_DIR / clean.removeprefix("/outputs/"))
        if clean == "/" or clean.startswith("/chinese-tts/"):
            rel = clean.removeprefix("/chinese-tts/").lstrip("/")
            return str((WEB_DIR / (rel or "index.html")).resolve())
        return str((ROOT / clean.lstrip("/")).resolve())

    def do_GET(self) -> None:
        if self.path.startswith("/api/health"):
            json_response(
                self,
                {
                    "ok": True,
                    "model_loaded": model is not None,
                    "model": MODEL_PATH,
                    "max_text_chars": MAX_TEXT_CHARS,
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if not self.path.startswith("/api/tts"):
            json_response(self, {"ok": False, "error": "not_found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            payload, uploaded_prompt_wav = read_tts_request(self, stamp)
            text = str(payload.get("text", "")).strip()
            voice = str(payload.get("voice", "")).strip()
            prompt_text = str(payload.get("prompt_text", "")).strip()
            cfg_value = float(payload.get("cfg_value", 2.0))
            inference_timesteps = int(payload.get("inference_timesteps", 10))
            stable_voice = bool_value(payload.get("stable_voice", True), True)

            if not text:
                json_response(self, {"ok": False, "error": "empty_text"}, HTTPStatus.BAD_REQUEST)
                return

            voice_prefix = f"({voice})" if voice else ""
            chunk_limit = max(40, MAX_TEXT_CHARS - len(voice_prefix))
            chunks = split_text(text, chunk_limit)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tts = load_model()
            sample_rate = int(tts.tts_model.sample_rate)
            total_started = time.perf_counter()
            segments: list[dict] = []
            segment_wavs: list[np.ndarray] = []
            anchor_wav_path: Path | None = None
            anchor_prompt_text = ""

            for chunk in chunks:
                use_uploaded_prompt = uploaded_prompt_wav is not None
                use_anchor = stable_voice and anchor_wav_path is not None
                prompt_wav_path = uploaded_prompt_wav if use_uploaded_prompt else anchor_wav_path
                active_prompt_text = prompt_text if use_uploaded_prompt else anchor_prompt_text
                model_text = chunk.text if use_uploaded_prompt or use_anchor else f"{voice_prefix}{chunk.text}"
                started = time.perf_counter()
                wav = tts.generate(
                    text=model_text,
                    prompt_wav_path=str(prompt_wav_path) if prompt_wav_path else None,
                    prompt_text=active_prompt_text or None,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    denoise=False,
                )
                elapsed = time.perf_counter() - started
                audio_sec = float(len(wav)) / float(sample_rate)
                file_name = f"{stamp}-part{chunk.index:02d}-{clean_name(chunk.text)}.wav"
                out_path = OUTPUT_DIR / file_name
                sf.write(out_path, wav, sample_rate)
                segment_wavs.append(np.asarray(wav, dtype=np.float32))
                if stable_voice and anchor_wav_path is None:
                    anchor_wav_path = out_path
                    anchor_prompt_text = chunk.text
                segments.append(
                    {
                        "index": chunk.index,
                        "text": chunk.text,
                        "url": f"/outputs/{file_name}",
                        "file": str(out_path),
                        "elapsed_sec": round(elapsed, 4),
                        "audio_sec": round(audio_sec, 4),
                        "rtf": round(elapsed / audio_sec, 4) if audio_sec else None,
                        "chars": len(chunk.text),
                    }
                )

            elapsed_total = time.perf_counter() - total_started
            audio_total = sum(float(item["audio_sec"]) for item in segments)
            first = segments[0] if segments else {}
            combined_url = first.get("url", "")
            combined_file = first.get("file", "")
            if len(segment_wavs) > 1:
                combined_wav = concatenate_wavs(segment_wavs, sample_rate)
                combined_name = f"{stamp}-combined-{len(segment_wavs):02d}parts-{clean_name(text)}.wav"
                combined_path = OUTPUT_DIR / combined_name
                sf.write(combined_path, combined_wav, sample_rate)
                combined_url = f"/outputs/{combined_name}"
                combined_file = str(combined_path)
                audio_total = float(len(combined_wav)) / float(sample_rate)
            json_response(
                self,
                {
                    "ok": True,
                    "url": combined_url,
                    "file": combined_file,
                    "combined_url": combined_url,
                    "combined_file": combined_file,
                    "elapsed_sec": round(elapsed_total, 4),
                    "audio_sec": round(audio_total, 4),
                    "rtf": round(elapsed_total / audio_total, 4) if audio_total else None,
                    "sample_rate": sample_rate,
                    "chars": len(text),
                    "segments": segments,
                    "segment_count": len(segments),
                    "chunk_limit": chunk_limit,
                    "stable_voice": stable_voice,
                    "uploaded_prompt": bool(uploaded_prompt_wav),
                    "uploaded_prompt_file": str(uploaded_prompt_wav) if uploaded_prompt_wav else "",
                    "has_prompt_text": bool(prompt_text),
                    "anchor_file": str(anchor_wav_path) if anchor_wav_path else "",
                },
            )
        except Exception as exc:
            traceback.print_exc()
            json_response(self, {"ok": False, "error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> int:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ReusableThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[tts] listening on http://{HOST}:{PORT}/chinese-tts/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
