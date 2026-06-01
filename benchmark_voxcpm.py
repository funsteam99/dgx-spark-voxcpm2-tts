#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import soundfile as sf


def read_prompts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "text" not in item:
                raise ValueError(f"{path}:{line_no} must contain id and text")
            rows.append({"id": str(item["id"]), "text": str(item["text"])})
    return rows


def probe_torch() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            info["cuda_mem_free_gb"] = round(free_bytes / 1024**3, 3)
            info["cuda_mem_total_gb"] = round(total_bytes / 1024**3, 3)
    except Exception as exc:
        info["torch_probe_error"] = repr(exc)
    return info


def audio_duration_seconds(wav, sample_rate: int) -> float:
    try:
        return float(len(wav)) / float(sample_rate)
    except Exception:
        return 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VoxCPM2 on DGX Spark")
    parser.add_argument("--model", default="openbmb/VoxCPM2")
    parser.add_argument("--prompts", type=Path, default=Path("prompts/baseline.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-denoiser", action="store_true")
    parser.add_argument("--denoise-input", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--reference-wav", default=None)
    parser.add_argument("--prompt-wav", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audio").mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(args.prompts)
    env_info = probe_torch()
    env_info.update(
        {
            "model": args.model,
            "prompts": str(args.prompts),
            "device": args.device,
            "load_denoiser": not args.no_denoiser,
            "optimize": not args.no_optimize,
            "cfg_value": args.cfg_value,
            "inference_timesteps": args.inference_timesteps,
            "reference_wav": args.reference_wav,
            "prompt_wav": args.prompt_wav,
            "has_prompt_text": bool(args.prompt_text),
        }
    )
    (args.output_dir / "env.json").write_text(json.dumps(env_info, indent=2, ensure_ascii=False), encoding="utf-8")

    from voxcpm import VoxCPM

    load_start = time.perf_counter()
    model = VoxCPM.from_pretrained(
        args.model,
        load_denoiser=not args.no_denoiser,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        optimize=not args.no_optimize,
        device=args.device,
    )
    load_seconds = time.perf_counter() - load_start
    sample_rate = int(model.tts_model.sample_rate)

    rows: list[dict[str, Any]] = []
    for repeat_idx in range(args.repeat):
        for prompt in prompts:
            case_id = prompt["id"]
            text = prompt["text"]
            out_wav = args.output_dir / "audio" / f"{case_id}_r{repeat_idx + 1}.wav"
            print(f"[run] {case_id} repeat={repeat_idx + 1}", flush=True)

            started = time.perf_counter()
            error = None
            duration = 0.0
            rtf = None
            try:
                wav = model.generate(
                    text=text,
                    prompt_wav_path=args.prompt_wav,
                    prompt_text=args.prompt_text,
                    reference_wav_path=args.reference_wav,
                    cfg_value=args.cfg_value,
                    inference_timesteps=args.inference_timesteps,
                    denoise=args.denoise_input,
                )
                elapsed = time.perf_counter() - started
                duration = audio_duration_seconds(wav, sample_rate)
                rtf = elapsed / duration if duration > 0 else None
                sf.write(out_wav, wav, sample_rate)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                error = repr(exc)

            row = {
                "id": case_id,
                "repeat": repeat_idx + 1,
                "text_chars": len(text),
                "elapsed_sec": round(elapsed, 4),
                "audio_sec": round(duration, 4),
                "rtf": round(rtf, 4) if rtf is not None else None,
                "sample_rate": sample_rate,
                "load_seconds": round(load_seconds, 4),
                "output_wav": str(out_wav) if error is None else "",
                "error": error or "",
            }
            rows.append(row)
            write_jsonl(args.output_dir / "results.jsonl", rows)
            write_csv(args.output_dir / "results.csv", rows)

    rtfs = [float(row["rtf"]) for row in rows if row.get("rtf") is not None]
    summary = {
        "cases": len(rows),
        "successes": sum(1 for row in rows if not row["error"]),
        "failures": sum(1 for row in rows if row["error"]),
        "load_seconds": round(load_seconds, 4),
        "rtf_mean": round(statistics.mean(rtfs), 4) if rtfs else None,
        "rtf_median": round(statistics.median(rtfs), 4) if rtfs else None,
        "rtf_min": round(min(rtfs), 4) if rtfs else None,
        "rtf_max": round(max(rtfs), 4) if rtfs else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
