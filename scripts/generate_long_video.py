#!/usr/bin/env python3
"""
generate_long_video.py — Chunked Long Video Generation with LTX-Video
=====================================================================

Generates long-duration videos (up to 5+ minutes) by chaining LTX-Video
text-to-video and image-to-video calls. Each chunk produces ~65 frames;
the last frame of chunk N conditions chunk N+1 for visual continuity.

Target hardware: NVIDIA L40S (48 GB VRAM)

Usage:
    python scripts/generate_long_video.py \
        --prompt "A woman walks through a sunlit garden" \
        --duration_seconds 300 --fps 24 \
        --height 480 --width 704 \
        --output videos/5min_video.mp4

Author: Dan Abergel
"""

import argparse
import json
import logging
import os
import sys
import time
from math import ceil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure root logger with console + optional file output."""
    logger = logging.getLogger("ltx_longvideo")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def log_gpu_info(logger: logging.Logger):
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — generation will be very slow on CPU")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            "GPU %d: %s — %.1f GB VRAM",
            i, props.name, props.total_mem / 1e9,
        )


def log_vram(logger: logging.Logger, tag: str = ""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    logger.debug("VRAM [%s] allocated=%.2f GB  reserved=%.2f GB", tag, alloc, reserved)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipelines(model_id: str, device: str, dtype: torch.dtype, logger: logging.Logger):
    """Load text-to-video and image-to-video LTX pipelines."""
    from diffusers import LTXPipeline, LTXImageToVideoPipeline

    logger.info("Loading text-to-video pipeline from %s ...", model_id)
    t2v_pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype)
    t2v_pipe.to(device)
    log_vram(logger, "after t2v load")

    logger.info("Loading image-to-video pipeline from %s ...", model_id)
    i2v_pipe = LTXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    i2v_pipe.to(device)
    log_vram(logger, "after i2v load")

    return t2v_pipe, i2v_pipe


# ---------------------------------------------------------------------------
# Frame extraction helper
# ---------------------------------------------------------------------------

def frames_from_output(output) -> list[np.ndarray]:
    """Extract list of HWC uint8 numpy frames from a diffusers pipeline output."""
    # diffusers returns output.frames as list of lists of PIL Images
    frames = output.frames[0]  # first (only) batch element
    return [np.array(f) for f in frames]


# ---------------------------------------------------------------------------
# Chunk generation
# ---------------------------------------------------------------------------

FRAMES_PER_CHUNK = 65  # validated default for LTX-Video
NEW_FRAMES_PER_CHUNK = FRAMES_PER_CHUNK - 1  # first frame is overlap from prev chunk


def generate_first_chunk(
    t2v_pipe,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Generate the first chunk using text-to-video."""
    logger.info("Generating first chunk (text-to-video, %d frames) ...", num_frames)
    gen = torch.Generator(device=t2v_pipe.device).manual_seed(seed)
    output = t2v_pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )
    return frames_from_output(output)


def generate_continuation_chunk(
    i2v_pipe,
    prompt: str,
    last_frame: np.ndarray,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Generate a continuation chunk conditioned on the last frame."""
    image = Image.fromarray(last_frame).resize((width, height))
    gen = torch.Generator(device=i2v_pipe.device).manual_seed(seed)
    output = i2v_pipe(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )
    # Skip the first frame (it's the conditioning frame / overlap)
    return frames_from_output(output)[1:]


# ---------------------------------------------------------------------------
# Quality metrics (optional)
# ---------------------------------------------------------------------------

def compute_clip_metrics(
    frames: list[np.ndarray],
    prompt: str,
    sample_every: int = 100,
    logger: logging.Logger | None = None,
) -> dict:
    """Compute CLIP score and frame consistency on sampled frames."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        if logger:
            logger.warning("transformers not installed — skipping CLIP metrics")
        return {}

    if logger:
        logger.info("Computing CLIP metrics (sampling every %d frames) ...", sample_every)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    sampled_indices = list(range(0, len(frames), sample_every))
    sampled_frames = [Image.fromarray(frames[i]) for i in sampled_indices]

    # CLIP score: cosine similarity between prompt and each sampled frame
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    text_feat = clip_model.get_text_features(**text_inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    clip_scores = []
    frame_features = []
    for img in sampled_frames:
        img_inputs = processor(images=img, return_tensors="pt").to(device)
        img_feat = clip_model.get_image_features(**img_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        frame_features.append(img_feat)
        score = (text_feat @ img_feat.T).item()
        clip_scores.append(score)

    # Frame consistency: cosine similarity between consecutive sampled frames
    consistencies = []
    for i in range(len(frame_features) - 1):
        sim = (frame_features[i] @ frame_features[i + 1].T).item()
        consistencies.append(sim)

    metrics = {
        "clip_score_mean": float(np.mean(clip_scores)),
        "clip_score_std": float(np.std(clip_scores)),
        "frame_consistency_mean": float(np.mean(consistencies)) if consistencies else None,
        "frame_consistency_std": float(np.std(consistencies)) if consistencies else None,
        "num_sampled_frames": len(sampled_indices),
    }

    if logger:
        logger.info("  CLIP score: %.4f ± %.4f", metrics["clip_score_mean"], metrics["clip_score_std"])
        if metrics["frame_consistency_mean"] is not None:
            logger.info("  Frame consistency: %.4f ± %.4f",
                        metrics["frame_consistency_mean"], metrics["frame_consistency_std"])

    # Free CLIP model VRAM
    del clip_model
    torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------

def export_video(frames: list[np.ndarray], output_path: str, fps: int, logger: logging.Logger):
    """Write frames to MP4 using imageio + ffmpeg."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info("Exporting %d frames to %s at %d fps ...", len(frames), output_path, fps)
    writer = iio.imopen(output_path, "w", plugin="pyav")
    writer.write(
        np.stack(frames),
        codec="h264",
        fps=fps,
    )
    file_size_mb = os.path.getsize(output_path) / 1e6
    logger.info("  Output: %s (%.1f MB)", output_path, file_size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate long videos with LTX-Video via chunked image-to-video continuation",
    )
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt describing the video")
    p.add_argument("--duration_seconds", type=float, default=300,
                   help="Target video duration in seconds (default: 300 = 5 min)")
    p.add_argument("--fps", type=int, default=24,
                   help="Frames per second (default: 24)")
    p.add_argument("--height", type=int, default=480,
                   help="Video height in pixels (default: 480)")
    p.add_argument("--width", type=int, default=704,
                   help="Video width in pixels (default: 704)")
    p.add_argument("--output", type=str, default="videos/5min_video.mp4",
                   help="Output MP4 path (default: videos/5min_video.mp4)")
    p.add_argument("--model_id", type=str, default="Lightricks/LTX-Video-0.9.1",
                   help="HuggingFace model ID (default: Lightricks/LTX-Video-0.9.1)")
    p.add_argument("--num_inference_steps", type=int, default=30,
                   help="Denoising steps per chunk (default: 30)")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="Classifier-free guidance scale (default: 7.5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--frames_per_chunk", type=int, default=65,
                   help="Frames per generation call (default: 65)")
    p.add_argument("--skip_metrics", action="store_true",
                   help="Skip CLIP quality metrics at the end")
    p.add_argument("--log_file", type=str, default=None,
                   help="Optional log file path (default: <output>.log)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve output path relative to script directory's parent (repo root)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if not os.path.isabs(args.output):
        args.output = str(repo_root / args.output)

    # Default log file next to the output video
    if args.log_file is None:
        args.log_file = str(Path(args.output).with_suffix(".log"))

    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("LTX-Video Long Video Generator")
    logger.info("=" * 60)

    # ── Compute plan ──────────────────────────────────────────────
    total_frames = int(args.duration_seconds * args.fps)
    new_per_chunk = args.frames_per_chunk - 1  # 1 frame overlap
    num_chunks = 1 + ceil((total_frames - args.frames_per_chunk) / new_per_chunk)
    actual_frames = args.frames_per_chunk + (num_chunks - 1) * new_per_chunk

    logger.info("  Prompt:         %s", args.prompt)
    logger.info("  Duration:       %.1f s (%d frames @ %d fps)", args.duration_seconds, total_frames, args.fps)
    logger.info("  Resolution:     %d x %d", args.width, args.height)
    logger.info("  Frames/chunk:   %d (%d new + 1 overlap)", args.frames_per_chunk, new_per_chunk)
    logger.info("  Total chunks:   %d", num_chunks)
    logger.info("  Actual frames:  %d (%.1f s)", actual_frames, actual_frames / args.fps)
    logger.info("  Output:         %s", args.output)
    logger.info("  Model:          %s", args.model_id)
    logger.info("  Steps/chunk:    %d", args.num_inference_steps)
    logger.info("  Guidance:       %.1f", args.guidance_scale)
    logger.info("  Seed:           %d", args.seed)

    # ── GPU info ──────────────────────────────────────────────────
    log_gpu_info(logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ── Load pipelines ────────────────────────────────────────────
    t0_load = time.time()
    t2v_pipe, i2v_pipe = load_pipelines(args.model_id, device, dtype, logger)
    load_time = time.time() - t0_load
    logger.info("Pipelines loaded in %.1f s", load_time)

    # ── Generate chunks ───────────────────────────────────────────
    all_frames: list[np.ndarray] = []
    chunk_times: list[float] = []
    t0_gen = time.time()

    pbar = tqdm(total=num_chunks, desc="Chunks", unit="chunk")

    for chunk_idx in range(num_chunks):
        t_chunk = time.time()
        chunk_seed = args.seed + chunk_idx  # vary seed per chunk for diversity

        if chunk_idx == 0:
            frames = generate_first_chunk(
                t2v_pipe, args.prompt,
                args.height, args.width, args.frames_per_chunk,
                args.num_inference_steps, args.guidance_scale,
                chunk_seed, logger,
            )
            all_frames.extend(frames)
        else:
            last_frame = all_frames[-1]
            frames = generate_continuation_chunk(
                i2v_pipe, args.prompt, last_frame,
                args.height, args.width, args.frames_per_chunk,
                args.num_inference_steps, args.guidance_scale,
                chunk_seed, logger,
            )
            all_frames.extend(frames)

        elapsed = time.time() - t_chunk
        chunk_times.append(elapsed)
        log_vram(logger, f"chunk {chunk_idx}")
        logger.info(
            "  Chunk %d/%d done — %d frames in %.1f s (total: %d frames)",
            chunk_idx + 1, num_chunks, len(frames), elapsed, len(all_frames),
        )
        pbar.update(1)

    pbar.close()
    gen_time = time.time() - t0_gen

    # Trim to exact target frame count
    all_frames = all_frames[:total_frames]
    logger.info("Generation complete: %d frames in %.1f s", len(all_frames), gen_time)

    # ── Free pipeline VRAM before export ──────────────────────────
    del t2v_pipe, i2v_pipe
    torch.cuda.empty_cache()

    # ── Export video ──────────────────────────────────────────────
    t0_export = time.time()
    export_video(all_frames, args.output, args.fps, logger)
    export_time = time.time() - t0_export

    # ── Quality metrics ───────────────────────────────────────────
    metrics = {}
    if not args.skip_metrics:
        metrics = compute_clip_metrics(all_frames, args.prompt, sample_every=100, logger=logger)

    # ── Summary ───────────────────────────────────────────────────
    total_time = time.time() - t0_load
    summary = {
        "prompt": args.prompt,
        "duration_seconds": len(all_frames) / args.fps,
        "total_frames": len(all_frames),
        "resolution": f"{args.width}x{args.height}",
        "fps": args.fps,
        "num_chunks": num_chunks,
        "frames_per_chunk": args.frames_per_chunk,
        "model_id": args.model_id,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "timings": {
            "pipeline_load_s": round(load_time, 1),
            "generation_s": round(gen_time, 1),
            "export_s": round(export_time, 1),
            "total_s": round(total_time, 1),
            "avg_chunk_s": round(float(np.mean(chunk_times)), 1),
        },
        "output_file": args.output,
        "output_size_mb": round(os.path.getsize(args.output) / 1e6, 1),
        "metrics": metrics,
    }

    json_path = str(Path(args.output).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total frames:   %d", summary["total_frames"])
    logger.info("  Duration:       %.1f s", summary["duration_seconds"])
    logger.info("  Chunks:         %d", summary["num_chunks"])
    logger.info("  Avg chunk time: %.1f s", summary["timings"]["avg_chunk_s"])
    logger.info("  Total time:     %.1f s (%.1f min)", total_time, total_time / 60)
    logger.info("  Output video:   %s", args.output)
    logger.info("  Output JSON:    %s", json_path)
    logger.info("  Output log:     %s", args.log_file)
    if metrics:
        logger.info("  CLIP score:     %.4f", metrics.get("clip_score_mean", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
