"""Cache OpenCLIP frame, video, and text embeddings for MSR-VTT 1K-A."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch

from visionseek.embedding import OpenCLIPEmbedding, mean_pool_frame_embeddings
from visionseek.video import sample_uniform_frames

DEFAULT_DATA_ROOT = Path(os.environ.get("VISIONSEEK_DATA_ROOT", "data/raw"))
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "msrvtt/test_1k/captions.csv"
DEFAULT_OUTPUT = (
    DEFAULT_DATA_ROOT
    / "msrvtt/test_1k/embeddings/openclip_vit_b32_openai_frames8.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--video-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(os.environ["HF_HOME"]) if "HF_HOME" in os.environ else None,
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_manifest(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    return rows


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest, args.limit)
    video_ids = [row["video_id"] for row in rows]
    video_paths = [Path(row["video_path"]) for row in rows]
    captions = [row["caption"] for row in rows]

    print("Loading OpenCLIP...")
    encoder = OpenCLIPEmbedding(cache_dir=args.model_cache)
    print(f"device: {encoder.device}")
    print(f"videos: {len(video_paths)}")
    print(f"frames per video: {args.frames}")

    synchronize_cuda()
    video_start = time.perf_counter()
    all_frame_embeddings: list[torch.Tensor] = []

    for start in range(0, len(video_paths), args.video_batch_size):
        batch_paths = video_paths[start : start + args.video_batch_size]
        frame_batches = [
            sample_uniform_frames(path, num_frames=args.frames)[0]
            for path in batch_paths
        ]
        batch_embeddings = encoder.encode_video_batch(frame_batches)
        all_frame_embeddings.append(batch_embeddings.cpu())

        completed = min(start + len(batch_paths), len(video_paths))
        print(f"encoded videos: {completed}/{len(video_paths)}")

    synchronize_cuda()
    video_seconds = time.perf_counter() - video_start
    frame_embeddings = torch.cat(all_frame_embeddings)
    video_embeddings = mean_pool_frame_embeddings(frame_embeddings)

    synchronize_cuda()
    text_start = time.perf_counter()
    all_text_embeddings = []
    for start in range(0, len(captions), args.text_batch_size):
        text_batch = captions[start : start + args.text_batch_size]
        all_text_embeddings.append(encoder.encode_text(text_batch).cpu())
    synchronize_cuda()
    text_seconds = time.perf_counter() - text_start
    text_embeddings = torch.cat(all_text_embeddings)

    cache = {
        "dataset": "MSR-VTT 1K-A",
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "frames_per_video": args.frames,
        "video_ids": video_ids,
        "captions": captions,
        "frame_embeddings": frame_embeddings,
        "video_embeddings": video_embeddings,
        "text_embeddings": text_embeddings,
        "video_encoding_seconds": video_seconds,
        "text_encoding_seconds": text_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)

    print(f"frame embeddings: {tuple(frame_embeddings.shape)}")
    print(f"video embeddings: {tuple(video_embeddings.shape)}")
    print(f"text embeddings: {tuple(text_embeddings.shape)}")
    print(f"video encoding seconds: {video_seconds:.2f}")
    print(f"text encoding seconds: {text_seconds:.2f}")
    print(f"saved cache: {args.output}")


if __name__ == "__main__":
    main()
