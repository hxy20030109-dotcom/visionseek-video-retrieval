"""Evaluate a cached MSR-VTT text-to-video retrieval baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from visionseek.metrics import rank_summary, recall_at_k
from visionseek.retrieval import cosine_similarity_matrix

DEFAULT_DATA_ROOT = Path(os.environ.get("VISIONSEEK_DATA_ROOT", "data/raw"))
DEFAULT_CACHE = (
    DEFAULT_DATA_ROOT
    / "msrvtt/test_1k/embeddings/openclip_vit_b32_openai_frames8.pt"
)
DEFAULT_RESULTS = Path("outputs/msrvtt_openclip_baseline.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    video_embeddings = cache["video_embeddings"]
    text_embeddings = cache["text_embeddings"]

    scores = cosine_similarity_matrix(text_embeddings, video_embeddings)
    targets = torch.arange(scores.shape[0], dtype=torch.long)
    valid_ks = tuple(k for k in (1, 5, 10) if k <= scores.shape[1])
    metrics = {
        **recall_at_k(scores, targets, ks=valid_ks),
        **rank_summary(scores, targets),
    }

    results = {
        "dataset": cache["dataset"],
        "model_name": cache["model_name"],
        "pretrained": cache["pretrained"],
        "frames_per_video": cache["frames_per_video"],
        "num_videos": len(cache["video_ids"]),
        "video_encoding_seconds": cache["video_encoding_seconds"],
        "text_encoding_seconds": cache["text_encoding_seconds"],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"saved results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
