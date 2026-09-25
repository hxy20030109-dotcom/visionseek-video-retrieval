"""OpenCLIP encoding and temporal aggregation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def mean_pool_frame_embeddings(frame_embeddings: Tensor) -> Tensor:
    """Return one normalized embedding per video.

    Args:
        frame_embeddings: Tensor shaped [videos, frames, embedding_dim].
    """
    if frame_embeddings.ndim != 3:
        raise ValueError(
            "Expected frame embeddings with shape [videos, frames, dim], "
            f"got {tuple(frame_embeddings.shape)}"
        )

    normalized_frames = F.normalize(frame_embeddings, dim=-1)
    pooled_embeddings = normalized_frames.mean(dim=1)
    return F.normalize(pooled_embeddings, dim=-1)


def max_frame_similarity(query_embeddings: Tensor, frame_embeddings: Tensor) -> Tensor:
    """Return the maximum query-to-frame similarity for every video."""
    if query_embeddings.ndim != 2:
        raise ValueError("Query embeddings must have shape [queries, dim].")
    if frame_embeddings.ndim != 3:
        raise ValueError("Frame embeddings must have shape [videos, frames, dim].")
    if query_embeddings.shape[-1] != frame_embeddings.shape[-1]:
        raise ValueError("Query and frame embedding dimensions must match.")

    normalized_queries = F.normalize(query_embeddings, dim=-1)
    normalized_frames = F.normalize(frame_embeddings, dim=-1)
    frame_scores = torch.einsum(
        "qd,vfd->qvf",
        normalized_queries,
        normalized_frames,
    )
    return frame_scores.max(dim=-1).values


class OpenCLIPEmbedding:
    """Encode RGB frames and text with one frozen OpenCLIP model."""

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        import open_clip

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
            cache_dir=cache_dir,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_frames(self, frame_batch: Tensor) -> Tensor:
        """Encode RGB uint8 frames shaped [frames, 3, height, width]."""
        from PIL import Image

        if frame_batch.ndim != 4 or frame_batch.shape[1] != 3:
            raise ValueError(
                "Expected RGB frames with shape [frames, 3, height, width], "
                f"got {tuple(frame_batch.shape)}"
            )
        if frame_batch.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 frames, got {frame_batch.dtype}")

        processed_frames = torch.stack(
            [
                self.preprocess(
                    Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
                )
                for frame in frame_batch
            ]
        ).to(self.device)

        with torch.inference_mode():
            embeddings = self.model.encode_image(processed_frames)
        return F.normalize(embeddings, dim=-1)

    def encode_video_batch(self, video_frame_batches: Sequence[Tensor]) -> Tensor:
        """Encode equally sampled videos with possibly different resolutions.

        Args:
            video_frame_batches: Sequence of uint8 tensors, each shaped
                [frames, 3, height, width].

        Returns:
            Normalized frame embeddings shaped [videos, frames, embedding_dim].
        """
        from PIL import Image

        if not video_frame_batches:
            raise ValueError("At least one video frame batch is required.")

        frames_per_video = video_frame_batches[0].shape[0]
        processed_frames = []
        for frame_batch in video_frame_batches:
            if frame_batch.ndim != 4 or frame_batch.shape[1] != 3:
                raise ValueError("Every video must have shape [frames, 3, height, width].")
            if frame_batch.dtype != torch.uint8:
                raise ValueError("Every video frame batch must use uint8 dtype.")
            if frame_batch.shape[0] != frames_per_video:
                raise ValueError("Every video must contain the same number of sampled frames.")

            processed_frames.extend(
                self.preprocess(
                    Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
                )
                for frame in frame_batch
            )

        image_batch = torch.stack(processed_frames).to(self.device)
        with torch.inference_mode():
            flat_embeddings = self.model.encode_image(image_batch)
        flat_embeddings = F.normalize(flat_embeddings, dim=-1)
        return flat_embeddings.reshape(
            len(video_frame_batches),
            frames_per_video,
            -1,
        )

    def encode_text(self, texts: Sequence[str]) -> Tensor:
        """Encode a non-empty sequence of text queries."""
        if not texts:
            raise ValueError("At least one text query is required.")

        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode():
            embeddings: Any = self.model.encode_text(tokens)
        return F.normalize(embeddings, dim=-1)
