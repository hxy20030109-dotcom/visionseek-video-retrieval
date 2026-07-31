"""Tensor utilities for first-stage retrieval."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def l2_normalize(embeddings: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize a 2D embedding tensor along its feature dimension."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape [items, dim], got {tuple(embeddings.shape)}"
        )
    return F.normalize(embeddings, p=2, dim=-1, eps=eps)


def cosine_similarity_matrix(query_embeddings: Tensor, item_embeddings: Tensor) -> Tensor:
    """Return all pairwise cosine similarities with shape [queries, items]."""
    if query_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise ValueError("Query and item embeddings must both be 2D tensors.")
    if query_embeddings.shape[-1] != item_embeddings.shape[-1]:
        raise ValueError(
            "Embedding dimensions must match: "
            f"{query_embeddings.shape[-1]} != {item_embeddings.shape[-1]}"
        )

    normalized_queries = l2_normalize(query_embeddings)
    normalized_items = l2_normalize(item_embeddings)
    return normalized_queries @ normalized_items.transpose(0, 1)


def top_k(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Return descending Top-K values and indices for each query."""
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape [queries, items], got {scores.shape}")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be in [1, {scores.shape[1]}], got {k}")
    return torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)
