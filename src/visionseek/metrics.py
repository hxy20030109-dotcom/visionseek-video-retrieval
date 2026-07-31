"""Retrieval metrics for one relevant item per query."""

from __future__ import annotations

import torch
from torch import Tensor


def retrieval_ranks(scores: Tensor, target_indices: Tensor) -> Tensor:
    """Return one-indexed rank of the relevant item for every query."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape [queries, items]")
    if target_indices.ndim != 1 or target_indices.shape[0] != scores.shape[0]:
        raise ValueError("target_indices must have shape [queries]")
    if target_indices.dtype != torch.long:
        raise ValueError("target_indices must use torch.long dtype")
    if torch.any(target_indices < 0) or torch.any(target_indices >= scores.shape[1]):
        raise ValueError("target_indices contains an out-of-range item index")

    ranking = torch.argsort(scores, dim=-1, descending=True)
    matches = ranking.eq(target_indices.unsqueeze(1))
    return matches.to(torch.int64).argmax(dim=1) + 1


def recall_at_k(
    scores: Tensor,
    target_indices: Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K percentages for one relevant item per query."""
    ranks = retrieval_ranks(scores, target_indices)
    metrics: dict[str, float] = {}
    for k in ks:
        if not 1 <= k <= scores.shape[1]:
            raise ValueError(f"Each k must be in [1, {scores.shape[1]}], got {k}")
        metrics[f"R@{k}"] = float((ranks <= k).float().mean().item() * 100.0)
    return metrics


def rank_summary(scores: Tensor, target_indices: Tensor) -> dict[str, float]:
    """Return mean and median one-indexed rank."""
    ranks = retrieval_ranks(scores, target_indices).float()
    return {
        "mean_rank": float(ranks.mean().item()),
        "median_rank": float(ranks.median().item()),
    }
