"""Routing policies for deciding when second-stage reranking is useful."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def confidence_features(candidate_scores: Tensor, eps: float = 1e-12) -> Tensor:
    """Extract [top1, margin, entropy, std] from candidate scores.

    Args:
        candidate_scores: Tensor with shape [queries, candidates].

    Returns:
        Tensor with shape [queries, 4].
    """
    if candidate_scores.ndim != 2:
        raise ValueError("candidate_scores must have shape [queries, candidates]")
    if candidate_scores.shape[1] < 2:
        raise ValueError("At least two candidates are required to compute a score margin")

    sorted_scores = torch.sort(candidate_scores, dim=-1, descending=True).values
    top1 = sorted_scores[:, 0]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]

    probabilities = torch.softmax(candidate_scores, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    score_std = candidate_scores.std(dim=-1, unbiased=False)

    return torch.stack((top1, margin, entropy, score_std), dim=-1)


class MarginRouter(nn.Module):
    """Route low-margin queries to the reranker."""

    def __init__(self, margin_threshold: float) -> None:
        super().__init__()
        self.margin_threshold = margin_threshold

    def forward(self, candidate_scores: Tensor) -> Tensor:
        features = confidence_features(candidate_scores)
        margin = features[:, 1]
        return margin < self.margin_threshold


class LearnedRouter(nn.Module):
    """Small MLP that predicts whether reranking is worth its cost."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.network[0].in_features:
            raise ValueError(
                "features must have shape "
                f"[queries, {self.network[0].in_features}], got {tuple(features.shape)}"
            )
        return self.network(features).squeeze(-1)
