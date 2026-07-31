"""End-to-end synthetic retrieval smoke test."""

from __future__ import annotations

import torch

from visionseek.metrics import rank_summary, recall_at_k
from visionseek.retrieval import cosine_similarity_matrix, top_k
from visionseek.routing import MarginRouter, confidence_features


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_videos = 32
    embedding_dim = 64

    video_embeddings = torch.randn(num_videos, embedding_dim, device=device)
    query_embeddings = video_embeddings + 0.02 * torch.randn_like(video_embeddings)
    target_indices = torch.arange(num_videos, dtype=torch.long, device=device)

    scores = cosine_similarity_matrix(query_embeddings, video_embeddings)
    metrics = recall_at_k(scores, target_indices, ks=(1, 5, 10))
    summary = rank_summary(scores, target_indices)

    candidate_scores, candidate_indices = top_k(scores, k=10)
    features = confidence_features(candidate_scores)
    route_mask = MarginRouter(margin_threshold=0.05)(candidate_scores)

    assert scores.shape == (num_videos, num_videos)
    assert candidate_indices.shape == (num_videos, 10)
    assert features.shape == (num_videos, 4)
    assert metrics["R@1"] >= 90.0

    print("Synthetic VisionSeek smoke test")
    print(f"device: {device}")
    print(f"scores: {tuple(scores.shape)}")
    print(f"router features: {tuple(features.shape)}")
    print(f"metrics: {metrics | summary}")
    print(f"queries routed: {int(route_mask.sum())}/{num_videos}")


if __name__ == "__main__":
    main()
