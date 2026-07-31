import pytest
import torch

from visionseek.metrics import rank_summary, recall_at_k
from visionseek.retrieval import cosine_similarity_matrix, l2_normalize, top_k


def test_l2_normalize_produces_unit_vectors() -> None:
    embeddings = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    normalized = l2_normalize(embeddings)
    assert torch.allclose(normalized.norm(dim=-1), torch.ones(2))


def test_cosine_similarity_retrieves_matching_items() -> None:
    items = torch.eye(3)
    queries = torch.eye(3)
    scores = cosine_similarity_matrix(queries, items)
    _, indices = top_k(scores, k=1)
    assert torch.equal(indices.squeeze(1), torch.arange(3))


def test_metrics_for_perfect_ranking() -> None:
    scores = torch.eye(3)
    targets = torch.arange(3, dtype=torch.long)
    assert recall_at_k(scores, targets, ks=(1, 2)) == {"R@1": 100.0, "R@2": 100.0}
    assert rank_summary(scores, targets) == {"mean_rank": 1.0, "median_rank": 1.0}


def test_similarity_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        cosine_similarity_matrix(torch.randn(2, 4), torch.randn(3, 5))
