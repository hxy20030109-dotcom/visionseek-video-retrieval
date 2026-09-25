import pytest
import torch

from visionseek.embedding import max_frame_similarity, mean_pool_frame_embeddings


def test_mean_pool_returns_one_unit_vector_per_video() -> None:
    frame_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )

    pooled = mean_pool_frame_embeddings(frame_embeddings)

    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled.norm(dim=-1), torch.ones(2))
    assert torch.allclose(pooled, torch.eye(2))


def test_max_frame_similarity_finds_the_best_frame_in_each_video() -> None:
    queries = torch.tensor([[1.0, 0.0]])
    frame_embeddings = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ]
    )

    scores = max_frame_similarity(queries, frame_embeddings)

    assert scores.shape == (1, 2)
    assert torch.allclose(scores, torch.tensor([[1.0, 0.0]]))


def test_mean_pool_rejects_a_2d_tensor() -> None:
    with pytest.raises(ValueError, match="videos, frames, dim"):
        mean_pool_frame_embeddings(torch.randn(2, 4))
