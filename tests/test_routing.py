import pytest
import torch

from visionseek.routing import LearnedRouter, MarginRouter, confidence_features


def test_confidence_features_have_expected_values() -> None:
    scores = torch.tensor([[4.0, 1.0, 0.0], [1.0, 0.9, 0.8]])
    features = confidence_features(scores)

    assert features.shape == (2, 4)
    assert torch.allclose(features[:, 0], torch.tensor([4.0, 1.0]))
    assert torch.allclose(features[:, 1], torch.tensor([3.0, 0.1]))
    assert features[0, 2] < features[1, 2]


def test_margin_router_routes_only_uncertain_query() -> None:
    scores = torch.tensor([[4.0, 1.0, 0.0], [1.0, 0.9, 0.8]])
    routed = MarginRouter(margin_threshold=0.5)(scores)
    assert torch.equal(routed, torch.tensor([False, True]))


def test_learned_router_output_shape() -> None:
    router = LearnedRouter(input_dim=4, hidden_dim=8)
    logits = router(torch.randn(5, 4))
    assert logits.shape == (5,)


def test_confidence_features_require_two_candidates() -> None:
    with pytest.raises(ValueError, match="At least two candidates"):
        confidence_features(torch.randn(3, 1))
