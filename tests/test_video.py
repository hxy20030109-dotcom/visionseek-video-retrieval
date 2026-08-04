import pytest

from visionseek.video import VideoMetadata, uniform_frame_indices


def test_uniform_frame_indices_cover_the_complete_video() -> None:
    indices = uniform_frame_indices(frame_count=24, num_frames=4)
    assert indices == [0, 7, 15, 23]


def test_uniform_frame_indices_keep_a_fixed_output_count() -> None:
    indices = uniform_frame_indices(frame_count=2, num_frames=4)
    assert len(indices) == 4
    assert indices[0] == 0
    assert indices[-1] == 1


@pytest.mark.parametrize(
    ("frame_count", "num_frames"),
    [
        (0, 4),
        (24, 0),
    ],
)
def test_uniform_frame_indices_reject_invalid_counts(
    frame_count: int,
    num_frames: int,
) -> None:
    with pytest.raises(ValueError):
        uniform_frame_indices(frame_count=frame_count, num_frames=num_frames)


def test_video_metadata_duration() -> None:
    metadata = VideoMetadata(frame_count=24, fps=8.0, width=320, height=240)
    assert metadata.duration_seconds == 3.0
