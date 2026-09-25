"""Video metadata and frame-sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

SUPPORTED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata required by the first-stage video pipeline."""

    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        """Return video duration in seconds."""
        return self.frame_count / self.fps


def discover_video_files(video_dir: str | Path) -> list[Path]:
    """Return supported video files in a directory, sorted by filename."""
    directory = Path(video_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    )


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Read and validate basic video metadata without decoding every frame."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if frame_count < 1 or fps <= 0 or width < 1 or height < 1:
        raise ValueError(f"Video has invalid metadata: {path}")

    return VideoMetadata(
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )


def uniform_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    """Return fixed-count frame indices spread across the complete video."""
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    return np.linspace(0, frame_count - 1, num_frames, dtype=int).tolist()


def sample_uniform_frames(
    video_path: str | Path,
    num_frames: int,
) -> tuple[Tensor, list[int], VideoMetadata]:
    """Decode uniformly spaced RGB frames.

    Returns:
        A uint8 tensor with shape [frames, channels, height, width], the selected
        frame indices, and source-video metadata.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count < 1 or fps <= 0 or width < 1 or height < 1:
        capture.release()
        raise ValueError(f"Video has invalid metadata: {path}")

    metadata = VideoMetadata(
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )
    frame_indices = uniform_frame_indices(frame_count, num_frames)
    frames: list[Tensor] = []

    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, bgr_frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(f"Could not decode frame {frame_index} from {path}")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(rgb_frame).permute(2, 0, 1).contiguous()
        frames.append(frame)

    capture.release()
    return torch.stack(frames), frame_indices, metadata
