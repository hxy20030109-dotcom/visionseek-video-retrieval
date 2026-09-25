"""Scan local videos and create an annotation manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from visionseek.video import discover_video_files, probe_video

FIELDNAMES = [
    "video_id",
    "relative_path",
    "duration_seconds",
    "fps",
    "frame_count",
    "width",
    "height",
    "query_en",
    "query_zh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/raw/local_videos"),
        help="Directory containing self-owned video files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/local_videos_manifest.csv"),
        help="CSV manifest to create or update.",
    )
    return parser.parse_args()


def load_existing_annotations(manifest_path: Path) -> dict[str, tuple[str, str]]:
    """Preserve text annotations when the manifest is regenerated."""
    if not manifest_path.is_file():
        return {}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            row["relative_path"]: (row.get("query_en", ""), row.get("query_zh", ""))
            for row in csv.DictReader(file)
        }


def main() -> None:
    args = parse_args()
    args.video_dir.mkdir(parents=True, exist_ok=True)

    video_paths = discover_video_files(args.video_dir)
    if not video_paths:
        print(f"No videos found in: {args.video_dir.resolve()}")
        print("Copy 5-10 self-owned MP4 files into that folder, then run this command again.")
        return

    existing_annotations = load_existing_annotations(args.manifest)
    rows: list[dict[str, str | int | float]] = []

    for video_index, video_path in enumerate(video_paths):
        metadata = probe_video(video_path)
        relative_path = video_path.relative_to(args.video_dir).as_posix()
        query_en, query_zh = existing_annotations.get(relative_path, ("", ""))
        rows.append(
            {
                "video_id": f"local_{video_index:04d}",
                "relative_path": relative_path,
                "duration_seconds": round(metadata.duration_seconds, 3),
                "fps": round(metadata.fps, 3),
                "frame_count": metadata.frame_count,
                "width": metadata.width,
                "height": metadata.height,
                "query_en": query_en,
                "query_zh": query_zh,
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Imported videos: {len(rows)}")
    print(f"Manifest: {args.manifest.resolve()}")
    for row in rows:
        print(
            f"- {row['relative_path']}: {row['duration_seconds']}s, "
            f"{row['width']}x{row['height']}"
        )


if __name__ == "__main__":
    main()
