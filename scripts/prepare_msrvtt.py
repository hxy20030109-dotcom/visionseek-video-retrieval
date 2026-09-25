"""Download MSR-VTT and extract the standard 1K test split."""

from __future__ import annotations

import csv
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "friedrichor/MSR-VTT"
VIDEO_ARCHIVE = "MSRVTT_Videos.zip"
TEST_ANNOTATIONS = "msrvtt_test_1k.json"
DEFAULT_DATA_ROOT = Path(os.environ.get("VISIONSEEK_DATA_ROOT", "data/raw"))


def load_split(annotation_path: Path) -> tuple[set[str], dict[str, list[str]]]:
    """Read video IDs and captions from common MSR-VTT JSON structures."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    video_ids: set[str] = set()
    captions: dict[str, list[str]] = defaultdict(list)

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or "video_id" not in item:
                continue
            video_id = str(item["video_id"])
            video_ids.add(video_id)
            caption = item.get("caption") or item.get("sentence")
            if caption:
                captions[video_id].append(str(caption))

    elif isinstance(data, dict):
        for video in data.get("videos", []):
            if isinstance(video, dict) and "video_id" in video:
                video_ids.add(str(video["video_id"]))

        for sentence in data.get("sentences", []):
            if not isinstance(sentence, dict) or "video_id" not in sentence:
                continue
            video_id = str(sentence["video_id"])
            caption = sentence.get("caption") or sentence.get("sentence")
            video_ids.add(video_id)
            if caption:
                captions[video_id].append(str(caption))

    if not video_ids:
        raise ValueError(f"Unsupported annotation structure: {annotation_path}")
    return video_ids, captions


def extract_test_videos(archive_path: Path, video_ids: set[str], output_dir: Path) -> int:
    """Extract only archive members belonging to the requested split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or Path(member.filename).stem not in video_ids:
                continue
            target_path = output_dir / Path(member.filename).name
            if not target_path.exists():
                with archive.open(member) as source, target_path.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            extracted += 1

    return extracted


def write_manifest(
    manifest_path: Path,
    video_ids: set[str],
    captions: dict[str, list[str]],
    video_dir: Path,
) -> None:
    """Write one row per caption for retrieval evaluation."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video_id", "video_path", "caption"])
        writer.writeheader()
        for video_id in sorted(video_ids):
            matches = sorted(video_dir.glob(f"{video_id}.*"))
            if not matches:
                continue
            for caption in captions.get(video_id, []):
                writer.writerow(
                    {
                        "video_id": video_id,
                        "video_path": str(matches[0].resolve()),
                        "caption": caption,
                    }
                )


def main() -> None:
    dataset_root = DEFAULT_DATA_ROOT / "msrvtt"
    downloads_dir = dataset_root / "downloads"
    test_video_dir = dataset_root / "test_1k" / "videos"
    manifest_path = dataset_root / "test_1k" / "captions.csv"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MSR-VTT test annotations...")
    annotation_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=TEST_ANNOTATIONS,
            local_dir=downloads_dir,
        )
    )
    video_ids, captions = load_split(annotation_path)
    print(f"Test video IDs: {len(video_ids)}")

    print("Downloading the video archive (about 2.2 GB)...")
    archive_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=VIDEO_ARCHIVE,
            local_dir=downloads_dir,
        )
    )

    extracted = extract_test_videos(archive_path, video_ids, test_video_dir)
    write_manifest(manifest_path, video_ids, captions, test_video_dir)

    print(f"Extracted test videos: {extracted}/{len(video_ids)}")
    print(f"Videos: {test_video_dir}")
    print(f"Manifest: {manifest_path}")

    if extracted != len(video_ids):
        raise RuntimeError("The extracted video count does not match the test split.")


if __name__ == "__main__":
    main()
