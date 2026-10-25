
import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ingestion"


def inspect_video(video_path: Path, sample_interval: float = 10.0) -> dict:
    video_path = video_path.resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if sample_interval <= 0:
        raise ValueError("Sample interval must be greater than zero.")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0.0

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        frame_dir = OUTPUT_DIR / f"{video_path.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        saved_frames = []
        timestamp = 0.0
        index = 0

        while timestamp < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = capture.read()

            if success:
                frame_name = f"frame_{timestamp:09.2f}s.jpg"
                frame_path = frame_dir / frame_name

                if cv2.imwrite(str(frame_path), frame):
                    saved_frames.append({
                        "timestamp_seconds": round(timestamp, 2),
                        "path": str(frame_path.relative_to(PROJECT_ROOT)),
                    })

            index += 1
            timestamp = index * sample_interval

        metadata = {
            "video": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2),
            "sample_interval_seconds": sample_interval,
            "saved_frame_count": len(saved_frames),
            "frames": saved_frames,
        }

        metadata_path = OUTPUT_DIR / f"{video_path.stem}_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print(f"Video: {video_path.name}")
        print(f"Resolution: {width} x {height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Frames saved: {len(saved_frames)}")
        print(f"Metadata: {metadata_path}")

        return metadata

    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a football video and extract sample frames."
    )
    parser.add_argument("video", help="Path to a recorded video")
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between sampled frames (default: 10)",
    )
    args = parser.parse_args()

    inspect_video(Path(args.video), args.interval)
