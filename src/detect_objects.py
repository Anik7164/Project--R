
import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "videos" / "test1.mp4"
MODEL_PATH = ROOT / "yolo26n.pt"
OUTPUT_DIR = ROOT / "outputs" / "detection"
OUTPUT_PATH = OUTPUT_DIR / "test1_detected.mp4"


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

    frame_number = 0
    detected_frames = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(
                source=frame,
                device=device,
                imgsz=640,
                conf=0.25,
                verbose=False,
            )

            result = results[0]
            annotated_frame = result.plot()

            if result.boxes is not None and len(result.boxes) > 0:
                detected_frames += 1

            writer.write(annotated_frame)
            frame_number += 1

            if frame_number % 30 == 0:
                print(f"Processed {frame_number}/{total_frames} frames")

    finally:
        cap.release()
        writer.release()

    print("\nDetection finished!")
    print(f"Frames processed: {frame_number}")
    print(f"Frames with detections: {detected_frames}")
    print(f"Output video: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
