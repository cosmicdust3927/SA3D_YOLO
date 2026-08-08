from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = REPO_ROOT / "yolov8x-seg.pt"


def load_yolo_model():
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {YOLO_MODEL_PATH}. "
            "Place yolov8x-seg.pt in the repository root."
        )
    return YOLO(str(YOLO_MODEL_PATH))
