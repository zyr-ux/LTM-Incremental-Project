from pathlib import Path
from typing import Any, Dict, List
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = ROOT_DIR / "models" / "yolov8n.pt"

_detector = None

def get_detector() -> YOLO:
    global _detector
    if _detector is None:
        _detector = YOLO(str(YOLO_MODEL_PATH))
    return _detector

def detect_catalog_objects(image_input, conf: float = 0.25) -> List[Dict[str, Any]]:
    model = get_detector()
    results = model.predict(image_input, conf=conf, verbose=False)
    detections = []

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "label": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "box": [round(coord, 2) for coord in box.xyxy[0].tolist()],
                })

    return detections

def run_detection_demo():
    sample_image_url = "https://ultralytics.com/images/bus.jpg"
    print(f"Running detection on sample image ({sample_image_url})...")
    detections = detect_catalog_objects(sample_image_url, conf=0.3)
    print(f"Found {len(detections)} objects:")
    for d in detections:
        print(f" - {d['label']} (conf: {d['confidence']}) at bbox: {d['box']}")
    return detections

if __name__ == "__main__":
    run_detection_demo()
