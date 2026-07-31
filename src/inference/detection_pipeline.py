import cv2
import os
from src.models.yolo_sahi_detector import YoloSahiDetector


class AttentionMonitor:
    def __init__(self, model_path="models/yolov8n_best.pt"):
        mode = os.getenv("DETECTION_MODE", "normal")
        sahi_enabled = os.getenv("SAHI_ENABLED", "False").lower() == "true"
        if not sahi_enabled:
            mode = "normal"

        self.detector = YoloSahiDetector(model_path, mode=mode)

    def reset(self):
        pass

    def process_frame(self, frame, run_inference=True):
        annotated_frame = frame.copy()

        if not run_inference:
            return annotated_frame

        boxes = self.detector.predict(frame)

        color = (255, 0, 0)  # Blue (BGR)
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            label = box["cls_name"]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - 24), (int(x1) + w, int(y1)), color, -1)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame
