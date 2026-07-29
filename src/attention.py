import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time


class AttentionMonitor:
    def __init__(self, model_path="models/phone-classification.pt"):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        
        # Detection persistence: keep showing boxes for a few frames after they disappear
        self.last_detections = []        # List of (box, cls, conf) from last successful detection
        self.last_detection_time = 0     # Timestamp of last successful detection
        self.persistence_seconds = 0.6   # How long to keep showing old boxes (in seconds)
        
    def process_frame(self, frame):
        """
        Processes a single BGR frame, detects objects, and draws bounding boxes.
        Uses detection persistence to prevent flickering.
        Returns the annotated frame.
        """
        # Run inference at the model's native resolution (default 640)
        results = self.model(frame, conf=0.4, verbose=False)
        
        current_boxes = results[0].boxes
        now = time.time()
        
        if current_boxes is not None and len(current_boxes) > 0:
            # We have fresh detections — update our cache
            self.last_detections = []
            for box in current_boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_name = results[0].names[cls]
                self.last_detections.append((xyxy, cls_name, conf))
            self.last_detection_time = now
            
            # Use YOLO's built-in plot (no confidence scores shown)
            annotated_frame = results[0].plot(conf=False)
        elif (now - self.last_detection_time) < self.persistence_seconds and self.last_detections:
            # No current detection, but we had one recently — draw the cached boxes
            annotated_frame = frame.copy()
            for (xyxy, cls_name, conf) in self.last_detections:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                
                # Draw label background
                label = cls_name
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (255, 100, 0), -1)
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No detection and persistence expired — show clean frame
            annotated_frame = frame.copy()
            self.last_detections = []
            
        return annotated_frame

    def predict(self, image):
        """
        Run prediction on a PIL Image or numpy array.
        """
        results = self.model(image, verbose=False)
        return results[0]
