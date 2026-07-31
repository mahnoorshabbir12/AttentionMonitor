import cv2
import time
import os
import numpy as np
from src.models.yolo_sahi_detector import YoloSahiDetector

class TrackedDetection:
    def __init__(self, xyxy, cls_name, conf):
        self.xyxy = np.array(xyxy)
        self.cls_name = cls_name
        self.conf = conf
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.hit_count = 1
        self.miss_count = 0

    def update(self, xyxy, cls_name, conf):
        alpha = 0.85
        self.xyxy = (alpha * np.array(xyxy) + (1 - alpha) * self.xyxy).astype(int)
        self.cls_name = cls_name
        self.conf = conf
        self.last_seen = time.time()
        self.miss_count = 0
        self.hit_count += 1

    def mark_missed(self):
        """Mark this detection as not found in the current frame."""
        self.miss_count += 1


class AttentionMonitor:
    def __init__(self, model_path="models/yolov8n_best.pt"):
        import threading
        self.lock = threading.Lock()
        mode = os.getenv("DETECTION_MODE", "adaptive")
        sahi_enabled = os.getenv("SAHI_ENABLED", "True").lower() == "true"
        if not sahi_enabled:
            mode = "normal"
            
        self.detector = YoloSahiDetector(model_path, mode=mode)

        self.tracked = []
        self.iou_threshold = 0.3
        self.max_miss_frames = 8

    def reset(self):
        with self.lock:
            self.tracked = []
        
    def _match_and_update(self, new_detections):
        """
        Match new detections to existing tracked objects using IoU.
        """
        matched_new = set()
        matched_tracked = set()

        with self.lock:
            # Simple greedy matching
            for t_idx, track in enumerate(self.tracked):
                best_iou = 0
                best_n_idx = -1

                for n_idx, nd in enumerate(new_detections):
                    if n_idx in matched_new:
                        continue
                    
                    iou = self._calculate_iou(track.xyxy, nd['xyxy'])
                    if iou > best_iou:
                        best_iou = iou
                        best_n_idx = n_idx

                if best_iou > self.iou_threshold:
                    # Match found
                    track.update(new_detections[best_n_idx]['xyxy'], 
                                 new_detections[best_n_idx]['cls_name'],
                                 new_detections[best_n_idx]['conf'])
                    matched_new.add(best_n_idx)
                    matched_tracked.add(t_idx)

            # Increment miss count for unmatched tracks
            for t_idx, track in enumerate(self.tracked):
                if t_idx not in matched_tracked:
                    track.mark_missed()

            # Add new tracks
            for n_idx, nd in enumerate(new_detections):
                if n_idx not in matched_new:
                    self.tracked.append(TrackedDetection(nd['xyxy'], nd['cls_name'], nd['conf']))

            # Remove dead tracks (only based on miss_count now)
            self.tracked = [
                t for t in self.tracked 
                if t.miss_count <= self.max_miss_frames
            ]

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_frame(self, frame, run_inference=True):
        """
        Process a single frame. Runs inference if requested.
        Returns the annotated frame.
        """
        annotated_frame = frame.copy()

        if run_inference:
            boxes = self.detector.predict(frame)
            self._match_and_update(boxes)
        else:
            pass

        with self.lock:
            tracks_copy = list(self.tracked)

        # Draw currently tracked bounding boxes
        color = (255, 0, 0)  # Blue
        for track in tracks_copy:
            x1, y1, x2, y2 = track.xyxy
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            label = track.cls_name

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - 24), (int(x1) + w, int(y1)), color, -1)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame
