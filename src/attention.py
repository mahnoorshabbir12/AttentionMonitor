import cv2
import numpy as np
from ultralytics import YOLO
import time


def _compute_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


class TrackedDetection:
    """Tracks a single detection across frames."""
    def __init__(self, xyxy, cls_name, conf):
        self.xyxy = xyxy
        self.cls_name = cls_name
        self.conf = conf
        self.last_seen = time.time()
        self.miss_count = 0         # Consecutive frames without a match
        self.hit_count = 1          # Total frames this was detected

    def update(self, xyxy, cls_name, conf):
        """Update with a new matching detection."""
        # Smooth the box position to reduce jitter
        alpha = 0.6  # Weight for new detection (0-1)
        self.xyxy = (alpha * xyxy + (1 - alpha) * self.xyxy).astype(int)
        self.cls_name = cls_name
        self.conf = conf
        self.last_seen = time.time()
        self.miss_count = 0
        self.hit_count += 1

    def mark_missed(self):
        """Mark this detection as not found in the current frame."""
        self.miss_count += 1


class AttentionMonitor:
    def __init__(self, model_path="models/exp.pt"):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)

        # Per-detection tracking
        self.tracked = []                # List of TrackedDetection objects
        self.iou_threshold = 0.3         # IoU threshold to match detections across frames
        self.max_miss_frames = 15        # Keep showing a box for this many missed frames
        self.max_miss_seconds = 1.0      # Max time to persist a missed detection

    def _match_and_update(self, new_detections):
        """
        Match new detections to existing tracked objects using IoU.
        Unmatched new detections become new tracks.
        Unmatched old tracks get their miss count incremented.
        """
        now = time.time()
        used_new = set()
        used_old = set()

        # Build IoU matrix
        if self.tracked and new_detections:
            for old_idx, trk in enumerate(self.tracked):
                best_iou = 0
                best_new_idx = -1
                for new_idx, (xyxy, cls_name, conf) in enumerate(new_detections):
                    if new_idx in used_new:
                        continue
                    iou = _compute_iou(trk.xyxy, xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_new_idx = new_idx

                if best_iou >= self.iou_threshold and best_new_idx >= 0:
                    # Match found — update tracked detection
                    xyxy, cls_name, conf = new_detections[best_new_idx]
                    trk.update(xyxy, cls_name, conf)
                    used_new.add(best_new_idx)
                    used_old.add(old_idx)

        # Mark unmatched old tracks as missed
        for old_idx, trk in enumerate(self.tracked):
            if old_idx not in used_old:
                trk.mark_missed()

        # Create new tracks for unmatched new detections
        for new_idx, (xyxy, cls_name, conf) in enumerate(new_detections):
            if new_idx not in used_new:
                self.tracked.append(TrackedDetection(xyxy, cls_name, conf))

        # Remove tracks that have been missing too long
        self.tracked = [
            trk for trk in self.tracked
            if trk.miss_count <= self.max_miss_frames
        ]

    def process_frame(self, frame, run_inference=True):
        """
        Processes a single BGR frame, detects objects, and draws bounding boxes.
        Uses per-detection tracking to prevent flickering.
        If run_inference is False, simply draws the last known bounding boxes.
        Returns the annotated frame.
        """
        if run_inference:
            # Run inference (lowered conf to 0.25 to catch more people)
            results = self.model(frame, conf=0.25, verbose=False)

            # Extract current detections
            current_boxes = results[0].boxes
            new_detections = []
            if current_boxes is not None and len(current_boxes) > 0:
                for box in current_boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls_name = results[0].names[cls]
                    new_detections.append((xyxy, cls_name, conf))

            # Match with existing tracks
            self._match_and_update(new_detections)

        # Draw all active tracked detections
        annotated_frame = frame.copy()
        
        # Dynamically scale thickness and font size based on frame resolution
        # A 4K frame needs much thicker lines and larger text to be visible
        height = frame.shape[0]
        scale = max(1.0, height / 720.0)
        base_thickness = max(2, int(2 * scale))
        font_scale = max(0.7, 0.7 * scale)

        for trk in self.tracked:
            x1, y1, x2, y2 = trk.xyxy

            # Fade the color slightly for persisted (missed) detections
            if trk.miss_count == 0:
                color = (0, 100, 255)       # Orange for active detections
                thickness = base_thickness
            else:
                # Gradually fade as miss count increases
                fade = max(0.4, 1.0 - (trk.miss_count / self.max_miss_frames))
                color = (
                    int(0 * fade),
                    int(100 * fade),
                    int(255 * fade),
                )
                thickness = base_thickness

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label = trk.cls_name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, base_thickness)
            cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), base_thickness)

        return annotated_frame

    def predict(self, image):
        """
        Run prediction on a PIL Image or numpy array.
        """
        results = self.model(image, conf=0.25, verbose=False)
        return results[0]
