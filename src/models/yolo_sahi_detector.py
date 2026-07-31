import os
import logging
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logger = logging.getLogger(__name__)

class YoloSahiDetector:
    def __init__(self, model_path: str, mode: str = "adaptive"):
        self.model_path = model_path
        self.mode = mode.lower()
        
        self.slice_size = int(os.getenv("SLICE_SIZE", "512"))
        self.overlap = float(os.getenv("OVERLAP", "0.25"))
        self.conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.20"))
        self.adaptive_conf_threshold = float(os.getenv("ADAPTIVE_CONF_THRESHOLD", "0.40"))
        
        # Initialize Ultralytics native model for 'normal' and 'adaptive' initial pass
        if self.mode in ["normal", "adaptive"]:
            logger.info(f"Loading native YOLO model from {self.model_path}")
            self.model_native = YOLO(model_path)
            
        # Initialize SAHI model for 'sahi' and 'adaptive' fallback pass
        if self.mode in ["sahi", "adaptive"]:
            logger.info(f"Loading SAHI AutoDetectionModel from {self.model_path}")
            self.model_sahi = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=self.conf_threshold,
                device="cpu"
            )

    def predict(self, frame):
        """
        Runs inference on the frame based on the configured DETECTION_MODE.
        Returns standardized bounding boxes: list of dicts with 'xyxy', 'conf', 'cls', 'cls_name'
        """
        if self.mode == "normal":
            return self._predict_normal(frame)
        elif self.mode == "sahi":
            return self._predict_sahi(frame)
        elif self.mode == "adaptive":
            # Fast path
            boxes = self._predict_normal(frame)
            
            # Check if we should fallback
            fallback_needed = False
            if not boxes:
                fallback_needed = True
            else:
                max_conf = max(b["conf"] for b in boxes)
                if max_conf < self.adaptive_conf_threshold:
                    fallback_needed = True
                    
            if fallback_needed:
                # logger.debug("Adaptive mode: Low confidence or no detections, falling back to SAHI...")
                return self._predict_sahi(frame)
            return boxes
        else:
            raise ValueError(f"Unknown detection mode: {self.mode}")

    def _predict_normal(self, frame):
        # We don't hardcode imgsz=1280 here. YOLOv8n native imgsz (640) will be used by default 
        # or it can dynamically scale depending on the Ultralytics configuration.
        results = self.model_native(frame, conf=self.conf_threshold, iou=0.45, verbose=False)
        boxes_out = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = self.model_native.names
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                cls_name = names[cls_id]
                boxes_out.append({
                    "xyxy": xyxy,
                    "conf": conf,
                    "cls": cls_id,
                    "cls_name": cls_name
                })
        return boxes_out

    def _predict_sahi(self, frame):
        result = get_sliced_prediction(
            frame,
            self.model_sahi,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            verbose=0
        )
        
        boxes_out = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy() # list [x1, y1, x2, y2]
            conf = float(obj.score.value)
            cls_id = obj.category.id
            cls_name = obj.category.name
            boxes_out.append({
                "xyxy": bbox,
                "conf": conf,
                "cls": cls_id,
                "cls_name": cls_name
            })
        return boxes_out
