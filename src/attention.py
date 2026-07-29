import cv2
import numpy as np
from ultralytics import YOLO

class AttentionMonitor:
    def __init__(self, model_path="models/phone-classification.pt"):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        
    def process_frame(self, frame):
        """
        Processes a single BGR frame, detects objects, and draws bounding boxes.
        Returns the annotated frame.
        """
        # Run inference on the frame
        results = self.model(frame, verbose=False)
        
        # Plot the results on the frame
        annotated_frame = results[0].plot()
        return annotated_frame

    def predict(self, image):
        """
        Run prediction on a PIL Image or numpy array.
        """
        results = self.model(image, verbose=False)
        return results[0]
