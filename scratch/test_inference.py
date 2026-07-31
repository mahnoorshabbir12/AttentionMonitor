import cv2
import numpy as np
from ultralytics import YOLO
import time

def main():
    model = YOLO("models/exp.onnx")
    print("Model:", model.model.__class__.__name__)
    print("Task:", model.task)
    
    # Create a dummy image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    t0 = time.time()
    results = model(img, verbose=False)
    t1 = time.time()
    
    print("Inference time:", t1 - t0)
    print("imgsz used:", results[0].orig_shape, results[0].orig_img.shape)

if __name__ == "__main__":
    main()
