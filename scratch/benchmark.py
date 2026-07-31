import time
import cv2
import sys
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import warnings
warnings.filterwarnings('ignore')

image_path = r"C:\Users\Mahnoor Shabbir\Downloads\img1211_jpg.rf.56faddade9c4ea80f3bc86a5cbe4d4cf.jpg"
video_path = "test.mp4"

def benchmark_normal(model_path, imgsz):
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    for _ in range(5):
        ret, frame = cap.read()
        if not ret: break
        model(frame, imgsz=imgsz, verbose=False, conf=0.25)
        
    cap = cv2.VideoCapture(video_path)
    frames_processed = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or frames_processed >= 15:
            break
        model(frame, imgsz=imgsz, verbose=False, conf=0.25)
        frames_processed += 1
    t1 = time.time()
    cap.release()
    
    avg_latency = (t1 - t0) / frames_processed if frames_processed > 0 else 0
    fps = 1.0 / avg_latency if avg_latency > 0 else 0
    
    img = cv2.imread(image_path)
    if img is None:
        return fps, avg_latency, 0
    results = model(img, imgsz=imgsz, verbose=False, conf=0.25)
    detections = len(results[0].boxes)
    
    return fps, avg_latency, detections

def benchmark_sahi(model_path, slice_size, overlap):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.25,
        device="cpu" 
    )
    
    cap = cv2.VideoCapture(video_path)
    for _ in range(1):
        ret, frame = cap.read()
        if not ret: break
        get_sliced_prediction(
            frame,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0
        )
        
    cap = cv2.VideoCapture(video_path)
    frames_processed = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or frames_processed >= 5: 
            break
        get_sliced_prediction(
            frame,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0
        )
        frames_processed += 1
    t1 = time.time()
    cap.release()
    
    avg_latency = (t1 - t0) / frames_processed if frames_processed > 0 else 0
    fps = 1.0 / avg_latency if avg_latency > 0 else 0
    
    img = cv2.imread(image_path)
    if img is None:
        return fps, avg_latency, 0
    result = get_sliced_prediction(
        img,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0
    )
    detections = len(result.object_prediction_list)
    
    return fps, avg_latency, detections

def main():
    print("| Method | FPS | Avg Latency | Detections |")
    print("|---|---|---|---|")
    
    fps, lat, dets = benchmark_normal("models/yolov8l_best.pt", 640)
    print(f"| YOLOv8l (Normal, 640) | {fps:.2f} | {lat:.3f}s | {dets} |")
    
    fps, lat, dets = benchmark_normal("models/yolov8l_best.pt", 960)
    print(f"| YOLOv8l (Normal, 960) | {fps:.2f} | {lat:.3f}s | {dets} |")
    
    fps, lat, dets = benchmark_normal("models/yolov8n_best.pt", 640)
    print(f"| YOLOv8n (Normal, 640) | {fps:.2f} | {lat:.3f}s | {dets} |")
    
    fps, lat, dets = benchmark_normal("models/yolov8n_best.pt", 960)
    print(f"| YOLOv8n (Normal, 960) | {fps:.2f} | {lat:.3f}s | {dets} |")
    
    fps, lat, dets = benchmark_sahi("models/yolov8n_best.pt", 512, 0.25)
    print(f"| YOLOv8n (SAHI, 512, 0.25) | {fps:.2f} | {lat:.3f}s | {dets} |")
    
    fps, lat, dets = benchmark_sahi("models/yolov8n_best.pt", 640, 0.20)
    print(f"| YOLOv8n (SAHI, 640, 0.20) | {fps:.2f} | {lat:.3f}s | {dets} |")
    
    fps, lat, dets = benchmark_sahi("models/yolov8l_best.pt", 640, 0.20)
    print(f"| YOLOv8l (SAHI, 640, 0.20) | {fps:.2f} | {lat:.3f}s | {dets} |")

if __name__ == "__main__":
    main()
