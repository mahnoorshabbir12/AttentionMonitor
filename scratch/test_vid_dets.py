import cv2
from ultralytics import YOLO

def main():
    model = YOLO("models/exp.onnx")
    cap = cv2.VideoCapture("test.mp4")
    
    frame_count = 0
    found_any = False
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 5: # just test first 5 frames
            break
            
        print(f"--- Frame {frame_count} ---")
        
        # Test with very low confidence
        results = model(frame, conf=0.01, verbose=False)
        boxes = results[0].boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                c = box.conf.item()
                cls = int(box.cls.item())
                xyxy = box.xyxy.cpu().numpy().tolist()
                print(f"Detected Cls: {cls}, Conf: {c:.4f}, Box: {xyxy}")
                found_any = True
        else:
            print("No detections at all.")
            
        frame_count += 1

    cap.release()
    
    if not found_any:
        print("SUMMARY: The model detected absolutely nothing in the first 5 frames.")

if __name__ == "__main__":
    main()
